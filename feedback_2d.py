import base64
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
import imagehash
import numpy as np
from PIL import Image

from constants import constants
from elasticsearch_constants import es_constant
from feedback_base import RocchioFeedbackBase
from model_loader_utils import base64_to_image


class RocchioFeedback2D(RocchioFeedbackBase):
    def __init__(self, elasticsearch_db, similarity_threshold=0.98):
        super().__init__(elasticsearch_db, similarity_threshold)

    def _load_original_image_b64(self, project_id: Any) -> str:
        res = self.elasticsearch_db.find(
            indice_name=constants.ZONE_ENCODED_DATA_PHYSICAL_OBJECT,
            field_name="object_id",
            value=project_id,
        )
        data = res["hits"]["hits"][0]["_source"]["data"]
        dtype = res["hits"]["hits"][0]["_source"]["type"]
        return data, dtype

    @staticmethod
    def compute_phash_vector_from_image(
        img: Image.Image,
        hash_size: int = 32,
        normalise: bool = True,
        dtype: type = np.float32,
        binary_output: bool = True,
    ) -> np.ndarray:
        """Compute phash vector from a PIL Image."""
        img_hash = imagehash.phash(img, hash_size=hash_size)
        bits = bin(int(str(img_hash), 16))[2:].zfill(hash_size * hash_size)
        v = np.array([int(b) for b in bits], dtype=dtype if binary_output else int)

        if normalise:
            v = v / (np.linalg.norm(v) + 1e-8)

        return v

    @staticmethod
    def compute_phash_vector(
        img_b64: str,
        hash_size: int = 32,
        normalise: bool = True,
        dtype: type = np.float32,
        binary_output: bool = True,
    ) -> np.ndarray:
        """Compute phash from base64-encoded image."""
        img = base64_to_image(img_b64)
        return RocchioFeedback2D.compute_phash_vector_from_image(
            img,
            hash_size=hash_size,
            normalise=normalise,
            dtype=dtype,
            binary_output=binary_output,
        )

    @staticmethod
    def infer_phash_from_data(
        data: str,
        dtype: str,
        hash_size: int = 32,
        normalise: bool = True,
        vector_dtype: type = np.float32,
        binary_output: bool = True,
    ) -> np.ndarray:
        """Infer phash vector from data, either image or PDF."""

        if dtype not in ["image", "pdf"]:
            raise NotImplementedError(f"dtype={dtype} not supported")

        if dtype == "image":
            return RocchioFeedback2D.compute_phash_vector(
                data,
                hash_size=hash_size,
                normalise=normalise,
                dtype=vector_dtype,
                binary_output=binary_output,
            )

        if dtype == "pdf":
            # Step 1: Open PDF from base64
            pdf_bytes = base64.b64decode(data)
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            if pdf_doc.page_count == 0:
                raise ValueError("PDF has no pages")

            # Step 2: Render only the first page to a PIL image
            page = pdf_doc[0]
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Step 3: Compute phash directly from image
            return RocchioFeedback2D.compute_phash_vector_from_image(
                img,
                hash_size=hash_size,
                normalise=normalise,
                dtype=vector_dtype,
                binary_output=binary_output,
            )

    def _apply_rocchio_decision(
        self,
        query_vectors: List[np.ndarray],
        feedback_list: List[Tuple],
        embedding_index: str,
        org_id: str,
        object_id: str,
        phash_2d: Optional[List],
        matches: Dict,
    ):
        if feedback_list:
            pos_vecs, neg_vecs = self._extract_feedback_vectors(
                feedback_list,
                embedding_index,
                embedding_vector_field="embedding_vector_v2",
                routing_key=f"v2___{str(org_id)}",
            )

            updated, pos_mean, neg_mean = self.rocchio_update(
                query_vectors, pos_vecs, neg_vecs, return_intermediates=True
            )

            pos_mean = self.add_epsilon(pos_mean)
            neg_mean = self.add_epsilon(neg_mean)

            interactions_save_in_es = self.convert_feedback_to_interactions(
                feedback_list,
                es_constant.SCHEMA_ROCCHIO_HISTORY_PHYSICAL_OBJECT["mappings"][
                    "properties"
                ]["interactions"]["properties"].keys(),
            )

            if (
                matches["similarity_score"] is not None
                and matches["similarity_score"] >= self.similarity_threshold
            ):
                # UPDATE MODE
                doc_id = matches["doc_id"]

                # update_body = {
                #     "doc": {
                #         "rocchio_pos_vec_2d": pos_mean.tolist(),
                #         "rocchio_neg_vec_2d": neg_mean.tolist(),
                #         "interactions": interactions_save_in_es,
                #         "updated_at": datetime.now(),
                #     }
                # }

                update_body = {
                    "script": {
                        "lang": "painless",
                        "source": """
                            if (ctx._source.interactions == null) {
                                ctx._source.interactions = [];
                            }

                            Map seen = new HashMap();

                            // existing interactions
                            for (item in ctx._source.interactions) {
                                String key = item.physical_id.toString();
                                seen.put(key, item);
                            }

                            // incoming interactions
                            for (item in params.interactions) {
                                String key = item.physical_id.toString();

                                if (seen.containsKey(key)) {
                                    def oldItem = seen.get(key);

                                    // if reaction is the same -> keep old one
                                    if (oldItem.reaction == item.reaction) {
                                        continue;
                                    }
                                }

                                // otherwise overwrite (latest wins)
                                seen.put(key, item);
                            }

                            ctx._source.interactions = new ArrayList(seen.values());

                            ctx._source.rocchio_pos_vec_2d = params.rocchio_pos_vec_2d;
                            ctx._source.rocchio_neg_vec_2d = params.rocchio_neg_vec_2d;
                            ctx._source.updated_at = params.updated_at;
                        """,
                        "params": {
                            "interactions": interactions_save_in_es,
                            "rocchio_pos_vec_2d": pos_mean.tolist(),
                            "rocchio_neg_vec_2d": neg_mean.tolist(),
                            "updated_at": datetime.now(),
                        },
                    }
                }

                self.elasticsearch_db.client.update(
                    index=constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
                    id=doc_id,
                    body=update_body,
                    refresh="wait_for",
                )
            else:
                # CREATE MODE
                new_doc = {
                    "id": object_id,
                    "phash_2d": phash_2d,
                    "rocchio_pos_vec_2d": pos_mean.tolist(),
                    "rocchio_neg_vec_2d": neg_mean.tolist(),
                    "type": "2d",
                    "interactions": interactions_save_in_es,
                    "org_id": org_id,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                }

                self.elasticsearch_db.client.index(
                    index=constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
                    document=new_doc,
                    refresh="wait_for",
                )

            return updated

        if (
            matches["similarity_score"] is not None
            and matches["similarity_score"] >= self.similarity_threshold
        ):
            rocchio_pos, rocchio_neg = [], []

            # Only append vectors if they exist in ES
            if matches["rocchio_pos_vec_2d"] is not None:
                rocchio_pos.append(matches["rocchio_pos_vec_2d"])

            if matches["rocchio_neg_vec_2d"] is not None:
                rocchio_neg.append(matches["rocchio_neg_vec_2d"])

            # If no vectors are present, return original vectors
            if not rocchio_pos and not rocchio_neg:
                return query_vectors

            return self.rocchio_update(query_vectors, rocchio_pos, rocchio_neg)

        return query_vectors

    def retrieve_matches(self, project_id: str, org_id: str) -> List:
        """
        Retrieve a match using project_id
        """

        data, dtype = self._load_original_image_b64(project_id)
        hash_vector = self.infer_phash_from_data(data, dtype)

        matches = self.find_similar_rocchio_record(
            query_vector=hash_vector,
            search_field="phash_2d",
            check_and_create=True,
            filters=[{"term": {"type": "2d"}}, {"term": {"org_id": org_id}}],
        )

        return hash_vector, matches

    def run(
        self,
        query_vectors: List[np.ndarray],
        project_id: Union[int, str],
        org_id: str,
        embedding_index: str,
        feedback_list: List[Tuple] = [],
    ) -> List[np.ndarray]:
        """Execute retrieval, feedback cleaning, and Rocchio-based query update."""

        hash_vector, matches = self.retrieve_matches(project_id, org_id)

        past_interactions = [
            (inter["physical_id"], inter["reaction"])
            for inter in matches["interactions"]
        ]

        feedback_list = self.remove_neutral_and_duplicate_negative_interactions(
            feedback_list, past_interactions
        )

        # ------------------ Unified decision logic ------------------
        return self._apply_rocchio_decision(
            query_vectors,
            feedback_list,
            embedding_index,
            org_id,
            project_id,
            hash_vector,
            matches,
        )
