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
        match: Dict,
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
                match["similarity_score"] is not None
                and match["similarity_score"] >= self.similarity_threshold
            ):
                # UPDATE MODE
                doc_id = match["doc_id"]

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
            match["similarity_score"] is not None
            and match["similarity_score"] >= self.similarity_threshold
        ):
            rocchio_pos, rocchio_neg = [], []

            # Only append vectors if they exist in ES
            if match["rocchio_pos_vec_2d"] is not None:
                rocchio_pos.append(match["rocchio_pos_vec_2d"])

            if match["rocchio_neg_vec_2d"] is not None:
                rocchio_neg.append(match["rocchio_neg_vec_2d"])

            # If no vectors are present, return original vectors
            if not rocchio_pos and not rocchio_neg:
                return query_vectors

            return self.rocchio_update(query_vectors, rocchio_pos, rocchio_neg)

        return query_vectors

    def retrieve_2d_exact_match(self, project_id: str, org_id: str) -> List:
        """
        Retrieve 2D exact match in Rocchio index using project_id.
        Filtered by org_id and similarity_threshold.
        """

        data, dtype = self._load_original_image_b64(project_id)
        hash_vector = self.infer_phash_from_data(data, dtype)

        match = self.find_similar_rocchio_record(
            query_vector=hash_vector,
            search_field="phash_2d",
            check_and_create=True,
            filters=[{"term": {"type": "2d"}}, {"term": {"org_id": org_id}}],
        )

        return hash_vector, match

    def _load_disliked_cluster_info(self, disliked_cluster_ids: List[str]) -> dict:
        """
        Return:
        {
            cluster_id: {
                "physical_id": ...,
                "rep_emb": ...,
                "version": ...
            }
        }
        """
        if not disliked_cluster_ids:
            return {}

        es = self.elasticsearch_db.client

        res = es.mget(
            index=constants.SIMILARITY_CLUSTERS,
            body={"ids": disliked_cluster_ids},
        )

        disliked_cluster_info = {}

        for doc in res["docs"]:
            if not doc.get("found"):
                continue

            src = doc["_source"]

            # Use _id if cluster_id is not stored in _source
            cluster_id = src.get("cluster_id", doc["_id"])

            if src.get("version") == "v3":
                version = "v3"
                emb_field = "embedding_vector_2d"
            elif src.get("version") == "3d":
                version = "3d"
                emb_field = "embedding_vector_3d"
            else:
                raise ValueError

            disliked_cluster_info[cluster_id] = {
                "physical_ids": src["physical_ids"],
                "rep_emb": src[emb_field],
                "version": version,
            }

        return disliked_cluster_info

    def run(
        self,
        query_vectors: List[np.ndarray],
        project_id: Union[int, str],
        org_id: str,
        embedding_index: str,
        feedback_list: List[Tuple] = [],
    ) -> List[np.ndarray]:
        """Execute retrieval, feedback cleaning, and Rocchio-based query update."""

        feedback_list = [
            (physical_id, reaction)
            for (physical_id, reaction) in feedback_list
            if reaction != 0
        ]

        # Step 1: Retrieve exact match in Rocchio index
        hash_vector, match = self.retrieve_2d_exact_match(project_id, org_id)

        # match = {"disliked_cluster_ids": ["v6bXSpwBtsGT_mLwgDRk"], "similarity_score": None,}

        print(f"match: {match}")
        # Step 2: Use match["disliked_cluster_ids"] to get all disliked_physical_ids.
        disliked_cluster_info = self._load_disliked_cluster_info(
            [item["cluster_id"] for item in match["disliked_cluster_ids"]]
        )

        # Step 3: Forward phase
        # Do clustering for remaining disliked_phys_ids and append it to the host Rocchio
        known_disliked_physical_ids = []
        for v in disliked_cluster_info.values():
            known_disliked_physical_ids.extend(v["physical_ids"])

        remaining_feedback_list = [
            (physical_id, reaction)
            for (physical_id, reaction) in feedback_list
            if physical_id not in known_disliked_physical_ids
        ]

        print(
            f"remaining_feedback_list: {remaining_feedback_list}, known_disliked_phys_ids: {known_disliked_physical_ids}"
        )

        if remaining_feedback_list:
            rem_feedback_info = (
                self.fetch_representation_embeddings_from_raijin_search_indexer(
                    physical_ids=[
                        physical_id
                        for (physical_id, reaction) in remaining_feedback_list
                    ],
                    company_id=org_id,
                )
            )  # {physical_id: {"version": str, "embedding": list}}

            disliked_cluster_info_new = {}

            for physical_id, info in rem_feedback_info.items():
                rep_vec = info["embedding"]
                version = info["version"]

                embedding_field = (
                    "embedding_vector_3D" if version == "3d" else "embedding_vector_2D"
                )
                # 1. Search for similar cluster
                res = self.elasticsearch_db.client.search(
                    index=constants.SIMILARITY_CLUSTERS,
                    size=1,
                    query={
                        "bool": {
                            "filter": [
                                {"term": {"org_id": org_id}},
                                {
                                    "term": {"type": version}
                                },  # IMPORTANT: your field is "type", not "version"
                            ],
                            "must": {
                                "knn": {
                                    "field": embedding_field,
                                    "query_vector": rep_vec,
                                    "k": 1,
                                    "num_candidates": 50,
                                }
                            },
                        }
                    },
                )

                hits = res.get("hits", {}).get("hits", [])
                print(f"hits: {hits}")
                if hits and hits[0]["_score"] >= self.similarity_threshold:
                    # Found existing cluster
                    src = hits[0]["_source"]
                    cluster_id = src.get("cluster_id", hits[0]["_id"])

                    disliked_cluster_info_new[cluster_id] = {
                        "physical_ids": src["physical_ids"],
                        "embedding": np.array(src[embedding_field], dtype=np.float32),
                        "version": src.get("version"),
                    }
                else:

                    # Create a new cluster and do the clustering
                    created_doc = self._create_cluster_from_vector(
                        rep_vec, version=version, org_id=org_id
                    )

                    disliked_cluster_info_new[created_doc["doc_id"]] = {
                        "physical_ids": created_doc["physical_ids"],
                        "embedding": created_doc["embedding"],
                        "version": version,
                    }

                # Update the host's disliked_cluster_ids
                now = datetime.now()
                if (
                    match["similarity_score"] is not None
                    and match["similarity_score"] >= self.similarity_threshold
                ):
                    print("We are in Rocchio Update Mode")
                    # UPDATE MODE
                    self.elasticsearch_db.client.update(
                        index=constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
                        id=match["doc_id"],
                        refresh="wait_for",
                        script={
                            "source": """
                            if (ctx._source.disliked_cluster_ids == null) {
                                ctx._source.disliked_cluster_ids = [];
                            }

                            for (c in params.new_clusters) {
                                boolean exists = false;
                                for (e in ctx._source.disliked_cluster_ids) {
                                    if (e.cluster_id == c.cluster_id) {
                                        exists = true;
                                        break;
                                    }
                                }
                                if (!exists) {
                                    ctx._source.disliked_cluster_ids.add(c);
                                }
                            }

                            ctx._source.updated_at = params.now;
                            """,
                            "params": {
                                "new_clusters": [
                                    {"cluster_id": cid, "timestamp": now}
                                    for cid in disliked_cluster_info_new.keys()
                                ],
                                "now": now,
                            },
                        },
                    )

                else:
                    # CREATE MODE
                    print("We are in Rocchio CREATE MODE")
                    new_doc = {
                        "id": project_id,
                        "phash_2d": hash_vector,
                        "type": "2d",
                        "disliked_cluster_ids": [
                            {"cluster_id": cluster_id, "timestamp": now}
                            for cluster_id in disliked_cluster_info_new.keys()
                        ],
                        "org_id": org_id,
                        "created_at": now,
                        "updated_at": now,
                    }

                    self.elasticsearch_db.client.index(
                        index=constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
                        document=new_doc,
                        refresh="wait_for",
                    )

        return
