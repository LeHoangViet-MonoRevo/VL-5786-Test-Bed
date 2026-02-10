from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from constants import constants
from elasticsearch_constants import es_constant
from interaction import ElasticsearchBase


class RocchioFeedbackBase:
    def __init__(
        self, elasticsearch_db: ElasticsearchBase, similarity_threshold: float = 0.98
    ):
        self.elasticsearch_db = elasticsearch_db
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def l2_normalize(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-8)

    @staticmethod
    def add_epsilon(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        return v + eps

    @staticmethod
    def convert_feedback_to_interactions(
        feedback_list: List[Tuple],
        field_names: Tuple[str, str] = ("physical_id", "reaction", "timestamp"),
    ) -> List[Dict[str, int]]:
        """
        Convert a list of tuples into Elasticsearch nested objects with custom field names.
        Only keep reactions = 1 or -1 (ignore 0 / neutral).

        Parameters:
            feedback_list: List of tuples, e.g. [(1175, 1), (1234, -1), (555, 0)]
            field_names: Tuple of field names (default: ("physical_id", "reaction"))

        Returns:
            List of dicts: [{field1: value1, field2: value2}, ...]
        """
        field1, field2, field3 = field_names

        timestamp = datetime.now()

        return [
            {field1: int(phys_id), field2: int(reaction), field3: timestamp}
            for phys_id, reaction in feedback_list
            if reaction in (-1, 1)
        ]

    @staticmethod
    def rocchio_update(
        vecs: List[np.ndarray],
        pos_vecs: List[np.ndarray],
        neg_vecs: List[np.ndarray],
        alpha: float = 1.0,
        beta: float = 0.75,
        gamma: float = 0.25,
        return_intermediates: bool = False,
    ):
        """Rocchio update with optional returning of pos_mean/neg_mean."""
        d = vecs[0].shape[-1]

        pos_mean = (
            np.mean(pos_vecs, axis=0).astype(np.float32)
            if len(pos_vecs)
            else np.zeros(d, dtype=np.float32)
        )
        neg_mean = (
            np.mean(neg_vecs, axis=0).astype(np.float32)
            if len(neg_vecs)
            else np.zeros(d, dtype=np.float32)
        )

        updated = []
        for vec in vecs:
            vec = RocchioFeedbackBase.l2_normalize(vec.astype(np.float32))
            vec_new = alpha * vec + beta * pos_mean - gamma * neg_mean
            updated.append(RocchioFeedbackBase.l2_normalize(vec_new))

        if return_intermediates:
            return updated, pos_mean, neg_mean
        return updated

    def _extract_feedback_vectors(
        self,
        feedback_list: List[Tuple],
        embedding_index: str,
        embedding_vector_field="embedding_vector_v2",
        routing_key=None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:

        pos_vecs, neg_vecs = [], []

        for product_id, reaction in feedback_list:
            res = self.elasticsearch_db.find(
                indice_name=embedding_index,
                field_name="product_id",
                value=product_id,
                routing_field=routing_key,
            )
            for hit in res["hits"]["hits"]:
                if hit["_source"]["version"] != embedding_vector_field.split("_")[-1]:
                    continue
                vec = np.array(hit["_source"][embedding_vector_field], dtype=np.float32)
                if reaction == 1:
                    pos_vecs.append(vec)
                elif reaction == -1:
                    neg_vecs.append(vec)

        return pos_vecs, neg_vecs

    def _ensure_rocchio_index(self) -> None:
        self.elasticsearch_db.check_indice_existance(
            indice_name=constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
            create=True,
            body=es_constant.SCHEMA_ROCCHIO_HISTORY_PHYSICAL_OBJECT,
        )

    def find_similar_rocchio_record(
        self,
        query_vector: np.ndarray,
        search_field: str,
        check_and_create: bool = True,
        filters: Optional[List[Dict]] = None,
    ) -> Optional[dict]:
        """
        Search for a similar Rocchio record and return its stored 2D/3D vectors.
        """

        if check_and_create:
            self._ensure_rocchio_index()

        result = {
            "object_id": None,
            "similarity_score": None,
            "rocchio_pos_vec_3d": None,
            "rocchio_neg_vec_3d": None,
            "rocchio_pos_vec_2d": None,
            "rocchio_neg_vec_2d": None,
            "interactions": [],
            "disliked_cluster_ids": [],
        }

        # Step 1: Find matches in ES db
        hist_res = self.elasticsearch_db.search_vector_w_filters(
            indice_name=constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
            query_vector=query_vector,
            number_retrieval_vector=1,
            search_field=search_field,
            filters=filters,
            selected_cols=[
                "rocchio_pos_vec_2d",
                "rocchio_neg_vec_2d",
                "rocchio_pos_vec_3d",
                "rocchio_neg_vec_3d",
                "interactions",
                "disliked_cluster_ids",
            ],
            score_threshold=self.similarity_threshold,
        )

        # No hits at all
        if not hist_res or not hist_res["hits"]["hits"]:
            return result

        es_hit = hist_res["hits"]["hits"][0]
        similarity_score = es_hit["_score"]

        source = es_hit["_source"]

        # Extract vectors if exist
        result["doc_id"] = es_hit["_id"]
        result["similarity_score"] = similarity_score

        # Convert any existing ES dense vectors to numpy
        if "rocchio_pos_vec_2d" in source:
            result["rocchio_pos_vec_2d"] = np.array(
                source["rocchio_pos_vec_2d"], dtype=np.float32
            )

        if "rocchio_neg_vec_2d" in source:
            result["rocchio_neg_vec_2d"] = np.array(
                source["rocchio_neg_vec_2d"], dtype=np.float32
            )

        if "rocchio_pos_vec_3d" in source:
            result["rocchio_pos_vec_3d"] = np.array(
                source["rocchio_pos_vec_3d"], dtype=np.float32
            )

        if "rocchio_neg_vec_3d" in source:
            result["rocchio_neg_vec_3d"] = np.array(
                source["rocchio_neg_vec_3d"], dtype=np.float32
            )

        if "interactions" in source:
            result["interactions"] = source["interactions"]

        if "disliked_cluster_ids" in source:
            result["disliked_cluster_ids"] = source["disliked_cluster_ids"]

        return result

    def remove_interactions_by_physical_ids(
        self,
        index: str,
        doc_id: str,
        physical_ids: set | list,
    ) -> None:
        """
        Remove interaction objects whose physical_id is in `physical_ids`.

        Args:
            index (str): Elasticsearch index name
            doc_id (str): Document ID to update
            physical_ids (set | list): physical_ids to remove from interactions
        """
        if not physical_ids:
            return  # nothing to do

        self.elasticsearch_db.client.update(
            index=index,
            id=doc_id,
            body={
                "script": {
                    "lang": "painless",
                    "source": """
                        if (ctx._source.interactions != null) {
                            ctx._source.interactions.removeIf(
                                i -> params.ids.contains(i.physical_id)
                            );
                        }
                    """,
                    "params": {"ids": list(physical_ids)},
                }
            },
            refresh="wait_for",
        )

    @staticmethod
    def remove_neutral_and_duplicate_negative_interactions(
        feedback_list: List[Tuple[Any, int]],
        past_interactions: List[Tuple[Any, int]],
    ) -> List[Tuple[Any, int]]:
        """
        Remove:
        1. Neutral reactions (reaction == 0)
        2. Negative reactions (-1) that already exist in past_interactions
        """

        # Collect physical_ids that already had negative (-1)
        past_negative_ids = {
            physical_id for physical_id, reaction in past_interactions if reaction == -1
        }

        # Filter feedback
        return [
            (physical_id, reaction)
            for physical_id, reaction in feedback_list
            if reaction != 0
            and not (reaction == -1 and physical_id in past_negative_ids)
        ]

    def fetch_representation_embeddings_from_raijin_search_indexer(
        self, physical_ids: List, company_id: int
    ) -> Dict[int, Dict[str, Union[str, List[float]]]]:
        """
        Fetch representation (2D: Phash, 3D: embedding vector) for disliked physical_ids.

        Returns:
            Dict[int, Dict[str, Any]]:
            {physical_id: {"version": str, "embedding": list}}
        """

        if not physical_ids:
            return {}

        resp = self.elasticsearch_db.client.search(
            index=constants.ELASTICSEARCH_PREFIX,
            body={
                "size": 10000,
                "_source": [
                    "embedding_vector_v3",
                    "embedding_vector_3d",
                    "version",
                    "product_id",
                ],
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {"product_id": physical_ids}},
                            {"terms": {"version": ["v3", "3d"]}},
                            {"term": {"organization_id": company_id}},
                        ]
                    }
                },
            },
        )

        results = {}

        hits = resp.get("hits", {}).get("hits", [])
        for hit in hits:
            src = hit["_source"]
            physical_id = src["product_id"]
            version = src["version"]

            if version == "v3":
                embedding = src.get("embedding_vector_v3")
            elif version == "3d":
                embedding = src.get("embedding_vector_3d")
            else:
                continue  # safety, should not happen due to query filter

            results[physical_id] = {
                "version": version,
                "embedding": embedding,
            }

        return results
