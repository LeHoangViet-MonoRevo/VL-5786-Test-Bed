from typing import Dict, List, Optional

from constants import Constants
from elasticsearch_constants import ESConstant
from interaction import ElasticsearchBase


class SimilarityClusters:
    MODE_CONFIG = {
        "2d": {
            "product_search_field": "embedding_vector_v3",
            "product_version": "v3",
            "cluster_embedding_field": "embedding_vector_2D",
            "type": "2d",
        },
        "3d": {
            "product_search_field": "embedding_vector_3d",
            "product_version": "3d",
            "cluster_embedding_field": "embedding_vector_3D",
            "type": "3d",
        },
    }

    def __init__(
        self,
        elasticsearch_db: ElasticsearchBase,
        mode: str,  # "2d" or "3d"
        cluster_similarity_threshold: float = 0.98,
        retrieved_prod_similarity_threshold: float = 0.8,
    ):
        if mode not in self.MODE_CONFIG:
            raise ValueError(
                f"Invalid mode '{mode}', expected one of {list(self.MODE_CONFIG)}"
            )

        self.elasticsearch_db = elasticsearch_db
        self.mode = mode
        self.cluster_similarity_threshold = cluster_similarity_threshold
        self.retrieved_prod_similarity_threshold = retrieved_prod_similarity_threshold

        # Freeze mode config into instance attributes
        cfg = self.MODE_CONFIG[mode]
        self.product_search_field = cfg["product_search_field"]
        self.product_version = cfg["product_version"]
        self.cluster_embedding_field = cfg["cluster_embedding_field"]
        self.cluster_type = cfg["type"]

    def _ensure_existence(self):
        """Ensure the similarity clusters index exists (create if missing)."""
        self.elasticsearch_db.check_indice_existance(
            indice_name=Constants.SIMILARITY_CLUSTERS,
            create=True,
            body=ESConstant.SCHEMA_SIMILARITY_CLUSTERS,
        )

    def _create_cluster_from_vector(
        self,
        vector,
        org_id: int | str,
    ) -> Dict:
        """
        Create a new similarity cluster from a vector by retrieving
        similar product IDs and storing them in a new document.
        """

        similar_resp = self.elasticsearch_db.search_vector_w_filters(
            indice_name=Constants.ELASTICSEARCH_PREFIX,
            query_vector=vector,
            number_retrieval_vector=10000,
            search_field=self.product_search_field,
            filters=[
                {"term": {"organization_id": org_id}},
                {"term": {"version": self.product_version}},
            ],
            selected_cols=["product_id"],
            score_threshold=self.retrieved_prod_similarity_threshold,
        )

        hits = (similar_resp or {}).get("hits", {}).get("hits", [])

        physical_ids = [
            hit["_source"]["product_id"]
            for hit in hits
            if "product_id" in hit.get("_source", {})
        ]

        doc_body = {
            self.cluster_embedding_field: vector,
            "type": self.cluster_type,
            "org_id": org_id,
            "physical_ids": physical_ids,
        }

        create_resp = self.elasticsearch_db.client.index(
            index=Constants.SIMILARITY_CLUSTERS,
            document=doc_body,
            refresh="wait_for",
        )

        return {
            "doc_id": create_resp["_id"],
            "similarity_score": None,
            "physical_ids": physical_ids,
        }

    def search_cluster(
        self,
        vector,
        org_id: int | str,
    ) -> Dict:
        """
        Find an existing cluster for the vector, or create one if none exists.
        """

        self._ensure_existence()

        resp = self.elasticsearch_db.search_vector_w_filters(
            indice_name=Constants.SIMILARITY_CLUSTERS,
            query_vector=vector,
            number_retrieval_vector=1,
            search_field=self.cluster_embedding_field,
            filters=[
                {"term": {"org_id": org_id}},
                {"term": {"type": self.cluster_type}},
            ],
            selected_cols=["physical_ids"],
            vector_method="l2",
            score_threshold=self.cluster_similarity_threshold,
        )

        if not resp or not resp["hits"]["hits"]:
            return self._create_cluster_from_vector(
                vector=vector,
                org_id=org_id,
            )

        es_hit = resp["hits"]["hits"][0]
        source = es_hit["_source"]

        return {
            "doc_id": es_hit["_id"],
            "similarity_score": es_hit["_score"],
            "physical_ids": source.get("physical_ids"),
        }
