from typing import Dict, List, Optional

from constants import Constants
from elasticsearch_constants import ESConstant
from interaction import ElasticsearchBase


class SimilarityClusters:
    def __init__(
        self,
        elasticsearch_db: ElasticsearchBase,
        cluster_similarity_threshold: float = 0.98,
        retrieved_prod_similarity_threshold: float = 0.8,
        product_search_field: str = "embedding_vector_v3",
        product_version: str = "v3",
    ):
        self.elasticsearch_db = elasticsearch_db
        self.cluster_similarity_threshold = cluster_similarity_threshold
        self.retrieved_prod_similarity_threshold = retrieved_prod_similarity_threshold
        self.product_search_field = product_search_field
        self.product_version = product_version

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
        search_field: str | None = None,
        version: str | None = None,
    ) -> Dict:
        """
        Create a new similarity cluster from a vector by retrieving
        similar product IDs and storing them in a new document.
        """

        search_field = search_field or self.product_search_field
        version = version or self.product_version

        result = {
            "doc_id": None,
            "similarity_score": None,
            "physical_ids": None,
        }

        similar_resp = self.elasticsearch_db.search_vector_w_filters(
            indice_name=Constants.ELASTICSEARCH_PREFIX,
            query_vector=vector,
            number_retrieval_vector=10000,
            search_field=search_field,
            filters=[
                {"term": {"organization_id": org_id}},
                {"term": {"version": version}},
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
            "embedding_vector_2D": vector,
            "org_id": org_id,
            "physical_ids": physical_ids,
        }

        create_resp = self.elasticsearch_db.client.index(
            index=Constants.SIMILARITY_CLUSTERS,
            document=doc_body,
            refresh="wait_for",
        )

        result["doc_id"] = create_resp["_id"]
        result["physical_ids"] = physical_ids

        return result

    def search_cluster(
        self,
        vector,
        org_id: int | str,
        search_field: str,
        product_search_field: str | None = None,
        product_version: str | None = None,
    ) -> Dict:
        """
        Find an existing cluster for the vector, or create one if none exists.
        """

        self._ensure_existence()

        result = {
            "doc_id": None,
            "similarity_score": None,
            "physical_ids": None,
        }

        resp = self.elasticsearch_db.search_vector_w_filters(
            indice_name=Constants.SIMILARITY_CLUSTERS,
            query_vector=vector,
            number_retrieval_vector=1,
            search_field=search_field,
            filters=[{"term": {"org_id": org_id}}],
            selected_cols=["physical_ids"],
            vector_method="l2",
            score_threshold=self.cluster_similarity_threshold,
        )

        if not resp or not resp["hits"]["hits"]:
            return self._create_cluster_from_vector(
                vector=vector,
                org_id=org_id,
                search_field=product_search_field,
                version=product_version,
            )

        es_hit = resp["hits"]["hits"][0]
        source = es_hit["_source"]

        result["doc_id"] = es_hit["_id"]
        result["similarity_score"] = es_hit["_score"]

        if "physical_ids" in source:
            result["physical_ids"] = source["physical_ids"]

        return result
