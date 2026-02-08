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
    ):
        self.elasticsearch_db = elasticsearch_db
        self.cluster_similarity_threshold = cluster_similarity_threshold
        self.retrieved_prod_similarity_threshold = retrieved_prod_similarity_threshold

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

        result = {
            "doc_id": None,
            "similarity_score": None,
            "physical_ids": None,
        }

        similar_resp = self.elasticsearch_db.search_vector_w_filters(
            indice_name=Constants.ELASTICSEARCH_PREFIX,
            query_vector=vector,
            number_retrieval_vector=10000,
            search_field="embedding_vector_v3",
            filters=[
                {"term": {"organization_id": org_id}},
                {"term": {"version": "v3"}},
            ],
            selected_cols=["product_id"],
            score_threshold=self.retrieved_prod_similarity_threshold,
        )

        physical_ids = [
            hit["_source"]["product_id"]
            for hit in similar_resp["hits"]["hits"]
            if "product_id" in hit["_source"]
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
    ) -> Dict:
        """
        Find an existing cluster for the vector, or create one if none exists.
        """

        # Step 1: Make sure the index exists
        self._ensure_existence()

        result = {
            "doc_id": None,
            "similarity_score": None,
            "physical_ids": None,
        }

        # Step 2: Use the vector to search for the identical document.

        resp = self.elasticsearch_db.search_vector_w_filters(
            indice_name=Constants.SIMILARITY_CLUSTERS,
            query_vector=vector,
            number_retrieval_vector=1,
            search_field=search_field,  # "embedding_vector_2D" or "embedding_vector_3D"
            filters=[{"term": {"org_id": org_id}}],
            selected_cols=["physical_ids"],
            vector_method="l2",
            score_threshold=self.cluster_similarity_threshold,
        )

        # No hits at all
        if not resp or not resp["hits"]["hits"]:
            return self._create_cluster_from_vector(vector=vector, org_id=org_id)

        es_hit = resp["hits"]["hits"][0]

        source = es_hit["_source"]

        # Extract vectors if exist
        result["doc_id"] = es_hit["_id"]
        result["similarity_score"] = es_hit["_score"]

        if "physical_ids" in source:
            result["physical_ids"] = source["physical_ids"]

        # Step 2: Return the dictionary containing the result.

        return result
