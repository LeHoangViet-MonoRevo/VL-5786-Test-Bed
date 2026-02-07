from typing import Dict, List, Optional

from constants import Constants
from elasticsearch_constants import ESConstant
from interaction import ElasticsearchBase


class SimilarityClusters:
    def __init__(self, elasticsearch_db: ElasticsearchBase):
        self.elasticsearch_db = elasticsearch_db

    def _ensure_existence(self):
        """
        Ensure the SimilarityClusters exists, otherwise, create it.
        """
        self.elasticsearch_db.check_indice_existance(
            indice_name=Constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
            create=True,
            body=ESConstant.SCHEMA_SIMILARITY_CLUSTERS,
        )

    def _create_cluster_from_vector(self, vector, org_id: int | str, score_threshold: float = 0.8) -> Dict:
        """
        Create a new similarity cluster by searching similar products first.
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
            score_threshold=score_threshold
        )

        if not similar_resp or not similar_resp["hits"]["hits"]:
            return result

        physical_ids = [
            hit["_source"]["physical_id"]
            for hit in similar_resp["hits"]["hits"]
            if "physical_id" in hit["_source"]
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
        filters: Optional[List[Dict]],
    ) -> Dict:
        """Search or create new cluster"""

        result = {
            "doc_id": None,
            "similarity_score": None,
            "physical_ids": None,
        }

        # Step 1: Use the vector to search for the identical document.

        resp = self.elasticsearch_db.search_vector_w_filters(
            indice_name=Constants.SIMILARITY_CLUSTERS,
            query_vector=vector,
            number_retrieval_vector=1,
            search_field=search_field,  # "embedding_vector_2d" or "embedding_vector_3d"
            filters=filters,
            selected_cols=[],  # TODO: Test later
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
