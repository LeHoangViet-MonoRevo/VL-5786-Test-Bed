from typing import Dict, List, Optional

from elasticsearch.helpers import BulkIndexError, bulk

import logger
from connection import es_client
from constants import constants
from search_updater_constants import search_updater_constants
from vector_db import VectorDataBaseInteraction

logger = logger.get_logger(logger_name="ELASTICSEARCH_DB")


class ElasticsearchBase(VectorDataBaseInteraction):
    def __init__(self):
        self.client = es_client

    def search_vector(
        self,
        indice_name,
        organization_id,
        version,
        query_vector,
        number_retrieval_vector=100,
        selected_cols=["product_id", "original_image"],
    ):
        try:
            field_embedding_vector_name = f"embedding_vector_{version}"
            logger.info(f"{field_embedding_vector_name = }")
            results = self.client.search(
                index=indice_name,
                body={
                    "size": number_retrieval_vector,
                    "_source": selected_cols,
                    "query": {
                        "script_score": {
                            "query": {
                                "bool": {
                                    "filter": [
                                        {"term": {"version": str(version)}},
                                        {
                                            "term": {
                                                "organization_id": str(organization_id)
                                            }
                                        },
                                    ]
                                }
                            },
                            "script": {
                                "source": f"dotProduct(params.query_vector, '{field_embedding_vector_name}')",
                                "params": {"query_vector": query_vector},
                            },
                        }
                    },
                },
                routing=search_updater_constants.create_routing(
                    version, organization_id
                ),
            )

            return results
        except Exception as e:
            logger.error(
                f"Failed to search data from indice {indice_name}. " f"Reason {e}"
            )

    def search_vector_w_filters(
        self,
        indice_name,
        query_vector,
        number_retrieval_vector=100,
        search_field: str = "embedding_vector",
        filters: Optional[List[Dict]] = None,
        selected_cols=["product_id", "original_image"],
        vector_method: str = "l2",  # "cosine", "dot", "l2"
        score_threshold: float | None = None,
    ):
        try:
            if filters is None:
                filters = []

            # Map method to Elasticsearch script
            if vector_method == "cosine":
                script_source = (
                    f"cosineSimilarity(params.query_vector, '{search_field}') + 1.0"
                )
            elif vector_method == "dot":
                script_source = f"dotProduct(params.query_vector, '{search_field}')"
            elif vector_method == "l2":
                script_source = (
                    f"1 / (1 + l2norm(params.query_vector, '{search_field}'))"
                )
            else:
                raise ValueError(f"Unknown vector_method: {vector_method}")

            query_body = {
                "size": number_retrieval_vector,
                "_source": selected_cols,
                "query": {
                    "script_score": {
                        "query": {"bool": {"filter": filters}},
                        "script": {
                            "source": script_source,
                            "params": {"query_vector": query_vector},
                        },
                    }
                },
            }

            # 🔥 Apply threshold
            if score_threshold is not None:
                query_body["min_score"] = score_threshold

            results = self.client.search(index=indice_name, body=query_body)
            return results

        except Exception as e:
            logger.error(f"Failed to search data from indice {indice_name}. Reason {e}")
            return None

    def find(
        self, indice_name, field_name=None, value=None, query=None, routing_field=None
    ):
        if query is None and field_name is not None and value is not None:
            if isinstance(value, (list, tuple, set)):
                query = {"query": {"terms": {field_name: list(value)}}}
            else:
                query = {"query": {"term": {field_name: value}}}

        logger.info(f"query ES with command {query}, routing={routing_field}")

        try:
            response = self.client.search(
                index=indice_name, body=query, size=10000, routing=routing_field
            )
            logger.info(f"Query '{indice_name}' Successfully.")
            return response
        except Exception as e:
            logger.error(
                f"Failed to query data from indice {indice_name}. " f"Reason {e}"
            )
            raise

    def insert_vector(self, index_name, document_id):
        pass

    def create_indice(self, indice_name, body=None):
        if body == None:
            body = {
                "mappings": {
                    "properties": {
                        "version": {"type": "keyword"},
                        "organization_id": {"type": "keyword"},
                        "product_id": {"type": "keyword"},
                        "images_paths": {"type": "keyword"},
                        "original_image": {"type": "keyword"},
                        "number_objects": {"type": "integer"},
                        "embedding_vector_v2": {
                            "type": "dense_vector",
                            "dims": constants.FEATURE_EXTRACT_DIMS_DICT[
                                constants.MODEL_EXTRACTION_DICT["v2"]
                            ],
                            "index": True,
                            "similarity": "dot_product",
                        },
                        "embedding_vector_v3": {
                            "type": "dense_vector",
                            "dims": constants.FEATURE_EXTRACT_DIMS_DICT[
                                constants.MODEL_EXTRACTION_DICT["v3"]
                            ],
                            "index": True,
                            "similarity": "dot_product",
                        },
                        "embedding_vector_3d": {
                            "type": "dense_vector",
                            "dims": constants.FEATURE_EXTRACT_DIMS_DICT[
                                constants.MODEL_EXTRACTION_DICT["3d"]
                            ],
                            "index": True,
                            "similarity": "dot_product",
                        },
                    }
                }
            }
        try:
            self.client.indices.create(index=indice_name, body=body)
        except Exception as e:
            logger.error(
                f"Failed to create indice {indice_name} from Elasticsearch. "
                f"Reason {e}"
            )

    def check_indice_existance(self, indice_name, create=False, body=None):
        try:
            if self.client.indices.exists(index=indice_name):
                logger.info(f"✅ Index '{indice_name}' exists.")
                return True
            else:
                logger.info(f"❌ Index '{indice_name}' does not exist.")
                if create:
                    self.create_indice(indice_name=indice_name, body=body)
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to fetch data from Elasticsearch. " f"Reason {e}")
            return False

    def bulk_insert_vector(self, indice_name, docs, overwrite=True, id_column=None):
        if not overwrite:
            _op_type = "create"
        else:
            _op_type = "index"
        actions = []
        for doc in docs:
            routing = doc.pop("routing", None)
            action = {"_op_type": _op_type, "_index": indice_name, "_source": doc}
            if routing:
                action["_routing"] = routing
            if id_column and id_column in doc:
                action["_id"] = doc[
                    id_column
                ]  # Only set _id if id_column is defined and exists
            actions.append(action)
        try:
            success, failed = bulk(
                self.client,
                actions,
                stats_only=True,
                chunk_size=500,
                request_timeout=60,
            )
            logger.info(f"✅ Indexed: {success}, ❌ Failed: {failed}")
        except BulkIndexError as e:
            logger.error("❌ Bulk index error:")
            for err in e.errors:
                logger.error(err)

    def bulk_delete(self, indice_name, key, value):
        try:
            response = self.client.delete_by_query(
                index=indice_name, body={"query": {"terms": {key: value}}}
            )
        except Exception as e:
            logger.error(f"❌ ES delete error: {e}")

    def bulk_delete_set_conditions(self, indice_name, set_of_condition):
        try:
            if not set_of_condition:
                return False
            should_conditions = []
            routing_values = set()
            for cond in set_of_condition:
                must_conditions = []
                for key, value in cond.items():
                    if key == "routing":
                        routing_values.add(value)
                        continue
                    must_conditions.append({"term": {key: str(value)}})
                should_conditions.append({"bool": {"must": must_conditions}})
            body = {
                "query": {
                    "bool": {"should": should_conditions, "minimum_should_match": 1}
                }
            }
            routing_param = ",".join(routing_values) if routing_values else None
            response = self.client.delete_by_query(
                index=indice_name,
                body=body,
                routing=routing_param,
                refresh=True,
                conflicts="proceed",
            )
            return response
        except Exception as e:
            logger.error(f"❌ ES delete error: {e}")

    def bulk_update(self, indice_name, docs, id_column="object_id"):
        actions = []
        for doc in docs:
            doc_id = doc.get(id_column)
            action = {
                "_op_type": "update",
                "_index": indice_name,
                "_id": doc_id,
                "doc": doc,
                "doc_as_upsert": True,
            }
            actions.append(action)

        try:
            success, failed = bulk(
                self.client,
                actions,
                stats_only=True,
                chunk_size=500,
                request_timeout=60,
            )
            logger.info(f"✅ Indexed: {success}, ❌ Failed: {failed}")
        except BulkIndexError as e:
            logger.info("❌ Bulk index error:")
            for err in e.errors:
                logger.info(err)


elasticsearch_db = ElasticsearchBase()
