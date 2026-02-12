from datetime import datetime
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

    def remove_physical_id_from_neutralisations_in_rocchio(
        self,
        physical_id: int,
        org_id: Optional[int] = None,
    ) -> Dict:
        """
        Remove a physical_id from neutralisations array in Rocchio index.
        """

        try:
            query = {
                "bool": {
                    "must": [
                        {
                            "nested": {
                                "path": "neutralisations",
                                "query": {
                                    "term": {"neutralisations.physical_id": physical_id}
                                },
                            }
                        }
                    ]
                }
            }

            if org_id is not None:
                query["bool"]["must"].append({"term": {"org_id": str(org_id)}})

            body = {
                "query": query,
                "script": {
                    "lang": "painless",
                    "source": """
                        if (ctx._source.containsKey('neutralisations')) {
                            ctx._source.neutralisations.removeIf(n -> n.physical_id == params.physical_id);
                        }
                        ctx._source.updated_at = params.now;
                    """,
                    "params": {"physical_id": physical_id, "now": "now"},
                },
            }

            response = self.client.update_by_query(
                index=constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
                body=body,
                refresh="wait_for",
                conflicts="proceed",
            )

            total = response.get("total", 0)
            updated = response.get("updated", 0)
            conflicts = response.get("version_conflicts", 0)

            if total == 0:
                logger.info(
                    f"ℹ️ No Rocchio docs contain neutralised physical_id={physical_id}"
                )
            else:
                logger.info(
                    f"✅ Removed physical_id={physical_id} from neutralisations "
                    f"(matched={total}, updated={updated}, conflicts={conflicts})"
                )

            return response

        except Exception as e:
            logger.error(
                f"❌ Failed to remove physical_id={physical_id} from neutralisations. Reason: {e}"
            )
            raise

    def remove_physical_id_from_cluster(
        self,
        indice_name: str,
        physical_id: int,
        org_id: Optional[int] = None,
    ) -> Dict:
        """
        Remove a physical_id from the physical_ids list in documents.
        If the list becomes empty after removal, delete the document.
        Args:
            indice_name: Name of the Elasticsearch index
            physical_id: The physical ID to remove from the list
            org_id: Optional organization ID to filter documents
        """
        try:
            query_filters = [{"term": {"physical_ids": int(physical_id)}}]
            if org_id is not None:
                query_filters.append({"term": {"org_id": str(org_id)}})

            query = {"query": {"bool": {"filter": query_filters}}}

            # Find all documents with this physical_id
            response = self.client.search(
                index=indice_name, body=query, size=10000, _source=["physical_ids"]
            )
            hits = response.get("hits", {}).get("hits", [])

            if not hits:
                logger.info(f"No documents found with physical_id {physical_id}")
            updated_count = 0

            for hit in hits:
                doc_id = hit["_id"]
                current_physical_ids = hit["_source"].get("physical_ids", [])

                # Remove the physical_id from the list
                new_physical_ids = [
                    pid for pid in current_physical_ids if pid != physical_id
                ]

                # Update the document with the new list and updated_at timestamp
                self.client.update(
                    index=indice_name,
                    id=doc_id,
                    body={
                        "doc": {
                            "physical_ids": new_physical_ids,
                            "updated_at": datetime.now().isoformat(),
                        }
                    },
                )
                updated_count += 1
                logger.info(
                    f"Updated document {doc_id}, removed physical_id {physical_id}"
                )

            # Refresh index to make changes visible
            self.client.indices.refresh(index=indice_name)

            logger.info(
                f"✅ Removed physical_id {physical_id}: Updated {updated_count}"
            )

        except Exception as e:
            logger.error(
                f"❌ Failed to remove physical_id {physical_id} from {indice_name}. Reason: {e}"
            )

    def find_closest_cluster_and_append_physical_id(
        self,
        indice_name: str,
        embedding_vector: List[float],
        physical_id: int,
        org_id: int,
        version: str,
        similarity_threshold: float = 0.98,
    ) -> Dict:
        """
        Find the closest cluster based on embedding vector and append physical_id to it.
        If no cluster is found above the threshold, create a new cluster.
        Args:
            indice_name: Name of the Elasticsearch index (similarity_clusters)
            embedding_vector: The embedding vector to search with (phash for 2d, 3d embedding for 3d)
            physical_id: The physical ID to append to the cluster
            org_id: Organization ID
            version: "2d" or "3d" to determine which embedding field to use
            similarity_threshold: Minimum similarity score to consider a match
        """
        try:
            # Determine embedding field based on version
            if version == "v3":
                embedding_field = "embedding_vector_2d"
            elif version == "3d":
                embedding_field = "embedding_vector_3d"
            else:
                raise ValueError(f"Invalid version '{version}', expected 'v3' or '3d'")

            # Search for closest cluster
            response = self.search_vector_w_filters(
                indice_name=indice_name,
                query_vector=embedding_vector,
                number_retrieval_vector=1,
                search_field=embedding_field,
                filters=[
                    {"term": {"org_id": str(org_id)}},
                    {"term": {"version": version}},
                ],
                selected_cols=["physical_ids"],
                vector_method="l2",
                score_threshold=similarity_threshold,
            )

            if response and response.get("hits", {}).get("hits"):
                # Found a matching cluster - append physical_id
                hit = response["hits"]["hits"][0]
                doc_id = hit["_id"]
                current_physical_ids = hit["_source"].get("physical_ids", [])

                # Check if physical_id already exists
                if physical_id in current_physical_ids:
                    logger.info(
                        f"Physical ID {physical_id} already exists in cluster {doc_id}"
                    )

                # Append physical_id to the list
                new_physical_ids = current_physical_ids + [physical_id]
                self.client.update(
                    index=indice_name,
                    id=doc_id,
                    body={
                        "doc": {
                            "physical_ids": new_physical_ids,
                            "updated_at": datetime.utcnow().isoformat(),
                        }
                    },
                )
                logger.info(
                    f"✅ Appended physical_id {physical_id} to cluster {doc_id}"
                )
            else:
                now = datetime.utcnow().isoformat()
                doc_body = {
                    embedding_field: embedding_vector,
                    "version": version,
                    "org_id": str(org_id),
                    "physical_ids": [physical_id],
                    "created_at": now,
                    "updated_at": now,
                }
                create_resp = self.client.index(
                    index=indice_name,
                    document=doc_body,
                    refresh="wait_for",
                )

                logger.info(
                    f"✅ Created new cluster {create_resp['_id']} with physical_id {physical_id}"
                )

        except Exception as e:
            logger.error(
                f"❌ Failed to find/append cluster for physical_id {physical_id}. Reason: {e}"
            )


elasticsearch_db = ElasticsearchBase()
