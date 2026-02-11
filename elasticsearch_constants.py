from constants import constants


class ESConstant:
    SCHEMA_OCR_INFORMATION = {
        "product_information": {
            "mappings": {
                "properties": {
                    "product_id": {"type": "keyword"},
                    "original_image": {"type": "keyword"},
                    "ocr_product_code": {"type": "keyword"},
                    "ocr_product_name": {"type": "keyword"},
                    "ocr_drawing_number": {"type": "keyword"},
                    "ocr_drawing_issuer": {"type": "keyword"},
                }
            }
        },
        "raijin_object": {
            "mappings": {
                "properties": {
                    "object_id": {"type": "keyword"},
                    "base64": {
                        "type": "text",
                    },
                }
            }
        },
    }

    SCHEMA_ENCODED_DATA_PHYSICAL_OBJECT = {
        "mappings": {
            "properties": {
                "data": {"type": "binary"},
                "shape": {"type": "keyword"},
                "type": {"type": "keyword"},
                "object_id": {"type": "keyword"},
            }
        }
    }

    SCHEMA_METADATA_PHYSICAL_OBJECT = {
        "mappings": {
            "properties": {
                "physical_id": {"type": "long"},
                "location": {"type": "keyword"},
                "extension": {"type": "keyword"},
                "file_type": {"type": "keyword"},
            }
        }
    }

    SCHEMA_ENCODED_DATA_PHYSICAL_OBJECT_CROP = {
        "mappings": {
            "properties": {
                "data": {"type": "binary"},
                "object_id": {"type": "keyword"},
            }
        }
    }

    SCHEMA_ROCCHIO_HISTORY_PHYSICAL_OBJECT = {
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "rocchio_pos_vec_2d": {
                    "type": "dense_vector",
                    "dims": constants.FEATURE_EXTRACT_DIMS_DICT[
                        constants.MODEL_EXTRACTION_NAME
                    ],
                },
                "rocchio_neg_vec_2d": {
                    "type": "dense_vector",
                    "dims": constants.FEATURE_EXTRACT_DIMS_DICT[
                        constants.MODEL_EXTRACTION_NAME
                    ],
                },
                "phash_2d": {
                    "type": "dense_vector",
                    "dims": constants.FEATURE_EXTRACT_DIMS_DICT[
                        constants.MODEL_EXTRACTION_NAME_V3
                    ],
                },
                "embedding_vector_3d": {
                    "type": "dense_vector",
                    "dims": constants.FEATURE_EXTRACT_DIMS_DICT[
                        constants.ELASTICSEARCH_EXTRACTION_MODEL
                    ],
                },
                "rocchio_pos_vec_3d": {
                    "type": "dense_vector",
                    "dims": constants.FEATURE_EXTRACT_DIMS_DICT[
                        constants.ELASTICSEARCH_EXTRACTION_MODEL
                    ],
                },
                "rocchio_neg_vec_3d": {
                    "type": "dense_vector",
                    "dims": constants.FEATURE_EXTRACT_DIMS_DICT[
                        constants.ELASTICSEARCH_EXTRACTION_MODEL
                    ],
                },
                "type": {"type": "keyword"},
                "interactions": {
                    "type": "nested",
                    "properties": {
                        "physical_id": {"type": "integer"},
                        "reaction": {"type": "integer"},
                        "timestamp": {"type": "date"},
                    },
                },
                "neutralisations": {
                    "type": "nested",
                    "properties": {
                        "physical_id": {"type": "integer"},
                        "timestamp": {"type": "date"},
                    },
                },
                "disliked_cluster_ids": {
                    "type": "nested",
                    "properties": {
                        "cluster_id": {"type": "keyword"},
                        "timestamp": {"type": "date"},
                    },
                },
                "org_id": {"type": "keyword"},
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"},
            }
        }
    }

    SCHEMA_SIMILARITY_CLUSTERS = {
        "mappings": {
            "properties": {
                "embedding_vector_2d": {
                    "type": "dense_vector",
                    "dims": constants.FEATURE_EXTRACT_DIMS_DICT[
                        constants.MODEL_EXTRACTION_NAME_V3
                    ],
                },
                "embedding_vector_3d": {
                    "type": "dense_vector",
                    "dims": constants.FEATURE_EXTRACT_DIMS_DICT[
                        constants.ELASTICSEARCH_EXTRACTION_MODEL
                    ],
                },
                "version": {"type": "keyword"},  # "v3" or "3d"
                "physical_ids": {
                    "type": "keyword"
                },  # supports int or str; ES arrays are implicit
                "org_id": {"type": "keyword"},  # supports str or int safely
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"},
            }
        }
    }


es_constant = ESConstant()
