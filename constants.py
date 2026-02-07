class Constants:
    LIST_VERSION = ["v2", "v3"]
    MODEL_EXTRACTION_NAME: str = "vgg19"
    MODEL_EXTRACTION_NAME_V3: str = "phash"
    ELASTICSEARCH_EXTRACTION_MODEL = "pointnet2"
    MODEL_EXTRACTION_DICT = {
        "v2": MODEL_EXTRACTION_NAME,
        "v3": MODEL_EXTRACTION_NAME_V3,
        "3d": ELASTICSEARCH_EXTRACTION_MODEL,
        "3d_to_2d": MODEL_EXTRACTION_NAME,
    }
    FEATURE_EXTRACT_DIMS_DICT = {
        MODEL_EXTRACTION_NAME: 4096,
        MODEL_EXTRACTION_NAME_V3: 1024,
        ELASTICSEARCH_EXTRACTION_MODEL: 512,
    }
    TITLE: str = "Raijin API"
    VERSION: str = "0.0.1"
    MONGO_COLLECTION_PROJECT_RELEVANCE: str = "project_relevance_with_score"
    MONGO_COLLECTION_PROJECT_3D_RELEVANCE: str = "project_3D_relevance_with_score"

    MONGO_COLLECTION_PROJECT_RELEVANCE_V3: str = "project_relevance_with_score_v3"
    MONGO_COLLECTION_PRODUCT_OCR: str = "project_product_ocr"
    MONGO_COLLECTION_COUNT_PROJECT_OBJECTS: str = "project_objects_quantity"
    MONGO_COLLECTION_SYNONYM_INFO: str = "information_mapping_synonym"
    MONGO_COLLECTION_TRANSLATED_INFO: str = "information_mapping_translated"
    MONGO_DATABASE_CACHED: str = "cached_information"
    MONGO_COLLECTION_MATERIAL_CACHED: str = "material_knowledge"
    MONGO_COLLECTION_PHYSICAL_ID_METADATA: str = "physical_id_metadata"
    NUMBER_SEARCH_IMAGE: int = 15
    NUMBER_RANKING_IMAGE_OUTPUT_DEFAULT: int = 30
    CONF = 0.4
    THRESHOLD_TEXT_MATCHING = 80
    THRESHOLD_TEXT_CLASSIFICATION_QUESTION = 0.7
    THRESHOLD_CLASSFICATION_DIAGRAM = 0.545
    REGION: str = "ap-northeast-1"
    THUMBNAIL_SIZE = 1200

    EXTENSION_SUPPORT_TYPE = {
        "2d": ("pdf", "jpeg", "jpg", "tif", "tiff", "png", "dxf", "dwg"),
        "3d": ("stl", "stp", "step", "obj", "iges", "igs"),
    }

    ELASTICSEARCH_PRODUCT_INFORMATION: str = "raijin_product_information"
    ELASTICSEARCH_PREFIX: str = "raijin_search_indexer"

    ZONE_METADATA_PHYSICAL_OBJECT = "physical_id_metadata"
    ZONE_ENCODED_DATA_PHYSICAL_OBJECT = "encoded_data_physical_object"
    ZONE_ENCODED_DATA_PHYSICAL_OBJECT_2D = "encoded_data_physical_object_2d"
    ZONE_MATERIAL_CACHED = "material_knowledge"
    ROCCHIO_HISTORY_PHYSICAL_OBJECT = "rocchio_history_physical_object_1"
    SIMILARITY_CLUSTERS = "similarity_clusters"


constants = Constants()
