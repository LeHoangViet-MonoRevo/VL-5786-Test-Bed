from constants import constants


class SearchUpdaterConstants:
    search_updater_constants_dict = {
        "v2": {
            "collection_project_relevance": constants.MONGO_COLLECTION_PROJECT_RELEVANCE,
            "model_extract_name": constants.MODEL_EXTRACTION_NAME,
        },
        "v3": {
            "collection_project_relevance": constants.MONGO_COLLECTION_PROJECT_RELEVANCE_V3,
            "model_extract_name": constants.MODEL_EXTRACTION_NAME_V3,
        },
    }

    @staticmethod
    def create_routing(version, company_id):
        return f"{version}___{company_id}"


search_updater_constants = SearchUpdaterConstants()
