import traceback
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple, Union

import imagehash
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torchvision import transforms

import logger
from common_schemas import ErrorCodeAISystem
from config import settings
from constants import constants
from exception_utils import AISystemError, SearchException
from feedback_2d import RocchioFeedback2D
from interaction import elasticsearch_db
from model_loader_utils import base64_to_image, model_loader
from similarity_clusters import SimilarityClusters

logger = logger.get_logger(logger_name="SERVICE_SIMILARITY_SEARCH")


class SimilaritySearchV3:
    def search_similar_project_v3(
        self, project_id: int | str, company_id: int | str
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Search for visually similar projects using perceptual image hashing (v3).

        Computes a normalised perceptual hash vector from project's image and
        queries Elasticsearch for similar items.
        """

        image_info = elasticsearch_db.find(
            indice_name=constants.ZONE_ENCODED_DATA_PHYSICAL_OBJECT,
            field_name="object_id",
            value=project_id,
        )
        physical_id_metadata = [
            x for x in image_info["hits"]["hits"] if x["_source"]["type"] == "image"
        ]
        image_base64 = physical_id_metadata[0]["_source"]["data"]

        image = base64_to_image(image_base64)
        image_hash = imagehash.phash(image, hash_size=32)
        hash_vector = np.array(
            [int(b) for b in bin(int(str(image_hash), 16))[2:].zfill(1024)],
            dtype=np.float32,
        )
        hash_vector = hash_vector / np.linalg.norm(hash_vector)

        # ✅ Single empty return definition
        empty_return = (
            pd.DataFrame(columns=["physical_id", "score", "parent"]),
            hash_vector,
        )

        indice_name = constants.ELASTICSEARCH_PREFIX
        if not elasticsearch_db.check_indice_existance(indice_name=indice_name):
            logger.warning(f"Index {indice_name} does not exist")
            return empty_return

        selected_cols = ["product_id", "images_paths", "original_image"]

        search_results = elasticsearch_db.search_vector(
            indice_name=indice_name,
            organization_id=company_id,
            version="v3",
            query_vector=hash_vector,
            selected_cols=selected_cols,
        )

        search_results = search_results if search_results is not None else {}
        hits = search_results.get("hits", {}).get("hits", [])
        if not hits:
            logger.info(
                f"No similar images found for project_id={project_id}, company_id={company_id}"
            )
            return empty_return

        df_res = pd.json_normalize(hits)
        if df_res.empty:
            logger.info(
                f"Empty DataFrame after normalization for project_id={project_id}"
            )
            return empty_return

        agg_df = df_res.copy()
        agg_df.columns = [col.replace("_source.", "") for col in agg_df.columns]
        required_cols = ["_score"] + selected_cols
        for col in required_cols:
            if col not in agg_df.columns:
                logger.warning(f"Column {col} not found in search results")
                return empty_return

        agg_df = agg_df[required_cols]
        agg_df = agg_df.rename(columns={"_score": "score"})

        result_similarity_v3 = agg_df.drop(columns=["images_paths"])
        result_similarity_v3 = result_similarity_v3.rename(
            columns={"product_id": "physical_id", "original_image": "parent"}
        )

        result_similarity_v3.loc[result_similarity_v3["score"] >= 0.96, "score"] = 1
        result_similarity_v3 = result_similarity_v3[
            result_similarity_v3["score"] >= 0.93
        ]

        if result_similarity_v3.empty:
            logger.info(
                f"No similar images above score threshold for project_id={project_id}"
            )
            return empty_return

        result_similarity_v3 = (
            result_similarity_v3.sort_values(by="score", ascending=False)
            .drop_duplicates(subset=["physical_id"], keep="first")
            .reset_index(drop=True)
        )

        return result_similarity_v3[["physical_id", "score", "parent"]], hash_vector


similarity_search_v3 = SimilaritySearchV3()


class SimilaritySearchService:
    def __init__(self):
        self.device = settings.DEVICE
        self.model_text_classification = model_loader.load_word_classification_model()
        self.model_extract = model_loader.load_model_extraction(
            model_name=constants.MODEL_EXTRACTION_NAME
        )
        self.rocchio_feedback_2d = RocchioFeedback2D(elasticsearch_db)
        self.similarity_clusters = SimilarityClusters(elasticsearch_db)

    def check_production_info(self, word):
        probabilities = self.model_text_classification.predict_proba([word])
        if probabilities[0][1] >= constants.THRESHOLD_TEXT_CLASSIFICATION_QUESTION:
            predicted_label = "question"
        else:
            predicted_label = "answer"
        return predicted_label

    def extract_feature(self, img) -> np.array:
        img = img.resize((224, 224))
        img = img.convert("RGB")

        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        x = preprocess(img)
        x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False).to(
            self.device
        )

        feature = self.model_extract(x)
        feature = feature.data.cpu().numpy().flatten()
        return feature / np.linalg.norm(feature)

    def search_similar_project(
        self,
        list_query_vector,
        company_id: str,
        project_id: str,
        feedback_list: List[Tuple] = [],
    ) -> pd.DataFrame:
        indice_name = constants.ELASTICSEARCH_PREFIX
        if not elasticsearch_db.check_indice_existance(indice_name=indice_name):
            return None

        list_query_vector = self.rocchio_feedback_2d.run(
            list_query_vector, project_id, company_id, indice_name, feedback_list
        )

        result_similarity = pd.DataFrame()
        selected_cols = [
            "product_id",
            "images_paths",
            "original_image",
            "number_objects",
        ]
        for query_vector in list_query_vector:
            search_results = elasticsearch_db.search_vector(
                indice_name=indice_name,
                organization_id=company_id,
                version="v2",
                query_vector=query_vector,
                selected_cols=selected_cols,
            )

            # ✅ Safely handle None or empty responses
            if (
                not search_results
                or "hits" not in search_results
                or "hits" not in search_results["hits"]
            ):
                continue

            hits = search_results["hits"]["hits"]
            if not hits:  # empty list
                continue

            df_res = pd.json_normalize(hits)
            if df_res.empty:
                continue

            result_similarity = pd.concat(
                [result_similarity, df_res], ignore_index=True
            )

        # If retrievals are empty
        if len(result_similarity) == 0:
            return pd.DataFrame(
                columns=[
                    "index",
                    "product_id",
                    "parent",
                    "score",
                    "number_objects",
                    "original_image",
                ]
            )

        result_similarity.columns = [
            col.replace("_source.", "") for col in result_similarity.columns
        ]
        result_similarity = result_similarity[["_score"] + selected_cols]
        result_similarity = result_similarity.rename(columns={"_score": "score"})
        result_similarity = result_similarity.sort_values(
            by=["images_paths", "score"], ascending=[True, False]
        )
        result_similarity = result_similarity.drop_duplicates(
            subset=["images_paths"], keep="first"
        )
        result_similarity["parent"] = (
            result_similarity["images_paths"].str.rsplit(pat="_", n=1).str[0]
        )
        result_similarity = result_similarity.reset_index(drop=True)
        result_similarity["index"] = result_similarity.index
        result_similarity = result_similarity[
            [
                "index",
                "product_id",
                "parent",
                "score",
                "number_objects",
                "original_image",
            ]
        ]
        result_similarity = result_similarity.astype({"product_id": int})
        return result_similarity

    @staticmethod
    def sort_by_score_and_physical_id(df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort the DataFrame by score (descending) and then by physical_id (descending).

        The score column is the primary sort key.
        physical_id is used as a tie-breaker when scores are equal.
        """
        return df.sort_values(
            by=["score", "physical_id"],
            ascending=[False, False],  # score desc, physical_id desc
        ).reset_index(drop=True)

    @staticmethod
    def limit_and_merge_by_reaction(
        df: pd.DataFrame,
        reaction_col: str = "reaction",
        max_each: int = 20,
    ) -> pd.DataFrame:
        """
        Keep at most `max_each` non-disliked and `max_each` disliked rows.
        Disliked rows are always placed at the bottom.

        Total rows <= 2 * max_each.
        """

        # normal / liked
        not_disliked_df = df[df[reaction_col] != -1].head(max_each)

        # disliked
        disliked_df = df[df[reaction_col] == -1].head(max_each)

        # merge with disliked at bottom
        return pd.concat([not_disliked_df, disliked_df], ignore_index=True)

    @staticmethod
    def final_sort(result: List[Dict]) -> List[Dict]:
        """
        Sort items with reaction != -1 by score (desc) and physical_id (desc),
        keep disliked items at the bottom, and cap each group at 20 rows.

        Note: disliked is already sorted by date (not by score like the normal)
              -> order is kept in this function.
        """
        df = pd.DataFrame(result)

        normal, disliked = df[df["reaction"] != -1], df[df["reaction"] == -1]

        # Step 1: Sort the reaction != -1 by score and physical_id (secondary)
        normal = SimilaritySearchService.sort_by_score_and_physical_id(normal)

        # Step 2: Merge normal with disliked
        out = pd.concat([normal, disliked], ignore_index=True)

        out = SimilaritySearchService.limit_and_merge_by_reaction(out)

        return out.to_dict("records")

    @staticmethod
    def build_disliked_retrievals(
        disliked_phys_ids: Set[Tuple[int, datetime]],
        org_id: int,
        version: Union[str, List[str]] = ["3d", "v3"],
    ) -> pd.DataFrame:
        """
        Build a DataFrme of disliked items by fetching their metadata
        from ElasticSearch.
        """

        if not disliked_phys_ids:
            return pd.DataFrame(
                columns=["physical_id", "score", "parent", "reaction", "datetime"]
            )

        phys_id_to_dt = {pid: dt for pid, dt in disliked_phys_ids}
        phys_id_list = list(phys_id_to_dt.keys())

        # normalise version to list
        if isinstance(version, str):
            version_filter = {"term": {"version": version}}
        else:
            version_filter = {"terms": {"version": version}}

        resp = elasticsearch_db.client.search(
            index=constants.ELASTICSEARCH_PREFIX,
            body={
                "size": 10000,
                "_source": ["original_image", "path", "product_id"],
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {"product_id": phys_id_list}},
                            version_filter,
                            {"term": {"organization_id": org_id}},
                        ]
                    }
                },
            },
        )

        rows = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            pid = src["product_id"]

            # fallback: use path if exists, else original_image
            parent = src.get("path") or src.get("original_image")

            rows.append(
                {
                    "physical_id": pid,
                    "score": 0.0,
                    "parent": parent,
                    "reaction": -1,
                    "datetime": phys_id_to_dt.get(pid),
                }
            )

        df = pd.DataFrame(
            rows, columns=["physical_id", "score", "parent", "reaction", "datetime"]
        )
        return df.sort_values("datetime", ascending=False).reset_index(drop=True)

    def filter_and_sync_neutral_feedback_2d(
        self,
        feedback_list: List[Tuple[Any, int]],
        disliked_phys_ids: Set,
        matches: Dict,
    ) -> Set:
        """
        Remove neutralised dislikes and sync changes to Elasticsearch.
        """

        # Step 1: find neutral physical_ids in feedback_list
        neutral_phys_ids = {
            phys_id for phys_id, reaction in feedback_list if reaction == 0
        }

        # Step 1.5: Find disliked items that are now neutral
        neutral_in_disliked = {
            (phys_id, ts)
            for phys_id, ts in disliked_phys_ids
            if phys_id in neutral_phys_ids
        }

        # Step 2: Remove neutral from disliked_phys_ids
        if neutral_in_disliked:
            disliked_phys_ids -= neutral_in_disliked

            # Step 3: update ES (remove from interactions)
            self.rocchio_feedback_2d.remove_interactions_by_physical_ids(
                constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
                doc_id=matches["doc_id"],
                physical_ids=[physical_id for (physical_id, _) in neutral_in_disliked],
            )

        return disliked_phys_ids

    @staticmethod
    def apply_disliked_drawings(
        final_result: pd.DataFrame,
        disliked_phys_ids: Set,
        company_id: int,
        show_disliked_drawings: bool,
    ) -> pd.DataFrame:
        """Filter disliked drawings from results and optionally append them back."""

        if not disliked_phys_ids:
            return final_result

        disliked_id_set = {pid for pid, _ in disliked_phys_ids}

        # Always remove disliked from main result
        filtered = final_result[
            ~final_result["physical_id"].isin(disliked_id_set)
        ].reset_index(drop=True)

        if not show_disliked_drawings:
            return filtered

        disliked_df = SimilaritySearchService.build_disliked_retrievals(
            disliked_phys_ids=disliked_phys_ids, org_id=company_id
        ).drop(columns=["datetime"])

        return pd.concat([filtered, disliked_df], ignore_index=True)

    @staticmethod
    def build_product_embedding_map(resp: Dict) -> Dict[int, np.ndarray]:
        """
        Build a mapping from product_id to embedding_vector_v3 (as numpy array).

        Skips records where embedding_vector_v3 is None or missing.
        """

        result: Dict[int, np.ndarray] = {}

        hits = resp["hits"]["hits"]
        for hit in hits:
            src = hit["_source"]
            product_id = src.get("product_id")
            emb = src.get("embedding_vector_v3")

            if product_id is None or emb is None:
                continue

            result[product_id] = np.array(emb, dtype=np.float32)

        return result

    def ranking_project_ref(
        self,
        project_id: str,
        company_id: str,
        ocr_result,
        basic_info_metadata: pd.DataFrame,
        feedback_list: List[Tuple] = [],
        show_disliked_drawings: bool = True,
    ):
        try:
            image_info = elasticsearch_db.find(
                indice_name=constants.ZONE_ENCODED_DATA_PHYSICAL_OBJECT_2D,
                field_name="object_id",
                value=project_id,
            )
            crop_base64 = [x for x in image_info["hits"]["hits"]]
            list_query = []
            for image in crop_base64:
                # crop = base64_to_image(image["image_base64"])
                crop = base64_to_image(image["_source"]["data"])
                list_query.append(self.extract_feature(crop))
            similar_df = self.search_similar_project(
                list_query, company_id, project_id, feedback_list
            )
            if similar_df is None:
                return [], []
        except Exception as e:
            logger.error(traceback.format_exc())
            raise SearchException(
                f"search failed {e}", errors=ErrorCodeAISystem.RAI_SYS_003.name
            )

        try:
            # query product information
            try:
                if basic_info_metadata.empty:
                    result_similarity_by_ocr = pd.DataFrame(
                        columns=["physical_id", "parent", "score"]
                    )
                else:
                    basic_info_metadata = basic_info_metadata.rename(
                        columns={
                            "id": "physical_id",
                            "product_no": "ocr_product_code",
                            "product_name": "ocr_product_name",
                            "figure_number": "ocr_drawing_number",
                            "customer_name": "ocr_drawing_issuer",
                            "location": "parent",
                        }
                    )
                    mask_should = pd.Series(False, index=basic_info_metadata.index)
                    if ocr_result.ocr_product_name:
                        mask_should |= (
                            basic_info_metadata["ocr_product_name"]
                            == ocr_result.ocr_product_name
                        )
                    if ocr_result.ocr_product_code:
                        mask_should |= (
                            basic_info_metadata["ocr_product_code"]
                            == ocr_result.ocr_product_code
                        )
                    if ocr_result.ocr_drawing_number:
                        mask_should |= (
                            basic_info_metadata["ocr_drawing_number"]
                            == ocr_result.ocr_drawing_number
                        )
                    if ocr_result.ocr_drawing_issuer:
                        mask_should |= (
                            basic_info_metadata["ocr_drawing_issuer"]
                            == ocr_result.ocr_drawing_issuer
                        )

                    result_similarity_by_ocr = basic_info_metadata[mask_should]
                    if result_similarity_by_ocr.empty:
                        result_similarity_by_ocr = pd.DataFrame(
                            columns=["physical_id", "parent", "score"]
                        )
                    else:
                        result_similarity_by_ocr["score"] = 1
                        result_similarity_by_ocr[
                            "ocr_product_name"
                        ] = result_similarity_by_ocr["ocr_product_name"].apply(
                            lambda x: (
                                self.check_production_info(x) if pd.notna(x) else x
                            )
                        )
                        result_similarity_by_ocr[
                            "ocr_product_code"
                        ] = result_similarity_by_ocr["ocr_product_code"].apply(
                            lambda x: (
                                self.check_production_info(x) if pd.notna(x) else x
                            )
                        )
                        result_similarity_by_ocr[
                            "ocr_drawing_number"
                        ] = result_similarity_by_ocr["ocr_drawing_number"].apply(
                            lambda x: (
                                self.check_production_info(x) if pd.notna(x) else x
                            )
                        )
                        result_similarity_by_ocr[
                            "ocr_drawing_issuer"
                        ] = result_similarity_by_ocr["ocr_drawing_issuer"].apply(
                            lambda x: (
                                self.check_production_info(x) if pd.notna(x) else x
                            )
                        )
                        result_similarity_by_ocr = result_similarity_by_ocr[
                            (result_similarity_by_ocr["ocr_product_name"] == "answer")
                            | (result_similarity_by_ocr["ocr_product_code"] == "answer")
                            | (
                                result_similarity_by_ocr["ocr_drawing_number"]
                                == "answer"
                            )
                            | (
                                result_similarity_by_ocr["ocr_drawing_issuer"]
                                == "answer"
                            )
                        ]
                        result_similarity_by_ocr = result_similarity_by_ocr[
                            ["physical_id", "parent", "score"]
                        ]
            except Exception as e:
                logger.warning(traceback.format_exc())
                result_similarity_by_ocr = pd.DataFrame(
                    columns=["physical_id", "parent", "score"]
                )
            similar_df["number_detected_object"] = len(crop_base64)
            similar_df_score_higher = similar_df[similar_df["score"] >= 0.7]
            similar_df_score_lower = similar_df[similar_df["score"] < 0.7]
            similar_df_score_lower = similar_df_score_lower[
                ~similar_df_score_lower["product_id"].isin(
                    similar_df_score_higher["product_id"].to_list()
                )
            ]

            similar_df = pd.concat(
                [similar_df_score_higher, similar_df_score_lower], ignore_index=True
            )

            parent_df = similar_df[["original_image", "parent"]].drop_duplicates()

            final_df = (
                similar_df.groupby(
                    ["parent", "number_objects", "number_detected_object"]
                )
                .agg({"index": "count", "score": "sum"})
                .reset_index()
                .rename(columns={"index": "count_appearance"})
                .sort_values(by="count_appearance", ascending=False)
                .reset_index(drop=True)
            )
            product_id_keep = similar_df[["parent", "product_id"]].drop_duplicates(
                subset="parent"
            )
            final_df["denominator"] = final_df[
                ["number_objects", "number_detected_object", "count_appearance"]
            ].max(axis=1)
            final_df["score"] = final_df["score"] / (
                final_df["denominator"]
                + abs(final_df["number_detected_object"] - final_df["number_objects"])
            )
            final_df = final_df.sort_values(by="score", ascending=False)

            results = pd.merge(final_df, product_id_keep, on="parent")
            results = results.rename(columns={"product_id": "physical_id"})
            results = results.drop_duplicates(subset=["physical_id"], keep="first")
            results = results.sort_values(by="score", ascending=False).reset_index()

            results = pd.merge(
                results[["physical_id", "score", "parent"]], parent_df, on="parent"
            )
            results = results.drop(columns=["parent"])
            results = results.rename(columns={"original_image": "parent"})
            final_result_ocr = pd.concat([result_similarity_by_ocr, results])
            final_result_ocr = (
                final_result_ocr.sort_values(by="score", ascending=False)
                .drop_duplicates(subset=["parent", "physical_id"], keep="first")
                .reset_index()
            )
            final_result_ocr["physical_id"] = final_result_ocr["physical_id"].astype(
                int
            )
            results_v2 = final_result_ocr[["physical_id", "score", "parent"]]
            results_v2["version"] = "v2"

        except Exception as e:
            logger.error(traceback.format_exc())
            raise AISystemError(
                f"system processed failed {e}",
                errors=ErrorCodeAISystem.RAI_SYS_004.name,
            )

        ### Raijin search V3 (safe version)
        try:
            results_v3, image_phash = similarity_search_v3.search_similar_project_v3(
                project_id, company_id
            )

            # Ensure both results DataFrames exist and have consistent columns
            required_cols = ["physical_id", "score", "parent", "version"]
            for df, version_name in [(results_v2, "v2"), (results_v3, "v3")]:
                if df is None:
                    df = pd.DataFrame(columns=required_cols)
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = None
                df["version"] = version_name
                # Assign back if it's results_v2 or results_v3
                if version_name == "v2":
                    results_v2 = df
                else:
                    results_v3 = df

            # Combine both sets
            final_result = pd.concat([results_v3, results_v2], ignore_index=True)

            # Ensure numeric score column
            final_result["score"] = pd.to_numeric(
                final_result["score"], errors="coerce"
            )

            # Early exit if both are empty
            if final_result["score"].isna().all():
                logger.warning(
                    "Both results_v2 and results_v3 are empty → returning []"
                )
                return [], []

            # Sort and deduplicate
            final_result = SimilaritySearchService.sort_by_score_and_physical_id(
                final_result
            )
            final_result = final_result.drop_duplicates(
                subset=["physical_id"], keep="first"
            ).reset_index(drop=True)

            # Remove disliked drawings
            _, matches = self.rocchio_feedback_2d.retrieve_matches(
                project_id, company_id
            )
            disliked_phys_ids = set(
                (inter["physical_id"], datetime.fromisoformat(inter["timestamp"]))
                for inter in matches["interactions"]
                if inter["reaction"] == -1
            )

            disliked_phys_ids = self.filter_and_sync_neutral_feedback_2d(
                feedback_list, disliked_phys_ids, matches
            )

            # ---- Two-Way Response START ----

            # -- Step 1: Get dislikes' phases (Currently, only support 2D) --
            disliked_info_response = elasticsearch_db.find(
                indice_name=constants.ELASTICSEARCH_PREFIX,
                query={
                    "query": {
                        "bool": {
                            "filter": [
                                {
                                    "terms": {
                                        "product_id": [
                                            phys_id for phys_id, _ in disliked_phys_ids
                                        ]
                                    }
                                },
                                {"term": {"organization_id": company_id}},
                                {
                                    "term": {"version": "v3"}
                                },  # Currently, only assume 2D is disliked
                            ]
                        }
                    }
                },
            )

            disliked_phashes = SimilaritySearchService.build_product_embedding_map(
                disliked_info_response
            )

            print(f"disliked_phashes: {disliked_phashes}")

            # -- Step 1: End --

            # -- Step 2: Get cluster info (Currently, only support 2D) --
            cluster_info = self.similarity_clusters.search_cluster(
                vector=image_phash,
                org_id=company_id,
                search_field="embedding_vector_2D",
            )

            # -- Step 2: End --

            # -- Step 3: Add the cluster_id to the Rocchio documents of these dislikes (Currently, only support 2D) --
            for key, val in disliked_phashes.items():
                new_doc = {
                    "phash_2d": val,
                    "type": "2d",
                    "disliked_cluster_ids": [
                        {
                            "cluster_id": cluster_info["doc_id"],
                            "timestamp": datetime.now(),
                        }
                    ],
                    "org_id": company_id,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                }

                # From this phash, find similar document in the Rocchio index
                # What function does this in feedback_2d.py

                matches = self.rocchio_feedback_2d.find_similar_rocchio_record(
                    query_vector=val,
                    search_field="phash_2d",
                    check_and_create=True,
                    filters=[
                        {"term": {"type": "2d"}},
                        {"term": {"org_id": company_id}},
                    ],
                )

                # If there is 1 match exists, use update
                if matches.get("doc_id", None) is not None:
                    new_cluster_id = cluster_info["doc_id"]
                    now = datetime.now()

                    update_body = {
                        "script": {
                            "lang": "painless",
                            "source": """
                                if (ctx._source.disliked_cluster_ids == null) {
                                    ctx._source.disliked_cluster_ids = [];
                                }

                                boolean exists = false;
                                for (item in ctx._source.disliked_cluster_ids) {
                                    if (item.cluster_id == params.cluster_id) {
                                        exists = true;
                                        break;
                                    }
                                }

                                if (!exists) {
                                    ctx._source.disliked_cluster_ids.add([
                                        "cluster_id": params.cluster_id,
                                        "timestamp": params.timestamp
                                    ]);
                                }

                                ctx._source.updated_at = params.timestamp;
                            """,
                            "params": {
                                "cluster_id": new_cluster_id,
                                "timestamp": now,
                            },
                        }
                    }

                    elasticsearch_db.client.update(
                        index=constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
                        id=matches["doc_id"],
                        body=update_body,
                        refresh="wait_for",
                    )

                else:
                    new_doc = {
                        "phash_2d": val,
                        "type": "2d",
                        "disliked_cluster_ids": [
                            {
                                "cluster_id": cluster_info["doc_id"],
                                "timestamp": datetime.now(),
                            }
                        ],
                        "org_id": company_id,
                        "created_at": datetime.now(),
                        "updated_at": datetime.now(),
                    }

                    elasticsearch_db.client.index(
                        index=constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
                        document=new_doc,
                        refresh="wait_for",
                    )

            # -- Step 3: End --

            # ---- Two-Way Response END ----

            final_result["reaction"] = 0
            final_result = self.apply_disliked_drawings(
                final_result, disliked_phys_ids, company_id, show_disliked_drawings
            )

            return final_result[["physical_id", "score", "parent", "reaction"]].to_dict(
                "records"
            ), final_result[
                ["physical_id", "score", "parent", "version", "reaction"]
            ].to_dict(
                "records"
            )

        except Exception as e:
            logger.error(traceback.format_exc())
            raise AISystemError(
                f"system processed failed {e}",
                errors=ErrorCodeAISystem.RAI_SYS_004.name,
            )


class DummyOCRResult:
    def __init__(
        self,
        ocr_product_name=None,
        ocr_product_code=None,
        ocr_drawing_number=None,
        ocr_drawing_issuer=None,
    ):
        self.ocr_product_name = ocr_product_name
        self.ocr_product_code = ocr_product_code
        self.ocr_drawing_number = ocr_drawing_number
        self.ocr_drawing_issuer = ocr_drawing_issuer


if __name__ == "__main__":
    import pandas as pd

    from get_diagram_ocr import get_diagram_ocr_physical_types

    similarity_search_service = SimilaritySearchService()

    project_id = "1770382286045"
    company_id = "3"
    ocr_result = DummyOCRResult()
    basic_info_metadata = get_diagram_ocr_physical_types(company_id)
    feedback_list = []
    show_disliked_drawings = True

    res = similarity_search_service.ranking_project_ref(
        project_id=project_id,
        company_id=company_id,
        ocr_result=ocr_result,
        basic_info_metadata=basic_info_metadata,
        feedback_list=feedback_list,
        show_disliked_drawings=show_disliked_drawings,
    )

    print(f"res: {res}")
