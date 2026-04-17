import argparse
import logging
import time
from datetime import datetime
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torchvision import transforms

import logger
from config import settings
from constants import constants
from feedback_2d import RocchioFeedback2D
from interaction import elasticsearch_db
from model_loader_utils import model_loader
from similarity_clusters import SimilarityClusters

logger = logging.getLogger("benchmark_dislikes")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

file_handler = logging.FileHandler("benchmark_process_dislikes_2d.txt")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# IMPORTANT: prevent propagation to root logger
logger.propagate = False


class SimilaritySearchService:
    def __init__(self):
        self.device = settings.DEVICE
        self.model_text_classification = model_loader.load_word_classification_model()
        self.model_extract = model_loader.load_model_extraction(
            model_name=constants.MODEL_EXTRACTION_NAME
        )
        self.rocchio_feedback_2d = RocchioFeedback2D(elasticsearch_db)
        self.similarity_clusters_2d = SimilarityClusters(
            elasticsearch_db=elasticsearch_db, mode="2d"
        )

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

    @staticmethod
    def _es_search_sequential(list_query_vector, company_id, selected_cols):
        indice_name = constants.ELASTICSEARCH_PREFIX
        result_similarity = pd.DataFrame()

        for query_vector in list_query_vector:
            search_results = elasticsearch_db.search_vector(
                indice_name=indice_name,
                organization_id=company_id,
                version="v2",
                query_vector=query_vector,
                selected_cols=selected_cols,
            )

            if (
                not search_results
                or "hits" not in search_results
                or "hits" not in search_results["hits"]
            ):
                continue

            hits = search_results["hits"]["hits"]
            if not hits:
                continue

            df_res = pd.json_normalize(hits)
            if df_res.empty:
                continue

            result_similarity = pd.concat(
                [result_similarity, df_res], ignore_index=True
            )

        return result_similarity

    @staticmethod
    def _es_search_batch(list_query_vector, company_id, selected_cols):
        indice_name = constants.ELASTICSEARCH_PREFIX
        result_similarity = pd.DataFrame()

        batch_results = elasticsearch_db.search_vectors_batch(
            indice_name=indice_name,
            organization_id=company_id,
            version="v2",
            list_query_vector=list_query_vector,
            selected_cols=selected_cols,
        )

        for res in batch_results["responses"]:
            if "hits" not in res or not res["hits"]["hits"]:
                continue

            hits = res["hits"]["hits"]
            if not hits:
                continue

            df_res = pd.json_normalize(hits)
            if df_res.empty:
                continue

            result_similarity = pd.concat(
                [result_similarity, df_res], ignore_index=True
            )

        return result_similarity

    def search_similar_project(
        self,
        list_query_vector,
        company_id: str,
    ) -> pd.DataFrame:
        indice_name = constants.ELASTICSEARCH_PREFIX
        if not elasticsearch_db.check_indice_existance(indice_name=indice_name):
            return None

        result_similarity = pd.DataFrame()
        selected_cols = [
            "product_id",
            "images_paths",
            "original_image",
            "number_objects",
        ]

        # ───────────── Benchmark batch (_msearch) ─────────────
        start_time = time.perf_counter()
        result_similarity = self._es_search_batch(
            list_query_vector=list_query_vector,
            company_id=company_id,
            selected_cols=selected_cols,
        )
        end_time = time.perf_counter()
        logger.info(f"[ES Batch] {end_time - start_time:.4f}s")

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
        if not result:
            return result

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

    @staticmethod
    def apply_disliked_drawings(
        final_result: pd.DataFrame,
        disliked_phys_ids: Set[Tuple[int, datetime]],
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
    def merge_latest_by_phys_id(
        phys_id_ts_pairs: Set[Tuple[int, datetime]],
    ) -> Set[Tuple[int, datetime]]:
        """
        Deduplicate (physical_id, timestamp) pairs by keeping the newest timestamp.
        """
        latest = {}

        for phys_id, ts in phys_id_ts_pairs:
            if phys_id not in latest or ts > latest[phys_id]:
                latest[phys_id] = ts

        return {(pid, ts) for pid, ts in latest.items()}

    def process_dislikes_2d(
        self,
        result: pd.DataFrame,
        project_id: int,
        company_id: int,
        show_disliked_drawings: bool,
    ) -> pd.DataFrame:
        """Adjust results using Rocchio 2D dislike feedback."""

        t0 = time.perf_counter()

        # Step 1: Retrieve match
        t1 = time.perf_counter()
        _, match = self.rocchio_feedback_2d.retrieve_2d_exact_match(
            project_id, company_id
        )
        t2 = time.perf_counter()

        # Step 2: Build cluster_timestamp_map
        disliked_cluster_info = match.get("disliked_cluster_ids", [])
        cluster_timestamp_map = {
            info["cluster_id"]: datetime.fromisoformat(info["timestamp"])
            for info in disliked_cluster_info
            if "cluster_id" in info and "timestamp" in info
        }
        t3 = time.perf_counter()

        if not cluster_timestamp_map:
            logger.info(
                "[benchmark] process_dislikes_2d early exit | total=%.4f",
                t3 - t0,
            )
            return result

        # Step 3: Elasticsearch mget
        resp = elasticsearch_db.client.mget(
            index=constants.SIMILARITY_CLUSTERS,
            body={"ids": list(cluster_timestamp_map.keys())},
        )
        t4 = time.perf_counter()

        # Step 4: Neutralisations
        neutralisations = match.get("neutralisations", [])
        neutralised_physical_ids = {
            n["physical_id"] for n in neutralisations if "physical_id" in n
        }
        t5 = time.perf_counter()

        # Step 5: Build disliked_phys_ids
        disliked_phys_ids: Set[Tuple[int, datetime]] = set()

        for doc in resp.get("docs", []):
            if not doc.get("found"):
                continue

            cluster_id = doc["_id"]
            cluster_timestamp = cluster_timestamp_map.get(cluster_id)

            for pid in doc.get("_source", {}).get("physical_ids", []):
                if pid in neutralised_physical_ids:
                    continue
                disliked_phys_ids.add((pid, cluster_timestamp))

        t6 = time.perf_counter()

        # Step 6: Apply handler
        result = self.apply_disliked_drawings(
            final_result=result,
            disliked_phys_ids=disliked_phys_ids,
            company_id=company_id,
            show_disliked_drawings=show_disliked_drawings,
        )
        t7 = time.perf_counter()

        # Final log
        logger.info(
            "[benchmark] process_dislikes_2d | total=%.4f | "
            "retrieve=%.4f | map=%.4f | es=%.4f | neutral=%.4f | loop=%.4f | apply=%.4f | "
            "clusters=%d | disliked=%d",
            t7 - t0,
            t2 - t1,
            t3 - t2,
            t4 - t3,
            t5 - t4,
            t6 - t5,
            t7 - t6,
            len(cluster_timestamp_map),
            len(disliked_phys_ids),
        )

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run process_dislikes_2d benchmark")

    parser.add_argument(
        "--project_id",
        type=str,
        required=True,
        help="Project ID",
    )

    parser.add_argument(
        "--company_id",
        type=str,
        required=True,
        help="Company ID",
    )

    parser.add_argument(
        "--show_disliked",
        action="store_true",
        help="Whether to show disliked drawings",
    )

    args = parser.parse_args()

    similarity_search_service = SimilaritySearchService()

    result = similarity_search_service.process_dislikes_2d(
        result=pd.DataFrame(),
        project_id=args.project_id,
        company_id=args.company_id,
        show_disliked_drawings=args.show_disliked,
    )

    print(result)
