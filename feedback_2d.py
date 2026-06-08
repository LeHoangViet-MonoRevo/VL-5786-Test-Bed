import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import fitz  # PyMuPDF
import imagehash
import numpy as np
from elasticsearch import helpers as es_helpers
from PIL import Image

from constants import constants
from elasticsearch_constants import es_constant
from feedback_base import RocchioFeedbackBase
from model_loader_utils import base64_to_image


class RocchioFeedback2D(RocchioFeedbackBase):
    def __init__(self, elasticsearch_db, similarity_threshold=0.98):
        super().__init__(elasticsearch_db, similarity_threshold)

    def _load_original_image_b64(self, project_id: Any) -> str:
        res = self.elasticsearch_db.find(
            indice_name=constants.ZONE_ENCODED_DATA_PHYSICAL_OBJECT,
            field_name="object_id",
            value=project_id,
        )
        data = res["hits"]["hits"][0]["_source"]["data"]
        dtype = res["hits"]["hits"][0]["_source"]["type"]
        return data, dtype

    @staticmethod
    def compute_phash_vector_from_image(
        img: Image.Image,
        hash_size: int = 32,
        normalise: bool = True,
        dtype: type = np.float32,
        binary_output: bool = True,
    ) -> np.ndarray:
        """Compute phash vector from a PIL Image."""
        img_hash = imagehash.phash(img, hash_size=hash_size)
        bits = bin(int(str(img_hash), 16))[2:].zfill(hash_size * hash_size)
        v = np.array([int(b) for b in bits], dtype=dtype if binary_output else int)

        if normalise:
            v = v / (np.linalg.norm(v) + 1e-8)

        return v

    @staticmethod
    def compute_phash_vector(
        img_b64: str,
        hash_size: int = 32,
        normalise: bool = True,
        dtype: type = np.float32,
        binary_output: bool = True,
    ) -> np.ndarray:
        """Compute phash from base64-encoded image."""
        img = base64_to_image(img_b64)
        return RocchioFeedback2D.compute_phash_vector_from_image(
            img,
            hash_size=hash_size,
            normalise=normalise,
            dtype=dtype,
            binary_output=binary_output,
        )

    @staticmethod
    def infer_phash_from_data(
        data: str,
        dtype: str,
        hash_size: int = 32,
        normalise: bool = True,
        vector_dtype: type = np.float32,
        binary_output: bool = True,
    ) -> np.ndarray:
        """Infer phash vector from data, either image or PDF."""

        if dtype not in ["image", "pdf"]:
            raise NotImplementedError(f"dtype={dtype} not supported")

        if dtype == "image":
            return RocchioFeedback2D.compute_phash_vector(
                data,
                hash_size=hash_size,
                normalise=normalise,
                dtype=vector_dtype,
                binary_output=binary_output,
            )

        if dtype == "pdf":
            # Step 1: Open PDF from base64
            pdf_bytes = base64.b64decode(data)
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            if pdf_doc.page_count == 0:
                raise ValueError("PDF has no pages")

            # Step 2: Render only the first page to a PIL image
            page = pdf_doc[0]
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Step 3: Compute phash directly from image
            return RocchioFeedback2D.compute_phash_vector_from_image(
                img,
                hash_size=hash_size,
                normalise=normalise,
                dtype=vector_dtype,
                binary_output=binary_output,
            )

    def _apply_rocchio_decision(
        self,
        query_vectors: List[np.ndarray],
        feedback_list: List[Tuple],
        embedding_index: str,
        org_id: str,
        object_id: str,
        phash_2d: Optional[List],
        match: Dict,
    ):
        if feedback_list:
            pos_vecs, neg_vecs = self._extract_feedback_vectors(
                feedback_list,
                embedding_index,
                embedding_vector_field="embedding_vector_v2",
                routing_key=f"v2___{str(org_id)}",
            )

            updated, pos_mean, neg_mean = self.rocchio_update(
                query_vectors, pos_vecs, neg_vecs, return_intermediates=True
            )

            pos_mean = self.add_epsilon(pos_mean)
            neg_mean = self.add_epsilon(neg_mean)

            interactions_save_in_es = self.convert_feedback_to_interactions(
                feedback_list,
                es_constant.SCHEMA_ROCCHIO_HISTORY_PHYSICAL_OBJECT["mappings"][
                    "properties"
                ]["interactions"]["properties"].keys(),
            )

            if (
                match["similarity_score"] is not None
                and match["similarity_score"] >= self.similarity_threshold
            ):
                # UPDATE MODE
                doc_id = match["doc_id"]

                # update_body = {
                #     "doc": {
                #         "rocchio_pos_vec_2d": pos_mean.tolist(),
                #         "rocchio_neg_vec_2d": neg_mean.tolist(),
                #         "interactions": interactions_save_in_es,
                #         "updated_at": datetime.now(),
                #     }
                # }

                update_body = {
                    "script": {
                        "lang": "painless",
                        "source": """
                            if (ctx._source.interactions == null) {
                                ctx._source.interactions = [];
                            }

                            Map seen = new HashMap();

                            // existing interactions
                            for (item in ctx._source.interactions) {
                                String key = item.physical_id.toString();
                                seen.put(key, item);
                            }

                            // incoming interactions
                            for (item in params.interactions) {
                                String key = item.physical_id.toString();

                                if (seen.containsKey(key)) {
                                    def oldItem = seen.get(key);

                                    // if reaction is the same -> keep old one
                                    if (oldItem.reaction == item.reaction) {
                                        continue;
                                    }
                                }

                                // otherwise overwrite (latest wins)
                                seen.put(key, item);
                            }

                            ctx._source.interactions = new ArrayList(seen.values());

                            ctx._source.rocchio_pos_vec_2d = params.rocchio_pos_vec_2d;
                            ctx._source.rocchio_neg_vec_2d = params.rocchio_neg_vec_2d;
                            ctx._source.updated_at = params.updated_at;
                        """,
                        "params": {
                            "interactions": interactions_save_in_es,
                            "rocchio_pos_vec_2d": pos_mean.tolist(),
                            "rocchio_neg_vec_2d": neg_mean.tolist(),
                            "updated_at": datetime.now(),
                        },
                    }
                }

                self.elasticsearch_db.client.update(
                    index=constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
                    id=doc_id,
                    body=update_body,
                    refresh="wait_for",
                )
            else:
                # CREATE MODE
                new_doc = {
                    "id": object_id,
                    "phash_2d": phash_2d,
                    "rocchio_pos_vec_2d": pos_mean.tolist(),
                    "rocchio_neg_vec_2d": neg_mean.tolist(),
                    "type": "2d",
                    "interactions": interactions_save_in_es,
                    "org_id": org_id,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                }

                self.elasticsearch_db.client.index(
                    index=constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
                    document=new_doc,
                    refresh="wait_for",
                )

            return updated

        if (
            match["similarity_score"] is not None
            and match["similarity_score"] >= self.similarity_threshold
        ):
            rocchio_pos, rocchio_neg = [], []

            # Only append vectors if they exist in ES
            if match["rocchio_pos_vec_2d"] is not None:
                rocchio_pos.append(match["rocchio_pos_vec_2d"])

            if match["rocchio_neg_vec_2d"] is not None:
                rocchio_neg.append(match["rocchio_neg_vec_2d"])

            # If no vectors are present, return original vectors
            if not rocchio_pos and not rocchio_neg:
                return query_vectors

            return self.rocchio_update(query_vectors, rocchio_pos, rocchio_neg)

        return query_vectors

    def retrieve_2d_exact_match(self, project_id: str, org_id: str) -> List:
        """
        Retrieve 2D exact match in Rocchio index using project_id.
        Filtered by org_id and similarity_threshold.
        """

        # TODO: Add the lru_cache to the hash_vector
        data, dtype = self._load_original_image_b64(project_id)
        hash_vector = self.infer_phash_from_data(data, dtype)

        # TODO: Add the option to search by id to save time
        match = self.find_similar_rocchio_record(
            query_vector=hash_vector,
            search_field="phash_2d",
            check_and_create=True,
            filters=[{"term": {"type": "2d"}}, {"term": {"org_id": org_id}}],
        )

        return hash_vector, match

    def _load_disliked_cluster_info(self, disliked_cluster_ids: List[str]) -> dict:
        """
        Return:
        {
            cluster_id: {
                "physical_ids": ...,
                "embedding": ...,
                "version": ...
            }
        }
        """
        if not disliked_cluster_ids:
            return {}

        es = self.elasticsearch_db.client

        # ES 9 strips dense_vector from _source and mget cannot request
        # `fields`. Use a search with an ids query so the embedding is
        # recoverable via `fields`; _hydrate_vector_fields merges it into
        # _source only when missing, keeping older indices working.
        res = es.search(
            index=constants.SIMILARITY_CLUSTERS,
            body={
                "size": len(disliked_cluster_ids),
                "query": {"ids": {"values": disliked_cluster_ids}},
                "_source": ["cluster_id", "physical_ids", "version"],
                "fields": ["embedding_vector_2d", "embedding_vector_3d"],
            },
        )
        res = self.elasticsearch_db._hydrate_vector_fields(res)

        disliked_cluster_info = {}

        for doc in res["hits"]["hits"]:
            src = doc["_source"]

            # Use _id if cluster_id is not stored in _source
            cluster_id = src.get("cluster_id", doc["_id"])

            if src.get("version") == "v3":
                version = "v3"
                emb_field = "embedding_vector_2d"
            elif src.get("version") == "3d":
                version = "3d"
                emb_field = "embedding_vector_3d"
            else:
                raise ValueError

            disliked_cluster_info[cluster_id] = {
                "physical_ids": src["physical_ids"],
                "embedding": src[emb_field],
                "version": version,
            }

        return disliked_cluster_info

    def _load_disliked_clusters_from_match(
        self, match: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Load cluster info from Rocchio match."""
        cluster_ids = [item["cluster_id"] for item in match["disliked_cluster_ids"]]
        return self._load_disliked_cluster_info(cluster_ids)

    def _get_remaining_feedback(
        self,
        feedback_list: List[Tuple[int, int]],
        disliked_cluster_info: Dict[str, Dict[str, Any]],
    ) -> List[Tuple[int, int]]:
        """Remove known disliked physical IDs."""
        known_ids = []
        for v in disliked_cluster_info.values():
            known_ids.extend(v["physical_ids"])

        return [(pid, r) for pid, r in feedback_list if pid not in known_ids]

    def _process_remaining_feedback(
        self,
        remaining_feedback: List[Tuple[int, int]],
        org_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Cluster remaining disliked items."""
        rem_info = self.fetch_representation_embeddings_from_raijin_search_indexer(
            physical_ids=[pid for pid, _ in remaining_feedback],
            company_id=org_id,
        )

        new_clusters: Dict[str, Dict[str, Any]] = {}

        # TODO: Parallelize _find_or_create_cluster calls with ThreadPoolExecutor (same pattern as _backward_update_disliked_clusters)
        for physical_id, info in rem_info.items():
            cluster_id, cluster_data = self._find_or_create_cluster(
                info["embedding"], info["version"], org_id
            )
            new_clusters[cluster_id] = cluster_data

        return new_clusters

    def _find_or_create_cluster(
        self,
        rep_vec: np.ndarray,
        version: str,
        org_id: str,
    ) -> Tuple[str, Dict[str, Any]]:

        embedding_field = (
            "embedding_vector_3d" if version == "3d" else "embedding_vector_2d"
        )

        filters = [
            {"term": {"org_id": org_id}},
            {"term": {"version": version}},
        ]

        res = self.elasticsearch_db.search_vector_w_filters(
            indice_name=constants.SIMILARITY_CLUSTERS,
            query_vector=rep_vec,
            number_retrieval_vector=1,
            search_field=embedding_field,
            filters=filters,
            vector_method="l2",  # or "l2"
            score_threshold=self.similarity_threshold,
            selected_cols=["physical_ids", "version"],
        )

        print(f"res: {res}")

        hits = res.get("hits", {}).get("hits", []) if res else []

        # ✅ MATCH FOUND
        if hits:
            hit = hits[0]
            src = hit["_source"]
            cluster_id = hit["_id"]

            return cluster_id, {
                "physical_ids": src.get("physical_ids", []),
                "embedding": rep_vec,
                "version": src.get("version", version),
            }

        # ❌ NO MATCH → CREATE
        created = self._create_cluster_from_vector(
            rep_vec, version=version, org_id=org_id
        )

        return created["doc_id"], {
            "physical_ids": created["physical_ids"],
            "embedding": created["embedding"],
            "version": version,
        }

    def _update_or_create_host_doc(
        self,
        match: Dict[str, Any],
        emb_vector: np.ndarray,
        new_clusters: Dict[str, Dict[str, Any]],
        project_id: Union[int, str],
        org_id: str,
        type: str = "2d",  # "2d" or "3d"
    ) -> None:
        """Update or create Rocchio doc."""
        now = datetime.now()

        if (
            match["similarity_score"] is not None
            and match["similarity_score"] >= self.similarity_threshold
        ):
            self._update_host_doc(match["doc_id"], new_clusters, now)
        else:
            self._create_host_doc(
                emb_vector, new_clusters, project_id, org_id, now, type
            )

        # Mirror the ES write into the in-memory match so callers can reuse it
        # without re-fetching (process_dislikes_2d parses timestamp via
        # datetime.fromisoformat → store the ISO string, matching ES read shape).
        match.setdefault("disliked_cluster_ids", [])
        existing_ids = {c.get("cluster_id") for c in match["disliked_cluster_ids"]}
        for cid in new_clusters.keys():
            if cid not in existing_ids:
                match["disliked_cluster_ids"].append(
                    {"cluster_id": cid, "timestamp": now.isoformat()}
                )

    def _update_host_doc(
        self,
        doc_id: str,
        new_clusters: Dict[str, Dict[str, Any]],
        now: datetime,
    ) -> None:
        """Append new clusters."""
        self.elasticsearch_db.client.update(
            index=constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
            id=doc_id,
            refresh="wait_for",
            script={
                "source": """
                if (ctx._source.disliked_cluster_ids == null) {
                    ctx._source.disliked_cluster_ids = [];
                }

                for (c in params.new_clusters) {
                    boolean exists = false;
                    for (e in ctx._source.disliked_cluster_ids) {
                        if (e.cluster_id == c.cluster_id) {
                            exists = true;
                            break;
                        }
                    }
                    if (!exists) {
                        ctx._source.disliked_cluster_ids.add(c);
                    }
                }

                ctx._source.updated_at = params.now;
                """,
                "params": {
                    "new_clusters": [
                        {"cluster_id": cid, "timestamp": now}
                        for cid in new_clusters.keys()
                    ],
                    "now": now,
                },
            },
        )

    def _create_host_doc(
        self,
        vector: np.ndarray,
        new_clusters: Dict[str, Dict[str, Any]],
        project_id: Union[int, str],
        org_id: str,
        now: datetime,
        type: str = "2d",  # "2d" or "3d"
    ) -> None:
        """Create Rocchio doc for both 2D and 3D."""

        if type == "2d":
            vector_field = "phash_2d"
        elif type == "3d":
            vector_field = "embedding_vector_3d"
        else:
            raise ValueError(f"Unsupported version: {type}")

        new_doc = {
            "id": project_id,
            vector_field: vector,
            "type": type,
            "disliked_cluster_ids": [
                {"cluster_id": cid, "timestamp": now} for cid in new_clusters.keys()
            ],
            "org_id": org_id,
            "created_at": now,
            "updated_at": now,
        }

        self.elasticsearch_db.client.index(
            index=constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
            document=new_doc,
            refresh="wait_for",
        )

    def _resolve_rocchio_doc(
        self,
        disliked_cluster_id: str,
        info: Dict[str, Any],
        org_id: str,
    ) -> Dict[str, Any]:
        """Resolve the Rocchio doc_id for a single disliked cluster (runs in thread)."""
        version = info["version"]

        if version == "v3":
            embedding_field = "phash_2d"
            doc_type = "2d"
            filters = [{"term": {"type": "2d"}}, {"term": {"org_id": org_id}}]
        elif version == "3d":
            embedding_field = "embedding_vector_3d"
            doc_type = "3d"
            filters = [{"term": {"type": "3d"}}, {"term": {"org_id": org_id}}]
        else:
            raise ValueError(f"Unknown version: {version}")

        disliked_match = self.find_similar_rocchio_record(
            query_vector=info["embedding"],
            search_field=embedding_field,
            check_and_create=True,
            filters=filters,
        )

        return {
            "disliked_cluster_id": disliked_cluster_id,
            "doc_id": disliked_match.get("doc_id"),
            "embedding_field": embedding_field,
            "doc_type": doc_type,
            "info": info,
        }

    def _build_append_action(
        self,
        doc_id: str,
        host_cluster_id: str,
        now: datetime,
    ) -> Dict[str, Any]:
        """Build a bulk update action (append host cluster to existing doc)."""
        return {
            "_op_type": "update",
            "_index": constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
            "_id": doc_id,
            "script": {
                "lang": "painless",
                "source": """
                    if (ctx._source.disliked_cluster_ids == null) {
                        ctx._source.disliked_cluster_ids = [];
                    }

                    boolean exists = false;
                    for (e in ctx._source.disliked_cluster_ids) {
                        if (e.cluster_id == params.cluster_id) {
                            exists = true;
                            break;
                        }
                    }

                    if (!exists) {
                        ctx._source.disliked_cluster_ids.add(
                            ["cluster_id": params.cluster_id, "timestamp": params.now]
                        );
                    }

                    ctx._source.updated_at = params.now;
                """,
                "params": {
                    "cluster_id": host_cluster_id,
                    "now": now,
                },
            },
        }

    def _build_create_action(
        self,
        embedding_field: str,
        doc_type: str,
        embedding: np.ndarray,
        host_cluster_id: str,
        org_id: str,
        now: datetime,
    ) -> Dict[str, Any]:
        """Build a bulk index action (create new backward Rocchio doc)."""
        return {
            "_op_type": "index",
            "_index": constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
            "_source": {
                embedding_field: embedding,
                "type": doc_type,
                "org_id": org_id,
                "disliked_cluster_ids": [
                    {"cluster_id": host_cluster_id, "timestamp": now}
                ],
                "created_at": now,
                "updated_at": now,
            },
        }

    def _backward_update_disliked_clusters(
        self,
        combined_disliked_clusters: Dict[str, Dict[str, Any]],
        host_cluster_id: str,
        org_id: str,
        max_workers: int = 8,
    ) -> None:
        """Append host cluster to disliked cluster Rocchio docs.

        Optimised: parallel kNN lookups via ThreadPoolExecutor,
        then a single bulk write — no per-doc refresh blocking.
        """
        if not combined_disliked_clusters:
            return

        now = datetime.now()

        # ── Step 1: Parallel kNN lookups ──────────────────────────────────────
        resolved: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._resolve_rocchio_doc,
                    disliked_cluster_id,
                    info,
                    org_id,
                ): disliked_cluster_id
                for disliked_cluster_id, info in combined_disliked_clusters.items()
            }

            for future in as_completed(futures):
                try:
                    resolved.append(future.result())
                except Exception as e:
                    cluster_id = futures[future]
                    print(
                        f"[backward_update] Failed to resolve cluster {cluster_id}: {e}"
                    )

        # ── Step 2: Build bulk actions ─────────────────────────────────────────
        bulk_actions: List[Dict[str, Any]] = []

        for r in resolved:
            if r["doc_id"] is not None:
                bulk_actions.append(
                    self._build_append_action(
                        doc_id=r["doc_id"],
                        host_cluster_id=host_cluster_id,
                        now=now,
                    )
                )
            else:
                bulk_actions.append(
                    self._build_create_action(
                        embedding_field=r["embedding_field"],
                        doc_type=r["doc_type"],
                        embedding=r["info"]["embedding"],
                        host_cluster_id=host_cluster_id,
                        org_id=org_id,
                        now=now,
                    )
                )

        # ── Step 3: Single bulk write ──────────────────────────────────────────
        if bulk_actions:
            es_helpers.bulk(
                self.elasticsearch_db.client,
                bulk_actions,
                refresh=False,  # no per-doc blocking refresh
                raise_on_error=False,  # log failures, don't crash the whole batch
            )

    def _collect_disliked_physical_ids(
        self, combined_disliked_clusters: Dict[str, Dict[str, Any]]
    ) -> Set[int]:
        """Return all disliked physical_ids."""
        return {
            pid
            for info in combined_disliked_clusters.values()
            for pid in info.get("physical_ids", [])
        }

    def _build_neutralisations_to_add(
        self,
        neutral_feedback_list: List[Tuple[int, int]],
        disliked_physical_ids: Set[int],
        now: datetime,
    ) -> List[Dict[str, Any]]:
        """Build neutralisation docs to add."""
        return [
            {"physical_id": pid, "timestamp": now}
            for pid, _ in neutral_feedback_list
            if pid in disliked_physical_ids
        ]

    def _build_neutralisations_to_remove(
        self, feedback_list: List[Tuple[int, int]]
    ) -> List[int]:
        """Build physical_ids to remove."""
        return [pid for pid, reaction in feedback_list if reaction == -1]

    def _update_neutralisations_in_es(
        self,
        host_doc_id: str,
        to_add: List[Dict[str, Any]],
        to_remove: List[int],
        now: datetime,
    ) -> None:
        """Append/remove neutralisations."""
        if not to_add and not to_remove:
            return  # No-op guard: skip the round-trip entirely

        # Removals are a set lookup — cheaper than iterating a list in Painless
        to_remove_set = list(set(to_remove))  # deduplicate before sending

        self.elasticsearch_db.client.update(
            index=constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
            id=host_doc_id,
            refresh=False,  # caller controls refresh; don't block the write path
            retry_on_conflict=3,  # handle optimistic concurrency without crashing
            body={
                "script": {
                    "lang": "painless",
                    "source": """
                        if (ctx._source.neutralisations == null) {
                            ctx._source.neutralisations = new ArrayList();
                        }

                        // Build a removal set for O(1) lookups
                        Set toRemove = new HashSet(params.to_remove);

                        // Remove in a single pass — avoids full HashMap rebuild
                        ctx._source.neutralisations.removeIf(
                            n -> toRemove.contains(n.physical_id)
                        );

                        // Append only truly new entries (avoid duplication)
                        Set existing = new HashSet();
                        for (n in ctx._source.neutralisations) {
                            existing.add(n.physical_id);
                        }

                        for (n in params.to_add) {
                            if (!existing.contains(n.physical_id)) {
                                ctx._source.neutralisations.add(n);
                            }
                        }

                        ctx._source.updated_at = params.now;
                    """,
                    "params": {
                        "to_add": to_add,
                        "to_remove": to_remove_set,
                        "now": now,
                    },
                }
            },
        )

    def execute_feedback_update_2d(
        self,
        project_id: Union[int, str],
        org_id: str,
        feedback_list: List[Tuple[int, int]] = [],
    ) -> Dict:
        """Main Rocchio execution."""

        # Ensure existstance
        self.elasticsearch_db.check_indice_existance(
            indice_name=constants.SIMILARITY_CLUSTERS,
            create=True,
            body=es_constant.SCHEMA_SIMILARITY_CLUSTERS,
        )

        self.elasticsearch_db.check_indice_existance(
            indice_name=constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
            create=True,
            body=es_constant.SCHEMA_ROCCHIO_HISTORY_PHYSICAL_OBJECT,
        )

        neutral_feedback_list = [
            (physical_id, reaction)
            for physical_id, reaction in feedback_list
            if reaction == 0
        ]

        # Step 0: Filter neutral feedback -> Get disliked feedback
        disliked_feedback_list = self.filter_disliked_feedback(feedback_list)

        # Step 1: Retrieve exact match in Rocchio index
        hash_vector, match = self.retrieve_2d_exact_match(project_id, org_id)

        # Step 2: Get all disliked_cluster info (fan out search_cluster in parallel — it only needs hash_vector)
        with ThreadPoolExecutor(max_workers=1) as executor:
            fut_host_cluster = (
                executor.submit(self.search_cluster, hash_vector, "v3", org_id)
                if disliked_feedback_list
                else None
            )
            disliked_cluster_info = self._load_disliked_clusters_from_match(match)

        # Step 3: Forward (Update host in Rocchio + Update dislikes in similarity_clusters + Add disliked clusters to host in Rocchio)
        remaining_feedback = self._get_remaining_feedback(
            disliked_feedback_list, disliked_cluster_info
        )

        if remaining_feedback:
            print(f"remaining_feedback: {remaining_feedback}")
            new_disliked_clusters = self._process_remaining_feedback(
                remaining_feedback, org_id
            )
            self._update_or_create_host_doc(
                match=match,
                emb_vector=hash_vector,
                new_clusters=new_disliked_clusters,
                project_id=project_id,
                org_id=org_id,
                type="2d",
            )

        # Combine
        combined_disliked_clusters = {
            **disliked_cluster_info,
            **(new_disliked_clusters if remaining_feedback else {}),
        }

        # Step 4: Backward (Update dislikes in Rocchio + Append the host cluster to dislikes in Rocchio)
        # Only run when this run actually contains dislikes (reaction == -1). Likes never touch clusters, and
        # neutrals (dislike -> undislike) are handled by Step 5 on an already-existing cluster. Gate on
        # disliked_feedback_list (this run's dislikes), NOT combined_disliked_clusters — the latter carries
        # historical dislikes from the match doc and would wrongly find/create the host cluster on a
        # likes/neutrals-only run.
        if disliked_feedback_list:
            host_cluster_id = fut_host_cluster.result()

            self._backward_update_disliked_clusters(
                combined_disliked_clusters=combined_disliked_clusters,
                host_cluster_id=host_cluster_id["doc_id"],
                org_id=org_id,
            )

        # Step 5: Handle neutralisations
        # If a cluster is newly added in this run → its physical_ids must NOT participate in neutralisation.
        host_doc_id = match.get("doc_id")
        if host_doc_id:
            now = datetime.now()

            # ONLY use clusters that existed BEFORE this run
            disliked_physical_ids_before = self._collect_disliked_physical_ids(
                disliked_cluster_info
            )

            to_add = self._build_neutralisations_to_add(
                neutral_feedback_list,
                disliked_physical_ids_before,
                now,
            )

            to_remove = self._build_neutralisations_to_remove(disliked_feedback_list)

            # TODO: Fuse this call with _update_host_doc (UPDATE mode) into a single Painless script to save one ES round-trip
            self._update_neutralisations_in_es(
                host_doc_id=host_doc_id,
                to_add=to_add,
                to_remove=to_remove,
                now=now,
            )

            # Mirror the ES neutralisation write into the in-memory match so the
            # returned host_match is fresh and reusable by process_dislikes_2d.
            match.setdefault("neutralisations", [])
            if to_remove:
                remove_set = set(to_remove)
                match["neutralisations"] = [
                    n
                    for n in match["neutralisations"]
                    if n.get("physical_id") not in remove_set
                ]
            existing_pids = {n.get("physical_id") for n in match["neutralisations"]}
            for n in to_add:
                if n["physical_id"] not in existing_pids:
                    match["neutralisations"].append(n)

        return match

    def update_vectors_w_rocchio_2d(
        self,
        query_vectors: List[np.ndarray],
        feedback_list: List[Tuple[int, int]],
        host_match: Dict,
        org_id: Union[int, str],
    ) -> List[np.ndarray]:

        # Step 1: Previous positive centroid (shape = (4096,))
        prev_pos_vec = host_match.get("rocchio_pos_vec_2d", None)
        if prev_pos_vec is not None:
            prev_pos_vec = np.array(prev_pos_vec, dtype=np.float32)

        # Step 2: Only positive feedback
        pos_feedback_list = [
            (physical_id, reaction)
            for (physical_id, reaction) in feedback_list
            if reaction == 1
        ]

        # If nothing new and no previous centroid → no update
        if not pos_feedback_list and prev_pos_vec is None:
            return query_vectors

        # Step 3: Extract vectors from positive feedback
        pos_vecs = []
        if pos_feedback_list:
            pos_vecs, _ = self._extract_feedback_vectors(
                feedback_list=pos_feedback_list,
                embedding_index=constants.ELASTICSEARCH_PREFIX,
                embedding_vector_field="embedding_vector_v2",
                routing_key=f"v2___{str(org_id)}",
            )

        # Step 4: Add previous centroid as ONE vector
        if prev_pos_vec is not None:
            pos_vecs = pos_vecs + [prev_pos_vec]

        # Step 5: Rocchio (no negatives for 2D)
        updated, pos_mean, _ = self.rocchio_update(
            vecs=query_vectors,
            pos_vecs=pos_vecs,
            neg_vecs=[],
            return_intermediates=True,
        )

        pos_mean = self.add_epsilon(pos_mean)

        # Step 6: Persist
        if (
            host_match.get("similarity_score") is not None
            and host_match["similarity_score"] >= self.similarity_threshold
        ):
            doc_id = host_match["doc_id"]

            update_body = {
                "doc": {
                    "rocchio_pos_vec_2d": pos_mean.tolist(),
                    "updated_at": datetime.now(),
                }
            }

            self.elasticsearch_db.client.update(
                index=constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
                id=doc_id,
                body=update_body,
                refresh="wait_for",
            )

        return updated
