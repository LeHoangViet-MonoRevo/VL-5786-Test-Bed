from datetime import datetime
from typing import Dict, List, Tuple, Union

import numpy as np

from constants import constants
from elasticsearch_constants import es_constant
from feedback_2d import RocchioFeedback2D
from feedback_base import RocchioFeedbackBase


class RocchioFeedback3D(RocchioFeedback2D):
    def __init__(self, elasticsearch_db, similarity_threshold=0.98):
        super().__init__(elasticsearch_db, similarity_threshold)

    def apply_rocchio_on_3d_vector(
        self,
        query_vectors: List[np.ndarray],
        feedback_list: List[Tuple],
        embedding_index: str,
        org_id: str,
        object_id: str,
        matches: Dict,
    ):
        """Apply Rocchio feedback update to 3D query vectors and persist history in Elasticsearch."""

        # Filter out the neutral reactions and already processed reactions
        past_interactions = [
            (inter["physical_id"], inter["reaction"]) for inter in matches["interactions"]
        ]
        feedback_list = self.remove_neutral_and_duplicate_negative_interactions(
            feedback_list, past_interactions
        )

        if feedback_list:
            pos_vecs, neg_vecs = self._extract_feedback_vectors(
                feedback_list,
                embedding_index,
                embedding_vector_field="embedding_vector_3d",
                routing_key=f"3d___{str(org_id)}",
            )

            updated, pos_mean, neg_mean = self.rocchio_update(
                query_vectors, pos_vecs, neg_vecs, return_intermediates=True
            )

            pos_mean = self.add_epsilon(pos_mean)
            neg_mean = self.add_epsilon(neg_mean)

            interactions_save_in_es = self.convert_feedback_to_interactions(
                feedback_list,
                es_constant.SCHEMA_ROCCHIO_HISTORY_PHYSICAL_OBJECT["mappings"]["properties"][
                    "interactions"
                ]["properties"].keys(),
            )

            # Save the pos_mean to `rocchio_pos_vec_3d`, neg_mean to `rocchio_neg_vec_3d`
            if (
                matches["similarity_score"] is not None
                and matches["similarity_score"] >= self.similarity_threshold
            ):
                # UPDATE MODE
                doc_id = matches["doc_id"]

                # update_body = {
                #     "doc": {
                #         "rocchio_pos_vec_3d": pos_mean.tolist(),
                #         "rocchio_neg_vec_3d": neg_mean.tolist(),
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

                            ctx._source.rocchio_pos_vec_3d = params.rocchio_pos_vec_3d;
                            ctx._source.rocchio_neg_vec_3d = params.rocchio_neg_vec_3d;
                            ctx._source.updated_at = params.updated_at;
                        """,
                        "params": {
                            "interactions": interactions_save_in_es,
                            "rocchio_pos_vec_3d": pos_mean.tolist(),
                            "rocchio_neg_vec_3d": neg_mean.tolist(),
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
                    "embedding_vector_3d": query_vectors[0].tolist(),
                    "rocchio_pos_vec_3d": pos_mean.tolist(),
                    "rocchio_neg_vec_3d": neg_mean.tolist(),
                    "type": "3d",
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
            matches["similarity_score"] is not None
            and matches["similarity_score"] >= self.similarity_threshold
        ):

            rocchio_pos, rocchio_neg = [], []

            # Only append vectors if they exist in ES
            if matches["rocchio_pos_vec_3d"] is not None:
                rocchio_pos.append(matches["rocchio_pos_vec_3d"])

            if matches["rocchio_neg_vec_3d"] is not None:
                rocchio_neg.append(matches["rocchio_neg_vec_3d"])

            # If no vectors are present, return original vectors
            if not rocchio_pos and not rocchio_neg:
                return query_vectors

            return self.rocchio_update(query_vectors, rocchio_pos, rocchio_neg)

        return query_vectors

    def apply_rocchio_on_2d_vector(
        self,
        query_vectors: List[np.ndarray],
        feedback_list: List[Tuple],
        embedding_index: str,
        matches: Dict,
        org_id: str,
    ):
        """Apply Rocchio feedback update to 2D query vectors using stored feedback vectors."""

        # Filter out the neutral reactions and the already processed reactions
        past_interactions = [
            (inter["physical_id"], inter["reaction"]) for inter in matches["interactions"]
        ]
        feedback_list = self.remove_neutral_and_duplicate_negative_interactions(
            feedback_list, past_interactions
        )

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

            if (
                matches["similarity_score"] is not None
                and matches["similarity_score"] >= self.similarity_threshold
            ):
                # UPDATE MODE
                doc_id = matches["doc_id"]

                update_body = {
                    "doc": {
                        "rocchio_pos_vec_2d": pos_mean.tolist(),
                        "rocchio_neg_vec_2d": neg_mean.tolist(),
                        "updated_at": datetime.now(),
                    }
                }

                self.elasticsearch_db.client.update(
                    index=constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
                    id=doc_id,
                    body=update_body,
                )

            # 2D does not have create mode
            return updated

        if (
            matches["similarity_score"] is not None
            and matches["similarity_score"] >= self.similarity_threshold
        ):
            rocchio_pos, rocchio_neg = [], []

            # Only append vectors if they exist in ES
            if matches["rocchio_pos_vec_2d"] is not None:
                rocchio_pos.append(matches["rocchio_pos_vec_2d"])

            if matches["rocchio_neg_vec_2d"] is not None:
                rocchio_neg.append(matches["rocchio_neg_vec_2d"])

            # If no vectors are present, return original vectors
            if not rocchio_pos and not rocchio_neg:
                return query_vectors

            return self.rocchio_update(query_vectors, rocchio_pos, rocchio_neg)

        return query_vectors

    def retrieve_3d_exact_match(self, emb_3d: Union[np.ndarray, List], org_id: int) -> List:
        """
        Retrieve 3D exact match in Rocchio index using embedding_vector_3d.
        Filtered by org_id and self.similarity_threshold.
        """

        match = self.find_similar_rocchio_record(
            query_vector=emb_3d,
            search_field="embedding_vector_3d",
            check_and_create=True,
            filters=[{"term": {"type": "3d"}}, {"term": {"org_id": org_id}}],
        )

        return match

    def execute_feedback_update_3d(
        self,
        emb_3d: Union[List, np.ndarray],
        project_id: Union[int, str],
        org_id: str,
        feedback_list: List[Tuple[int, int]] = [],
    ) -> None:
        """
        Add later
        """

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
            (physical_id, reaction) for physical_id, reaction in feedback_list if reaction == 0
        ]

        # Step 0: Filter neutral feedback
        feedback_list = self.filter_feedback(
            feedback_list
        )  # Refactor and put this function in RocchioBase later

        # Step 1: Retrieve exact match in Rocchio index
        match = self.retrieve_3d_exact_match(
            emb_3d=emb_3d,
            org_id=org_id,
        )

        # Step 2: Get all disliked_cluster info
        disliked_cluster_info = self._load_disliked_clusters_from_match(match)

        # Step 3: Forward (Update host in Rocchio + Update dislikes in similarity_clusters + Add disliked clusters to host in Rocchio)
        remaining_feedback = self._get_remaining_feedback(feedback_list, disliked_cluster_info)

        if remaining_feedback:
            print(f"remaining_feedback: {remaining_feedback}")
            new_disliked_clusters = self._process_remaining_feedback(remaining_feedback, org_id)
            self._update_or_create_host_doc(
                match=match,
                emb_vector=emb_3d,
                new_clusters=new_disliked_clusters,
                project_id=project_id,
                org_id=org_id,
                type="3d",
            )

        # Step 4: Backward (Update dislikes in Rocchio + Append the host cluster to dislikes in Rocchio)
        host_cluster_id = self.search_cluster(
            vector=emb_3d,
            version="3d",  # Host is 3D
            org_id=org_id,
        )

        # Combine
        combined_disliked_clusters = {
            **disliked_cluster_info,
            **(new_disliked_clusters if remaining_feedback else {}),
        }

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

            to_remove = self._build_neutralisations_to_remove(feedback_list)

            self._update_neutralisations_in_es(
                host_doc_id=host_doc_id,
                to_add=to_add,
                to_remove=to_remove,
                now=now,
            )

        return
