from datetime import datetime

import numpy as np
from PIL import Image

from constants import constants
from feedback_2d import RocchioFeedback2D
from interaction import elasticsearch_db
from similarity_clusters import SimilarityClusters
from two_way_response_utils import (build_product_embedding_map,
                                    extract_feature_v3,
                                    unique_vectors_by_similarity)

if __name__ == "__main__":

    similarity_clusters = SimilarityClusters(elasticsearch_db)
    rocchio_feedback_2d = RocchioFeedback2D(elasticsearch_db)
    disliked_phys_ids = {(1303, datetime.now()), (1229, datetime.now())}
    company_id = 1

    image = Image.open("./2. 少し雑.jpg")
    image_phash = extract_feature_v3(image)

    # ---- Two-Way Response START ----
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

    disliked_phashes = build_product_embedding_map(disliked_info_response)

    print(f"disliked_phashes: {disliked_phashes}")

    # Create A's cluster and find all similar images and put them in
    cluster_info = similarity_clusters.search_cluster(
        vector=image_phash,
        org_id=company_id,
        search_field="embedding_vector_2D",
    )

    # Create documents for disliked physical_ids
    # Put the A cluster in as "disliked_cluster_ids"
    phys_ids = list(disliked_phashes.keys())
    vectors = np.array(list(disliked_phashes.values()))

    unique_vectors, keep_indices = unique_vectors_by_similarity(vectors)

    # Rebuild dict with only unique ones
    unique_disliked_phashes = {phys_ids[i]: vectors[i] for i in keep_indices}

    for key, val in unique_disliked_phashes.items():
        new_doc = {
            "phash_2d": val,
            "type": "2d",
            "disliked_cluster_ids": [
                {"cluster_id": cluster_info["doc_id"], "timestamp": datetime.now()}
            ],
            "org_id": company_id,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

        # From this phash, find similar document in the Rocchio index
        # What function does this in feedback_2d.py

        matches = rocchio_feedback_2d.find_similar_rocchio_record(
            query_vector=val,
            search_field="phash_2d",
            check_and_create=True,
            filters=[{"term": {"type": "2d"}}, {"term": {"org_id": company_id}}],
        )

        print(f"matches: {matches}")

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
                    {"cluster_id": cluster_info["doc_id"], "timestamp": datetime.now()}
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

    # ---- Two-Way Response END ----
