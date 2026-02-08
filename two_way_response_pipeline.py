from datetime import datetime

import numpy as np
from PIL import Image

from constants import constants
from interaction import elasticsearch_db
from similarity_clusters import SimilarityClusters
from two_way_response_utils import (build_product_embedding_map,
                                    extract_feature_v3,
                                    unique_vectors_by_similarity)

if __name__ == "__main__":

    similarity_clusters = SimilarityClusters(elasticsearch_db)
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

    # TODO 2-way response: Dont create duplicate document
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

        elasticsearch_db.client.index(
            index=constants.ROCCHIO_HISTORY_PHYSICAL_OBJECT,
            document=new_doc,
            refresh="wait_for",
        )

    # ---- Two-Way Response END ----
