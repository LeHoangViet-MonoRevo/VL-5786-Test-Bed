from datetime import datetime
from typing import Dict

import numpy as np
import imagehash
from PIL import Image

from constants import constants
from interaction import elasticsearch_db
from similarity_clusters import SimilarityClusters


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

def extract_feature_v3(image):
    image_hash = imagehash.phash(image, hash_size=32)
    hash_str = image_hash.__str__()
    hash_vector = np.array([int(b) for b in bin(int(hash_str, 16))[2:].zfill(64)], dtype=np.float32)
    hash_vector = hash_vector / np.linalg.norm(hash_vector)
    return hash_vector


if __name__ == "__main__":

    similarity_clusters = SimilarityClusters(elasticsearch_db)
    disliked_phys_ids = {(1303, datetime.now())}
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
                        {"term": {"version": "v3"}},
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

    # TODO: Handle the case where disliked_phashes are similar
    for key, val in disliked_phashes.items():
        new_doc = {
            "phash_2d": val,
            "type": "2d",
            "disliked_cluster_ids": [cluster_info["doc_id"]],
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
