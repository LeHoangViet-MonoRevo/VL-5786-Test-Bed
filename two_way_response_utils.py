from typing import Dict, Tuple

import imagehash
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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
    hash_vector = np.array(
        [int(b) for b in bin(int(hash_str, 16))[2:].zfill(64)], dtype=np.float32
    )
    hash_vector = hash_vector / np.linalg.norm(hash_vector)
    return hash_vector

def unique_vectors_by_similarity(vectors: np.ndarray, threshold: float = 0.98) -> tuple:
    """
    Returns vectors that are not too similar to each other
    based on cosine similarity.
    """
    
    unique = []
    unique_indices = []

    for i, v in enumerate(vectors):
        if not unique:
            unique.append(v)
            unique_indices.append(i)
            continue

        sims = cosine_similarity([v], unique)[0]
        if np.max(sims) < threshold:
            unique.append(v)
            unique_indices.append(i)

    return np.array(unique), unique_indices
