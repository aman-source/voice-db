from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid
import os
import numpy as np
from dotenv import load_dotenv
DIM = 192
COLLECTION = "voice_embeddings"


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Try cloud first, fall back to local in-memory mode
try:
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    # Quick connectivity check
    client.get_collections()
    print("[OK] Connected to Qdrant Cloud")
except Exception as e:
    print(f"[WARN] Qdrant Cloud unavailable ({e}), using local in-memory mode")
    client = QdrantClient(":memory:")


def init_collection():
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=DIM,
                distance=Distance.COSINE
            )
        )

def add_embedding(embedding, person_name):
    try:
        vector = normalize(embedding)

        client.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload={"person_name": person_name.lower()}
                )
            ]
        )

    except Exception as e:
        print("[ERROR] QDRANT REGISTER ERROR:", e)



def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def identify_speaker(embedding):
    try:
        query = normalize(embedding)

        response = client.query_points(
            collection_name=COLLECTION,
            query=query.tolist(),
            limit=1
        )

        # SAFETY CHECKS
        if response is None:
            print("[WARN] Qdrant response is None")
            return None, 0.0

        if not hasattr(response, "points"):
            print("[WARN] No 'points' in response:", response)
            return None, 0.0

        if len(response.points) == 0:
            print("[WARN] No matching points found")
            return None, 0.0

        top = response.points[0]

        if top.payload is None:
            print("[WARN] Payload missing")
            return None, float(top.score or 0.0)

        name = top.payload.get("person_name")
        score = float(top.score or 0.0)

        return name, score

    except Exception as e:
        print("[ERROR] QDRANT MATCH ERROR:", e)
        return None, 0.0


def check_name_exists(name: str) -> tuple:
    """
    Check if a person name exists in the database using fuzzy matching.
    Returns (True, matched_name) if found, (False, None) otherwise.
    Handles spelling variations like "sumant" matching "sumanth".
    """
    if not name:
        return False, None

    try:
        # Get all registered names
        all_names = get_all_registered_names()

        name_lower = name.lower().strip()

        # Exact match first
        if name_lower in all_names:
            return True, name_lower

        # Fuzzy match: check if input is substring or similar
        for registered_name in all_names:
            # Check if one contains the other (handles missing letters)
            if name_lower in registered_name or registered_name in name_lower:
                return True, registered_name

            # Check similarity (Levenshtein-like: allow 1-2 character difference)
            if _is_similar(name_lower, registered_name, max_distance=2):
                return True, registered_name

        return False, None

    except Exception as e:
        print(f"[ERROR] QDRANT CHECK NAME ERROR: {e}")
        return False, None


def _is_similar(s1: str, s2: str, max_distance: int = 2) -> bool:
    """
    Check if two strings are similar (within max_distance edits).
    Simple implementation for short names.
    """
    if abs(len(s1) - len(s2)) > max_distance:
        return False

    # Simple character difference count
    longer = s1 if len(s1) >= len(s2) else s2
    shorter = s2 if len(s1) >= len(s2) else s1

    differences = 0
    j = 0
    for i, char in enumerate(longer):
        if j < len(shorter) and char == shorter[j]:
            j += 1
        else:
            differences += 1

    differences += (len(shorter) - j)  # remaining unmatched chars

    return differences <= max_distance


def get_all_registered_names() -> list:
    """
    Get all registered person names from the database.
    """
    try:
        response = client.scroll(
            collection_name=COLLECTION,
            limit=100
        )

        points, _ = response
        names = set()
        for point in points:
            if point.payload and "person_name" in point.payload:
                names.add(point.payload["person_name"].lower())

        return list(names)

    except Exception as e:
        print(f"[ERROR] QDRANT GET NAMES ERROR: {e}")
        return []
