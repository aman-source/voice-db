from google.cloud import aiplatform
from google.cloud import firestore
from google.cloud.aiplatform_v1.types import IndexDatapoint
import uuid
import os
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv(override=True)

DIM = 192
GCP_PROJECT_ID        = os.getenv("GCP_PROJECT_ID")
GCP_REGION            = os.getenv("GCP_REGION", "us-central1")
GCP_INDEX_ID          = os.getenv("GCP_INDEX_ID")
GCP_INDEX_ENDPOINT_ID = os.getenv("GCP_INDEX_ENDPOINT_ID")
GCP_DEPLOYED_INDEX_ID = os.getenv("GCP_DEPLOYED_INDEX_ID")
FIRESTORE_COLLECTION  = "voice_speakers"

_db             = None
_index_endpoint = None
_index          = None


def init_gcp():
    global _db, _index_endpoint, _index

    project_id = os.getenv("GCP_PROJECT_ID", "").strip()
    if not project_id:
        index_resource = os.getenv("GCP_INDEX_ID", "")
        parts = index_resource.split("/")
        if len(parts) >= 2:
            project_id = parts[1]
            print(f"[DEBUG] GCP_PROJECT_ID extracted from GCP_INDEX_ID: '{project_id}'")
        else:
            print("[ERROR] GCP_PROJECT_ID is empty and could not be extracted from GCP_INDEX_ID")
    else:
        print(f"[DEBUG] GCP_PROJECT_ID = '{project_id}'")

    _db = firestore.Client(project=project_id, database="(default)")
    print("[OK] Firestore client initialized")

    _index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=os.getenv("GCP_INDEX_ENDPOINT_ID")
    )
    _index = aiplatform.MatchingEngineIndex(
        index_name=os.getenv("GCP_INDEX_ID")
    )
    print("[OK] Vertex AI Vector Search initialized")


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def add_embedding(embedding: np.ndarray, person_name: str) -> None:
    try:
        vector = normalize(embedding)
        datapoint_id = str(uuid.uuid4())
        name_lower = person_name.lower()

        _index.upsert_datapoints(
            datapoints=[
                IndexDatapoint(
                    datapoint_id=datapoint_id,
                    feature_vector=vector.tolist()
                )
            ]
        )
        print(f"[OK] Vector upserted to Vertex AI for '{name_lower}' ID={datapoint_id}")

        _db.collection(FIRESTORE_COLLECTION).document(datapoint_id).set({
            "person_name": name_lower,
            "created_at": datetime.now(timezone.utc),
            "embedding": vector.tolist()
        })
        print(f"[OK] Registered speaker '{name_lower}' with ID {datapoint_id}")

        _update_centroid(name_lower)

    except Exception as e:
        import traceback
        print(f"[ERROR] GCP REGISTER ERROR: {e}")
        traceback.print_exc()


def _update_centroid(person_name: str) -> None:
    try:
        docs = _db.collection(FIRESTORE_COLLECTION) \
                  .where("person_name", "==", person_name) \
                  .stream()

        embeddings = []
        for doc in docs:
            data = doc.to_dict()
            if data.get("embedding") and not data.get("is_centroid", False):
                embeddings.append(np.array(data["embedding"]))

        if not embeddings:
            return

        centroid = normalize(np.mean(embeddings, axis=0))
        centroid_id = f"{person_name}_centroid"

        _index.upsert_datapoints(
            datapoints=[
                IndexDatapoint(
                    datapoint_id=centroid_id,
                    feature_vector=centroid.tolist()
                )
            ]
        )

        _db.collection(FIRESTORE_COLLECTION).document(centroid_id).set({
            "person_name": person_name,
            "is_centroid": True,
            "sample_count": len(embeddings),
            "updated_at": datetime.now(timezone.utc)
        })

        print(f"[OK] Centroid updated for '{person_name}' from {len(embeddings)} sample(s)")

    except Exception as e:
        import traceback
        print(f"[ERROR] CENTROID UPDATE ERROR: {e}")
        traceback.print_exc()


def identify_speaker(embedding: np.ndarray) -> tuple:
    try:
        query = normalize(embedding)

        response = _index_endpoint.find_neighbors(
            deployed_index_id=GCP_DEPLOYED_INDEX_ID,
            queries=[query.tolist()],
            num_neighbors=20
        )

        if not response or not response[0]:
            print("[WARN] No neighbors found")
            return None, 0.0

        for neighbor in response[0]:
            similarity = 1.0 - neighbor.distance
            doc = _db.collection(FIRESTORE_COLLECTION).document(neighbor.id).get()
            if doc.exists:
                person_name = doc.to_dict().get("person_name")
                print(f"[OK] Matched '{person_name}' (ID={neighbor.id}, similarity={similarity:.4f})")
                return person_name, similarity
            else:
                print(f"[WARN] Skipping orphaned vector ID={neighbor.id} (no Firestore doc)")

        print("[WARN] No neighbors with valid Firestore documents found")
        return None, 0.0

    except Exception as e:
        print(f"[ERROR] GCP MATCH ERROR: {e}")
        return None, 0.0


def verify_speaker(embedding: np.ndarray, expected_name: str) -> tuple:
    try:
        name_lower = expected_name.lower().strip()

        docs = _db.collection(FIRESTORE_COLLECTION).where("person_name", "==", name_lower).stream()
        valid_ids = {doc.id for doc in docs}

        if not valid_ids:
            print(f"[WARN] No registered vectors found for '{name_lower}'")
            return 0.0, False

        print(f"[DEBUG] '{name_lower}' has {len(valid_ids)} registered vector(s)")

        query = normalize(embedding)
        response = _index_endpoint.find_neighbors(
            deployed_index_id=GCP_DEPLOYED_INDEX_ID,
            queries=[query.tolist()],
            num_neighbors=50
        )

        if not response or not response[0]:
            print("[WARN] No neighbors found in Vertex AI")
            return 0.0, True

        best_similarity = 0.0
        for neighbor in response[0]:
            if neighbor.id in valid_ids:
                similarity = 1.0 - neighbor.distance
                if similarity > best_similarity:
                    best_similarity = similarity
                    print(f"[OK] Best match for '{name_lower}': ID={neighbor.id}, sim={similarity:.4f}")

        if best_similarity == 0.0:
            print(f"[WARN] None of the top 50 neighbors belong to '{name_lower}'")

        return best_similarity, True

    except Exception as e:
        print(f"[ERROR] VERIFY SPEAKER ERROR: {e}")
        return 0.0, False


def check_name_exists(name: str) -> tuple:
    if not name or len(name.strip()) < 2:
        return False, None

    try:
        all_names = get_all_registered_names()
        name_lower = name.lower().strip()

        if name_lower in all_names:
            return True, name_lower

        for registered_name in all_names:
            if len(name_lower) >= 3 and len(registered_name) >= 3:
                if name_lower in registered_name or registered_name in name_lower:
                    return True, registered_name
            if _is_similar(name_lower, registered_name, max_distance=2):
                return True, registered_name

        return False, None

    except Exception as e:
        print(f"[ERROR] GCP CHECK NAME ERROR: {e}")
        return False, None


def _is_similar(s1: str, s2: str, max_distance: int = 2) -> bool:
    if abs(len(s1) - len(s2)) > max_distance:
        return False
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i - 1] == s2[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n] <= max_distance


def get_all_registered_names() -> list:
    try:
        docs = _db.collection(FIRESTORE_COLLECTION).stream()
        names = set()
        for doc in docs:
            data = doc.to_dict()
            if data and "person_name" in data and not data.get("is_centroid", False):
                names.add(data["person_name"].lower())
        return list(names)

    except Exception as e:
        print(f"[ERROR] GCP GET NAMES ERROR: {e}")
        return []
