from fastapi import APIRouter, UploadFile, File
from app.services.embedding import generate_embedding_from_bytes
from app.services.qdrant_store import identify_speaker

router = APIRouter(prefix="/voice")

THRESHOLD = 0.5  # keep low for now


@router.post("/match")
async def match_voice(audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()
        embedding = generate_embedding_from_bytes(audio_bytes)

        name, score = identify_speaker(embedding)

        if not name:
            return {
                "match": "NOT_FOUND",
                "confidence": score
            }

        if score < THRESHOLD:
            return {
                "match": "LOW_CONFIDENCE",
                "person_name": name,
                "confidence": score
            }

        return {
            "match": "SUCCESS",
            "person_name": name,
            "confidence": score
        }

    except Exception as e:
        print("❌ MATCH API ERROR:", e)
        return {
            "match": "ERROR",
            "message": str(e)
        }
