from fastapi import APIRouter, UploadFile, File
from app.services.embedding import generate_embedding_from_bytes
from app.services.gcp_vector_store import identify_speaker
from app.services.gcs_storage import upload_match_audio

router = APIRouter(prefix="/voice")

THRESHOLD = 0.45


@router.post("/match")
async def match_voice(audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()

        # Upload to GCS for audit trail (non-fatal)
        try:
            upload_match_audio(audio_bytes)
        except Exception as e:
            print(f"[WARN] GCS upload failed (non-fatal): {e}")

        embedding = generate_embedding_from_bytes(audio_bytes)
        name, score = identify_speaker(embedding)

        if not name:
            return {"match": "NOT_FOUND", "confidence": score}

        if score < THRESHOLD:
            return {"match": "LOW_CONFIDENCE", "person_name": name, "confidence": score}

        return {"match": "SUCCESS", "person_name": name, "confidence": score}

    except Exception as e:
        print("❌ MATCH API ERROR:", e)
        return {"match": "ERROR", "message": str(e)}
