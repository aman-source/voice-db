from fastapi import APIRouter, UploadFile, File, Form
from app.services.embedding import generate_embedding_from_bytes
from app.services.gcp_vector_store import add_embedding
from app.services.gcs_storage import upload_registration_audio

router = APIRouter(prefix="/voice")


@router.post("/register-multi")
async def register_voice_multi(
    person_name: str = Form(...),
    audio1: UploadFile = File(...),
    audio2: UploadFile = File(...),
    audio3: UploadFile = File(...)
):
    """
    Multi-sample registration.
    Stores each of the 3 voice samples individually so the centroid
    is computed from 3 real vectors for maximum accuracy.
    """
    audio_bytes1 = await audio1.read()
    audio_bytes2 = await audio2.read()
    audio_bytes3 = await audio3.read()

    # Upload to GCS for audit trail (non-fatal)
    for audio_bytes in [audio_bytes1, audio_bytes2, audio_bytes3]:
        try:
            upload_registration_audio(audio_bytes, person_name)
        except Exception as e:
            print(f"[WARN] GCS upload failed (non-fatal): {e}")

    # Store each embedding individually — centroid auto-computed after each upsert
    emb1 = generate_embedding_from_bytes(audio_bytes1)
    emb2 = generate_embedding_from_bytes(audio_bytes2)
    emb3 = generate_embedding_from_bytes(audio_bytes3)

    add_embedding(emb1, person_name)
    add_embedding(emb2, person_name)
    add_embedding(emb3, person_name)

    return {
        "status": "registered",
        "person_name": person_name,
        "samples_used": 3,
        "method": "individual_embeddings_with_centroid"
    }
