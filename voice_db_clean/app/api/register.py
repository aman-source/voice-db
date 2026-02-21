from fastapi import APIRouter, UploadFile, File, Form
import numpy as np
from app.services.embedding import generate_embedding_from_bytes
from app.services.qdrant_store import add_embedding

router = APIRouter(prefix="/voice")


@router.post("/register")
async def register_voice(
    person_name: str = Form(...),
    audio: UploadFile = File(...)
):
    """Single sample registration (legacy)"""
    audio_bytes = await audio.read()
    embedding = generate_embedding_from_bytes(audio_bytes)

    add_embedding(embedding, person_name)

    return {"status": "registered", "person_name": person_name}


@router.post("/register-multi")
async def register_voice_multi(
    person_name: str = Form(...),
    audio1: UploadFile = File(...),
    audio2: UploadFile = File(...),
    audio3: UploadFile = File(...)
):
    """
    Multi-sample registration (industrial best practice).
    Records 3 voice samples and stores the averaged embedding.
    """
    # Read all audio samples
    audio_bytes1 = await audio1.read()
    audio_bytes2 = await audio2.read()
    audio_bytes3 = await audio3.read()

    # Generate embeddings for each sample
    emb1 = generate_embedding_from_bytes(audio_bytes1)
    emb2 = generate_embedding_from_bytes(audio_bytes2)
    emb3 = generate_embedding_from_bytes(audio_bytes3)

    # Average the embeddings (industrial best practice)
    averaged_embedding = np.mean([emb1, emb2, emb3], axis=0)

    # Store the averaged embedding
    add_embedding(averaged_embedding, person_name)

    return {
        "status": "registered",
        "person_name": person_name,
        "samples_used": 3,
        "method": "averaged_embedding"
    }
