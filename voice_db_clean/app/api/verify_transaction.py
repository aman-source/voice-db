from fastapi import APIRouter, UploadFile, File
from app.services.embedding import generate_embedding_from_bytes
from app.services.qdrant_store import identify_speaker, check_name_exists
from app.services.stt import speech_to_text
from app.services.nlp import extract_transaction_info

router = APIRouter(prefix="/voice")

THRESHOLD = 0.5


@router.post("/verify-transaction")
async def verify_transaction(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()

    # 1️⃣ Speaker recognition (voice matching)
    embedding = generate_embedding_from_bytes(audio_bytes)
    speaker, confidence = identify_speaker(embedding)

    # Check if voice matched above threshold
    if not speaker or confidence < THRESHOLD:
        voice_matched = False
        speaker = "unknown"
    else:
        voice_matched = True

    # 2️⃣ Speech-to-text
    transcript = speech_to_text(audio_bytes)

    # 3️⃣ Extract entities from speech
    info = extract_transaction_info(transcript)

    # 4️⃣ Check if sender and receiver exist in database (with fuzzy matching)
    sender_name = info["sender"]
    receiver_name = info["receiver"]

    sender_found, sender_matched = check_name_exists(sender_name)
    receiver_found, receiver_matched = check_name_exists(receiver_name)

    return {
        "voice_status": "MATCHED" if voice_matched else "NOT_MATCHED",
        "speaker": speaker,
        "confidence": confidence,
        "sender": {
            "name": sender_matched if sender_found else sender_name,
            "spoken_as": sender_name,
            "db_status": "found" if sender_found else "not_found"
        },
        "receiver": {
            "name": receiver_matched if receiver_found else receiver_name,
            "spoken_as": receiver_name,
            "db_status": "found" if receiver_found else "not_found"
        },
        "amount": info["amount"],
        "transcript": transcript
    }
