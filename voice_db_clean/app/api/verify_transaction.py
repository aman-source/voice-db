from fastapi import APIRouter, UploadFile, File, Form
from app.services.embedding import generate_embedding_from_bytes
from app.services.gcp_vector_store import identify_speaker, verify_speaker, check_name_exists
from app.services.stt import speech_to_text
from app.services.nlp import extract_transaction_info
from app.services.gcs_storage import upload_transaction_audio

router = APIRouter(prefix="/voice")

THRESHOLD = 0.45


@router.post("/verify-transaction")
async def verify_transaction(
    audio: UploadFile = File(...),
    person_name: str = Form(None)
):
    """
    Verify a spoken transaction.

    - With person_name: targeted verification — confirms the audio belongs to that
      specific registered person before processing the transaction.
    - Without person_name: blind speaker identification (original behaviour).
    """
    audio_bytes = await audio.read()

    # Upload audio to GCS for audit trail (non-fatal)
    gcs_uri = None
    try:
        gcs_uri = upload_transaction_audio(audio_bytes)
    except Exception as e:
        print(f"[WARN] GCS upload failed (non-fatal): {e}")

    # 1. Speaker recognition
    embedding = generate_embedding_from_bytes(audio_bytes)

    if person_name:
        confidence, is_registered = verify_speaker(embedding, person_name)
        if not is_registered:
            voice_matched = False
            speaker = "unknown"
            confidence = 0.0
        elif confidence >= THRESHOLD:
            voice_matched = True
            speaker = person_name.lower()
        else:
            voice_matched = False
            speaker = "unknown"
    else:
        speaker, confidence = identify_speaker(embedding)
        if not speaker or confidence < THRESHOLD:
            voice_matched = False
            speaker = "unknown"
        else:
            voice_matched = True

    # 2. Speech-to-text
    transcript = speech_to_text(audio_bytes)

    # 3. Extract entities from speech
    info = extract_transaction_info(transcript)

    # 4. Extract sender and receiver from NLP, then check each in DB
    sender_name = info["sender"]

    # If user said "I"/"me"/"my" instead of their name, use the biometric speaker
    FIRST_PERSON = {"i", "me", "my", "myself", "mine"}
    if sender_name in FIRST_PERSON and voice_matched:
        sender_name = speaker

    sender_found, sender_matched = check_name_exists(sender_name)

    receiver_name = info["receiver"]
    receiver_found, receiver_matched = check_name_exists(receiver_name)

    return {
        "voice_status": "MATCHED" if voice_matched else "NOT_MATCHED",
        "speaker": speaker,
        "confidence": round(confidence, 4),
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
        "transcript": transcript,
        "audio_stored": gcs_uri
    }
