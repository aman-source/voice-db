from google.cloud import storage
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
_client = storage.Client()
_bucket = _client.bucket(GCS_BUCKET_NAME)


def upload_audio(audio_bytes: bytes, folder: str, filename: str) -> str:
    blob = _bucket.blob(f"{folder}/{filename}")
    blob.upload_from_string(audio_bytes, content_type="audio/wav", timeout=8)
    return f"gs://{GCS_BUCKET_NAME}/{folder}/{filename}"


def upload_registration_audio(audio_bytes: bytes, person_name: str) -> str:
    return upload_audio(audio_bytes, f"registrations/{person_name.lower()}", f"{uuid.uuid4()}.wav")


def upload_match_audio(audio_bytes: bytes) -> str:
    return upload_audio(audio_bytes, "matches", f"{uuid.uuid4()}.wav")


def upload_transaction_audio(audio_bytes: bytes) -> str:
    return upload_audio(audio_bytes, "transactions", f"{uuid.uuid4()}.wav")
