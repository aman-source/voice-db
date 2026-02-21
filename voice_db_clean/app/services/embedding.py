from app.services.audio import load_audio_from_bytes
from app.models.speaker import SpeakerEncoder

encoder = SpeakerEncoder()

def generate_embedding_from_bytes(audio_bytes: bytes):
    waveform = load_audio_from_bytes(audio_bytes)
    embedding = encoder.encode(waveform)
    return embedding
