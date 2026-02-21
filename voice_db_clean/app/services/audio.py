import librosa
import io
import numpy as np

TARGET_SR = 16000

def load_audio_from_bytes(audio_bytes: bytes):
    audio, sr = librosa.load(
        io.BytesIO(audio_bytes),
        sr=TARGET_SR,
        mono=True
    )

    # SpeechBrain expects (batch, time)
    return np.expand_dims(audio, axis=0)
