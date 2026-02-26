import requests
import librosa
import soundfile
import io
import os
from dotenv import load_dotenv

load_dotenv(override=True)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_URL     = "https://api.sarvam.ai/speech-to-text"


def speech_to_text(audio_bytes: bytes) -> str:
    """
    Convert audio bytes to text using Sarvam AI saaras:v3.
    Purpose-built for 23 Indian languages — handles Telugu, Hindi, Tamil,
    Kannada names natively without keyword hints.
    Returns empty string on failure.
    """
    try:
        audio_np, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        wav_io = io.BytesIO()
        soundfile.write(wav_io, audio_np, 16000, format="WAV", subtype="PCM_16")
        wav_bytes = wav_io.getvalue()
        print(f"[DEBUG] Audio resampled: {len(wav_bytes)} bytes")

        response = requests.post(
            SARVAM_URL,
            headers={"api-subscription-key": SARVAM_API_KEY},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={
                "model":         "saaras:v3",
                "language_code": "en-IN",
                "mode":          "transcribe",
            },
            timeout=15,
        )

        if response.status_code != 200:
            print(f"[ERROR] Sarvam STT HTTP {response.status_code}: {response.text}")
            return ""

        transcript = response.json().get("transcript", "").strip().lower()
        print(f"[OK] Sarvam transcript: '{transcript}'")
        return transcript

    except Exception as e:
        import traceback
        print(f"[ERROR] Sarvam STT error: {e}")
        traceback.print_exc()
        return ""
