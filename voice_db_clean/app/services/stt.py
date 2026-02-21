import speech_recognition as sr
import tempfile
import os

recognizer = sr.Recognizer()

def speech_to_text(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text.lower()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"Google STT error: {e}")
        return ""
    finally:
        os.remove(temp_path)
