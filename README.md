# Voice DB — Speaker Recognition System

A voice-based speaker registration and verification system built for Indian language support. Users can register their voice, then authenticate by speaking to verify identity — including transaction verification via voice commands.

---

## What It Does

- **Register** a speaker with 3 voice samples
- **Identify** who is speaking from an audio clip
- **Verify a transaction** by voice — e.g. *"Send 500 to Rahul"* — authenticates the speaker and extracts transaction details simultaneously

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Voice Embeddings | SpeechBrain ECAPA-TDNN (VoxCeleb) |
| Vector Search | GCP Vertex AI Matching Engine |
| Speaker Metadata | GCP Firestore |
| Audio Storage | GCP Cloud Storage (GCS) |
| Speech-to-Text | Sarvam AI `saaras:v3` (Indian languages) |
| NLP / Entity Extraction | Gemini 2.5 Flash (Vertex AI) |

---

## Prerequisites

- Python 3.11+
- A GCP project with the following enabled:
  - Vertex AI (Matching Engine index + deployed endpoint)
  - Firestore (default database)
  - Cloud Storage (bucket)
- A GCP Service Account key (`service_account.json`)
- Sarvam AI API key

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/aman-source/voice-db.git
cd voice-db
```

### 2. Install dependencies

```bash
cd voice_db_clean
pip install -r requirements.txt
```

### 3. Add credentials

Place your GCP service account key at:
```
voice_db_clean/credentials/service_account.json
```

### 4. Configure environment variables

Create a `.env` file inside `voice_db_clean/`:

```env
GCP_PROJECT_ID=your-gcp-project-id
GCP_REGION=us-central1

GCP_INDEX_ID=projects/your-project/locations/us-central1/indexes/your-index-id
GCP_INDEX_ENDPOINT_ID=projects/your-project/locations/us-central1/indexEndpoints/your-endpoint-id
GCP_DEPLOYED_INDEX_ID=your-deployed-index-id

GCS_BUCKET_NAME=your-gcs-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=credentials/service_account.json

SARVAM_API_KEY=your-sarvam-api-key
```

---

## Running the App

From inside the `voice_db_clean/` directory:

```bash
cd voice_db_clean
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

> First startup takes 30–60 seconds — the SpeechBrain model loads into memory.

Once you see:
```
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

Open your browser at: **http://localhost:8000**

---

## API Overview

### `POST /voice/register-multi`
Register a new speaker with 3 audio samples.

| Field | Type |
|---|---|
| `person_name` | string (form) |
| `audio1`, `audio2`, `audio3` | WAV files |

---

### `POST /voice/match`
Identify who is speaking from an audio clip.

| Field | Type |
|---|---|
| `audio` | WAV file |

Returns: matched speaker name + confidence score.

---

### `POST /voice/verify-transaction`
Verify a spoken transaction command (e.g. *"Send 500 to Rahul"*).

| Field | Type |
|---|---|
| `audio` | WAV file |
| `person_name` | string (optional) |

Returns: voice match status, speaker identity, transcript, sender, receiver, and amount.

- With `person_name` → confirms the audio belongs to that specific person
- Without `person_name` → blind identification from all registered speakers

---

## How It Works (Simple Overview)

```
User speaks into mic
        ↓
Audio → ECAPA model → 192-dim voice vector
        ↓
Vector compared against registered speakers in Vertex AI
        ↓
  ┌─────────────────────────────────────┐
  │  /register  → save vector + centroid│
  │  /match     → who is this person?   │
  │  /verify    → auth + parse command  │
  └─────────────────────────────────────┘
        ↓ (for /verify-transaction only)
Audio → Sarvam STT → transcript
        ↓
Gemini 2.5 Flash → extract sender, receiver, amount
        ↓
Check names against Firestore DB
        ↓
Return full transaction verification result
```

**Confidence threshold:** 0.45 — anything below is rejected as unrecognized.

All audio is stored in GCS for audit trail.

---

## Notes

- `credentials/` and `.env` are excluded from git — never commit secrets
- Pretrained models (`pretrained_models/`) are also excluded — downloaded automatically on first run
- Run the app from inside `voice_db_clean/` — static file paths are relative
