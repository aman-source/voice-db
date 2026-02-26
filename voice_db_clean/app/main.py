import os
from dotenv import load_dotenv
load_dotenv(override=True)

from app.utils.windows_symlink_fix import apply_windows_symlink_fix
apply_windows_symlink_fix()

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.cloud import aiplatform

from app.api.register import router as register_router
from app.api.match import router as match_router
from app.api.verify_transaction import router as verify_transaction_router
from app.services.gcp_vector_store import init_gcp


app = FastAPI(title="Voice Matching System")


@app.on_event("startup")
def startup():
    project_id = os.getenv("GCP_PROJECT_ID", "").strip()
    if not project_id:
        index_resource = os.getenv("GCP_INDEX_ID", "")
        parts = index_resource.split("/")
        if len(parts) >= 2:
            project_id = parts[1]
    region = os.getenv("GCP_REGION", "us-central1").strip()
    aiplatform.init(project=project_id, location=region)
    print(f"[OK] Vertex AI initialized: project={project_id}, region={region}")
    init_gcp()


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

app.include_router(register_router)
app.include_router(match_router)
app.include_router(verify_transaction_router)
