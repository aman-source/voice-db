from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.register import router as register_router
from app.api.match import router as match_router
from app.api.verify_transaction import router as verify_transaction_router
from app.services.vector_store import load_store   # 👈 ADD THIS
from app.services.qdrant_store import init_collection


app = FastAPI(title="Voice Matching System")



@app.on_event("startup")
def startup():
    init_collection()



# Serve static files at /static
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

# APIs
app.include_router(register_router)
app.include_router(match_router)
app.include_router(verify_transaction_router)
