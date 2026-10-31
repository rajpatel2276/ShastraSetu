from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import os
import shutil
import asyncio
from agent import run_debate_agent
from ingest import ingest_pdf

app = FastAPI(title="Debate RAG Engine")

os.makedirs("data/images", exist_ok=True)
app.mount("/data", StaticFiles(directory="data"), name="data")

class ClaimRequest(BaseModel):
    claim: str

class VerdictResponse(BaseModel):
    verdict: str
    images: list[str]
    texts: list[str]

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Receives a PDF, saves it temporarily, and runs the vector embedding pipeline."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    temp_path = f"temp_{file.filename}"
    try:
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Trigger multimodal ingestion
        vectors_created = ingest_pdf(temp_path)
        
        return {"status": "success", "vectors": vectors_created}
    except Exception as e:
        logging.error(f"Ingestion Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary PDF
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/evaluate_claim", response_model=VerdictResponse)
async def evaluate_claim(request: ClaimRequest):
    try:
        # Run directly on the main thread to prevent Qdrant/SQLite database locking
        result = run_debate_agent(request.claim)
        
        return VerdictResponse(
            verdict=result["verdict"],
            images=result["retrieved_images"],
            texts=result["retrieved_texts"]
        )
    except Exception as e:
        print(f"🔥 CRITICAL ERROR IN PIPELINE: {str(e)}") # This prints the exact error to the terminal
        raise HTTPException(status_code=500, detail=str(e))