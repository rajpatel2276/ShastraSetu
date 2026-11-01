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
from qdrant_client import models
from retrieve import client, collection_name
from pydantic import BaseModel

app = FastAPI(title="Debate RAG Engine")

os.makedirs("data/images", exist_ok=True)
app.mount("/data", StaticFiles(directory="data"), name="data")

class ClaimRequest(BaseModel):
    claim: str

class VerdictResponse(BaseModel):
    verdict: str
    images: list[str]
    texts: list[str]

class ClaimRequest(BaseModel):
    claim: str

@app.post("/evaluate_claim")
async def evaluate_claim(request: ClaimRequest):
    """Executes the LangGraph agent to evaluate a claim against multi-document evidence."""
    try:
        from agent import app
        # Initialize your graph state and invoke the workflow
        initial_state = {"claim": request.claim, "text_context": [], "image_context": [], "final_response": ""}
        
        # Replace 'workflow' with whatever your compiled LangGraph workflow variable is named in agent.py
        output_state = app.invoke(initial_state)
        
        return {
            "status": "success",
            "final_response": output_state.get("final_response", "No response generated."),
            "image_context": output_state.get("image_context", []) or []
        }
    except Exception as e:
        print(f"⚠️ Evaluation Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r") as f:
        return f.read()


@app.get("/documents")
async def list_documents():
    """Returns a list of all currently active PDFs stored in the system."""
    pdf_dir = "data/pdfs"
    if not os.path.exists(pdf_dir):
        return {"documents": []}
    
    files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    return {"documents": files}

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
            
        # Trigger multi-document ingestion and pass the original filename for clean storage tracking
        vectors_created = ingest_pdf(temp_path, original_filename=file.filename)
        
        return {"status": "success", "vectors": vectors_created}
    except Exception as e:
        logging.error(f"Ingestion Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary FastAPI upload file
        if os.path.exists(temp_path):
            os.remove(temp_path)

from qdrant_client import models # Ensure models is imported

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Manually deletes a specific PDF from disk and purges its vectors from Qdrant."""
    file_path = os.path.join("data/pdfs", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
        
    try:
        # 1. Remove file from disk
        os.remove(file_path)
        
        # 2. Purge associated vectors from Qdrant using FilterSelector
        client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_file",
                            match=models.MatchValue(value=file_path)
                        )
                    ]
                )
            )
        )
        print(f"🗑️ Successfully deleted {filename} and purged its vectors.")
        return {"status": "success", "message": f"Deleted {filename}"}
    except Exception as e:
        print(f"⚠️ Delete Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))