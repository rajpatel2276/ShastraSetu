from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
from fastapi.responses import StreamingResponse
import google.generativeai as genai
import asyncio
import json
import re
import os
import shutil
from agent import run_debate_agent
from ingest import ingest_pdf
from qdrant_client import models
from dotenv import load_dotenv
from retrieve import client, collection_name

load_dotenv()
app = FastAPI(title="Debate RAG Engine")

os.makedirs("data/images", exist_ok=True)
app.mount("/data", StaticFiles(directory="data"), name="data")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
class ArtifactRequest(BaseModel):
    full_text: str
    metadata: list

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
    from agent import app
    import time
    import requests
    import os
    from dotenv import load_dotenv
    
    server_start = time.time()
    print(f"\n=== 🔍 BACKEND PROFILING STARTED ===")
    print("⏳ [PROFILE] Executing LangGraph (Qdrant Retrieval)...")
    
    initial_state = {"claim": request.claim, "text_context": [], "image_context": [], "final_response": "", "final_prompt": ""}
    output_state = await asyncio.to_thread(app.invoke, initial_state)
    
    langgraph_duration = time.time() - server_start
    print(f"✅ [PROFILE] LangGraph finished in {langgraph_duration:.2f} seconds.")
    
    text_context = output_state.get("text_context", [])
    metadata = output_state.get("retrieved_metadata", [])
    context_string = "\n\n".join(text_context)
    
    prompt = f"""You are an elite, impartial RAG Validation Engine.
Your task is to evaluate the CLAIM against the provided MULTI-DOCUMENT CONTEXT.

CRITICAL INSTRUCTIONS:
1. You MUST cross-reference evidence from different source files if they are present in the context.
2. Quote exactly from the text and cite the source using this exact format: [filename.pdf, Page X].
3. Identify if the claim is SUPPORTED, REFUTED, or INCONCLUSIVE based strictly on the text.
4. Output your response EXACTLY in this strict format:

### SUPPORTING EVIDENCE
- As stated in [filename.pdf, Page X]: "exact quote here"

### REFUTING EVIDENCE
- As stated in [filename.pdf, Page Y]: "exact quote here"

### FINAL VERDICT
[SUPPORTED/REFUTED/INCONCLUSIVE]. A concise 1-sentence synthesis comparing the findings across the documents.

CLAIM: {request.claim}

MULTI-DOCUMENT CONTEXT:
{context_string}"""

    # We use a synchronous generator; FastAPI will automatically run this in a background thread
    def event_generator():
        try:
            print("⏳ [PROFILE] Initializing Gemini REST API stream...")
            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")
            
            if not api_key:
                yield f"data: {json.dumps({'type': 'error', 'content': 'Critical Error: GOOGLE_API_KEY is missing.'})}\n\n"
                return
            
            # Using high-throughput lite model primary with raw REST SSE endpoint to bypass WSL2 gRPC network bugs
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:streamGenerateContent?alt=sse&key={api_key}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            
            llm_start = time.time()
            ttft_measured = False
            full_text = ""
            
            # Split timeouts: 10s connection cap, 60s read cap for dense contexts
            with requests.post(url, headers={"Content-Type": "application/json"}, json=payload, stream=True, timeout=(10, 60)) as res:
                if res.status_code != 200:
                    error_msg = f"Google API Error {res.status_code}: {res.text}"
                    print(f"❌ [PROFILE] {error_msg}")
                    yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
                    return

                for line in res.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded.startswith("data: "):
                            data_str = decoded[6:]
                            
                            # Google's REST stream sends [DONE] or empty strings at the end
                            if data_str.strip() == "" or data_str == "[DONE]":
                                continue
                                
                            try:
                                chunk_data = json.loads(data_str)
                                candidates = chunk_data.get("candidates", [])
                                if candidates:
                                    parts = candidates[0].get("content", {}).get("parts", [])
                                    if parts:
                                        text_chunk = parts[0].get("text", "")
                                        
                                        if not ttft_measured:
                                            ttft = time.time() - llm_start
                                            print(f"✅ [PROFILE] Gemini REST TTFT (Time to First Token): {ttft:.2f} seconds.")
                                            ttft_measured = True
                                            
                                        full_text += text_chunk
                                        yield f"data: {json.dumps({'type': 'chunk', 'content': text_chunk})}\n\n"
                            except Exception:
                                pass # Safely ignore partial/status JSON payloads
                
                total_llm = time.time() - llm_start
                print(f"✅ [PROFILE] Total Stream Complete in {total_llm:.2f} seconds.\n")
                yield f"data: {json.dumps({'type': 'complete', 'metadata': metadata, 'full_text': full_text})}\n\n"
                
        except requests.exceptions.ConnectTimeout:
            error_msg = "Connect Timeout: Could not establish connection to Google servers within 10s."
            print(f"❌ [PROFILE] {error_msg}")
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
        except requests.exceptions.ReadTimeout:
            error_msg = "Read Timeout: Google's API took >60s to process this multi-document context."
            print(f"❌ [PROFILE] {error_msg}")
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
        except Exception as e:
            print(f"❌ [PROFILE] LLM ERROR: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'content': f'LLM Stream Error: {str(e)}'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# 2. THE RESTORED TWO-PASS IMAGE ENDPOINT
@app.post("/generate_artifact")
async def generate_artifact(request: ArtifactRequest):
    from agent import generate_highlighted_citation
    import time
    import re
    
    quotes = re.findall(r'"([^"]*)"', request.full_text)
    images = []
    
    if quotes and request.metadata:
        target_quote = quotes[0]
        best_match = request.metadata[0]
        
        for meta in request.metadata:
            if target_quote[:20] in meta.get("chunk", ""):
                best_match = meta
                break
                
        img_path = generate_highlighted_citation(
            text_chunk=target_quote, 
            page_num=best_match["page"],
            output_filename=f"citation_multidoc_{int(time.time())}.png",
            source_pdf=best_match["source"]
        )
        
        if img_path:
            images.append(img_path)
            
    return {"status": "success", "image_context": images}


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