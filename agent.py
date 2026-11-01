import os
import time
import base64
import re
import mimetypes
import pymupdf
import requests
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from ddgs import DDGS
from retrieve import search_evidence
from dotenv import load_dotenv

load_dotenv()

# 1. Multimodal REST API Execution (With Automated Model Cascading & 404 Bypass)
def call_gemini(prompt: str, image_paths: List[str] = None) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Critical Error: GOOGLE_API_KEY is missing. Ensure it is in your .env file or exported.")
        
    parts = [{"text": prompt}]
    
    if image_paths:
        for img_path in image_paths:
            if os.path.exists(img_path):
                mime_type, _ = mimetypes.guess_type(img_path)
                with open(img_path, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("utf-8")
                
                parts.append({
                    "inlineData": {
                        "mimeType": mime_type or "image/jpeg",
                        "data": b64_data
                    }
                })

    # SPEED OPTIMIZATION: Strict maxOutputTokens forces the model to finish quickly
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 400 
        }
    }
    headers = {"Content-Type": "application/json"}
    
    # Priority cascade: Shifted 'lite' to primary for maximum speed during real-time UI inference
    models_to_try = [
        "gemini-3.5-flash-lite",  # High-throughput, lowest latency primary
        "gemini-3.8-flash",       # Standard flash fallback
        "gemini-4-argon"          # Flagship reasoning fallback
    ]
    
    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        # Try each model twice for standard 503 traffic spikes
        for attempt in range(2):
            try:
                # SPEED OPTIMIZATION: Hard 15-second network timeout. If it hangs, it dies and cascades immediately.
                response = requests.post(url, headers=headers, json=payload, timeout=15)
            except requests.exceptions.Timeout:
                print(f"    [Timeout] {model} network request hung for >15s. Cascading...")
                break # Breaks the retry loop, moves to the next model in models_to_try
            
            if response.status_code == 200:
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
                
            elif response.status_code == 503:
                time.sleep(2)
                continue
                
            elif response.status_code == 429:
                print(f"    [Quota Exhausted] {model} daily limit reached. Cascading...")
                break 
                
            elif response.status_code == 404:
                print(f"    [Model Unavailable] {model} is not accessible on this endpoint. Cascading...")
                break 
                
        # If it fails with a hard error that isn't quota, server load, or missing model, crash gracefully
        if 'response' in locals() and response.status_code not in [429, 503, 404]:
            raise Exception(f"API Error ({response.status_code}): {response.text}")
            
    raise Exception("Critical: All Google model fallbacks and daily quotas have been exhausted.")


from typing import TypedDict, List

# 2. Define Agent State Schema
class AgentState(TypedDict):
    claim: str
    text_context: List[str]
    retrieved_metadata: List[dict]
    image_context: List[str]
    needs_web_search: bool
    final_response: str


def generate_highlighted_citation(text_chunk, page_num, output_filename, source_pdf):
    """Layout-aware citation renderer that safely validates source file paths."""
    # Fallback safety: If source_pdf is missing or invalid, default to any available PDF in data/pdfs
    if not source_pdf or not os.path.exists(source_pdf):
        pdf_dir = "data/pdfs"
        if os.path.exists(pdf_dir):
            available_pdfs = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
            if available_pdfs:
                source_pdf = available_pdfs[0] # Pick the first available active PDF as fallback
            else:
                print(f"⚠️ Error: No active PDFs found in {pdf_dir}.")
                return None
        else:
            return None
            
    try:
        doc = pymupdf.open(source_pdf)
        page_index = int(page_num) - 1 
        
        if page_index < 0 or page_index >= len(doc):
            doc.close()
            return None
            
        page = doc[page_index]
        page_rect = page.rect
        valid_blocks = []
        blocks = page.get_text("blocks")
        
        for b in blocks:
            bx0, by0, bx1, by1, btext, bno, btype = b
            if btype == 0 and by0 > 50 and by1 < (page_rect.height - 50):
                valid_blocks.append(b)

        clean_chunk = " ".join(text_chunk.split())
        words = clean_chunk.split()
        phrases = [" ".join(words[i:i+4]) for i in range(0, len(words), 4)]
        
        highlight_drawn = False
        for phrase in phrases:
            if len(phrase) < 10:
                continue
                
            text_instances = page.search_for(phrase)
            for inst in text_instances:
                inst_y = (inst.y0 + inst.y1) / 2
                is_in_valid_margin = any(b[1] <= inst_y <= b[3] for b in valid_blocks)
                
                if is_in_valid_margin or len(text_instances) == 1:
                    annot = page.add_highlight_annot(inst)
                    annot.set_colors(stroke=(1, 0.9, 0))
                    annot.update()
                    highlight_drawn = True
            
        if not highlight_drawn:
            for phrase in phrases[:2]:
                if len(phrase) >= 10:
                    for inst in page.search_for(phrase):
                        annot = page.add_highlight_annot(inst)
                        annot.set_colors(stroke=(1, 0.9, 0))
                        annot.update()

        os.makedirs("data/images", exist_ok=True)
        output_path = f"data/images/{output_filename}"
        
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2)) 
        pix.save(output_path)
        doc.close()
        
        return output_path
    except Exception as e:
        print(f"⚠️ Citation Render Error: {e}")
        return None
# 3. Define Workflow Nodes

def retrieve_node(state: AgentState):
    """Retrieves top text chunks across all active documents in Qdrant with source tracking."""
    # Set limit=3 to fetch exactly 3 chunks PER DOCUMENT using our iterative search logic
    text_hits = search_evidence(state["claim"], limit_per_doc=5, item_type="text")
    
    texts = []
    metadata = []
    
    print("\n--- 🔍 DEBUG: CONTEXT SENT TO LLM ---")
    
    for hit in text_hits:
        chunk = hit.payload.get("content", "")
        page_num = hit.payload.get("page", "?")
        source_pdf = hit.payload.get("source_file", "")
        
        # Extract just the clean filename for the LLM to read easily
        doc_name = os.path.basename(source_pdf) if source_pdf else "Unknown Document"
        
        # Print to your terminal so you can verify BOTH test1 and test2 are here
        print(f"Loaded chunk from: {doc_name} (Page {page_num})")
        
        # Inject document name and page into the context stream using the exact format the prompt expects
        texts.append(f"Source: [{doc_name}, Page {page_num}]\nContent: {chunk}\n")
        
        metadata.append({
            "page": page_num, 
            "source": source_pdf, 
            "chunk": chunk,
            "doc_name": doc_name
        })
        
    print("--------------------------------------\n")
            
    return {
        "text_context": texts,
        "retrieved_metadata": metadata
    }

def generate_verdict_node(state: AgentState):
    claim = state["claim"]
    context = "\n\n".join(state["text_context"])
    metadata = state["retrieved_metadata"]
    
    prompt = f"""You are an elite, impartial RAG Validation Engine.
Your task is to evaluate the CLAIM against the provided MULTI-DOCUMENT CONTEXT.

CRITICAL INSTRUCTIONS:
1. You MUST cross-reference evidence from different source files if they are present in the context (e.g., comparing [test1.pdf] with [test2.pdf]).
2. Quote exactly from the text and cite the source using this exact format: [filename.pdf, Page X].
3. Identify if the claim is SUPPORTED, REFUTED, or INCONCLUSIVE based strictly on the text.
4. Output your response EXACTLY in this strict format:

### SUPPORTING EVIDENCE
- As stated in [filename.pdf, Page X]: "exact quote here"

### REFUTING EVIDENCE
- As stated in [filename.pdf, Page Y]: "exact quote here"

### FINAL VERDICT
[SUPPORTED/REFUTED/INCONCLUSIVE]. A concise 1-sentence synthesis comparing the findings across the documents.

CLAIM: {claim}

MULTI-DOCUMENT CONTEXT:
{context}"""
    
    print("⏳ Waiting for Gemini API response...")
    start_gemini = time.time() 
    
    final_text = call_gemini(prompt, image_paths=None) 
    
    end_gemini = time.time() 
    print(f"✅ Gemini API responded in {end_gemini - start_gemini:.2f} secondschauffer.")
    
    # Exact-quote visual targeting matching the primary quote back to its source PDF
    quotes = re.findall(r'"([^"]*)"', final_text)
    
    images = []
    if quotes and metadata:
        target_quote = quotes[0]
        
        best_match = metadata[0]
        for meta in metadata:
            if target_quote[:20] in meta["chunk"]:
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
            
    return {
        "final_response": final_text, 
        "image_context": images
    }

def grade_evidence_node(state: AgentState):
    claim = state["claim"]
    context = "\n\n".join(state["text_context"])
    
    if not context.strip() and not state["image_context"]:
        return {"needs_web_search": True}

    prompt = f"""You are a strict debate evidence evaluator.
Claim: {claim}

Retrieved Evidence Context:
{context}

Does the retrieved evidence context contain sufficient, specific facts to evaluate or verify this claim?
Respond strictly with ONLY 'YES' or 'NO'."""
    
    response_text = call_gemini(prompt).strip().upper()
    needs_search = "NO" in response_text
    
    return {"needs_web_search": needs_search}

def web_search_node(state: AgentState):
    claim = state["claim"]
    web_texts = []
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(claim, max_results=3))
            for r in results:
                web_texts.append(f"[Web Search] {r['body']}")
    except Exception as e:
        web_texts.append(f"[Web Search Error] Could not retrieve live web results: {str(e)}")
    
    updated_context = state["text_context"] + web_texts
    return {"text_context": updated_context}

# 4. Conditional Routing Logic
def decide_next_step(state: AgentState):
    if state["needs_web_search"]:
        return "web_search"
    return "generate_verdict"

# 5. Build LangGraph Workflow Graph
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_evidence", grade_evidence_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate_verdict", generate_verdict_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_evidence")

workflow.add_conditional_edges(
    "grade_evidence",
    decide_next_step,
    {
        "web_search": "web_search",
        "generate_verdict": "generate_verdict"
    }
)

workflow.add_edge("web_search", "generate_verdict")
workflow.add_edge("generate_verdict", END)

app = workflow.compile()

if __name__ == "__main__":
    test_claim = "Provide a graph or visual evidence showing how global surface temperature has changed since 1850."
    
    initial_state = {
        "claim": test_claim,
        "text_context": [],
        "image_context": [],
        "needs_web_search": False,
        "final_response": ""
    }
    
    print("\nExecuting Production Agent Graph with Quota Cascading...\n")
    output = app.invoke(initial_state)
    
    print("\n================ FINAL DEBATE VERDICT ================\n")
    print(output["final_response"])
    if output["image_context"]:
        print("\nReferenced Image Paths:", output["image_context"])



# ... (Keep all your existing LangGraph setup, nodes, and app = workflow.compile() logic) ...

def run_debate_agent(claim: str) -> dict:
    """Entry point for the FastAPI backend to execute the LangGraph."""
    initial_state = {
        "claim": claim,
        "text_context": [],
        "image_context": [],
        "needs_web_search": False,
        "final_response": ""
    }
    
    # Execute the graph
    output = app.invoke(initial_state)
    
    # Return structured data for the API response
    return {
        "verdict": output["final_response"],
        "retrieved_images": output["image_context"],
        "retrieved_texts": output["text_context"]
    }