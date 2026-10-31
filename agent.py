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
    """Dynamically highlights the entire exact quote across multiple lines."""
    if not source_pdf or not os.path.exists(source_pdf):
        print(f"⚠️ Error: Source PDF {source_pdf} not found.")
        return None
        
    try:
        doc = pymupdf.open(source_pdf)
        page_index = int(page_num) - 1 
        
        if page_index < 0 or page_index >= len(doc):
            return None
            
        page = doc[page_index]
        
        # 1. Clean the LLM quote of excess formatting and hidden characters
        clean_chunk = " ".join(text_chunk.split())
        
        # 2. Chop the full quote into small 5-word phrases. 
        # This completely bypasses physical line-break boundaries in the PDF.
        words = clean_chunk.split()
        phrases = [" ".join(words[i:i+4]) for i in range(0, len(words), 4)]
        
        # 3. Highlight every phrase individually to build a massive, continuous highlight block
        for phrase in phrases:
            # Skip phrases that are too short to accurately match
            if len(phrase) < 10:
                continue
                
            text_instances = page.search_for(phrase)
            for inst in text_instances:
                annot = page.add_highlight_annot(inst)
                annot.set_colors(stroke=(1, 0.9, 0)) # RGB for Yellow
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
    """Strictly retrieves text and metadata. Visual rendering is deferred to the verdict node."""
    text_hits = search_evidence(state["claim"], limit=7, item_type="text")
    
    texts = []
    metadata = []
    
    for hit in text_hits:
        chunk = hit.payload["content"]
        page_num = hit.payload.get("page", 1)
        source_pdf = hit.payload.get("source_file")
        
        texts.append(f"[Page {page_num}] {chunk}")
        
        # Store the mapping so the verdict node can match the LLM's exact quote to the physical file
        metadata.append({
            "page": page_num, 
            "source": source_pdf, 
            "chunk": chunk
        })
                
    return {
        "text_context": texts,
        "retrieved_metadata": metadata
    }

def generate_verdict_node(state: AgentState):
    claim = state["claim"]
    context = "\n\n".join(state["text_context"])
    metadata = state["retrieved_metadata"]
    
    prompt = f"""You are an impartial, highly rigorous debate judge analyzing a specific document.
Analyze the provided text evidence to render a strictly logical verdict on the user's claim.

CRITICAL INSTRUCTIONS:
1. If the provided evidence does not explicitly contain enough information to validate the point, you MUST declare the verdict INCONCLUSIVE. Do not invent logic.
2. You MUST quote the actual text to prove your points.

You MUST structure your exact response using these three specific headings:
### SUPPORTING EVIDENCE
(Provide direct quotes from the text that support the claim, or write "None found.")

### REFUTING EVIDENCE
(Provide direct quotes from the text that contradict the claim, or write "None found.")

### FINAL VERDICT
(Start with SUPPORTED, REFUTED, or INCONCLUSIVE, followed by 1-2 sentences of strict logical reasoning explaining why.)

Claim to Evaluate:
{claim}

Retrieved Text Evidence Context:
{context}"""
    
    print("⏳ Waiting for Gemini API response...")
    start_gemini = time.time() 
    
    # We no longer pass images to Gemini; it evaluates pure text logic first
    final_text = call_gemini(prompt, image_paths=None) 
    
    end_gemini = time.time() 
    print(f"✅ Gemini API responded in {end_gemini - start_gemini:.2f} seconds.")
    
    # --- EXACT-QUOTE VISUAL TARGETING ---
    # 1. Extract the exact text Gemini enclosed in quotes
    quotes = re.findall(r'"([^"]*)"', final_text)
    
    images = []
    if quotes and metadata:
        target_quote = quotes[0] # Target the primary piece of evidence
        
        # 2. Cross-reference the quote with our metadata to find the correct page
        best_match = metadata[0] # Default to the top hit
        for meta in metadata:
            # If the first 20 characters of the quote match this chunk, we found the page
            if target_quote[:20] in meta["chunk"]:
                best_match = meta
                break
                
        # 3. Render the physical bounding box on the exact quote
        img_path = generate_highlighted_citation(
            text_chunk=target_quote, 
            page_num=best_match["page"],
            output_filename=f"citation_exact_{int(time.time())}.png",
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