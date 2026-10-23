import os
import time
import base64
import mimetypes
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

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0}
    }
    headers = {"Content-Type": "application/json"}
    
    # Priority cascade: If one model's daily quota dies OR it doesn't exist, seamlessly route to the next
    # Priority cascade using active 2026 Gemini models
    models_to_try = [
        "gemini-3.8-flash",       # Primary model
        "gemini-3.5-flash-lite",  # High-throughput, cost-sensitive fallback
        "gemini-4-argon"          # Flagship reasoning fallback
    ]
    
    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        # Try each model twice for standard 503 traffic spikes
        for attempt in range(2):
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
                
            elif response.status_code == 503:
                time.sleep(2)
                continue
                
            elif response.status_code == 429:
                print(f"    [Quota Exhausted] {model} daily limit reached. Cascading...")
                break # Breaks the retry loop, moves to the next model in models_to_try
                
            elif response.status_code == 404:
                print(f"    [Model Unavailable] {model} is not accessible on this endpoint. Cascading...")
                break # Breaks the retry loop, moves to the next model
                
        # If it fails with a hard error that isn't quota, server load, or missing model, crash gracefully
        if response.status_code not in [429, 503, 404]:
            raise Exception(f"API Error ({response.status_code}): {response.text}")
            
    raise Exception("Critical: All Google model fallbacks and daily quotas have been exhausted.")

# 2. Define Agent State Schema
class AgentState(TypedDict):
    claim: str
    text_context: List[str]
    image_context: List[str]
    needs_web_search: bool
    final_response: str

# 3. Define Workflow Nodes
def retrieve_node(state: AgentState):
    """Page-Linked Retrieval: Bypasses CLIP's inability to read charts by linking images to high-scoring text."""
    # 1. Pull the best text evidence
    text_hits = search_evidence(state["claim"], limit=4, item_type="text")
    
    # 2. Identify which pages contained the critical text
    relevant_pages = list(set([hit.payload.get('page') for hit in text_hits if hit.payload.get('page')]))
    
    # 3. Pull images ONLY from those highly relevant pages
    images = []
    if relevant_pages:
        # We query the DB for images, then filter by the pages we know are relevant
        image_hits = search_evidence(state["claim"], limit=15, item_type="image")
        for hit in image_hits:
            if hit.payload.get('page') in relevant_pages and len(images) < 3:
                images.append(hit.payload["path"])
    
    texts = [f"[Page {hit.payload.get('page', 'N/A')}] {hit.payload['content']}" for hit in text_hits]
            
    return {
        "text_context": texts,
        "image_context": images
    }

def generate_verdict_node(state: AgentState):
    claim = state["claim"]
    context = "\n\n".join(state["text_context"])
    images = state["image_context"]
    
    prompt = f"""You are an impartial, highly rigorous debate judge.
Analyze the provided text evidence AND attached visual diagrams to render a verdict on the user's claim.

You MUST structure your exact response using these three specific headings:
### SUPPORTING EVIDENCE
(List the facts, text, and visual data that validate the claim)

### REFUTING EVIDENCE
(List the facts, text, and visual data that contradict or challenge the claim)

### FINAL VERDICT
(Conclude with a definitive SUPPORTED, REFUTED, or INCONCLUSIVE, followed by a harsh, logical summary)

Claim to Evaluate:
{claim}

Retrieved Text Evidence Context:
{context}"""
    
    final_text = call_gemini(prompt, image_paths=images)
    return {"final_response": final_text}

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