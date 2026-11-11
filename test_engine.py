import requests
import time

BASE_URL = "http://127.0.0.1:8000"

def run_diagnostics():
    print("=== 🚀 INITIATING RAG ENGINE DIAGNOSTICS ===\n")

    # 1. Check Server & Active Documents
    print("⏳ Checking Server & Vector Space...")
    try:
        res = requests.get(f"{BASE_URL}/documents")
        if res.status_code == 200:
            docs = res.json().get("documents", [])
            print("✅ SERVER: Online.")
            if docs:
                print(f"✅ DATABASE: Found {len(docs)} active document(s): {', '.join(docs)}")
            else:
                print("⚠️ DATABASE: No documents found. Upload a PDF via the UI first, then run this test again.")
                return
        else:
            print(f"❌ SERVER ERROR: {res.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION FAILED: Is your FastAPI Uvicorn server running?")
        return

    # 2. Test RAG Evaluation Pipeline
    print("\n⏳ Testing LangGraph Agent & Qdrant Retrieval...")
    claim_payload = {"claim": "What are the core topics discussed in the uploaded documents?"}
    
    start_time = time.time()
    res = requests.post(f"{BASE_URL}/evaluate_claim", json=claim_payload)
    elapsed = time.time() - start_time

    if res.status_code == 200:
        data = res.json()
        print(f"✅ EVALUATION SUCCESS ({elapsed:.2f}s):")
        print(f"   ↳ Status: {data.get('status')}")
        print(f"   ↳ Synthesis Length: {len(data.get('final_response', ''))} characters")
        if data.get("image_context"):
            print(f"   ↳ Visual Artifacts Retrieved: {len(data.get('image_context'))}")
        else:
            print("   ↳ Visual Artifacts Retrieved: 0")
    else:
        print(f"❌ EVALUATION FAILED (HTTP {res.status_code}):")
        print(res.text)

    print("\n=== 🏁 DIAGNOSTICS COMPLETE ===")

if __name__ == "__main__":
    run_diagnostics()