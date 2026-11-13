import requests
import time
import json

BASE_URL = "http://127.0.0.1:8000"

def run_diagnostics():
    print("=== 🚀 INITIATING END-TO-END PIPELINE DIAGNOSTICS ===\n")

    # 1. Check Server & Active Documents
    print("⏳ 1. Checking Server & Vector Space...")
    try:
        res = requests.get(f"{BASE_URL}/documents")
        if res.status_code == 200:
            docs = res.json().get("documents", [])
            print(f"✅ SERVER: Online. Active Documents: {len(docs)}")
            if not docs:
                print("⚠️ WARNING: No documents found. Upload a PDF via the UI first.")
                return
        else:
            print(f"❌ SERVER ERROR: {res.status_code}")
            return
    except Exception as e:
        print(f"❌ CONNECTION FAILED: Is Uvicorn running? Error: {e}")
        return

    # 2. Test Streaming Inference
    print("\n⏳ 2. Testing Streaming Inference (/evaluate_claim)...")
    claim_payload = {"claim": "Compare the memory overhead reduction of standard LoRA versus QLoRA."}
    
    start_time = time.time()
    ttft = None
    
    full_text = ""
    metadata = []
    
    # Open the streaming connection
    res = requests.post(f"{BASE_URL}/evaluate_claim", json=claim_payload, stream=True)
    
    for line in res.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: "):
                # Capture Time to First Token on the very first chunk
                if ttft is None:
                    ttft = time.time() - start_time
                    
                data = json.loads(decoded_line[6:])
                
                if data.get("type") == "error":
                    print(f"❌ STREAM ERROR: {data['content']}")
                    return
                elif data.get("type") == "complete":
                    full_text = data.get("full_text", "")
                    metadata = data.get("metadata", [])
    
    stream_end_time = time.time()
    stream_duration = stream_end_time - start_time
    print(f"✅ STREAM SUCCESS: Received {len(full_text)} characters.")

    # 3. Test Visual Artifact Generation (The Second Pass)
    print("\n⏳ 3. Testing Visual Artifact Generation (/generate_artifact)...")
    artifact_start = time.time()
    img_res = requests.post(
        f"{BASE_URL}/generate_artifact", 
        json={"full_text": full_text, "metadata": metadata}
    )
    artifact_duration = time.time() - artifact_start
    
    if img_res.status_code == 200:
        img_data = img_res.json()
        images = img_data.get("image_context", [])
        print(f"✅ ARTIFACT SUCCESS: Generated {len(images)} visual artifact(s).")
    else:
        print(f"❌ ARTIFACT FAILED: {img_res.status_code}")

    # 4. Final Performance Summary
    if ttft is None:
        ttft = 0.0 # Safety net if the stream yields 0 chunks
        
    total_time = stream_duration + artifact_duration
    print("\n=== 🏁 DIAGNOSTICS SUMMARY ===")
    print(f"⏱️ Time to First Token (TTFT): {ttft:.2f}s (UI begins typing)")
    print(f"⏱️ Total Stream Duration:      {stream_duration:.2f}s (Text synthesis complete)")
    print(f"⏱️ Artifact Generation Time:   {artifact_duration:.2f}s (Image appears in sidebar)")
    print(f"🚀 TOTAL PIPELINE EXECUTION:   {total_time:.2f}s")

if __name__ == "__main__":
    run_diagnostics()