import requests

print("⏳ Testing SSE Stream...")
res = requests.post(
    "http://127.0.0.1:8000/evaluate_claim", 
    json={"claim": "What are the core topics discussed in the documents?"}, 
    stream=True
)

for line in res.iter_lines():
    if line:
        print(line.decode('utf-8'))

print("\n✅ Stream complete.")