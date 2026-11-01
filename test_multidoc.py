import requests

# Test a query that might pull cross-document insights
payload = {
    "claim": "Compare the training requirements or architectural claims across the loaded documents."
}

try:
    response = requests.post("http://127.0.0.1:8001/evaluate_claim", json=payload)
    if response.status_code == 200:
        data = response.json()
        print("🔍 MULTI-DOC EVALUATION OUTPUT:\n")
        print(data.get("final_response", data))
    else:
        print(f"Error: {response.status_code}")
except Exception as e:
    print(f"Connection failed: {e}")