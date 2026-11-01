import requests
import json

# Define your benchmark test suite (derived from the PDF report)
BENCHMARK_TESTS = [
    {
        "id": "TC-01",
        "category": "Direct Support & Bounding Box Precision",
        "claim": "Dot-product attention requires scaling because large values push the softmax function into regions with extremely small gradients.",
        "expected_verdict": "SUPPORTED"
    },
    {
        "id": "TC-02",
        "category": "Direct Refutation (Architecture)",
        "claim": "The Transformer architecture relies heavily on recurrent neural networks to maintain the order of the input sequence.",
        "expected_verdict": "REFUTED"
    },
    {
        "id": "TC-03",
        "category": "Data Nuance Trap (Base vs Big Model)",
        "claim": "The base Transformer model took 3.5 days to train on 8 P100 GPUs.",
        "expected_verdict": "REFUTED"
    },
    {
        "id": "TC-04",
        "category": "Hallucination Rejection Trap",
        "claim": "The Transformer model uses the AdamW optimizer with aggressive weight decay to prevent overfitting.",
        "expected_verdict": "REFUTED" # Or INCONCLUSIVE depending on strict wording
    },
    {
        "id": "TC-05",
        "category": "Core Mechanism Validation",
        "claim": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
        "expected_verdict": "SUPPORTED"
    }
]

API_URL = "http://127.0.0.1:8001/evaluate_claim" # Adjust to your FastAPI port if different

def run_evaluation():
    print("🚀 Starting Automated RAG Evaluation Suite...")
    passed_tests = 0
    total_tests = len(BENCHMARK_TESTS)

    for test in BENCHMARK_TESTS:
        print(f"\n----------------------------------------")
        print(f"Running [{test['id']}]: {test['category']}")
        print(f"Claim: \"{test['claim']}\"")
        print(f"Expected: {test['expected_verdict']}")
        
        payload = {"claim": test["claim"]}
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                data = response.json()
                # Assuming your API returns the text response under 'final_response' or similar key
                # Adjust key based on your exact FastAPI response schema
                result_text = data.get("final_response", str(data))
                
                print(f"Agent Output Preview:\n{result_text[:250]}...")
                
                # Check if expected verdict is in the output text
                if test["expected_verdict"] in result_text.upper():
                    print(f"✅ RESULT: PASSED")
                    passed_tests += 1
                else:
                    print(f"❌ RESULT: FAILED (Expected verdict missing from response)")
            else:
                print(f"❌ API Error: Status code {response.status_code}")
        except Exception as e:
            print(f"❌ Connection Error: {e}")

    print(f"\n========================================")
    print(f"🏁 EVALUATION COMPLETE: {passed_tests}/{total_tests} Tests Passed ({(passed_tests/total_tests)*100:.1f}%)")
    print(f"========================================")

if __name__ == "__main__":
    run_evaluation()