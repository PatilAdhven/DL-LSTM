import requests
import json
import time

# Give server time to start
time.sleep(2)

url = "http://localhost:8000/predict"
data = {
    "seed_text": "deep learning is",
    "top_n": 3
}

try:
    response = requests.post(url, json=data)
    print("Status Code:", response.status_code)
    print("Response JSON:")
    print(json.dumps(response.json(), indent=4))
    
    with open("api_test_proof.txt", "w") as f:
        f.write(f"Status Code: {response.status_code}\n")
        f.write("Response JSON:\n")
        f.write(json.dumps(response.json(), indent=4))
    print("\nSaved proof to api_test_proof.txt")
except Exception as e:
    print("Error:", e)
