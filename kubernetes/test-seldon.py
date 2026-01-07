#!/usr/bin/env python3

import requests
import json
import time
import sys

print("=" * 60)
print("Testing Seldon Core Deployment")
print("=" * 60)

# Determine endpoint based on K8s platform
ENDPOINTS = {
    "port-forward": "http://localhost:8001",
    "nodeport": "http://localhost:30000",
    "minikube": None  # Will be detected
}

def get_minikube_ip():
    """Get minikube IP if available"""
    try:
        import subprocess
        result = subprocess.run(['minikube', 'ip'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return f"http://{result.stdout.strip()}:30000"
    except:
        pass
    return None

def detect_endpoint():
    """Try to detect which endpoint to use"""
    # Try port-forward first
    try:
        response = requests.get(f"{ENDPOINTS['port-forward']}/health", timeout=2)
        if response.status_code == 200:
            return ENDPOINTS['port-forward']
    except:
        pass
    
    # Try nodeport
    try:
        response = requests.get(f"{ENDPOINTS['nodeport']}/health", timeout=2)
        if response.status_code == 200:
            return ENDPOINTS['nodeport']
    except:
        pass
    
    # Try minikube
    minikube_url = get_minikube_ip()
    if minikube_url:
        try:
            response = requests.get(f"{minikube_url}/health", timeout=2)
            if response.status_code == 200:
                return minikube_url
        except:
            pass
    
    return None

print("\n[1/4] Detecting Seldon endpoint...")
endpoint = detect_endpoint()

if not endpoint:
    print("✗ Cannot connect to Seldon deployment")
    print("\nPlease ensure:")
    print("1. Seldon deployment is running:")
    print("   kubectl get sdep -n ml-models")
    print("")
    print("2. Port forwarding is active:")
    print("   kubectl port-forward svc/fraud-detector-default 8001:8000 -n ml-models")
    print("")
    print("   OR use NodePort:")
    print("   http://localhost:30000 (Docker Desktop / Kind)")
    print("   http://$(minikube ip):30000 (Minikube)")
    sys.exit(1)

print(f"✓ Connected to: {endpoint}")

print("\n[2/4] Checking health...")
try:
    health_response = requests.get(f"{endpoint}/health", timeout=5)
    print(f"✓ Health check: {health_response.status_code}")
except Exception as e:
    print(f"✗ Health check failed: {e}")
    sys.exit(1)

print("\n[3/4] Testing predictions...")

test_cases = [
    {
        "name": "Low Risk Transaction",
        "data": {
            "data": {
                "ndarray": [[45.0, 14, 2, 0, 0, 85.0, 2, 35]]
            }
        }
    },
    {
        "name": "High Risk Transaction", 
        "data": {
            "data": {
                "ndarray": [[1500.0, 3, 0, 1, 1, 25.0, 8, 22]]
            }
        }
    }
]

results = []

for test_case in test_cases:
    print(f"\n  Testing: {test_case['name']}")
    
    start = time.time()
    try:
        response = requests.post(
            f"{endpoint}/api/v1.0/predictions",
            json=test_case['data'],
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            result = response.json()
            prediction = result.get('data', {}).get('ndarray', [[]])[0]
            
            # Assuming binary classification with predict_proba
            if isinstance(prediction, list) and len(prediction) >= 2:
                fraud_prob = prediction[1]  # Probability of class 1 (fraud)
            else:
                fraud_prob = prediction[0] if isinstance(prediction, list) else prediction
            
            is_fraud = fraud_prob > 0.5
            
            print(f"    Result: {'FRAUD' if is_fraud else 'LEGIT'}")
            print(f"    Fraud probability: {fraud_prob:.3f}")
            print(f"    Latency: {latency:.2f}ms")
            
            results.append({
                "test": test_case['name'],
                "status": "SUCCESS",
                "latency": latency,
                "prediction": "FRAUD" if is_fraud else "LEGIT"
            })
        else:
            print(f"    ✗ Error: {response.status_code}")
            print(f"    Response: {response.text}")
            results.append({
                "test": test_case['name'],
                "status": "ERROR",
                "error": response.status_code
            })
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append({
            "test": test_case['name'],
            "status": "EXCEPTION",
            "error": str(e)
        })

print("\n[4/4] Summary")
print("=" * 60)

successful = [r for r in results if r["status"] == "SUCCESS"]
if successful:
    latencies = [r["latency"] for r in successful]
    print(f"✓ Successful predictions: {len(successful)}/{len(results)}")
    print(f"  Average latency: {sum(latencies)/len(latencies):.2f}ms")
    print(f"  Min latency: {min(latencies):.2f}ms")
    print(f"  Max latency: {max(latencies):.2f}ms")
else:
    print("✗ No successful predictions")

print("\n" + "=" * 60)
print("✓ Testing Complete!")
print("=" * 60)
print("\nCompare with FastAPI:")
print("  python poc/test_predictions.py")
print("\nView deployment:")
print("  kubectl get sdep -n ml-models")
print("  kubectl describe sdep fraud-detector -n ml-models")
print("=" * 60)
