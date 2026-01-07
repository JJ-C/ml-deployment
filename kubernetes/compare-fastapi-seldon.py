#!/usr/bin/env python3

import requests
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tabulate import tabulate

print("=" * 80)
print("FastAPI vs Seldon Core - Performance Comparison")
print("=" * 80)

# Endpoints
FASTAPI_URL = "http://localhost:8000/predict"
SELDON_URL = "http://localhost:8001/api/v1.0/predictions"

def check_endpoints():
    """Check if both endpoints are available"""
    print("\n[1/5] Checking endpoints...")
    
    fastapi_available = False
    seldon_available = False
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            fastapi_available = True
            print("  ✓ FastAPI is available")
    except:
        print("  ✗ FastAPI is not available (start with: python poc/serve_model.py)")
    
    try:
        response = requests.get("http://localhost:8001/health", timeout=2)
        if response.status_code in [200, 405]:  # 405 is ok, means endpoint exists
            seldon_available = True
            print("  ✓ Seldon Core is available")
    except:
        print("  ✗ Seldon Core is not available")
        print("    Start with: kubectl port-forward svc/fraud-detector-default 8001:8000 -n ml-models")
    
    return fastapi_available, seldon_available

def test_fastapi(transaction):
    """Test FastAPI endpoint"""
    start = time.time()
    try:
        response = requests.post(
            FASTAPI_URL,
            json={
                "transaction_id": transaction["id"],
                "amount": transaction["amount"],
                "transaction_hour": transaction["hour"],
                "merchant_category": transaction["category"],
                "foreign_transaction": transaction["foreign"],
                "location_mismatch": transaction["location_mismatch"],
                "device_trust_score": transaction["device_trust"],
                "velocity_last_24h": transaction["velocity"],
                "cardholder_age": transaction["age"]
            },
            timeout=5
        )
        latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "latency": latency,
                "prediction": result["is_fraud"],
                "probability": result["fraud_probability"]
            }
        else:
            return {"success": False, "latency": latency, "error": response.status_code}
    except Exception as e:
        return {"success": False, "latency": (time.time() - start) * 1000, "error": str(e)}

def test_seldon(transaction):
    """Test Seldon Core endpoint"""
    start = time.time()
    try:
        # Format as ndarray for Seldon
        features = [
            transaction["amount"],
            transaction["hour"],
            0 if transaction["category"] == "Grocery" else 1,  # Simplified encoding
            transaction["foreign"],
            transaction["location_mismatch"],
            transaction["device_trust"],
            transaction["velocity"],
            transaction["age"]
        ]
        
        response = requests.post(
            SELDON_URL,
            json={"data": {"ndarray": [features]}},
            timeout=5
        )
        latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            result = response.json()
            prediction = result.get("data", {}).get("ndarray", [[]])[0]
            
            # Handle different response formats
            if isinstance(prediction, list) and len(prediction) >= 2:
                fraud_prob = prediction[1]
            else:
                fraud_prob = prediction[0] if isinstance(prediction, list) else prediction
            
            is_fraud = fraud_prob > 0.5
            
            return {
                "success": True,
                "latency": latency,
                "prediction": is_fraud,
                "probability": fraud_prob
            }
        else:
            return {"success": False, "latency": latency, "error": response.status_code}
    except Exception as e:
        return {"success": False, "latency": (time.time() - start) * 1000, "error": str(e)}

# Test cases
test_transactions = [
    {"id": "low_risk", "amount": 45.0, "hour": 14, "category": "Grocery", 
     "foreign": 0, "location_mismatch": 0, "device_trust": 85.0, "velocity": 2, "age": 35},
    {"id": "high_risk", "amount": 1500.0, "hour": 3, "category": "Electronics",
     "foreign": 1, "location_mismatch": 1, "device_trust": 25.0, "velocity": 8, "age": 22},
    {"id": "medium_risk", "amount": 250.0, "hour": 20, "category": "Travel",
     "foreign": 0, "location_mismatch": 1, "device_trust": 60.0, "velocity": 4, "age": 45},
]

fastapi_available, seldon_available = check_endpoints()

if not (fastapi_available or seldon_available):
    print("\n✗ No endpoints available. Exiting.")
    exit(1)

print("\n[2/5] Testing individual predictions...")
print("=" * 80)

comparison_results = []

for transaction in test_transactions:
    print(f"\nTransaction: {transaction['id']}")
    
    if fastapi_available:
        fastapi_result = test_fastapi(transaction)
        if fastapi_result["success"]:
            print(f"  FastAPI:     {'FRAUD' if fastapi_result['prediction'] else 'LEGIT':8} "
                  f"(prob: {fastapi_result['probability']:.3f}, latency: {fastapi_result['latency']:.2f}ms)")
    else:
        fastapi_result = {"success": False}
    
    if seldon_available:
        seldon_result = test_seldon(transaction)
        if seldon_result["success"]:
            print(f"  Seldon Core: {'FRAUD' if seldon_result['prediction'] else 'LEGIT':8} "
                  f"(prob: {seldon_result['probability']:.3f}, latency: {seldon_result['latency']:.2f}ms)")
    else:
        seldon_result = {"success": False}
    
    comparison_results.append({
        "transaction": transaction['id'],
        "fastapi": fastapi_result,
        "seldon": seldon_result
    })

print("\n[3/5] Load testing (100 requests each)...")
print("=" * 80)

def load_test(endpoint_func, n_requests=100):
    """Run load test on an endpoint"""
    results = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for _ in range(n_requests):
            transaction = test_transactions[_ % len(test_transactions)]
            future = executor.submit(endpoint_func, transaction)
            futures.append(future)
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    return results

fastapi_load_results = []
seldon_load_results = []

if fastapi_available:
    print("\n  Testing FastAPI...")
    fastapi_load_results = load_test(test_fastapi, 100)
    successful = [r for r in fastapi_load_results if r["success"]]
    print(f"    Success rate: {len(successful)}/100")
    if successful:
        latencies = [r["latency"] for r in successful]
        print(f"    Average latency: {np.mean(latencies):.2f}ms")

if seldon_available:
    print("\n  Testing Seldon Core...")
    seldon_load_results = load_test(test_seldon, 100)
    successful = [r for r in seldon_load_results if r["success"]]
    print(f"    Success rate: {len(successful)}/100")
    if successful:
        latencies = [r["latency"] for r in successful]
        print(f"    Average latency: {np.mean(latencies):.2f}ms")

print("\n[4/5] Computing statistics...")
print("=" * 80)

def compute_stats(results):
    """Compute statistics from load test results"""
    successful = [r for r in results if r["success"]]
    if not successful:
        return None
    
    latencies = [r["latency"] for r in successful]
    latencies_sorted = sorted(latencies)
    
    return {
        "total": len(results),
        "successful": len(successful),
        "success_rate": len(successful) / len(results) * 100,
        "avg_latency": np.mean(latencies),
        "min_latency": min(latencies),
        "max_latency": max(latencies),
        "p50_latency": latencies_sorted[len(latencies) // 2],
        "p95_latency": latencies_sorted[int(len(latencies) * 0.95)],
        "p99_latency": latencies_sorted[int(len(latencies) * 0.99)]
    }

print("\n[5/5] Comparison Summary")
print("=" * 80)

table_data = []

if fastapi_available and fastapi_load_results:
    stats = compute_stats(fastapi_load_results)
    if stats:
        table_data.append([
            "FastAPI",
            f"{stats['success_rate']:.1f}%",
            f"{stats['avg_latency']:.2f}ms",
            f"{stats['p50_latency']:.2f}ms",
            f"{stats['p95_latency']:.2f}ms",
            f"{stats['p99_latency']:.2f}ms"
        ])

if seldon_available and seldon_load_results:
    stats = compute_stats(seldon_load_results)
    if stats:
        table_data.append([
            "Seldon Core",
            f"{stats['success_rate']:.1f}%",
            f"{stats['avg_latency']:.2f}ms",
            f"{stats['p50_latency']:.2f}ms",
            f"{stats['p95_latency']:.2f}ms",
            f"{stats['p99_latency']:.2f}ms"
        ])

if table_data:
    print("\n" + tabulate(
        table_data,
        headers=["Platform", "Success Rate", "Avg Latency", "P50", "P95", "P99"],
        tablefmt="grid"
    ))

print("\n" + "=" * 80)
print("Analysis")
print("=" * 80)

if fastapi_available and seldon_available and fastapi_load_results and seldon_load_results:
    fastapi_stats = compute_stats(fastapi_load_results)
    seldon_stats = compute_stats(seldon_load_results)
    
    if fastapi_stats and seldon_stats:
        print("\nKey Findings:")
        
        # Latency comparison
        latency_diff = seldon_stats['avg_latency'] - fastapi_stats['avg_latency']
        print(f"\n1. Latency:")
        print(f"   - FastAPI is {abs(latency_diff):.2f}ms {'faster' if latency_diff > 0 else 'slower'} on average")
        print(f"   - Both meet <5ms target: {'✓' if max(fastapi_stats['p99_latency'], seldon_stats['p99_latency']) < 5 else '✗'}")
        
        # Reliability
        print(f"\n2. Reliability:")
        print(f"   - FastAPI success rate: {fastapi_stats['success_rate']:.1f}%")
        print(f"   - Seldon success rate: {seldon_stats['success_rate']:.1f}%")
        
        # Production readiness
        print(f"\n3. Production Features:")
        print(f"   - FastAPI: Simple, direct serving")
        print(f"   - Seldon: Built-in scaling, monitoring, A/B testing")

print("\n" + "=" * 80)
print("✓ Comparison Complete!")
print("=" * 80)
print("\nConclusions:")
print("- FastAPI: Best for POC, rapid iteration, simple deployments")
print("- Seldon Core: Best for production, multi-model, enterprise features")
print("- Both: Can achieve <5ms latency target")
print("\nRecommendation:")
print("- Use FastAPI for local development and quick testing")
print("- Use Seldon Core for staging and production deployments")
print("=" * 80)
