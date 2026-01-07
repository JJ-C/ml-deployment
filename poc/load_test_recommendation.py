#!/usr/bin/env python3
"""
Load test for book recommendation API.
"""

import requests
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import argparse

print("=" * 60)
print("Load Testing Book Recommendation API")
print("=" * 60)

API_URL = "http://localhost:8002/recommend"

# Sample book IDs and titles to use for testing
# These will be populated from the API
SAMPLE_BOOKS = []

def fetch_sample_books():
    """Fetch sample books from the API for testing"""
    global SAMPLE_BOOKS
    
    search_queries = ["Python", "Fiction", "Cooking", "History", "Business", 
                     "Science", "Medical", "Education", "Computer", "Biography"]
    
    for query in search_queries:
        try:
            response = requests.get(
                "http://localhost:8002/books/search",
                params={"query": query, "limit": 5},
                timeout=5
            )
            if response.status_code == 200:
                results = response.json()['results']
                SAMPLE_BOOKS.extend([book['book_id'] for book in results])
        except:
            pass
    
    # Remove duplicates
    SAMPLE_BOOKS = list(set(SAMPLE_BOOKS))
    print(f"Loaded {len(SAMPLE_BOOKS)} sample books for testing")

def generate_request():
    """Generate random recommendation request"""
    if not SAMPLE_BOOKS:
        return None
    
    return {
        "book_id": random.choice(SAMPLE_BOOKS),
        "num_recommendations": random.randint(3, 10),
        "request_id": f"load_test_{random.randint(1, 1000000)}"
    }

def send_request():
    """Send single request and measure latency"""
    start = time.time()
    try:
        req = generate_request()
        if not req:
            return {"success": False, "latency": 0, "error": "No sample books"}
        
        response = requests.post(API_URL, json=req, timeout=10)
        latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            return {"success": True, "latency": latency, "response": response.json()}
        else:
            return {"success": False, "latency": latency, "error": response.status_code}
    except Exception as e:
        latency = (time.time() - start) * 1000
        return {"success": False, "latency": latency, "error": str(e)}

def run_load_test(rps, duration, workers):
    """Run load test with specified parameters"""
    total_requests = rps * duration
    
    print(f"\n[Configuration]")
    print(f"  - Target RPS: {rps}")
    print(f"  - Duration: {duration}s")
    print(f"  - Workers: {workers}")
    print(f"  - Total requests: {total_requests}")
    print(f"\n[Starting load test...]")
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for _ in range(total_requests):
            future = executor.submit(send_request)
            futures.append(future)
            time.sleep(1.0 / rps)
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            
            if i % (total_requests // 10) == 0:
                print(f"  Progress: {i}/{total_requests} ({i*100//total_requests}%)")
    
    elapsed = time.time() - start_time
    
    print(f"\n[Results]")
    print("=" * 60)
    
    successes = [r for r in results if r['success']]
    failures = [r for r in results if not r['success']]
    
    success_rate = len(successes) / len(results) * 100
    actual_rps = len(results) / elapsed
    
    latencies = [r['latency'] for r in successes]
    latencies.sort()
    
    print(f"\nRequests:")
    print(f"  - Total: {len(results)}")
    print(f"  - Success: {len(successes)}")
    print(f"  - Failed: {len(failures)}")
    print(f"  - Success rate: {success_rate:.2f}%")
    
    print(f"\nThroughput:")
    print(f"  - Actual RPS: {actual_rps:.1f}")
    print(f"  - Duration: {elapsed:.2f}s")
    
    if latencies:
        print(f"\nLatency:")
        print(f"  - Average: {sum(latencies)/len(latencies):.2f}ms")
        print(f"  - Min: {min(latencies):.2f}ms")
        print(f"  - Max: {max(latencies):.2f}ms")
        print(f"  - P50: {latencies[len(latencies)//2]:.2f}ms")
        print(f"  - P95: {latencies[int(len(latencies)*0.95)]:.2f}ms")
        print(f"  - P99: {latencies[int(len(latencies)*0.99)]:.2f}ms")
        
        under_20ms = sum(1 for l in latencies if l < 20.0)
        print(f"\n  - Under 20ms: {under_20ms}/{len(latencies)} ({under_20ms*100//len(latencies)}%)")
    
    if successes:
        num_recs = [r['response']['num_results'] for r in successes]
        print(f"\nRecommendations:")
        print(f"  - Average per request: {sum(num_recs)/len(num_recs):.1f}")
        print(f"  - Total generated: {sum(num_recs)}")
    
    if failures:
        print(f"\nErrors:")
        error_counts = defaultdict(int)
        for f in failures:
            error_counts[str(f['error'])] += 1
        for error, count in error_counts.items():
            print(f"  - {error}: {count}")
    
    print("\n" + "=" * 60)
    
    if success_rate >= 99 and latencies and latencies[int(len(latencies)*0.99)] < 20.0:
        print("✓ Load test PASSED!")
        print("  - Success rate >= 99%")
        print("  - P99 latency < 20ms")
    elif success_rate >= 99:
        print("⚠ Load test PARTIAL PASS")
        print("  - Success rate >= 99% ✓")
        print("  - P99 latency >= 20ms ✗")
    else:
        print("✗ Load test FAILED")
        print(f"  - Success rate: {success_rate:.2f}% (target: >= 99%)")
    
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test the book recommendation API")
    parser.add_argument("--rps", type=int, default=20, help="Requests per second (default: 20)")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds (default: 30)")
    parser.add_argument("--workers", type=int, default=10, help="Number of workers (default: 10)")
    
    args = parser.parse_args()
    
    print("\nChecking server availability...")
    try:
        response = requests.get("http://localhost:8002/health", timeout=5)
        if response.status_code != 200:
            print("✗ Server is not healthy")
            exit(1)
        print("✓ Server is healthy")
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        print("  Make sure the server is running: python poc/serve_recommendation_model.py")
        exit(1)
    
    print("\nFetching sample books...")
    fetch_sample_books()
    
    if not SAMPLE_BOOKS:
        print("✗ Could not fetch sample books for testing")
        exit(1)
    
    run_load_test(args.rps, args.duration, args.workers)
