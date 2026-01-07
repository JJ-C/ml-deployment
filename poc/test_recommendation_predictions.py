#!/usr/bin/env python3
"""
Test book recommendation API with various queries.
"""

import requests
import json
import time
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8002"

logger.info("=" * 60)
logger.info("Testing Book Recommendation API")
logger.info("=" * 60)

logger.info("\n[1/5] Checking server health...")
try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    if response.status_code == 200:
        logger.info("✓ Server is healthy")
    else:
        logger.error(f"✗ Server returned status {response.status_code}")
        sys.exit(1)
except Exception as e:
    logger.error(f"✗ Cannot connect to server: {e}")
    logger.error("  Make sure the server is running: python poc/serve_recommendation_model.py")
    sys.exit(1)

logger.info("\n[2/5] Getting model info...")
response = requests.get(f"{API_URL}/model/info")
model_info = response.json()
logger.info("✓ Model information:")
logger.info(f"  {json.dumps(model_info, indent=2)}")

logger.info("\n[3/5] Searching for books...")

search_queries = ["Python", "Machine Learning", "Fiction", "Cooking"]

for query in search_queries:
    response = requests.get(f"{API_URL}/books/search", params={"query": query, "limit": 3})
    if response.status_code == 200:
        results = response.json()
        logger.info(f"\n  Search: '{query}' - Found {results['num_results']} books")
        for book in results['results'][:3]:
            logger.info(f"    - {book['title']} by {book['authors']}")

logger.info("\n[4/5] Testing recommendations by book_id...")

# First, search for a book to get its ID
response = requests.get(f"{API_URL}/books/search", params={"query": "Python", "limit": 1})
if response.status_code == 200 and response.json()['num_results'] > 0:
    book = response.json()['results'][0]
    book_id = book['book_id']
    
    logger.info(f"\n  Query book: '{book['title']}'")
    
    # Get recommendations
    rec_request = {
        "book_id": book_id,
        "num_recommendations": 5,
        "request_id": "test_by_id"
    }
    
    start = time.time()
    response = requests.post(f"{API_URL}/recommend", json=rec_request, timeout=10)
    latency = (time.time() - start) * 1000
    
    if response.status_code == 200:
        result = response.json()
        logger.info(f"  ✓ Got {result['num_results']} recommendations (latency: {latency:.2f}ms)")
        logger.info(f"\n  Recommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            logger.info(f"    {i}. {rec['title']}")
            logger.info(f"       Authors: {rec['authors']}")
            logger.info(f"       Category: {rec['categories']}")
            logger.info(f"       Similarity: {rec['similarity_score']:.3f}")
    else:
        logger.error(f"  ✗ Error: {response.status_code}")
        logger.error(f"    {response.text}")

logger.info("\n[5/5] Testing recommendations by title...")

test_cases = [
    {
        "name": "Search by partial title",
        "data": {
            "title": "Machine Learning",
            "num_recommendations": 5,
            "request_id": "test_by_title_1"
        }
    },
    {
        "name": "Search for cooking book",
        "data": {
            "title": "Cooking",
            "num_recommendations": 3,
            "request_id": "test_by_title_2"
        }
    },
    {
        "name": "Search for fiction",
        "data": {
            "title": "Fiction",
            "num_recommendations": 5,
            "request_id": "test_by_title_3"
        }
    }
]

results = []

for test_case in test_cases:
    logger.info(f"\n  Testing: {test_case['name']}")
    
    start = time.time()
    response = requests.post(
        f"{API_URL}/recommend",
        json=test_case['data'],
        timeout=10
    )
    latency = (time.time() - start) * 1000
    
    if response.status_code == 200:
        result = response.json()
        logger.info(f"    ✓ Query: '{result['query_book']['title']}'")
        logger.info(f"      Category: {result['query_book']['categories']}")
        logger.info(f"      Got {result['num_results']} recommendations")
        logger.info(f"      Latency: {latency:.2f}ms")
        
        logger.info(f"      Top 3 recommendations:")
        for i, rec in enumerate(result['recommendations'][:3], 1):
            logger.info(f"        {i}. {rec['title']} (similarity: {rec['similarity_score']:.3f})")
        
        results.append({
            'name': test_case['name'],
            'success': True,
            'num_recommendations': result['num_results'],
            'latency': latency
        })
    else:
        logger.error(f"    ✗ Error: {response.status_code}")
        logger.error(f"      {response.text}")
        results.append({
            'name': test_case['name'],
            'success': False,
            'error': response.status_code
        })

logger.info("\n" + "=" * 60)
logger.info("Summary")
logger.info("=" * 60)

successes = [r for r in results if r['success']]
failures = [r for r in results if not r['success']]

logger.info(f"\nResults:")
logger.info(f"  - Total tests: {len(results)}")
logger.info(f"  - Passed: {len(successes)}")
logger.info(f"  - Failed: {len(failures)}")

if successes:
    latencies = [r['latency'] for r in successes]
    avg_latency = sum(latencies) / len(latencies)
    
    logger.info(f"\nPerformance:")
    logger.info(f"  - Average latency: {avg_latency:.2f}ms")
    logger.info(f"  - Min latency: {min(latencies):.2f}ms")
    logger.info(f"  - Max latency: {max(latencies):.2f}ms")

if failures:
    logger.warning(f"\nFailed tests:")
    for f in failures:
        logger.warning(f"  - {f['name']}: {f.get('error', 'Unknown error')}")

logger.info("\n" + "=" * 60)

if len(successes) == len(results):
    logger.info("✓ All tests passed!")
else:
    logger.warning(f"⚠ {len(failures)} test(s) failed")

logger.info("=" * 60)
