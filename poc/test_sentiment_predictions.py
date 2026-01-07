#!/usr/bin/env python3
"""
Test sentiment analysis API with various text samples.
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

API_URL = "http://localhost:8001"

logger.info("=" * 60)
logger.info("Testing Sentiment Analysis API")
logger.info("=" * 60)

logger.info("\n[1/4] Checking server health...")
try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    if response.status_code == 200:
        logger.info("✓ Server is healthy")
    else:
        logger.error(f"✗ Server returned status {response.status_code}")
        sys.exit(1)
except Exception as e:
    logger.error(f"✗ Cannot connect to server: {e}")
    logger.error("  Make sure the server is running: python poc/serve_sentiment_model.py")
    sys.exit(1)

logger.info("\n[2/4] Getting model info...")
response = requests.get(f"{API_URL}/model/info")
logger.info("✓ Model information:")
logger.info(f"  {json.dumps(response.json(), indent=2)}")

logger.info("\n[3/4] Testing predictions...")

test_cases = [
    {
        "name": "Positive - Product Review",
        "data": {
            "request_id": "test_1",
            "text": "This product is absolutely amazing! Best purchase I've ever made. Highly recommend to everyone!"
        }
    },
    {
        "name": "Negative - Customer Complaint",
        "data": {
            "request_id": "test_2",
            "text": "Terrible experience. The service was awful and the product broke after one day. Complete waste of money."
        }
    },
    {
        "name": "Neutral - Factual Statement",
        "data": {
            "request_id": "test_3",
            "text": "The package arrived on Tuesday. It contains three items as described in the listing."
        }
    },
    {
        "name": "Positive - Social Media",
        "data": {
            "request_id": "test_4",
            "text": "Just had the best day ever! So grateful for my friends and family. Life is good! 😊"
        }
    },
    {
        "name": "Negative - Political Comment",
        "data": {
            "request_id": "test_5",
            "text": "This policy is a disaster. It will hurt millions of people and solve nothing. Absolutely unacceptable."
        }
    },
    {
        "name": "Mixed - Restaurant Review",
        "data": {
            "request_id": "test_6",
            "text": "The food was decent but the service was slow. Prices are reasonable though."
        }
    },
    {
        "name": "Positive - Tech Review",
        "data": {
            "request_id": "test_7",
            "text": "Love the new features! The interface is intuitive and performance is excellent. Great job by the dev team!"
        }
    },
    {
        "name": "Negative - Movie Review",
        "data": {
            "request_id": "test_8",
            "text": "Boring plot, bad acting, terrible ending. Two hours of my life I'll never get back."
        }
    }
]

results = []

for test_case in test_cases:
    logger.info(f"\n  Testing: {test_case['name']}")
    
    start = time.time()
    response = requests.post(
        f"{API_URL}/predict",
        json=test_case['data'],
        timeout=5
    )
    latency = (time.time() - start) * 1000
    
    if response.status_code == 200:
        result = response.json()
        sentiment = result['sentiment']
        confidence = result['confidence']
        
        # Color coding for terminal output
        sentiment_emoji = {
            'positive': '😊',
            'neutral': '😐',
            'negative': '😞'
        }
        
        logger.info(f"    ✓ Sentiment: {sentiment.upper()} {sentiment_emoji.get(sentiment, '')}")
        logger.info(f"      Confidence: {confidence:.1%}")
        logger.info(f"      Probabilities: Pos={result['probabilities']['positive']:.2f}, "
                    f"Neu={result['probabilities']['neutral']:.2f}, "
                    f"Neg={result['probabilities']['negative']:.2f}")
        logger.info(f"      Latency: {latency:.2f}ms")
        
        results.append({
            'name': test_case['name'],
            'success': True,
            'sentiment': sentiment,
            'confidence': confidence,
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

logger.info("\n[4/4] Summary")
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
    
    # Sentiment distribution
    sentiment_counts = {}
    for r in successes:
        sentiment = r['sentiment']
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    
    logger.info(f"\nSentiment Distribution:")
    for sentiment, count in sorted(sentiment_counts.items()):
        logger.info(f"  - {sentiment.capitalize()}: {count}")

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
