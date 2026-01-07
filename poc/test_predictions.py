import requests
import json
import time
import logging
import sys
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("Testing Model Predictions (POC)")
logger.info("=" * 60)

API_URL = "http://localhost:8000"

logger.info("\n[1/4] Checking server health...")
try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    if response.status_code == 200:
        logger.info("✓ Server is healthy")
        logger.info(f"  {json.dumps(response.json(), indent=2)}")
    else:
        logger.error(f"✗ Server health check failed: {response.status_code}")
        sys.exit(1)
except Exception as e:
    logger.error(f"✗ Cannot connect to server: {e}")
    logger.error("  Make sure the server is running: python poc/serve_model.py")
    sys.exit(1)

logger.info("\n[2/4] Getting model info...")
response = requests.get(f"{API_URL}/model/info")
logger.info("✓ Model information:")
logger.info(f"  {json.dumps(response.json(), indent=2)}")

logger.info("\n[3/4] Testing predictions...")

test_cases = [
    {
        "name": "Low Risk Transaction (with all features)",
        "data": {
            "transaction_id": "test_1",
            "user_id": "user_1193",
            "amount": 45.0,
            "transaction_hour": 14,
            "merchant_category": "Grocery",
            "foreign_transaction": 0,
            "location_mismatch": 0,
            "device_trust_score": 85.0,
            "velocity_last_24h": 2,
            "cardholder_age": 35
        }
    },
    {
        "name": "High Risk Transaction (with all features)",
        "data": {
            "transaction_id": "test_2",
            "user_id": "user_1518",
            "amount": 1500.0,
            "transaction_hour": 3,
            "merchant_category": "Electronics",
            "foreign_transaction": 1,
            "location_mismatch": 1,
            "device_trust_score": 25.0,
            "velocity_last_24h": 8,
            "cardholder_age": 22
        }
    },
    {
        "name": "Medium Risk Transaction (with all features)",
        "data": {
            "transaction_id": "test_3",
            "user_id": "user_1854",
            "amount": 250.0,
            "transaction_hour": 20,
            "merchant_category": "Travel",
            "foreign_transaction": 0,
            "location_mismatch": 1,
            "device_trust_score": 60.0,
            "velocity_last_24h": 4,
            "cardholder_age": 45
        }
    },
    {
        "name": "Using Feast Features (fetch by user_id)",
        "data": {
            "transaction_id": "test_4",
            "user_id": "user_1193"
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
        results.append([
            test_case['name'],
            result['transaction_id'],
            "FRAUD" if result['is_fraud'] else "LEGIT",
            f"{result['fraud_probability']:.3f}",
            f"{result['latency_ms']:.2f}ms",
            result['feature_source']
        ])
        logger.info(f"    Result: {'FRAUD' if result['is_fraud'] else 'LEGIT'} (prob: {result['fraud_probability']:.3f})")
        logger.info(f"    Latency: {result['latency_ms']:.2f}ms")
    else:
        logger.error(f"    ✗ Error: {response.status_code}")
        results.append([test_case['name'], "N/A", "ERROR", "N/A", "N/A", "N/A"])

logger.info("\n[4/4] Summary of Results")
logger.info("=" * 60)
logger.info("\n" + tabulate(
    results,
    headers=["Test Case", "Transaction ID", "Prediction", "Probability", "Latency", "Source"],
    tablefmt="grid"
))

latencies = [float(r[4].replace('ms', '')) for r in results if r[4] != "N/A"]
if latencies:
    logger.info(f"\nLatency Statistics:")
    logger.info(f"  - Average: {sum(latencies)/len(latencies):.2f}ms")
    logger.info(f"  - Min: {min(latencies):.2f}ms")
    logger.info(f"  - Max: {max(latencies):.2f}ms")
    
    if max(latencies) < 5.0:
        logger.info(f"\n✓ All predictions under 5ms target!")
    else:
        logger.warning(f"\n⚠ Some predictions exceeded 5ms target")

logger.info("\n" + "=" * 60)
logger.info("✓ Testing Complete!")
logger.info("=" * 60)
logger.info("\nNext steps:")
logger.info("1. Run load test: python poc/load_test.py")
logger.info("2. View metrics: http://localhost:9090")
logger.info("=" * 60)
