#!/usr/bin/env python3

import sys
from pathlib import Path
from feast import FeatureStore
import pandas as pd
import time
from tabulate import tabulate

print("=" * 60)
print("Testing Feast Feature Retrieval")
print("=" * 60)

print("\n[1/4] Loading feature store...")
try:
    store = FeatureStore(repo_path="feature_repo")
    print("✓ Feature store loaded")
except Exception as e:
    print(f"✗ Failed to load feature store: {e}")
    print("  Run: python scripts/feast_setup.py")
    sys.exit(1)

print("\n[2/4] Loading sample transaction IDs...")
try:
    df = pd.read_parquet("data/credit_card_fraud_10k.parquet")
    sample_ids = df['transaction_id'].head(5).astype(str).tolist()
    print(f"✓ Loaded {len(sample_ids)} sample transaction IDs")
except Exception as e:
    print(f"✗ Failed to load data: {e}")
    sys.exit(1)

print("\n[3/4] Retrieving features from online store...")

feature_names = [
    "transaction_features:amount",
    "transaction_features:transaction_hour",
    "transaction_features:merchant_category",
    "transaction_features:foreign_transaction",
    "transaction_features:location_mismatch",
    "transaction_features:device_trust_score",
    "transaction_features:velocity_last_24h",
    "transaction_features:cardholder_age",
]

results = []
latencies = []

for txn_id in sample_ids:
    entity_rows = [{"transaction_id": txn_id}]
    
    start = time.time()
    try:
        features = store.get_online_features(
            features=feature_names,
            entity_rows=entity_rows,
        ).to_dict()
        
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        
        results.append({
            "transaction_id": txn_id,
            "amount": features.get("amount", [None])[0],
            "hour": features.get("transaction_hour", [None])[0],
            "category": features.get("merchant_category", [None])[0],
            "latency_ms": f"{latency:.2f}"
        })
        
        print(f"  ✓ Transaction {txn_id}: {latency:.2f}ms")
        
    except Exception as e:
        print(f"  ✗ Transaction {txn_id}: {e}")
        results.append({
            "transaction_id": txn_id,
            "amount": "ERROR",
            "hour": "ERROR",
            "category": "ERROR",
            "latency_ms": "N/A"
        })

print("\n[4/4] Results Summary")
print("=" * 60)

if results:
    print("\nSample Features Retrieved:")
    table_data = [[r["transaction_id"], r["amount"], r["hour"], r["category"], r["latency_ms"]] 
                  for r in results]
    print(tabulate(
        table_data,
        headers=["Transaction ID", "Amount", "Hour", "Category", "Latency"],
        tablefmt="grid"
    ))

if latencies:
    print(f"\nPerformance Metrics:")
    print(f"  Average latency: {sum(latencies)/len(latencies):.2f}ms")
    print(f"  Min latency: {min(latencies):.2f}ms")
    print(f"  Max latency: {max(latencies):.2f}ms")
    
    if sum(latencies)/len(latencies) < 5:
        print(f"  ✓ Meets <5ms target")
    else:
        print(f"  ⚠ Exceeds 5ms target")

print("\n" + "=" * 60)
print("✓ Feast Testing Complete!")
print("=" * 60)
print("\nNext steps:")
print("1. Serve model with Feast: python poc/serve_model_feast.py")
print("2. Compare with direct Cassandra: python scripts/compare_feast_cassandra.py")
print("=" * 60)
