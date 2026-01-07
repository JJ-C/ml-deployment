#!/usr/bin/env python3

import sys
from pathlib import Path
from datetime import datetime, timedelta
from feast import FeatureStore
import time
import logging

# Enable trace/debug logging for Feast
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('feast').setLevel(logging.DEBUG)

print("=" * 60)
print("Feast Feature Materialization")
print("=" * 60)

print("\n[1/4] Loading feature store...")
try:
    store = FeatureStore(repo_path="feature_repo")
    print("✓ Feature store loaded")
except Exception as e:
    print(f"✗ Failed to load feature store: {e}")  
    print("  Run: python scripts/feast_setup.py")
    sys.exit(1)

print("\n[2/4] Checking feature views...")
feature_views = store.list_feature_views()
if not feature_views:
    print("✗ No feature views found")
    sys.exit(1)

for fv in feature_views:
    print(f"  - {fv.name}: {len(fv.schema)} features")

print("\n[3/4] Materializing features to online store...")
print("  This will sync features from Parquet to Cassandra")

# Materialize features for the last 30 days
end_date = datetime.now()
start_date = end_date - timedelta(days=60)

print(f"  Time range: {start_date.date()} to {end_date.date()}")

try:
    start_time = time.time()
    
    store.materialize(
        start_date=start_date,
        end_date=end_date
    )
    
    duration = time.time() - start_time
    print(f"✓ Materialization complete ({duration:.2f}s)")
    
except Exception as e:
    print(f"✗ Materialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[4/4] Verifying online store...")
try:
    # Test feature retrieval
    from feast import FeatureStore
    store = FeatureStore(repo_path="feature_repo")
    
    # Get a sample transaction ID from the data
    import pandas as pd
    #df = pd.read_parquet("data/credit_card_fraud_10k.parquet")
    df = pd.read_parquet("data/google_books_dataset.parquet")
    sample_id = str(df['transaction_id'].iloc[0])
    
    # Retrieve features
    entity_rows = [{"transaction_id": sample_id}]
    
    features = store.get_online_features(
        features=[
            "transaction_features:amount",
            "transaction_features:transaction_hour",
            "transaction_features:merchant_category",
        ],
        entity_rows=entity_rows,
    ).to_dict()
    
    print(f"✓ Sample feature retrieval successful")
    print(f"  Transaction ID: {sample_id}")
    print(f"  Features retrieved: {list(features.keys())}")
    
except Exception as e:
    print(f"⚠ Verification warning: {e}")
    print("  Features may still be available, try test_feast.py")

print("\n" + "=" * 60)
print("✓ Feature Materialization Complete!")
print("=" * 60)
print("\nFeatures are now available in Cassandra online store")
print("\nNext steps:")
print("1. Test feature retrieval: python scripts/test_feast.py")
print("2. Serve model with Feast: python poc/serve_model_feast.py")
print("3. Compare with direct Cassandra: python scripts/compare_feast_cassandra.py")
print("=" * 60)
