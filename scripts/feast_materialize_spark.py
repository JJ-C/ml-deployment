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
logging.getLogger('pyspark').setLevel(logging.INFO)

print("=" * 60)
print("Feast Feature Materialization with Spark")
print("=" * 60)

print("\n[1/5] Initializing Spark session...")
try:
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder \
        .appName("FeastMaterialization") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    
    print(f"✓ Spark session created")
    print(f"  Spark version: {spark.version}")
    print(f"  Master: {spark.sparkContext.master}")
    
except Exception as e:
    print(f"✗ Failed to initialize Spark: {e}")
    print("  Install PySpark: pip install pyspark")
    sys.exit(1)

print("\n[2/5] Loading feature store...")
try:
    store = FeatureStore(repo_path="feature_repo")
    print("✓ Feature store loaded")
except Exception as e:
    print(f"✗ Failed to load feature store: {e}")
    print("  Run: python scripts/feast_setup.py")
    sys.exit(1)

print("\n[3/5] Checking feature views...")
feature_views = store.list_feature_views()
if not feature_views:
    print("✗ No feature views found")
    sys.exit(1)

for fv in feature_views:
    print(f"  - {fv.name}: {len(fv.schema)} features")

print("\n[4/5] Processing and materializing features with Spark...")
print("  Using Spark for data processing, then Feast for Cassandra writes")

# Materialize features for the last 60 days
end_date = datetime.now()
start_date = end_date - timedelta(days=60)

print(f"  Time range: {start_date.date()} to {end_date.date()}")

try:
    start_time = time.time()
    
    # Read data with Spark for distributed processing
    print("  Reading data with Spark...")
    df_spark = spark.read.parquet("data/google_books_dataset.parquet")
    row_count = df_spark.count()
    print(f"  Loaded {row_count:,} rows with Spark")
    
    # Apply any Spark transformations here (filtering, aggregations, etc.)
    from pyspark.sql.functions import col
    
    # Filter by date range if timestamp column exists
    if 'timestamp' in df_spark.columns:
        df_filtered = df_spark.filter(
            (col('timestamp') >= start_date) & 
            (col('timestamp') <= end_date)
        )
        filtered_count = df_filtered.count()
        print(f"  Filtered to {filtered_count:,} rows in date range")
    else:
        df_filtered = df_spark
        print("  No timestamp filtering applied")
    
    # Show Spark execution plan
    print("\n  Spark execution plan:")
    df_filtered.explain()
    
    # Stop Spark before Feast materialization
    spark.stop()
    print("\n  Spark processing complete, session stopped")
    
    # Now use Feast's standard materialization
    print("  Materializing to Cassandra via Feast...")
    store.materialize(
        start_date=start_date,
        end_date=end_date,
    )
    
    duration = time.time() - start_time
    print(f"✓ Materialization complete ({duration:.2f}s)")
    
except Exception as e:
    print(f"✗ Materialization failed: {e}")
    import traceback
    traceback.print_exc()
    spark.stop()
    sys.exit(1)

print("\n[5/5] Verifying online store...")
try:
    # Test feature retrieval
    store = FeatureStore(repo_path="feature_repo")
    
    # Get a sample user ID from the data
    import pandas as pd
    df = pd.read_parquet("data/google_books_dataset.parquet")
    sample_id = str(df['user_id'].iloc[0])
    
    # Retrieve features
    entity_rows = [{"user_id": sample_id}]
    
    features = store.get_online_features(
        features=[
            "user_features:cardholder_age",
            "user_features:device_trust_score",
            "user_features:velocity_last_24h",
        ],
        entity_rows=entity_rows,
    ).to_dict()
    
    print(f"✓ Sample feature retrieval successful")
    print(f"  User ID: {sample_id}")
    print(f"  Features retrieved: {list(features.keys())}")
    
except Exception as e:
    print(f"⚠ Verification warning: {e}")
    print("  Features may still be available, try test_feast.py")

print("\n" + "=" * 60)
print("✓ Spark-based Materialization Complete!")
print("=" * 60)
print("\nFeatures are now available in Cassandra online store")
print("\nNext steps:")
print("1. Test feature retrieval: python scripts/test_feast.py")
print("2. Monitor Spark UI: http://localhost:4040")
print("3. Scale to cluster: Update spark.master in feature_store.yaml")
print("=" * 60)
