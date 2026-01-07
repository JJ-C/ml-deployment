#!/usr/bin/env python3

import subprocess
import sys
import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

print("=" * 60)
print("Feast Feature Store Setup")
print("=" * 60)

# Change to project root
project_root = Path(__file__).parent.parent
os.chdir(project_root)

print("\n[1/6] Verifying Feast installation...")
try:
    import feast
    print(f"✓ Feast version: {feast.__version__}")
except ImportError:
    print("✗ Feast not installed")
    print("  Run: pip install -r requirements-poc.txt")
    sys.exit(1)

print("\n[2/6] Checking Cassandra connection...")
try:
    from cassandra.cluster import Cluster
    cluster = Cluster(['localhost'], port=9042)
    session = cluster.connect()
    print("✓ Cassandra is accessible")
    cluster.shutdown()
except Exception as e:
    print(f"✗ Cannot connect to Cassandra: {e}")
    print("  Make sure Cassandra is running: docker-compose up -d cassandra")
    sys.exit(1)

print("\n[3/6] Preparing data with timestamps...")
# Feast requires timestamp fields
parquet_file = "data/credit_card_fraud_10k.parquet"
if not Path(parquet_file).exists():
    print(f"✗ Data file not found: {parquet_file}")
    print("  Run: python scripts/convert_csv_to_parquet.py")
    sys.exit(1)

df = pd.read_parquet(parquet_file)

# Add timestamp fields if they don't exist
if 'timestamp' not in df.columns:
    print("  Adding timestamp field...")
    # Create timestamps spread over the last 30 days
    base_time = datetime.now() - timedelta(days=30)
    df['timestamp'] = [base_time + timedelta(hours=i % 720) for i in range(len(df))]

if 'created_timestamp' not in df.columns:
    print("  Adding created_timestamp field...")
    df['created_timestamp'] = datetime.now()

# Save updated parquet with timestamps
df.to_parquet(parquet_file, index=False)
print(f"✓ Data prepared with {len(df)} records")

print("\n[4/6] Creating Cassandra keyspace for Feast...")
try:
    from cassandra.cluster import Cluster
    cluster = Cluster(['localhost'], port=9042)
    session = cluster.connect()
    
    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS feast_online_store
        WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}
    """)
    print("✓ Cassandra keyspace created")
    cluster.shutdown()
except Exception as e:
    print(f"⚠ Keyspace creation: {e}")

print("\n[5/6] Applying Feast feature definitions...")
try:
    result = subprocess.run(
        ["feast", "-c", "feature_repo", "apply"],
        capture_output=True,
        text=True,
        check=True
    )
    print("✓ Feature definitions applied")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"✗ Failed to apply features: {e}")
    print(e.stderr)
    sys.exit(1)

print("\n[6/6] Verifying feature store...")
try:
    from feast import FeatureStore
    store = FeatureStore(repo_path="feature_repo")
    
    # List entities
    entities = store.list_entities()
    print(f"✓ Entities: {[e.name for e in entities]}")
    
    # List feature views
    feature_views = store.list_feature_views()
    print(f"✓ Feature views: {[fv.name for fv in feature_views]}")
    
except Exception as e:
    print(f"✗ Feature store verification failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ Feast Setup Complete!")
print("=" * 60)
print("\nNext steps:")
print("1. Materialize features: python scripts/feast_materialize.py")
print("2. Test feature retrieval: python scripts/test_feast.py")
print("3. Serve model with Feast: python poc/serve_model_feast.py")
print("=" * 60)
