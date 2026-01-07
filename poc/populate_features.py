import pandas as pd
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import json
import time
from tqdm import tqdm

print("=" * 60)
print("Populating Cassandra Feature Store (POC)")
print("=" * 60)

print("\n[1/5] Connecting to Cassandra...")
try:
    cluster = Cluster(['localhost'], port=9042)
    session = cluster.connect()
    print("✓ Connected to Cassandra")
except Exception as e:
    print(f"✗ Failed to connect to Cassandra: {e}")
    print("  Make sure Cassandra is running: docker-compose up -d cassandra")
    print("  Note: Cassandra takes 30-60 seconds to start up")
    exit(1)

print("\n[2/5] Creating keyspace and table...")
try:
    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS ml_features
        WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}
    """)
    session.set_keyspace('ml_features')
    
    session.execute("""
        CREATE TABLE IF NOT EXISTS transaction_features (
            transaction_id text PRIMARY KEY,
            amount double,
            transaction_hour int,
            merchant_category text,
            foreign_transaction int,
            location_mismatch int,
            device_trust_score double,
            velocity_last_24h int,
            cardholder_age int
        )
    """)
    print("✓ Keyspace and table ready")
except Exception as e:
    print(f"✗ Failed to create schema: {e}")
    exit(1)

print("\n[3/5] Loading transaction data...")
data_file = "data/credit_card_fraud_10k.parquet"
df = pd.read_parquet(data_file)
print(f"✓ Loaded {len(df)} transactions from Parquet")

print("\n[4/5] Computing and storing features...")

prepared = session.prepare("""
    INSERT INTO transaction_features (
        transaction_id, amount, transaction_hour, merchant_category,
        foreign_transaction, location_mismatch, device_trust_score,
        velocity_last_24h, cardholder_age
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
""")

latencies = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Storing features"):
    start = time.time()
    
    session.execute(prepared, (
        str(row['transaction_id']),
        float(row['amount']),
        int(row['transaction_hour']),
        str(row['merchant_category']),
        int(row['foreign_transaction']),
        int(row['location_mismatch']),
        float(row['device_trust_score']),
        int(row['velocity_last_24h']),
        int(row['cardholder_age'])
    ))
    
    latency = (time.time() - start) * 1000
    latencies.append(latency)

print(f"\n✓ Stored {len(df)} feature sets")

print("\n[5/5] Performance metrics...")
print(f"  - Average write latency: {sum(latencies)/len(latencies):.3f}ms")
print(f"  - P95 write latency: {sorted(latencies)[int(len(latencies)*0.95)]:.3f}ms")
print(f"  - P99 write latency: {sorted(latencies)[int(len(latencies)*0.99)]:.3f}ms")

print("\n" + "=" * 60)
print("Testing feature retrieval...")
print("=" * 60)

test_ids = df['transaction_id'].sample(100).tolist()
retrieval_latencies = []

for tid in test_ids:
    start = time.time()
    result = session.execute(
        "SELECT * FROM transaction_features WHERE transaction_id = %s",
        (str(tid),)
    )
    features = result.one()
    latency = (time.time() - start) * 1000
    retrieval_latencies.append(latency)

print(f"\n✓ Feature retrieval performance (100 samples):")
print(f"  - Average: {sum(retrieval_latencies)/len(retrieval_latencies):.3f}ms")
print(f"  - P95: {sorted(retrieval_latencies)[95]:.3f}ms")
print(f"  - P99: {sorted(retrieval_latencies)[99]:.3f}ms")

cluster.shutdown()

print("\n" + "=" * 60)
print("✓ Feature Store Ready!")
print("=" * 60)
print("\nNext steps:")
print("1. Start model server: python poc/serve_model.py")
print("2. Test predictions: python poc/test_predictions.py")
print("=" * 60)
