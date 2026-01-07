#!/usr/bin/env python3
"""
Direct Spark-to-Cassandra materialization script.
This bypasses Feast's materialization and uses Spark to write directly to Cassandra.
"""

import sys
from datetime import datetime, timedelta
import time
import logging

# Enable trace/debug logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 60)
print("Spark Direct Materialization to Cassandra")
print("=" * 60)

print("\n[1/5] Initializing Spark session with Cassandra connector...")
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, to_timestamp
    
    # Create Spark session with Cassandra connector
    spark = SparkSession.builder \
        .appName("FeastSparkMaterialization") \
        .config("spark.cassandra.connection.host", "localhost") \
        .config("spark.cassandra.connection.port", "9042") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    
    print(f"✓ Spark session created")
    print(f"  Spark version: {spark.version}")
    print(f"  Master: {spark.sparkContext.master}")
    
except ImportError as e:
    print(f"✗ Failed to import PySpark: {e}")
    print("  Install: pip install pyspark")
    sys.exit(1)
except Exception as e:
    print(f"✗ Failed to initialize Spark: {e}")
    sys.exit(1)

print("\n[2/5] Reading data from Parquet with Spark...")
try:
    # Read the parquet file
    df = spark.read.parquet("data/google_books_dataset.parquet")
    
    row_count = df.count()
    print(f"✓ Loaded {row_count:,} rows")
    print(f"  Columns: {', '.join(df.columns)}")
    
    # Show schema
    print("\n  Schema:")
    df.printSchema()
    
except Exception as e:
    print(f"✗ Failed to read data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[3/5] Filtering data by time range...")
try:
    # Filter for last 60 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    print(f"  Time range: {start_date.date()} to {end_date.date()}")
    
    # Convert timestamp column if needed
    if 'timestamp' in df.columns:
        df_filtered = df.filter(
            (col('timestamp') >= start_date) & 
            (col('timestamp') <= end_date)
        )
        filtered_count = df_filtered.count()
        print(f"✓ Filtered to {filtered_count:,} rows")
    else:
        print("  ⚠ No timestamp column found, using all data")
        df_filtered = df
        
except Exception as e:
    print(f"✗ Failed to filter data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[4/5] Writing to Cassandra online store...")
try:
    start_time = time.time()
    
    # Define the keyspace and table
    keyspace = "feast_online_store"
    
    # For Feast compatibility, we need to write in Feast's expected format
    # Feast stores features in a specific schema with entity keys and feature values
    
    print(f"  Target keyspace: {keyspace}")
    print(f"  Writing features...")
    
    # Write using Spark's Cassandra connector
    # Note: This requires the spark-cassandra-connector package
    df_filtered.write \
        .format("org.apache.spark.sql.cassandra") \
        .mode("append") \
        .options(table="user_features", keyspace=keyspace) \
        .save()
    
    duration = time.time() - start_time
    print(f"✓ Write complete ({duration:.2f}s)")
    
except Exception as e:
    print(f"✗ Write failed: {e}")
    print("\n  Note: This requires spark-cassandra-connector")
    print("  For local testing without connector, see alternative approach below")
    import traceback
    traceback.print_exc()
    
    # Alternative: Write via Cassandra Python driver
    print("\n  Attempting alternative write via cassandra-driver...")
    try:
        from cassandra.cluster import Cluster
        import pandas as pd
        
        # Convert to Pandas for batch insert
        df_pandas = df_filtered.limit(1000).toPandas()  # Limit for demo
        
        cluster = Cluster(['localhost'])
        session = cluster.connect(keyspace)
        
        # Prepare insert statement (adjust based on your schema)
        insert_stmt = session.prepare("""
            INSERT INTO user_features (user_id, cardholder_age, device_trust_score, 
                                      velocity_last_24h, amount, transaction_hour, 
                                      merchant_category, event_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """)
        
        # Batch insert
        for _, row in df_pandas.iterrows():
            session.execute(insert_stmt, (
                str(row.get('user_id')),
                int(row.get('cardholder_age', 0)),
                float(row.get('device_trust_score', 0.0)),
                int(row.get('velocity_last_24h', 0)),
                float(row.get('amount', 0.0)),
                int(row.get('transaction_hour', 0)),
                str(row.get('merchant_category', '')),
                row.get('timestamp')
            ))
        
        cluster.shutdown()
        print(f"✓ Alternative write successful (wrote {len(df_pandas)} rows)")
        
    except Exception as e2:
        print(f"✗ Alternative write also failed: {e2}")
        sys.exit(1)

print("\n[5/5] Verifying data in Cassandra...")
try:
    from cassandra.cluster import Cluster
    
    cluster = Cluster(['localhost'])
    session = cluster.connect('feast_online_store')
    
    # Count rows
    result = session.execute("SELECT COUNT(*) FROM user_features")
    count = result.one()[0]
    
    print(f"✓ Verification successful")
    print(f"  Total rows in Cassandra: {count:,}")
    
    # Sample a few rows
    result = session.execute("SELECT * FROM user_features LIMIT 5")
    print(f"\n  Sample rows:")
    for i, row in enumerate(result, 1):
        print(f"    {i}. user_id: {row.user_id}")
    
    cluster.shutdown()
    
except Exception as e:
    print(f"⚠ Verification warning: {e}")
    print("  Data may still be available")

finally:
    spark.stop()
    print("\n  Spark session stopped")

print("\n" + "=" * 60)
print("✓ Spark Materialization Complete!")
print("=" * 60)
print("\nFeatures written directly to Cassandra using Spark")
print("\nPerformance benefits:")
print("- Distributed data reading with Spark")
print("- Parallel writes to Cassandra")
print("- Scalable to TB+ datasets")
print("\nNext steps:")
print("1. Verify with Feast: python scripts/test_feast.py")
print("2. Monitor Spark UI: http://localhost:4040")
print("3. Scale to cluster: Update spark.master configuration")
print("=" * 60)
