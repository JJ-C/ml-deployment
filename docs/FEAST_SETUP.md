# Feast Feature Store Integration Guide

This guide explains how to use Feast for feature management in the ML platform POC.

## 🎯 What is Feast?

Feast (Feature Store) is an open-source feature store that:
- **Manages features** with versioning and lineage
- **Ensures consistency** between training and serving
- **Supports multiple stores** (offline: Parquet, online: Cassandra/Redis)
- **Provides point-in-time correctness** for historical features
- **Simplifies feature serving** with a unified API

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Feast Feature Store                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  Offline Store   │         │   Online Store   │         │
│  │   (Parquet)      │────────▶│   (Cassandra)    │         │
│  │                  │ Materialize                │         │
│  │  - Training data │         │  - Low latency   │         │
│  │  - Historical    │         │  - Real-time     │         │
│  │  - Batch jobs    │         │  - Serving       │         │
│  └──────────────────┘         └──────────────────┘         │
│           │                            │                     │
│           ▼                            ▼                     │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │ Model Training   │         │  Model Serving   │         │
│  │ (train_*.py)     │         │ (serve_*.py)     │         │
│  └──────────────────┘         └──────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements-poc.txt
```

This installs `feast==0.38.0` along with other dependencies.

### Step 2: Initialize Feast

```bash
python scripts/feast_setup.py
```

**What it does:**
1. Verifies Feast installation
2. Checks Cassandra connection
3. Adds timestamp fields to Parquet data
4. Applies feature definitions
5. Creates Cassandra keyspace for online store

**Expected output:**
```
✓ Feast version: 0.38.0
✓ Cassandra is accessible
✓ Data prepared with 10,000 records
✓ Feature definitions applied
✓ Entities: ['transaction']
✓ Feature views: ['transaction_features']
✓ Feast Setup Complete!
```

### Step 3: Materialize Features

```bash
python scripts/feast_materialize.py
```

**What it does:**
- Syncs features from offline store (Parquet) to online store (Cassandra)
- Makes features available for real-time serving
- Materializes last 30 days of data

**Expected output:**
```
✓ Materialization complete (2.5s)
✓ Sample feature retrieval successful
✓ Feature Materialization Complete!
```

### Step 4: Test Feature Retrieval

```bash
python scripts/test_feast.py
```

**Expected output:**
```
✓ Loaded 5 sample transaction IDs
✓ Transaction 1: 2.8ms
✓ Transaction 2: 2.5ms
...
Average latency: 2.6ms
✓ Meets <5ms target
```

### Step 5: Serve Model with Feast

```bash
python poc/serve_model_feast.py
```

Access at: http://localhost:8000

---

## 📁 Project Structure

```
ml_deployment/
├── feature_repo/                    # Feast feature repository
│   ├── feature_store.yaml          # Feast configuration
│   └── features.py                 # Feature definitions
├── scripts/
│   ├── feast_setup.py              # Initialize Feast
│   ├── feast_materialize.py        # Sync features to online store
│   ├── test_feast.py               # Test feature retrieval
│   └── compare_feast_cassandra.py  # Compare Feast vs direct Cassandra
├── poc/
│   ├── serve_model.py              # Direct Cassandra (original)
│   └── serve_model_feast.py        # With Feast integration
└── data/
    └── registry.db                 # Feast feature registry (auto-created)
```

---

## 🔧 Feature Definitions

### Feature Store Configuration

**File:** `feature_repo/feature_store.yaml`

```yaml
project: ml_platform_poc
registry: data/registry.db
provider: local
online_store:
  type: cassandra
  hosts:
    - localhost
  port: 9042
  keyspace: feast_online_store
offline_store:
  type: file
```

### Feature Definitions

**File:** `feature_repo/features.py`

```python
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

# Entity
transaction = Entity(
    name="transaction",
    join_keys=["transaction_id"],
)

# Data source
transaction_source = FileSource(
    path="data/credit_card_fraud_10k.parquet",
    timestamp_field="timestamp",
)

# Feature view
transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    schema=[
        Field(name="amount", dtype=Float64),
        Field(name="transaction_hour", dtype=Int64),
        Field(name="merchant_category", dtype=String),
        # ... more features
    ],
    online=True,
    source=transaction_source,
)
```

---

## 🎮 Usage Examples

### Retrieve Features in Python

```python
from feast import FeatureStore

# Initialize feature store
store = FeatureStore(repo_path="feature_repo")

# Get online features
entity_rows = [{"transaction_id": "12345"}]

features = store.get_online_features(
    features=[
        "transaction_features:amount",
        "transaction_features:transaction_hour",
        "transaction_features:merchant_category",
    ],
    entity_rows=entity_rows,
).to_dict()

print(features)
# {'amount': [500.0], 'transaction_hour': [14], ...}
```

### Batch Feature Retrieval

```python
# Get features for multiple transactions
entity_rows = [
    {"transaction_id": "12345"},
    {"transaction_id": "67890"},
    {"transaction_id": "11111"},
]

features = store.get_online_features(
    features=["transaction_features:amount"],
    entity_rows=entity_rows,
).to_df()

print(features)
#   transaction_id  amount
# 0          12345   500.0
# 1          67890   150.0
# 2          11111  1200.0
```

### Historical Features (for Training)

```python
from datetime import datetime

# Get point-in-time correct features
entity_df = pd.DataFrame({
    "transaction_id": ["12345", "67890"],
    "event_timestamp": [
        datetime(2024, 12, 1, 10, 0),
        datetime(2024, 12, 2, 14, 30),
    ]
})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["transaction_features:amount"],
).to_df()
```

---

## 📊 Comparison: Feast vs Direct Cassandra

### Run Comparison

```bash
python scripts/compare_feast_cassandra.py
```

### Expected Results

| Method | Avg Latency | P95 | P99 | Features |
|--------|-------------|-----|-----|----------|
| **Feast** | 2.8ms | 3.5ms | 4.2ms | ✅ Versioning, lineage, consistency |
| **Direct Cassandra** | 2.5ms | 3.0ms | 3.8ms | ⚠️ Manual management |

### Trade-offs

**Feast Advantages:**
- ✅ Feature versioning and lineage
- ✅ Point-in-time correctness
- ✅ Offline/online consistency
- ✅ Feature discovery and documentation
- ✅ Multi-store support
- ✅ Production-ready patterns

**Direct Cassandra Advantages:**
- ✅ Slightly lower latency (~0.3ms faster)
- ✅ Simpler setup
- ✅ Full control over schema

**Recommendation:** Use Feast for production (benefits outweigh small latency difference)

---

## 🔄 Workflow

### Development Workflow

1. **Define features** in `feature_repo/features.py`
2. **Apply changes:** `feast -c feature_repo apply`
3. **Materialize:** `python scripts/feast_materialize.py`
4. **Test:** `python scripts/test_feast.py`
5. **Serve:** `python poc/serve_model_feast.py`

### Adding New Features

```python
# Edit feature_repo/features.py
transaction_features = FeatureView(
    name="transaction_features",
    schema=[
        # Existing features...
        Field(name="new_feature", dtype=Float64),  # Add new feature
    ],
)
```

```bash
# Apply changes
feast -c feature_repo apply

# Re-materialize
python scripts/feast_materialize.py
```

### Updating Features

```bash
# Materialize incremental updates
python scripts/feast_materialize.py
```

---

## 🎯 Production Considerations

### Scaling Online Store

**Current (POC):** Single Cassandra node

**Production:** Cassandra cluster
```yaml
online_store:
  type: cassandra
  hosts:
    - cassandra-1.prod.internal
    - cassandra-2.prod.internal
    - cassandra-3.prod.internal
  port: 9042
  keyspace: feast_online_store
  replication_factor: 3
  consistency_level: LOCAL_QUORUM
```

### Scaling Offline Store

**Current (POC):** Local Parquet files

**Production:** Cloud data lake
```yaml
offline_store:
  type: bigquery  # or snowflake, redshift
  project_id: ml-platform-prod
  dataset: features
```

### Materialization at Scale

**Current (POC):** Manual script

**Production:** Scheduled jobs
```bash
# Airflow DAG
@dag(schedule_interval="@hourly")
def materialize_features():
    materialize_task = PythonOperator(
        task_id="materialize",
        python_callable=lambda: store.materialize_incremental(
            end_date=datetime.now()
        )
    )
```

### Feature Server

**Current (POC):** Embedded in FastAPI

**Production:** Dedicated Feast server
```bash
# Start Feast server
feast -c feature_repo serve

# Access via HTTP
curl -X POST http://feast-server:6566/get-online-features \
  -d '{"features": ["transaction_features:amount"], ...}'
```

---

## 🐛 Troubleshooting

### Issue: "Feast not found"

```bash
pip install feast==0.38.0
```

### Issue: "Cannot connect to Cassandra"

```bash
# Check Cassandra is running
docker-compose ps cassandra

# Wait for startup (30-60 seconds)
docker-compose logs -f cassandra
```

### Issue: "No features found"

```bash
# Re-run setup
python scripts/feast_setup.py

# Verify feature definitions
feast -c feature_repo feature-views list
```

### Issue: "Materialization failed"

```bash
# Check data has timestamps
python -c "import pandas as pd; df = pd.read_parquet('data/credit_card_fraud_10k.parquet'); print(df.columns)"

# Should include 'timestamp' and 'created_timestamp'
```

### Issue: "High latency"

**Causes:**
1. First request (cold start)
2. Network latency to Cassandra
3. Large feature sets

**Solutions:**
```python
# Use prepared statements (already implemented)
# Batch requests when possible
# Monitor Cassandra performance
```

---

## 📚 Feast CLI Commands

```bash
# List all feature views
feast -c feature_repo feature-views list

# Describe a feature view
feast -c feature_repo feature-views describe transaction_features

# List entities
feast -c feature_repo entities list

# Materialize features
feast -c feature_repo materialize 2024-11-01T00:00:00 2024-12-01T00:00:00

# Start feature server
feast -c feature_repo serve

# Validate feature definitions
feast -c feature_repo validate
```

---

## 🔗 Integration Points

### With MLflow

```python
# Log feature definitions with model
with mlflow.start_run():
    mlflow.log_param("feature_store", "feast")
    mlflow.log_param("feature_view", "transaction_features")
    mlflow.log_artifact("feature_repo/features.py")
```

### With Seldon Core

Update Seldon deployment to use Feast:
```yaml
env:
- name: FEAST_REPO_PATH
  value: "/mnt/feature_repo"
```

### With Airflow

```python
from airflow.operators.python import PythonOperator

def materialize_features():
    from feast import FeatureStore
    store = FeatureStore(repo_path="feature_repo")
    store.materialize_incremental(end_date=datetime.now())

materialize_task = PythonOperator(
    task_id="materialize_features",
    python_callable=materialize_features,
)
```

---

## 📈 Monitoring

### Key Metrics

1. **Feature Retrieval Latency**
   - Target: <5ms P99
   - Monitor: Prometheus metrics from serve_model_feast.py

2. **Materialization Duration**
   - Track time to sync offline → online
   - Alert if exceeds SLA

3. **Feature Freshness**
   - Time since last materialization
   - Alert if stale

4. **Online Store Size**
   - Monitor Cassandra disk usage
   - Plan capacity

### Prometheus Metrics

```python
# Already instrumented in serve_model_feast.py
feature_fetch_latency_seconds
prediction_latency_seconds
predictions_total
```

---

## 🎓 Learning Resources

- **Feast Docs:** https://docs.feast.dev/
- **Feast GitHub:** https://github.com/feast-dev/feast
- **Feast Slack:** https://slack.feast.dev/
- **Tutorials:** https://docs.feast.dev/tutorials/tutorials-overview

---

## 📋 Checklist

### POC Setup
- [x] Install Feast
- [x] Create feature repository
- [x] Define features
- [x] Configure Cassandra online store
- [x] Materialize features
- [x] Test feature retrieval
- [x] Integrate with model serving

### Production Readiness
- [ ] Set up Cassandra cluster (3+ nodes)
- [ ] Configure cloud offline store (BigQuery/Snowflake)
- [ ] Set up scheduled materialization (Airflow)
- [ ] Deploy Feast feature server
- [ ] Add monitoring and alerting
- [ ] Document feature definitions
- [ ] Set up feature discovery UI
- [ ] Implement feature validation

---

## 🆚 When to Use What

**Use Feast when:**
- ✅ Need feature versioning
- ✅ Multiple models share features
- ✅ Require offline/online consistency
- ✅ Want feature lineage tracking
- ✅ Production deployment

**Use Direct Cassandra when:**
- ✅ Simple POC/prototype
- ✅ Single model, single use case
- ✅ Need absolute minimum latency
- ✅ Full control over schema

---

## 🚀 Summary

**What You've Built:**
- ✅ Feast feature store with Cassandra online store
- ✅ Feature definitions for fraud detection
- ✅ Materialization pipeline
- ✅ Model serving with Feast integration
- ✅ Comparison with direct Cassandra approach

**Production Benefits:**
- Feature versioning and lineage
- Point-in-time correctness
- Offline/online consistency
- Multi-store support
- Industry-standard patterns

**Next Steps:**
1. Test both approaches (Feast vs direct Cassandra)
2. Measure performance differences
3. Plan production migration
4. Document feature definitions
5. Set up monitoring

---

**Ready for production-grade feature management!** 🎉
