# Setup Guide: User-ID Based Feature Store

## Quick Start

Follow these steps to migrate from transaction-based to user-based feature store:

### 1. Add user_id to Training Data

```bash
python scripts/add_user_id_to_data.py
```

**What it does:**
- Adds `user_id` column to `data/credit_card_fraud_10k.parquet`
- Simulates 500 users making multiple transactions
- Uses power-law distribution (some users more active than others)

### 2. Generate Development Feature Data

```bash
python scripts/generate_dev_features.py
```

**What it does:**
- Creates `data/features_dev.parquet` with 100 synthetic users
- Creates `data/features_test_users.parquet` with 5 known test users
- Creates `data/features_combined.parquet` (all combined)

**Test users created:**
- `test_user_normal` - Low risk, good history
- `test_user_suspicious` - High risk patterns  
- `test_user_high_value` - Legitimate high-value user
- `test_user_velocity` - High transaction frequency
- `test_user_new` - New user with no history

### 3. Apply Feast Configuration

```bash
# Make sure Cassandra is running
docker-compose up -d cassandra

# Apply feature definitions
python scripts/feast_setup.py
```

**What it does:**
- Creates `feast_online_store` keyspace in Cassandra
- Applies feature definitions with `user` entity
- Verifies feature store setup

### 4. Materialize Features

```bash
python scripts/feast_materialize.py
```

**What it does:**
- Syncs features from Parquet to Cassandra online store
- Makes features available for real-time lookup by `user_id`

### 5. Test the API

**Start the Feast-enabled server:**
```bash
python poc/serve_model_feast.py
```

**Test with all features provided:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_new_123",
    "user_id": "test_user_normal",
    "amount": 1500.0,
    "transaction_hour": 23,
    "merchant_category": "Electronics",
    "foreign_transaction": 1,
    "location_mismatch": 1,
    "device_trust_score": 30,
    "velocity_last_24h": 8,
    "cardholder_age": 28
  }'
```

**Test with Feast feature lookup:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_new_456",
    "user_id": "test_user_normal"
  }'
```

This will fetch all features from Feast for the user!

**Run automated tests:**
```bash
python poc/test_predictions.py
```

## What Changed

### Feature Store Architecture

**Before:**
```
Entity: transaction (key: transaction_id)
Problem: Each transaction_id is unique, can't lookup new transactions
```

**After:**
```
Entity: user (key: user_id)
Solution: Multiple transactions share same user_id, can lookup user features
```

### Files Modified

1. **`feature_repo/features.py`**
   - Changed entity from `transaction` to `user`
   - Changed join key from `transaction_id` to `user_id`
   - Renamed feature view from `transaction_features` to `user_features`

2. **`poc/serve_model_feast.py`**
   - Added `user_id` as required field in `PredictionRequest`
   - Changed feature lookup to use `user_id` instead of `transaction_id`
   - Updated feature view name from `transaction_features` to `user_features`

3. **`poc/test_predictions.py`**
   - Added `user_id` to all test cases
   - Updated test case names to reflect user-based lookup

### Files Created

1. **`scripts/add_user_id_to_data.py`**
   - Adds `user_id` column to existing training data
   - Simulates realistic user distribution

2. **`scripts/generate_dev_features.py`**
   - Generates synthetic user-level features for development
   - Creates known test users for testing

3. **`docs/USER_ID_MIGRATION.md`**
   - Detailed migration guide
   - Architecture explanation
   - Troubleshooting tips

4. **`SETUP_USER_ID_FEATURES.md`** (this file)
   - Quick start guide

## Understanding User Features

### Feature Types

**Profile Features** (relatively static):
- `cardholder_age` - User's age
- `device_trust_score` - Trust score for user's device

**Behavior Features** (updated frequently):
- `velocity_last_24h` - Number of transactions in last 24 hours
- `foreign_transaction` - Recent foreign transaction indicator
- `location_mismatch` - Recent location mismatch indicator

**Context Features** (latest transaction):
- `amount` - Last transaction amount
- `transaction_hour` - Last transaction hour
- `merchant_category` - Last merchant category

### How It Works in Production

1. **User makes transaction** → Feature pipeline computes/updates user features
2. **Features pushed to Feast** → Available in online store within milliseconds
3. **New transaction arrives** → Fetch user features by `user_id`
4. **Model predicts** → Using user features + transaction-specific data

## Verification

### Check Data Has user_id

```bash
python -c "import pandas as pd; df = pd.read_parquet('data/credit_card_fraud_10k.parquet'); print(df[['transaction_id', 'user_id', 'amount']].head())"
```

### Check Feast Entity

```bash
feast -c feature_repo entities list
```

Should show: `user`

### Check Feast Features

```bash
feast -c feature_repo feature-views list
```

Should show: `user_features`

### Check Cassandra

```bash
docker exec -it $(docker ps -q -f name=cassandra) cqlsh -e "USE feast_online_store; DESCRIBE TABLES;"
```

Should show Feast tables.

### Test Feature Retrieval

```python
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

# Fetch features for a test user
features = store.get_online_features(
    features=[
        "user_features:cardholder_age",
        "user_features:device_trust_score",
        "user_features:velocity_last_24h",
    ],
    entity_rows=[{"user_id": "test_user_normal"}]
).to_dict()

print(features)
```

## Troubleshooting

### "User not found" errors

**Problem:** Features return `None` for a user_id

**Solutions:**
1. Check if features are materialized: `python scripts/feast_materialize.py`
2. Verify user exists in data: Check parquet file
3. For new users, provide all features in request (don't rely on Feast)

### Cassandra connection errors

**Problem:** Cannot connect to Cassandra

**Solutions:**
```bash
# Check if running
docker ps | grep cassandra

# Restart if needed
docker-compose restart cassandra

# Wait 30-60 seconds for startup
docker-compose logs -f cassandra
```

### Type mismatch errors

**Problem:** Feature type doesn't match model expectation

**Solution:** Check dtypes in `serve_model_feast.py` match training data:
- `amount`: float64
- `transaction_hour`: int64
- `merchant_category`: int8 (after mapping)
- Others: int64

## Next Steps

1. ✅ Complete migration (Steps 1-4 above)
2. ✅ Test with known user_ids
3. 🔄 Update `poc/load_test.py` to include `user_id`
4. 🔄 Implement feature update pipeline for production
5. 🔄 Add monitoring for feature freshness
6. 🔄 Add caching layer for frequently accessed users

## Additional Resources

- Full migration guide: `docs/USER_ID_MIGRATION.md`
- Feast documentation: `docs/FEAST_SETUP.md`
- Feature definitions: `feature_repo/features.py`
- Serving logic: `poc/serve_model_feast.py`
