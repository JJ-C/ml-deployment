# Migration to User-ID Based Feature Store

This guide explains the migration from transaction-based to user-based feature store architecture.

## Overview

**Before:** Features were keyed by `transaction_id` (unique per transaction)
**After:** Features are keyed by `user_id` (reusable across transactions)

This change enables:
- Fetching user-level features for new transactions
- Tracking user behavior patterns over time
- More realistic production architecture

## Architecture Changes

### Entity Key
```
OLD: transaction_id (unique, not reusable)
NEW: user_id (shared across multiple transactions)
```

### Feature Store Structure

**User Features** (keyed by `user_id`):
- **Profile features**: `cardholder_age`, `device_trust_score`
- **Behavior features**: `velocity_last_24h`, `foreign_transaction`, `location_mismatch`
- **Latest context**: `amount`, `transaction_hour`, `merchant_category`

### Request Flow

**Before:**
```json
{
  "transaction_id": "txn_123"
}
```
→ Lookup fails (new transaction not in feature store)

**After:**
```json
{
  "transaction_id": "txn_123",
  "user_id": "user_0042"
}
```
→ Fetch user features by `user_id` ✓

## Migration Steps

### Step 1: Add user_id to Training Data

```bash
python scripts/add_user_id_to_data.py
```

This adds a `user_id` column to your training data, simulating 500 users making multiple transactions.

**Output:**
- Updates `data/credit_card_fraud_10k.parquet` with `user_id` column
- Users distributed with power-law (some users more active)
- Avg ~20 transactions per user

### Step 2: Update Feast Feature Definitions

Already updated in `feature_repo/features.py`:

```python
# OLD
transaction = Entity(name="transaction", join_keys=["transaction_id"])
transaction_features = FeatureView(name="transaction_features", entities=[transaction], ...)

# NEW
user = Entity(name="user", join_keys=["user_id"])
user_features = FeatureView(name="user_features", entities=[user], ...)
```

### Step 3: Apply Feast Changes

```bash
# Apply new feature definitions
python scripts/feast_setup.py

# Materialize features to online store
python scripts/feast_materialize.py
```

### Step 4: Generate Dev Feature Data (Optional)

For development/testing with synthetic users:

```bash
python scripts/generate_dev_features.py
```

This creates:
- `data/features_dev.parquet` - 100 synthetic users
- `data/features_test_users.parquet` - 5 known test users
- `data/features_combined.parquet` - Combined dataset

Known test users:
- `test_user_normal` - Low risk, good history
- `test_user_suspicious` - High risk patterns
- `test_user_high_value` - Legitimate high-value user
- `test_user_velocity` - High transaction frequency
- `test_user_new` - New user with no history

### Step 5: Update Model Training (if needed)

Your training script doesn't need changes since it uses the same features. The `user_id` column is just metadata for the feature store.

### Step 6: Test the Updated API

**Test with all features provided:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_new_123",
    "user_id": "user_0001",
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

This will fetch all features from Feast for `test_user_normal`.

## Updated Request Model

```python
class PredictionRequest(BaseModel):
    transaction_id: str              # Unique transaction ID (for tracking)
    user_id: str                     # Entity key for feature lookup
    amount: Optional[float] = None   # Optional: fetch from Feast if missing
    transaction_hour: Optional[int] = None
    merchant_category: Optional[str] = None
    foreign_transaction: Optional[int] = None
    location_mismatch: Optional[int] = None
    device_trust_score: Optional[float] = None
    velocity_last_24h: Optional[int] = None
    cardholder_age: Optional[int] = None
```

## Production Considerations

### Feature Updates

In production, you'd have a feature pipeline that:

1. **On transaction event:**
   ```python
   # Compute/update user features
   user_features = {
       'user_id': transaction.user_id,
       'velocity_last_24h': count_recent_transactions(user_id),
       'device_trust_score': get_device_score(device_id),
       'amount': transaction.amount,
       'transaction_hour': transaction.timestamp.hour,
       # ... other features
   }
   
   # Push to Feast online store
   feast_store.push(
       push_source_name="user_push_source",
       df=pd.DataFrame([user_features])
   )
   ```

2. **On prediction request:**
   ```python
   # Fetch latest user features
   features = feast_store.get_online_features(
       features=["user_features:*"],
       entity_rows=[{"user_id": request.user_id}]
   )
   ```

### Feature Freshness

- **Profile features** (age, device trust): Updated infrequently
- **Behavior features** (velocity, patterns): Updated after each transaction
- **Context features** (amount, hour): Transaction-specific, not stored

### Monitoring

Key metrics to track:
- Feature fetch latency (should be <5ms)
- Feature cache hit rate
- Feature staleness (time since last update)
- Missing feature rate (users not in store)

## Troubleshooting

### User not found in feature store

**Error:** Features return `None` for a user_id

**Solutions:**
1. Check if user exists: `feast_store.get_online_features(...)`
2. Materialize features: `python scripts/feast_materialize.py`
3. For new users, provide all features in request

### Feature type mismatch

**Error:** Type conversion errors during prediction

**Solution:** Ensure feature dtypes match training data:
- `amount`: float64
- `transaction_hour`: int64
- `merchant_category`: int8 (after mapping)
- Other features: int64

### Cassandra connection issues

**Error:** Cannot connect to Cassandra

**Solutions:**
1. Check Cassandra is running: `docker ps | grep cassandra`
2. Verify keyspace exists: `DESCRIBE KEYSPACES;` in cqlsh
3. Restart Cassandra: `docker-compose restart cassandra`

## Files Changed

- ✅ `feature_repo/features.py` - Entity changed to `user`
- ✅ `poc/serve_model_feast.py` - Fetch by `user_id`
- ✅ `scripts/add_user_id_to_data.py` - Add user_id to data
- ✅ `scripts/generate_dev_features.py` - Generate user-based features
- ⏳ `poc/test_predictions.py` - Update test cases with user_id
- ⏳ `poc/load_test.py` - Update load test with user_id

## Next Steps

1. Run migration scripts (Steps 1-3 above)
2. Test with known user_ids
3. Update test scripts to include user_id
4. Deploy to production with feature pipeline
5. Monitor feature freshness and latency
