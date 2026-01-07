# Feast Feature Repository

This directory contains the Feast feature store configuration and feature definitions for the ML platform POC.

## 📁 Structure

```
feature_repo/
├── feature_store.yaml    # Feast configuration
├── features.py           # Feature definitions
└── README.md            # This file
```

## 🔧 Configuration

**File:** `feature_store.yaml`

Defines:
- Project name: `ml_platform_poc`
- Registry location: `data/registry.db`
- Online store: Cassandra (localhost:9042)
- Offline store: File (Parquet)

## 📊 Features

**File:** `features.py`

Defines transaction features for fraud detection:
- Entity: `transaction` (keyed by `transaction_id`)
- Feature View: `transaction_features`
  - amount (Float64)
  - transaction_hour (Int64)
  - merchant_category (String)
  - foreign_transaction (Int64)
  - location_mismatch (Int64)
  - device_trust_score (Float64)
  - velocity_last_24h (Int64)
  - cardholder_age (Int64)

## 🚀 Usage

### Apply Feature Definitions

```bash
feast -c feature_repo apply
```

### List Features

```bash
feast -c feature_repo feature-views list
feast -c feature_repo entities list
```

### Materialize Features

```bash
# Using Python script (recommended)
python scripts/feast_materialize.py

# Or using Feast CLI
feast -c feature_repo materialize 2024-11-01T00:00:00 2024-12-22T00:00:00
```

### Start Feature Server

```bash
feast -c feature_repo serve
```

Access at: http://localhost:6566

## 📚 Documentation

See `FEAST_SETUP.md` in the project root for complete setup and usage instructions.

## 🔄 Workflow

1. **Define features** in `features.py`
2. **Apply changes:** `feast -c feature_repo apply`
3. **Materialize:** `python scripts/feast_materialize.py`
4. **Retrieve:** Use FeatureStore in Python or HTTP API

## 🎯 Production

For production deployment:
- Update `feature_store.yaml` with production Cassandra cluster
- Use cloud offline store (BigQuery, Snowflake, Redshift)
- Set up scheduled materialization (Airflow)
- Deploy dedicated Feast feature server
