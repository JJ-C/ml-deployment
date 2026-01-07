# ML Platform POC - Quick Start

This is a local Proof of Concept demonstrating the ML platform architecture before production deployment.

## 🚀 Quick Start (5 minutes)

```bash
# 0. Convert CSV to Parquet (one-time setup)
python scripts/convert_csv_to_parquet.py

# 1. Start infrastructure
docker-compose up -d

# 2. Install dependencies
# may require python -m ensurepip --upgrade if pip is not available
pip install -r requirements-poc.txt

# 3. Train model
python poc/train_fraud_model.py

# 4. Populate feature store
python poc/populate_features.py

# 5. Start model server
python poc/serve_model.py

# 6. Test (in another terminal)
python poc/test_predictions.py
```

## 📊 What's Included

### Core Components
- **MLflow** - Experiment tracking & model registry (http://localhost:5000)
- **Cassandra** - Online feature store (localhost:9042)
- **Feast** - Feature store framework (optional, production-grade)
- **Prometheus** - Metrics & monitoring (http://localhost:9090)
- **Fraud Detection Model** - XGBoost classifier with <5ms latency

### Model Serving Options
- **FastAPI + Direct Cassandra** - Simple serving (http://localhost:8000)
  - Best for: Quick development, testing
  - Setup time: 30 seconds
  - Feature store: Direct Cassandra access
  
- **FastAPI + Feast** - Production-grade serving (http://localhost:8000)
  - Best for: Feature versioning, consistency guarantees
  - Setup time: 5 minutes
  - Feature store: Feast with Cassandra backend
  
- **Seldon Core** - Kubernetes-based serving (http://localhost:8001)
  - Best for: Production parity, auto-scaling, A/B testing
  - Setup time: 10 minutes
  - Requires: Local Kubernetes cluster

## 📁 Project Structure

```
ml_deployment/
├── README.md                    # This file
├── docker-compose.yml          # Infrastructure setup
├── requirements-poc.txt        # Python dependencies
├── docs/                       # Documentation
│   ├── POC_LOCAL_SETUP.md      # Detailed setup guide
│   ├── ml_platform_design.md   # Production architecture design
│   ├── PARQUET_MIGRATION.md    # Parquet format guide
│   ├── FEAST_SETUP.md          # Feast feature store guide
│   └── SELDON_SETUP.md         # Kubernetes deployment guide
├── data/
│   ├── credit_card_fraud_10k.parquet  # Fraud detection data (Parquet)
│   ├── Twitter_Data.parquet           # Sentiment analysis data
│   └── ...                            # Other datasets
├── poc/
│   ├── train_fraud_model.py    # Model training
│   ├── populate_features.py    # Feature store setup
│   ├── serve_model.py          # Model serving (FastAPI)
│   ├── test_predictions.py     # Functional testing
│   └── load_test.py            # Performance testing
├── feature_repo/               # Feast feature definitions
├── kubernetes/                 # Kubernetes deployment files
├── scripts/                    # Utility scripts
└── mlflow_data/                # MLflow artifacts (created on first run)
```

## 🧪 Testing

```bash
# Functional tests
python poc/test_predictions.py

# Load test (100 RPS for 30 seconds)
python poc/load_test.py --rps 100 --duration 30

# High load test (1000 RPS)
python poc/load_test.py --rps 1000 --duration 10
```

## 📈 Expected Performance

- **Latency:** < 5ms (P99)
- **Throughput:** 300+ RPS per CPU core
- **Model Accuracy:** AUC > 0.94, F1 > 0.89
- **Feature Fetch:** ~2-3ms from Cassandra

## 🔍 Monitoring

- **MLflow UI:** http://localhost:5000 - View experiments and models
- **API Docs:** http://localhost:8000/docs - Interactive API documentation
- **Prometheus:** http://localhost:9090 - Query metrics
- **Metrics Endpoint:** http://localhost:8000/metrics - Raw metrics

## 📚 Documentation

- **Quick Start:** This file (README.md)
- **Detailed Setup:** See `docs/POC_LOCAL_SETUP.md`
- **Production Design:** See `docs/ml_platform_design.md`
- **Parquet Format:** See `docs/PARQUET_MIGRATION.md` (data optimization)
- **Feast Integration:** See `docs/FEAST_SETUP.md` (feature store framework)
- **Seldon Core Setup:** See `docs/SELDON_SETUP.md` (optional, for K8s deployment)

## 🛠️ Troubleshooting

**Server won't start:**
```bash
# Check if ports are available
lsof -i :5000  # MLflow
lsof -i :9042  # Cassandra
lsof -i :8000  # Model server
lsof -i :9090  # Prometheus
```

**Model not found:**
```bash
# Retrain the model
python poc/train_fraud_model.py
```

**Cassandra connection error:**
```bash
# Restart Cassandra
docker-compose restart cassandra

# Check Cassandra status
docker-compose ps cassandra

# Note: Cassandra takes 30-60 seconds to start
docker-compose logs -f cassandra
```

## 🧹 Cleanup

```bash
# Stop services
docker-compose down

# Remove all data
docker-compose down -v
rm -rf mlflow_data/
```

## ➡️ Next Steps

After validating the POC:

### Phase 1: Basic Validation
1. ✅ Confirm latency < 5ms with FastAPI
2. ✅ Verify model accuracy meets requirements
3. ✅ Test with your own data
4. 📝 Document lessons learned

### Phase 2: Advanced (Optional)
5. 🔧 Deploy with Seldon Core on local Kubernetes
6. 📊 Compare FastAPI vs Seldon Core performance
7. 🧪 Test auto-scaling and A/B testing features

### Phase 3: Production Planning
8. 🚀 Plan production migration (see `docs/ml_platform_design.md`)

## 🤝 Support

For questions or issues, refer to:
- `docs/POC_LOCAL_SETUP.md` for detailed instructions
- `docs/ml_platform_design.md` for architecture decisions
