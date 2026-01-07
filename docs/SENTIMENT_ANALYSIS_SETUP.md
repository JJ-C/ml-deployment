# Sentiment Analysis Model Setup

Complete guide for training and deploying the sentiment analysis model.

## Overview

**Model:** Logistic Regression + TF-IDF Vectorizer  
**Data:** Twitter (162K samples) + Reddit (37K samples) = ~200K total  
**Classes:** Negative (-1), Neutral (0), Positive (1)  
**API Port:** 8001 (different from fraud detection on 8000)

## Quick Start

```bash
# 1. Train the model
python poc/train_sentiment_model.py

# 2. Start the API server
python poc/serve_sentiment_model.py

# 3. Test predictions
python poc/test_sentiment_predictions.py

# 4. Load test
python poc/load_test_sentiment.py
```

## Data Details

### Twitter Data
- **File:** `data/Twitter_Data.parquet`
- **Samples:** 162,980
- **Columns:** `clean_text`, `category`
- **Nulls:** 4 texts, 7 labels

### Reddit Data
- **File:** `data/Reddit_Data.parquet`
- **Samples:** 37,249
- **Columns:** `clean_comment`, `category`
- **Nulls:** 100 comments, 0 labels

### Label Distribution (Combined)
- **Positive (1):** ~50%
- **Neutral (0):** ~35%
- **Negative (-1):** ~15%

## Training

### Model Architecture
```
Text Input
    ↓
TF-IDF Vectorizer
  - max_features: 10,000
  - ngram_range: (1, 2)
  - min_df: 2
  - max_df: 0.95
    ↓
Logistic Regression
  - max_iter: 1,000
  - class_weight: balanced
  - solver: saga
    ↓
Prediction (Negative/Neutral/Positive)
```

### Training Script
```bash
python poc/train_sentiment_model.py
```

**What it does:**
1. Loads Twitter and Reddit data
2. Combines and cleans datasets
3. Splits 80/20 train/test with stratification
4. Trains TF-IDF + Logistic Regression pipeline
5. Evaluates on test set
6. Logs to MLflow (experiment: `sentiment_analysis`)
7. Registers model as `sentiment_analyzer`

**Expected Performance:**
- Test Accuracy: ~75-85%
- Macro F1: ~0.70-0.80
- Training time: 2-5 minutes

### MLflow Tracking
- **Tracking URI:** http://localhost:5001
- **UI:** http://localhost:5000
- **Experiment:** sentiment_analysis
- **Model Name:** sentiment_analyzer

## Model Serving

### Start Server
```bash
python poc/serve_sentiment_model.py
```

Server runs on **http://localhost:8001** (port 8001 to avoid conflict with fraud detection)

### API Endpoints

#### Health Check
```bash
curl http://localhost:8001/health
```

#### Model Info
```bash
curl http://localhost:8001/model/info
```

#### Predict Sentiment
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This product is amazing! Best purchase ever!",
    "request_id": "test_123"
  }'
```

**Response:**
```json
{
  "request_id": "test_123",
  "text": "This product is amazing! Best purchase ever!",
  "sentiment": "positive",
  "sentiment_score": 1,
  "confidence": 0.92,
  "probabilities": {
    "positive": 0.92,
    "neutral": 0.06,
    "negative": 0.02
  },
  "latency_ms": 3.45,
  "model_version": "Production"
}
```

#### Metrics (Prometheus)
```bash
curl http://localhost:8001/metrics
```

## Testing

### Functional Tests
```bash
python poc/test_sentiment_predictions.py
```

Tests 8 different scenarios:
- Positive product review
- Negative customer complaint
- Neutral factual statement
- Positive social media post
- Negative political comment
- Mixed restaurant review
- Positive tech review
- Negative movie review

### Load Testing
```bash
# Default: 50 RPS for 30 seconds
python poc/load_test_sentiment.py

# Custom parameters
python poc/load_test_sentiment.py --rps 100 --duration 60 --workers 20
```

**Expected Performance:**
- P99 latency: <10ms
- Success rate: >99%
- Throughput: 50-100 RPS (single instance)

## Request/Response Schema

### Request
```python
{
  "text": str,              # Required: text to analyze
  "request_id": str | None  # Optional: tracking ID
}
```

### Response
```python
{
  "request_id": str | None,
  "text": str,                    # Truncated to 200 chars
  "sentiment": str,               # "positive", "neutral", "negative"
  "sentiment_score": int,         # 1, 0, -1
  "confidence": float,            # 0.0 to 1.0
  "probabilities": {
    "positive": float,
    "neutral": float,
    "negative": float
  },
  "latency_ms": float,
  "model_version": str
}
```

## Integration with Existing Platform

### Port Allocation
- **Fraud Detection API:** 8000
- **Sentiment Analysis API:** 8001
- **MLflow Tracking:** 5001
- **MLflow UI:** 5000

### Shared Infrastructure
- Same MLflow instance
- Same Docker Compose setup
- Same monitoring stack (Prometheus)

### Running Both Services
```bash
# Terminal 1: Fraud detection
python poc/serve_model.py

# Terminal 2: Sentiment analysis
python poc/serve_sentiment_model.py

# Both APIs running simultaneously
```

## Production Considerations

### Model Updates
1. Retrain with new data
2. Log to MLflow
3. Promote to Production stage
4. Restart server (auto-loads Production model)

### Monitoring
- Track prediction distribution
- Monitor confidence scores
- Alert on low confidence predictions
- Track latency metrics

### Scaling
- Horizontal scaling: Multiple server instances behind load balancer
- Caching: Cache predictions for identical texts
- Batch processing: For bulk sentiment analysis

### Feature Enhancements
1. **Multi-language support:** Train on multilingual data
2. **Aspect-based sentiment:** Identify sentiment for specific aspects
3. **Emotion detection:** Beyond positive/negative/neutral
4. **Confidence thresholds:** Flag low-confidence predictions for review

## Troubleshooting

### Model not loading
**Error:** "Model not loaded" or 503 errors

**Solutions:**
1. Check MLflow is running: `docker-compose ps`
2. Verify model exists: Check http://localhost:5000
3. Train model if missing: `python poc/train_sentiment_model.py`

### Port already in use
**Error:** "Address already in use" on port 8001

**Solutions:**
```bash
# Find process using port 8001
lsof -i :8001

# Kill the process
kill -9 <PID>

# Or use different port
uvicorn poc.serve_sentiment_model:app --port 8002
```

### Low accuracy
**Issue:** Model predictions seem incorrect

**Solutions:**
1. Check training data quality
2. Retrain with more data
3. Try different model (e.g., BERT-based)
4. Adjust TF-IDF parameters

### Slow predictions
**Issue:** Latency > 10ms

**Solutions:**
1. Reduce max_features in TF-IDF
2. Use sparse matrix operations
3. Add caching for common texts
4. Profile with `cProfile`

## Files Created

```
poc/
├── train_sentiment_model.py          # Training script
├── serve_sentiment_model.py          # FastAPI server
├── test_sentiment_predictions.py     # Functional tests
└── load_test_sentiment.py            # Load testing

docs/
└── SENTIMENT_ANALYSIS_SETUP.md       # This file

data/
├── Twitter_Data.parquet              # Twitter dataset
└── Reddit_Data.parquet               # Reddit dataset
```

## Next Steps

1. ✅ Train model
2. ✅ Start server
3. ✅ Run tests
4. 🔄 Deploy to production
5. 🔄 Add monitoring dashboards
6. 🔄 Implement A/B testing
7. 🔄 Add more data sources

## Comparison with Fraud Detection

| Feature | Fraud Detection | Sentiment Analysis |
|---------|----------------|-------------------|
| Model | XGBoost | Logistic Regression |
| Input | Structured features | Text |
| Output | Binary (fraud/not) | Multi-class (3) |
| Latency | ~3-5ms | ~5-10ms |
| Port | 8000 | 8001 |
| Data size | 10K samples | 200K samples |
| Feature Store | Feast (Cassandra) | N/A |

Both models share the same MLflow infrastructure and monitoring stack.
