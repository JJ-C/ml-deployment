from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
from feast import FeatureStore
import numpy as np
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import os
from typing import Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("Starting Model Server with Feast (POC)")
logger.info("=" * 60)

logger.info("[1/4] Loading model from MLflow...")
try:
    model_uri = "models:/fraud_detector/Production"
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("✓ Model loaded from registry")
except Exception as e:
    logger.warning(f"Could not load from registry: {e}")
    logger.info("Trying latest run...")
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5001")
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("fraud_detection_poc")
        runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
        if runs:
            model_uri = f"runs:/{runs[0].info.run_id}/model"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("✓ Model loaded from latest run")
        else:
            raise Exception("No trained models found. Run: python poc/train_fraud_model.py")
    except Exception as e2:
        logger.error(f"Failed to load model: {e2}")
        exit(1)

logger.info("[2/4] Connecting to Feast feature store...")
try:
    feast_store = FeatureStore(repo_path="feature_repo")
    logger.info("✓ Connected to Feast")
except Exception as e:
    logger.warning(f"Feast not available: {e}")
    feast_store = None

logger.info("[3/4] Starting FastAPI server...")

app = FastAPI(
    title="Fraud Detection API with Feast (POC)",
    description="Real-time fraud detection with Feast feature store",
    version="1.0.0"
)

predictions_counter = Counter('predictions_total', 'Total predictions made', ['outcome'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
feature_fetch_histogram = Histogram('feature_fetch_latency_seconds', 'Feature fetch latency')

MERCHANT_CATEGORY_MAP = {
    'Grocery': 0, 'Gas': 1, 'Restaurant': 2, 'Retail': 3, 'Other': 4,
    'Travel': 5, 'Entertainment': 6, 'Healthcare': 7, 'Electronics': 8
}

class PredictionRequest(BaseModel):
    transaction_id: str
    user_id: str  # Required: entity key for feature lookup
    amount: Optional[float] = None
    transaction_hour: Optional[int] = None
    merchant_category: Optional[str] = None
    foreign_transaction: Optional[int] = None
    location_mismatch: Optional[int] = None
    device_trust_score: Optional[float] = None
    velocity_last_24h: Optional[int] = None
    cardholder_age: Optional[int] = None

class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    latency_ms: float
    feature_source: str

@app.get("/")
async def root():
    return {
        "service": "Fraud Detection API with Feast",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "metrics": "/metrics",
            "model_info": "/model/info"
        }
    }

@app.get("/health")
async def health():
    feast_status = "healthy" if feast_store else "unavailable"
    return {
        "status": "healthy",
        "model": "loaded",
        "feast": feast_status
    }

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/model/info")
async def model_info():
    return {
        "model_name": "fraud_detector",
        "model_uri": model_uri,
        "feature_store": "feast",
        "features": [
            "amount", "transaction_hour", "merchant_category",
            "foreign_transaction", "location_mismatch",
            "device_trust_score", "velocity_last_24h", "cardholder_age"
        ]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()
    logger.info(f"Received prediction request for transaction_id: {request.transaction_id}")
    
    feature_source = "request"
    
    # Try to fetch features from Feast if any are missing
    if feast_store and any(v is None for v in [
        request.amount, request.transaction_hour, request.merchant_category,
        request.foreign_transaction, request.location_mismatch,
        request.device_trust_score, request.velocity_last_24h, request.cardholder_age
    ]):
        logger.info(f"Fetching features from Feast for user_id: {request.user_id}")
        fetch_start = time.time()
        try:
            entity_rows = [{"user_id": request.user_id}]
            
            features = feast_store.get_online_features(
                features=[
                    "user_features:amount",
                    "user_features:transaction_hour",
                    "user_features:merchant_category",
                    "user_features:foreign_transaction",
                    "user_features:location_mismatch",
                    "user_features:device_trust_score",
                    "user_features:velocity_last_24h",
                    "user_features:cardholder_age",
                ],
                entity_rows=entity_rows,
            ).to_dict()
            
            logger.debug(f"Feast features retrieved: {features}")
            
            # Fill in missing features from Feast
            request.amount = request.amount or features.get("amount", [None])[0]
            request.transaction_hour = request.transaction_hour or features.get("transaction_hour", [None])[0]
            request.merchant_category = request.merchant_category or features.get("merchant_category", [None])[0]
            request.foreign_transaction = request.foreign_transaction or features.get("foreign_transaction", [None])[0]
            request.location_mismatch = request.location_mismatch or features.get("location_mismatch", [None])[0]
            request.device_trust_score = request.device_trust_score or features.get("device_trust_score", [None])[0]
            request.velocity_last_24h = request.velocity_last_24h or features.get("velocity_last_24h", [None])[0]
            request.cardholder_age = request.cardholder_age or features.get("cardholder_age", [None])[0]
            
            feature_source = "feast"
            logger.info(f"Successfully fetched features from Feast in {(time.time() - fetch_start)*1000:.2f}ms")
        except Exception as e:
            logger.warning(f"Failed to fetch features from Feast: {e}")
        
        feature_fetch_histogram.observe(time.time() - fetch_start)
    
    if request.amount is None:
        logger.error("Missing required features")
        raise HTTPException(status_code=400, detail="Missing required features")
    
    merchant_code = MERCHANT_CATEGORY_MAP.get(request.merchant_category, 4)
    logger.debug(f"Merchant category '{request.merchant_category}' mapped to code: {merchant_code}")
    
    import pandas as pd
    features = pd.DataFrame({
        'amount': pd.Series([request.amount], dtype='float64'),
        'transaction_hour': pd.Series([request.transaction_hour or 12], dtype='int64'),
        'merchant_category': pd.Series([merchant_code], dtype='int8'),
        'foreign_transaction': pd.Series([request.foreign_transaction or 0], dtype='int64'),
        'location_mismatch': pd.Series([request.location_mismatch or 0], dtype='int64'),
        'device_trust_score': pd.Series([request.device_trust_score or 50], dtype='int64'),
        'velocity_last_24h': pd.Series([request.velocity_last_24h or 1], dtype='int64'),
        'cardholder_age': pd.Series([request.cardholder_age or 35], dtype='int64')
    })
    
    logger.debug(f"Feature DataFrame shape: {features.shape}, dtypes: {features.dtypes.to_dict()}")
    
    prediction = model.predict(features)
    fraud_probability = float(prediction[0])
    is_fraud = fraud_probability > 0.5
    
    logger.info(f"Prediction: is_fraud={is_fraud}, probability={fraud_probability:.4f}, source={feature_source}")
    
    latency_ms = (time.time() - start_time) * 1000
    
    predictions_counter.labels(outcome='fraud' if is_fraud else 'legit').inc()
    prediction_latency.observe(time.time() - start_time)
    
    return PredictionResponse(
        transaction_id=request.transaction_id,
        is_fraud=is_fraud,
        fraud_probability=fraud_probability,
        latency_ms=latency_ms,
        feature_source=feature_source
    )

logger.info("[4/4] Server ready!")
logger.info("=" * 60)
logger.info("✓ Model Server with Feast is running")
logger.info("=" * 60)
logger.info("Endpoints:")
logger.info("  - API: http://localhost:8000")
logger.info("  - Docs: http://localhost:8000/docs")
logger.info("  - Health: http://localhost:8000/health")
logger.info("  - Metrics: http://localhost:8000/metrics")
logger.info("Feature Store: Feast (Cassandra online store)")
logger.info("=" * 60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
