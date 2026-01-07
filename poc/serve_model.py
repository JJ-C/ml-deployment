from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
from cassandra.cluster import Cluster
import numpy as np
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import os

print("=" * 60)
print("Starting Model Server (POC)")
print("=" * 60)

mlflow.set_tracking_uri("http://localhost:5001")

print("\n[1/3] Loading model from MLflow...")
try:
    model = mlflow.pyfunc.load_model("models:/fraud_detector/Production")
    print("✓ Model loaded: fraud_detector/Production")
except Exception as e:
    print(f"⚠ Could not load from registry: {e}")
    print("  Trying to load from latest run...")
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("fraud_detection_poc")
        runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
        if runs:
            model_uri = f"runs:/{runs[0].info.run_id}/model"
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"✓ Model loaded from latest run")
        else:
            raise Exception("No trained models found. Run: python poc/train_fraud_model.py")
    except Exception as e2:
        print(f"✗ Failed to load model: {e2}")
        exit(1)

print("\n[2/3] Connecting to Cassandra feature store...")
try:
    cluster = Cluster(['localhost'], port=9042)
    cassandra_session = cluster.connect()
    cassandra_session.set_keyspace('ml_features')
    
    prepared_query = cassandra_session.prepare(
        "SELECT * FROM transaction_features WHERE transaction_id = ?"
    )
    print("✓ Connected to Cassandra")
except Exception as e:
    print(f"⚠ Cassandra not available: {e}")
    cassandra_session = None
    prepared_query = None

print("\n[3/3] Starting FastAPI server...")

app = FastAPI(
    title="Fraud Detection API (POC)",
    description="Real-time fraud detection with <5ms latency",
    version="1.0.0"
)

predictions_counter = Counter('predictions_total', 'Total predictions made', ['outcome'])
fraud_counter = Counter('fraud_predictions_total', 'Total fraud predictions')
latency_histogram = Histogram('prediction_latency_seconds', 'Prediction latency')
feature_fetch_histogram = Histogram('feature_fetch_latency_seconds', 'Feature fetch latency')

class PredictionRequest(BaseModel):
    transaction_id: str
    amount: float = None
    transaction_hour: int = None
    merchant_category: str = None
    foreign_transaction: int = None
    location_mismatch: int = None
    device_trust_score: float = None
    velocity_last_24h: int = None
    cardholder_age: int = None

class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_score: float
    latency_ms: float
    model_version: str
    feature_source: str

MERCHANT_CATEGORY_MAP = {
    'Electronics': 0,
    'Travel': 1,
    'Grocery': 2,
    'Dining': 3,
    'Other': 4
}

@app.get("/")
async def root():
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "metrics": "/metrics",
            "model_info": "/model/info"
        }
    }

@app.get("/health")
async def health():
    cassandra_status = "healthy" if cassandra_session else "unavailable"
    return {
        "status": "healthy",
        "model": "loaded",
        "cassandra": cassandra_status
    }

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/model/info")
async def model_info():
    return {
        "model_name": "fraud_detector",
        "model_stage": "Production",
        "framework": "XGBoost",
        "features": [
            "amount", "transaction_hour", "merchant_category",
            "foreign_transaction", "location_mismatch", 
            "device_trust_score", "velocity_last_24h", "cardholder_age"
        ]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()
    
    feature_source = "request"
    
    if cassandra_session and any(v is None for v in [
        request.amount, request.transaction_hour, request.merchant_category,
        request.foreign_transaction, request.location_mismatch,
        request.device_trust_score, request.velocity_last_24h, request.cardholder_age
    ]):
        fetch_start = time.time()
        try:
            result = cassandra_session.execute(
                prepared_query,
                (request.transaction_id,)
            )
            row = result.one()
            if row:
                request.amount = request.amount or row.amount
                request.transaction_hour = request.transaction_hour or row.transaction_hour
                request.merchant_category = request.merchant_category or row.merchant_category
                request.foreign_transaction = request.foreign_transaction or row.foreign_transaction
                request.location_mismatch = request.location_mismatch or row.location_mismatch
                request.device_trust_score = request.device_trust_score or row.device_trust_score
                request.velocity_last_24h = request.velocity_last_24h or row.velocity_last_24h
                request.cardholder_age = request.cardholder_age or row.cardholder_age
                feature_source = "cassandra"
        except Exception as e:
            pass
        
        feature_fetch_histogram.observe(time.time() - fetch_start)
    
    if request.amount is None:
        raise HTTPException(status_code=400, detail="Missing required features")
    
    merchant_code = MERCHANT_CATEGORY_MAP.get(request.merchant_category, 4)
    
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
    
    try:
        prediction = model.predict(features)
        fraud_prob = float(prediction[0])
    except Exception as e:
        import traceback
        print(f"ERROR: Prediction failed!")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    is_fraud = fraud_prob > 0.5
    
    latency_ms = (time.time() - start_time) * 1000
    
    latency_histogram.observe(time.time() - start_time)
    predictions_counter.labels(outcome='fraud' if is_fraud else 'legitimate').inc()
    if is_fraud:
        fraud_counter.inc()
    
    return PredictionResponse(
        transaction_id=request.transaction_id,
        is_fraud=is_fraud,
        fraud_probability=fraud_prob,
        risk_score=fraud_prob,
        latency_ms=round(latency_ms, 2),
        model_version="1",
        feature_source=feature_source
    )

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("✓ Server Ready!")
    print("=" * 60)
    print("\nEndpoints:")
    print("  - API: http://localhost:8000")
    print("  - Docs: http://localhost:8000/docs")
    print("  - Metrics: http://localhost:8000/metrics")
    print("\nTest with:")
    print('  curl -X POST http://localhost:8000/predict \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"transaction_id": "1", "amount": 500.0}\'')
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
