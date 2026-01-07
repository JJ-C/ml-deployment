#!/usr/bin/env python3
"""
FastAPI server for sentiment analysis model.
Loads model from MLflow and serves predictions.
"""

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import mlflow
import mlflow.sklearn
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MLflow configuration
MLFLOW_TRACKING_URI = "http://localhost:5001"
MODEL_NAME = "sentiment_analyzer"

app = FastAPI(
    title="Sentiment Analysis API",
    description="Real-time sentiment analysis for text",
    version="1.0.0"
)

# Prometheus metrics
prediction_counter = Counter('sentiment_predictions_total', 'Total predictions', ['sentiment'])
prediction_latency = Histogram('sentiment_prediction_latency_seconds', 'Prediction latency')
error_counter = Counter('sentiment_errors_total', 'Total errors', ['error_type'])

# Global model variable
model = None
model_version = None

def load_model():
    """Load model from MLflow registry"""
    global model, model_version
    
    logger.info("Loading sentiment analysis model from MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        # Try to load from registry (Production stage)
        model_uri = f"models:/{MODEL_NAME}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        model_version = "Production"
        logger.info(f"✓ Loaded model from registry: {model_uri}")
    except Exception as e:
        logger.warning(f"Could not load from registry: {e}")
        logger.info("Attempting to load latest model from runs...")
        
        # Fallback: load latest model from experiment
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name("sentiment_analysis")
            
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=1
                )
                
                if runs:
                    run_id = runs[0].info.run_id
                    model_uri = f"runs:/{run_id}/model"
                    model = mlflow.sklearn.load_model(model_uri)
                    model_version = f"run_{run_id[:8]}"
                    logger.info(f"✓ Loaded model from run: {run_id}")
                else:
                    raise Exception("No runs found in experiment")
            else:
                raise Exception("Experiment 'sentiment_analysis' not found")
        except Exception as e2:
            logger.error(f"Failed to load model: {e2}")
            raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("=" * 60)
    logger.info("Starting Sentiment Analysis API")
    logger.info("=" * 60)
    
    try:
        load_model()
        logger.info(f"Model version: {model_version}")
        logger.info("Server ready!")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

class SentimentRequest(BaseModel):
    text: str
    request_id: Optional[str] = None

class SentimentResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    request_id: Optional[str]
    text: str
    sentiment: str  # "positive", "neutral", "negative"
    sentiment_score: int  # 1, 0, -1
    confidence: float  # probability of predicted class
    probabilities: dict  # {"positive": 0.7, "neutral": 0.2, "negative": 0.1}
    latency_ms: float
    model_version: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Sentiment Analysis API",
        "version": "1.0.0",
        "status": "running",
        "model_version": model_version
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "model_type": "LogisticRegression + TfidfVectorizer",
        "classes": ["negative", "neutral", "positive"],
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI
    }

@app.post("/predict", response_model=SentimentResponse)
async def predict(request: SentimentRequest):
    """Predict sentiment for given text"""
    start_time = time.time()
    
    logger.info(f"Received prediction request: {request.request_id or 'no_id'}")
    logger.debug(f"Text preview: {request.text[:100]}...")
    
    if model is None:
        error_counter.labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text or not request.text.strip():
        error_counter.labels(error_type="empty_text").inc()
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Predict
        prediction = model.predict([request.text])[0]
        probabilities = model.predict_proba([request.text])[0]
        
        # Map prediction to sentiment
        sentiment_map = {-1: "negative", 0: "neutral", 1: "positive"}
        sentiment = sentiment_map[prediction]
        
        # Get confidence (probability of predicted class)
        class_index = {-1: 0, 0: 1, 1: 2}[prediction]
        confidence = float(probabilities[class_index])
        
        # Format probabilities
        probs_dict = {
            "negative": float(probabilities[0]),
            "neutral": float(probabilities[1]),
            "positive": float(probabilities[2])
        }
        
        latency = (time.time() - start_time) * 1000
        
        # Update metrics
        prediction_counter.labels(sentiment=sentiment).inc()
        prediction_latency.observe(time.time() - start_time)
        
        logger.info(f"Prediction: {sentiment} (confidence: {confidence:.3f}, latency: {latency:.2f}ms)")
        
        return SentimentResponse(
            request_id=request.request_id,
            text=request.text[:200],  # Truncate for response
            sentiment=sentiment,
            sentiment_score=int(prediction),
            confidence=confidence,
            probabilities=probs_dict,
            latency_ms=latency,
            model_version=model_version
        )
        
    except Exception as e:
        error_counter.labels(error_type="prediction_error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting server on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
