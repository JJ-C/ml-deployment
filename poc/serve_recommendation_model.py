#!/usr/bin/env python3
"""
FastAPI server for book recommendation model.
Loads model from MLflow and serves recommendations.
"""

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import mlflow
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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
EXPERIMENT_NAME = "book_recommendations"

app = FastAPI(
    title="Book Recommendation API",
    description="Content-based book recommendations",
    version="1.0.0"
)

# Prometheus metrics
recommendation_counter = Counter('book_recommendations_total', 'Total recommendations')
recommendation_latency = Histogram('book_recommendation_latency_seconds', 'Recommendation latency')
error_counter = Counter('book_recommendation_errors_total', 'Total errors', ['error_type'])

# Global model variables
model_artifacts = None
model_version = None

def load_model():
    """Load model from MLflow"""
    global model_artifacts, model_version
    
    logger.info("Loading recommendation model from MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        # Get the latest run from the experiment
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        
        if not experiment:
            raise Exception(f"Experiment '{EXPERIMENT_NAME}' not found")
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if not runs:
            raise Exception("No runs found in experiment")
        
        run_id = runs[0].info.run_id
        model_version = f"run_{run_id[:8]}"
        
        # Download the model artifact
        artifact_path = client.download_artifacts(run_id, "recommendation_model.pkl")
        
        # Load the pickle file
        with open(artifact_path, 'rb') as f:
            model_artifacts = pickle.load(f)
        
        logger.info(f"✓ Loaded model from run: {run_id}")
        logger.info(f"  - Books indexed: {len(model_artifacts['books_df'])}")
        logger.info(f"  - Vocabulary size: {len(model_artifacts['vectorizer'].vocabulary_)}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("=" * 60)
    logger.info("Starting Book Recommendation API")
    logger.info("=" * 60)
    
    try:
        load_model()
        logger.info(f"Model version: {model_version}")
        logger.info("Server ready!")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

class BookInfo(BaseModel):
    book_id: str
    title: str
    authors: str
    categories: str
    description: str
    average_rating: Optional[float]
    ratings_count: int
    thumbnail: Optional[str]
    similarity_score: float

class RecommendationRequest(BaseModel):
    book_id: Optional[str] = None
    title: Optional[str] = None
    num_recommendations: int = Field(default=5, ge=1, le=20)
    request_id: Optional[str] = None

class RecommendationResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    request_id: Optional[str]
    query_book: Optional[BookInfo]
    recommendations: List[BookInfo]
    num_results: int
    latency_ms: float
    model_version: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Book Recommendation API",
        "version": "1.0.0",
        "status": "running",
        "model_version": model_version,
        "total_books": len(model_artifacts['books_df']) if model_artifacts else 0
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    if model_artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model_artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_version": model_version,
        "model_type": "Content-Based (TF-IDF + Cosine Similarity)",
        "total_books": len(model_artifacts['books_df']),
        "vocabulary_size": len(model_artifacts['vectorizer'].vocabulary_),
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI
    }

@app.get("/books/search")
async def search_books(query: str, limit: int = 10):
    """Search for books by title"""
    if model_artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    df = model_artifacts['books_df']
    
    # Simple case-insensitive search
    matches = df[df['title'].str.lower().str.contains(query.lower(), na=False)]
    
    results = []
    for _, book in matches.head(limit).iterrows():
        results.append({
            "book_id": book['book_id'],
            "title": book['title'],
            "authors": book['authors'],
            "categories": book['categories']
        })
    
    return {
        "query": query,
        "num_results": len(results),
        "results": results
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """Get book recommendations"""
    start_time = time.time()
    
    logger.info(f"Received recommendation request: {request.request_id or 'no_id'}")
    
    if model_artifacts is None:
        error_counter.labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.book_id and not request.title:
        error_counter.labels(error_type="missing_input").inc()
        raise HTTPException(status_code=400, detail="Either book_id or title must be provided")
    
    try:
        df = model_artifacts['books_df']
        vectorizer = model_artifacts['vectorizer']
        tfidf_matrix = model_artifacts['tfidf_matrix']
        
        # Find the query book
        if request.book_id:
            query_book = df[df['book_id'] == request.book_id]
            if query_book.empty:
                error_counter.labels(error_type="book_not_found").inc()
                raise HTTPException(status_code=404, detail=f"Book with ID '{request.book_id}' not found")
            query_idx = query_book.index[0]
        else:
            # Search by title (case-insensitive, partial match)
            query_book = df[df['title'].str.lower().str.contains(request.title.lower(), na=False)]
            if query_book.empty:
                error_counter.labels(error_type="book_not_found").inc()
                raise HTTPException(status_code=404, detail=f"Book with title '{request.title}' not found")
            query_idx = query_book.index[0]
        
        query_book = df.iloc[query_idx]
        logger.info(f"Query book: '{query_book['title']}' by {query_book['authors']}")
        
        # Compute cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix[query_idx:query_idx+1], tfidf_matrix).flatten()
        
        # Get top N similar books (excluding the query book itself)
        similar_indices = cosine_sim.argsort()[::-1][1:request.num_recommendations+1]
        
        # Build recommendations
        recommendations = []
        for idx in similar_indices:
            book = df.iloc[idx]
            recommendations.append(BookInfo(
                book_id=book['book_id'],
                title=book['title'],
                authors=book['authors'],
                categories=book['categories'],
                description=book['description'][:200] if book['description'] else "",
                average_rating=float(book['average_rating']) if pd.notna(book['average_rating']) else None,
                ratings_count=int(book['ratings_count']),
                thumbnail=book['thumbnail'] if pd.notna(book['thumbnail']) else None,
                similarity_score=float(cosine_sim[idx])
            ))
        
        latency = (time.time() - start_time) * 1000
        
        # Update metrics
        recommendation_counter.inc()
        recommendation_latency.observe(time.time() - start_time)
        
        logger.info(f"Generated {len(recommendations)} recommendations (latency: {latency:.2f}ms)")
        
        return RecommendationResponse(
            request_id=request.request_id,
            query_book=BookInfo(
                book_id=query_book['book_id'],
                title=query_book['title'],
                authors=query_book['authors'],
                categories=query_book['categories'],
                description=query_book['description'][:200] if query_book['description'] else "",
                average_rating=float(query_book['average_rating']) if pd.notna(query_book['average_rating']) else None,
                ratings_count=int(query_book['ratings_count']),
                thumbnail=query_book['thumbnail'] if pd.notna(query_book['thumbnail']) else None,
                similarity_score=1.0
            ),
            recommendations=recommendations,
            num_results=len(recommendations),
            latency_ms=latency,
            model_version=model_version
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_counter.labels(error_type="recommendation_error").inc()
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    import pandas as pd
    
    logger.info("Starting server on http://localhost:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
