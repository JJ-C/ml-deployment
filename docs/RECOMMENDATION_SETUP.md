# Book Recommendation System Setup

Complete guide for training and deploying the content-based book recommendation model.

## Overview

**Model:** Content-Based Filtering (TF-IDF + Cosine Similarity)  
**Data:** Google Books Dataset (15K books)  
**Method:** Recommend similar books based on content features  
**API Port:** 8002 (different from fraud detection on 8000 and sentiment on 8001)

## Quick Start

```bash
# 1. Train the model
python poc/train_recommendation_model.py

# 2. Start the API server
python poc/serve_recommendation_model.py

# 3. Test recommendations
python poc/test_recommendation_predictions.py

# 4. Load test
python poc/load_test_recommendation.py
```

## Data Details

### Google Books Dataset
- **File:** `data/google_books_dataset.parquet`
- **Total Books:** 15,147
- **Columns:** 21 features including title, authors, description, categories, ratings

### Key Features
- **Title:** Book title
- **Authors:** Author names
- **Description:** Book description/summary
- **Categories:** Main category (Fiction, Computers, Business, etc.)
- **Search Category:** Specific subcategory (bestsellers 2024, romance, etc.)
- **Average Rating:** 1-5 stars (857 books have ratings)
- **Ratings Count:** Number of user ratings

### Top Categories
1. Fiction (994 books)
2. Computers (737 books)
3. Business & Economics (563 books)
4. Education (371 books)
5. History (365 books)

## How It Works

### Content-Based Filtering

The model recommends books similar to a query book based on content features:

1. **Feature Engineering:**
   - Combines title, authors, description, categories
   - Weights important fields (title 2x, authors 2x, categories 3x)
   - Creates unified text representation

2. **TF-IDF Vectorization:**
   - Converts text to numerical features
   - max_features: 5,000
   - ngram_range: (1, 2) - captures single words and pairs
   - Removes common English stop words

3. **Similarity Computation:**
   - Uses cosine similarity between TF-IDF vectors
   - Returns top N most similar books
   - Similarity score: 0.0 (different) to 1.0 (identical)

### Example
```
Query: "Python Programming for Beginners"
↓
TF-IDF Vector: [0.2, 0.5, 0.0, 0.8, ...]
↓
Compute Cosine Similarity with all books
↓
Top Recommendations:
1. "Learning Python" (similarity: 0.85)
2. "Python Crash Course" (similarity: 0.82)
3. "Automate with Python" (similarity: 0.78)
```

## Training

### Training Script
```bash
python poc/train_recommendation_model.py
```

**What it does:**
1. Loads Google Books dataset
2. Preprocesses and combines text features
3. Trains TF-IDF vectorizer
4. Computes TF-IDF matrix for all books
5. Validates with sample recommendations
6. Logs to MLflow (experiment: `book_recommendations`)
7. Saves model artifacts (vectorizer, matrix, book metadata)

**Expected Output:**
- Books indexed: ~15,000
- Vocabulary size: ~5,000 features
- Matrix sparsity: ~99% (very sparse)
- Training time: 1-2 minutes

### MLflow Tracking
- **Tracking URI:** http://localhost:5001
- **UI:** http://localhost:5000
- **Experiment:** book_recommendations

## Model Serving

### Start Server
```bash
python poc/serve_recommendation_model.py
```

Server runs on **http://localhost:8002** (port 8002 to avoid conflicts)

### API Endpoints

#### Health Check
```bash
curl http://localhost:8002/health
```

#### Model Info
```bash
curl http://localhost:8002/model/info
```

#### Search Books
```bash
curl "http://localhost:8002/books/search?query=Python&limit=5"
```

**Response:**
```json
{
  "query": "Python",
  "num_results": 5,
  "results": [
    {
      "book_id": "abc123",
      "title": "Python Programming",
      "authors": "John Doe",
      "categories": "Computers"
    }
  ]
}
```

#### Get Recommendations (by book_id)
```bash
curl -X POST http://localhost:8002/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "book_id": "abc123",
    "num_recommendations": 5,
    "request_id": "test_123"
  }'
```

#### Get Recommendations (by title)
```bash
curl -X POST http://localhost:8002/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Python Programming",
    "num_recommendations": 5,
    "request_id": "test_456"
  }'
```

**Response:**
```json
{
  "request_id": "test_123",
  "query_book": {
    "book_id": "abc123",
    "title": "Python Programming",
    "authors": "John Doe",
    "categories": "Computers",
    "description": "Learn Python from scratch...",
    "average_rating": 4.5,
    "ratings_count": 120,
    "thumbnail": "http://...",
    "similarity_score": 1.0
  },
  "recommendations": [
    {
      "book_id": "def456",
      "title": "Learning Python",
      "authors": "Mark Lutz",
      "categories": "Computers",
      "description": "Comprehensive Python guide...",
      "average_rating": 4.3,
      "ratings_count": 89,
      "thumbnail": "http://...",
      "similarity_score": 0.85
    }
  ],
  "num_results": 5,
  "latency_ms": 12.34,
  "model_version": "run_abc12345"
}
```

#### Metrics (Prometheus)
```bash
curl http://localhost:8002/metrics
```

## Testing

### Functional Tests
```bash
python poc/test_recommendation_predictions.py
```

Tests multiple scenarios:
- Search for books by title
- Get recommendations by book_id
- Get recommendations by partial title match
- Test different categories (ML, Cooking, Fiction)

### Load Testing
```bash
# Default: 20 RPS for 30 seconds
python poc/load_test_recommendation.py

# Custom parameters
python poc/load_test_recommendation.py --rps 50 --duration 60 --workers 20
```

**Expected Performance:**
- P99 latency: <20ms
- Success rate: >99%
- Throughput: 20-50 RPS (single instance)

## Request/Response Schema

### RecommendationRequest
```python
{
  "book_id": str | None,           # Either book_id or title required
  "title": str | None,             # Partial match supported
  "num_recommendations": int,      # 1-20, default 5
  "request_id": str | None         # Optional tracking ID
}
```

### RecommendationResponse
```python
{
  "request_id": str | None,
  "query_book": BookInfo,          # The book used for recommendations
  "recommendations": [BookInfo],   # List of similar books
  "num_results": int,
  "latency_ms": float,
  "model_version": str
}
```

### BookInfo
```python
{
  "book_id": str,
  "title": str,
  "authors": str,
  "categories": str,
  "description": str,              # Truncated to 200 chars
  "average_rating": float | None,
  "ratings_count": int,
  "thumbnail": str | None,
  "similarity_score": float        # 0.0 to 1.0
}
```

## Integration with Existing Platform

### Port Allocation
- **Fraud Detection API:** 8000
- **Sentiment Analysis API:** 8001
- **Book Recommendation API:** 8002
- **MLflow Tracking:** 5001
- **MLflow UI:** 5000

### Shared Infrastructure
- Same MLflow instance
- Same Docker Compose setup
- Same monitoring stack (Prometheus)

### Running All Services
```bash
# Terminal 1: Fraud detection
python poc/serve_model.py

# Terminal 2: Sentiment analysis
python poc/serve_sentiment_model.py

# Terminal 3: Book recommendations
python poc/serve_recommendation_model.py

# All three APIs running simultaneously
```

## Use Cases

### 1. E-commerce Book Store
```python
# User views a book
GET /books/search?query=Python

# Show "Customers who viewed this also viewed"
POST /recommend {"book_id": "abc123", "num_recommendations": 5}
```

### 2. Reading List Suggestions
```python
# User adds book to reading list
# Generate "You might also like" suggestions
POST /recommend {"title": "Machine Learning", "num_recommendations": 10}
```

### 3. Category Exploration
```python
# User browses Fiction category
GET /books/search?query=Fiction&limit=20

# For each book, get similar recommendations
POST /recommend {"book_id": "...", "num_recommendations": 3}
```

## Advanced Features

### Improving Recommendations

1. **Hybrid Filtering:**
   - Combine content-based with collaborative filtering
   - Use user ratings and purchase history
   - Weight by popularity and recency

2. **Personalization:**
   - Track user preferences
   - Learn from user interactions
   - Adjust recommendations over time

3. **Diversity:**
   - Add diversity penalty to avoid too-similar results
   - Mix categories in recommendations
   - Include serendipitous suggestions

4. **Real-time Updates:**
   - Incremental updates when new books added
   - Re-train periodically with new data
   - A/B test different algorithms

### Performance Optimization

1. **Caching:**
   - Cache popular book recommendations
   - Use Redis for hot recommendations
   - TTL-based cache invalidation

2. **Precomputation:**
   - Precompute top-N for popular books
   - Store in database for instant retrieval
   - Update nightly

3. **Approximate Search:**
   - Use FAISS or Annoy for fast similarity search
   - Trade accuracy for speed
   - Good for large catalogs (>100K books)

## Troubleshooting

### Model not loading
**Error:** "Model not loaded" or 503 errors

**Solutions:**
1. Check MLflow is running: `docker-compose ps`
2. Verify model exists: Check http://localhost:5000
3. Train model if missing: `python poc/train_recommendation_model.py`

### Book not found
**Error:** 404 "Book with ID/title not found"

**Solutions:**
1. Search first: `GET /books/search?query=...`
2. Use exact book_id from search results
3. Check for typos in title

### Slow recommendations
**Issue:** Latency > 20ms

**Solutions:**
1. Reduce num_recommendations (fewer = faster)
2. Implement caching for popular books
3. Use approximate similarity (FAISS)
4. Profile with `cProfile`

### Poor recommendation quality
**Issue:** Recommendations don't seem relevant

**Solutions:**
1. Adjust feature weights in training script
2. Increase max_features in TF-IDF
3. Add more text preprocessing
4. Consider collaborative filtering

## Files Created

```
poc/
├── train_recommendation_model.py          # Training script
├── serve_recommendation_model.py          # FastAPI server
├── test_recommendation_predictions.py     # Functional tests
└── load_test_recommendation.py            # Load testing

docs/
└── RECOMMENDATION_SETUP.md                # This file

data/
└── google_books_dataset.parquet           # Book catalog
```

## Model Comparison

| Feature | Fraud Detection | Sentiment Analysis | Book Recommendations |
|---------|----------------|-------------------|---------------------|
| Model   | XGBoost        | Logistic Regression | TF-IDF + Cosine Sim |
| Input   | Structured features | Text | Text + Metadata |
| Output  | Binary         | 3-class | Top-N list |
| Latency | ~3-5ms         | ~5-10ms | ~10-20ms |
| Port    | 8000         | 8001  | 8002    |
| Data size | 10K samples  | 200K samples | 15K books |
| Training time | 2-3 min  | 2-5 min | 1-2 min |

All three models share the same MLflow infrastructure and monitoring stack.

## Next Steps

1. ✅ Train model
2. ✅ Start server
3. ✅ Run tests
4. 🔄 Add user interaction tracking
5. 🔄 Implement collaborative filtering
6. 🔄 Add caching layer
7. 🔄 Deploy to production

## References

- **Content-Based Filtering:** Recommends items similar to what user liked
- **TF-IDF:** Term Frequency-Inverse Document Frequency
- **Cosine Similarity:** Measures similarity between vectors (0-1)
- **Collaborative Filtering:** Recommends based on similar users (not implemented yet)
