#!/usr/bin/env python3
"""
Train content-based book recommendation model.
Uses TF-IDF on book features + cosine similarity, logs to MLflow.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import mlflow.sklearn
import pickle
from pathlib import Path

print("=" * 60)
print("Book Recommendation Model Training")
print("=" * 60)

# Configuration
MLFLOW_TRACKING_URI = "http://localhost:5001"
MLFLOW_EXPERIMENT = "book_recommendations"
MAX_FEATURES = 5000

# Set MLflow tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

print("\n[1/5] Loading data...")
df = pd.read_parquet("data/google_books_dataset.parquet")
print(f"  - Total books: {len(df)}")
print(f"  - Features: {len(df.columns)} columns")

print("\n[2/5] Preprocessing data...")

# Fill missing values
df['title'] = df['title'].fillna('')
df['subtitle'] = df['subtitle'].fillna('')
df['authors'] = df['authors'].fillna('')
df['description'] = df['description'].fillna('')
df['categories'] = df['categories'].fillna('')
df['publisher'] = df['publisher'].fillna('')
df['search_category'] = df['search_category'].fillna('')

# Create combined text feature for content-based filtering
# Weight different fields by importance
df['combined_features'] = (
    df['title'] + ' ' + df['title'] + ' ' +  # Title twice (more important)
    df['subtitle'] + ' ' +
    df['authors'] + ' ' + df['authors'] + ' ' +  # Authors twice
    df['categories'] + ' ' + df['categories'] + ' ' + df['categories'] + ' ' +  # Categories 3x
    df['search_category'] + ' ' + df['search_category'] + ' ' +  # Search category 2x
    df['description']
)

# Clean the combined features
df['combined_features'] = df['combined_features'].str.lower().str.strip()

# Remove books with no useful features
df = df[df['combined_features'].str.len() > 10].reset_index(drop=True)

print(f"  - Books after cleaning: {len(df)}")
print(f"  - Average feature length: {df['combined_features'].str.len().mean():.0f} chars")

# Category distribution
print("\n  Top categories:")
for cat, count in df['categories'].value_counts().head(5).items():
    print(f"    {cat}: {count}")

print("\n[3/5] Building TF-IDF model...")

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8,
    strip_accents='unicode',
    lowercase=True,
    stop_words='english'
)

# Fit and transform
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

print(f"  - TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"  - Vocabulary size: {len(tfidf.vocabulary_)}")
print(f"  - Matrix sparsity: {(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100:.2f}%")

print("\n[4/5] Computing similarity metrics...")

# For validation, compute some sample similarities
sample_idx = 0
sample_book = df.iloc[sample_idx]
print(f"\n  Sample book: '{sample_book['title']}'")
print(f"  Category: {sample_book['categories']}")
print(f"  Authors: {sample_book['authors']}")

# Compute cosine similarity for sample
cosine_sim = cosine_similarity(tfidf_matrix[sample_idx:sample_idx+1], tfidf_matrix).flatten()
similar_indices = cosine_sim.argsort()[-6:-1][::-1]  # Top 5 similar (excluding itself)

print(f"\n  Top 5 similar books:")
for i, idx in enumerate(similar_indices, 1):
    sim_book = df.iloc[idx]
    print(f"    {i}. '{sim_book['title']}' (similarity: {cosine_sim[idx]:.3f})")
    print(f"       Category: {sim_book['categories']}, Authors: {sim_book['authors']}")

print("\n[5/5] Logging to MLflow...")

with mlflow.start_run(run_name="content_based_tfidf"):
    # Log parameters
    mlflow.log_param("model_type", "Content-Based (TF-IDF + Cosine Similarity)")
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("ngram_range", "(1, 2)")
    mlflow.log_param("total_books", len(df))
    mlflow.log_param("vocabulary_size", len(tfidf.vocabulary_))
    mlflow.log_param("matrix_sparsity", f"{(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100:.2f}%")
    
    # Log metrics
    mlflow.log_metric("num_books", len(df))
    mlflow.log_metric("avg_feature_length", df['combined_features'].str.len().mean())
    mlflow.log_metric("vocab_size", len(tfidf.vocabulary_))
    
    # Save the model components
    # We need: 1) TF-IDF vectorizer, 2) TF-IDF matrix, 3) Book metadata
    model_artifacts = {
        'vectorizer': tfidf,
        'tfidf_matrix': tfidf_matrix,
        'books_df': df[['book_id', 'title', 'authors', 'categories', 'description', 
                        'average_rating', 'ratings_count', 'thumbnail', 'combined_features']].copy()
    }
    
    # Save as pickle
    with open('recommendation_model.pkl', 'wb') as f:
        pickle.dump(model_artifacts, f)
    
    mlflow.log_artifact('recommendation_model.pkl')
    
    # Also log the vectorizer separately for inspection
    mlflow.sklearn.log_model(tfidf, "tfidf_vectorizer")
    
    run_id = mlflow.active_run().info.run_id
    print(f"  ✓ Logged to MLflow (run_id: {run_id})")

print("\n" + "=" * 60)
print("✓ Training Complete!")
print("=" * 60)
print(f"\nModel Details:")
print(f"  - Books indexed: {len(df)}")
print(f"  - Features: {len(tfidf.vocabulary_)}")
print(f"  - Model type: Content-Based Filtering")
print(f"\nMLflow:")
print(f"  - Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"  - Experiment: {MLFLOW_EXPERIMENT}")
print(f"  - Run ID: {run_id}")
print("\nNext steps:")
print("  1. View results: http://localhost:5000")
print("  2. Start serving: python poc/serve_recommendation_model.py")
print("  3. Test API: python poc/test_recommendation_predictions.py")
print("=" * 60)
