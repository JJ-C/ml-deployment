#!/usr/bin/env python3
"""
Train sentiment analysis model on Twitter and Reddit data.
Uses TF-IDF + Logistic Regression, logs to MLflow.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
from pathlib import Path

print("=" * 60)
print("Sentiment Analysis Model Training")
print("=" * 60)

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 10000
MLFLOW_TRACKING_URI = "http://localhost:5001"
MLFLOW_EXPERIMENT = "sentiment_analysis"

# Set MLflow tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

print("\n[1/6] Loading data...")

# Load Twitter data
twitter_df = pd.read_parquet("data/Twitter_Data.parquet")
twitter_df = twitter_df.dropna()
twitter_df = twitter_df.rename(columns={'clean_text': 'text'})
print(f"  - Twitter: {len(twitter_df)} samples")

# Load Reddit data
reddit_df = pd.read_parquet("data/Reddit_Data.parquet")
reddit_df = reddit_df.dropna()
reddit_df = reddit_df.rename(columns={'clean_comment': 'text'})
print(f"  - Reddit: {len(reddit_df)} samples")

# Combine datasets
df = pd.concat([twitter_df, reddit_df], ignore_index=True)
print(f"  - Combined: {len(df)} samples")

# Convert category to int
df['category'] = df['category'].astype(int)

print("\n[2/6] Data statistics...")
print(f"  - Total samples: {len(df)}")
print(f"  - Label distribution:")
label_counts = df['category'].value_counts().sort_index()
for label, count in label_counts.items():
    label_name = {-1: "Negative", 0: "Neutral", 1: "Positive"}[label]
    print(f"    {label_name:8s} ({label:2d}): {count:6d} ({count/len(df)*100:.1f}%)")

print("\n[3/6] Splitting data...")
X = df['text']
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"  - Training set: {len(X_train)} samples")
print(f"  - Test set: {len(X_test)} samples")

print("\n[4/6] Training model...")
print(f"  - Vectorizer: TF-IDF (max_features={MAX_FEATURES})")
print(f"  - Classifier: Logistic Regression")

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents='unicode',
        lowercase=True
    )),
    ('classifier', LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        solver='saga',
        n_jobs=-1
    ))
])

# Train
pipeline.fit(X_train, y_train)
print("  ✓ Model trained")

print("\n[5/6] Evaluating model...")

# Predictions
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# Metrics
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"  - Training accuracy: {train_accuracy:.4f}")
print(f"  - Test accuracy: {test_accuracy:.4f}")

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred_test, labels=[-1, 0, 1], average=None
)

print("\n  Per-class metrics (test set):")
for i, label in enumerate([-1, 0, 1]):
    label_name = {-1: "Negative", 0: "Neutral", 1: "Positive"}[label]
    print(f"    {label_name:8s}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}, N={support[i]}")

# Macro averages
macro_precision = precision.mean()
macro_recall = recall.mean()
macro_f1 = f1.mean()

print(f"\n  Macro averages:")
print(f"    Precision: {macro_precision:.4f}")
print(f"    Recall: {macro_recall:.4f}")
print(f"    F1-score: {macro_f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test, labels=[-1, 0, 1])
print("\n  Confusion Matrix:")
print("              Predicted")
print("              Neg  Neu  Pos")
print(f"    Actual Neg {cm[0][0]:4d} {cm[0][1]:4d} {cm[0][2]:4d}")
print(f"           Neu {cm[1][0]:4d} {cm[1][1]:4d} {cm[1][2]:4d}")
print(f"           Pos {cm[2][0]:4d} {cm[2][1]:4d} {cm[2][2]:4d}")

print("\n[6/6] Logging to MLflow...")

with mlflow.start_run(run_name="sentiment_logistic_regression"):
    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("vectorizer", "TfidfVectorizer")
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("ngram_range", "(1, 2)")
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))
    mlflow.log_param("twitter_samples", len(twitter_df))
    mlflow.log_param("reddit_samples", len(reddit_df))
    
    # Log metrics
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("macro_precision", macro_precision)
    mlflow.log_metric("macro_recall", macro_recall)
    mlflow.log_metric("macro_f1", macro_f1)
    
    # Log per-class metrics
    for i, label in enumerate([-1, 0, 1]):
        label_name = {-1: "negative", 0: "neutral", 1: "positive"}[label]
        mlflow.log_metric(f"{label_name}_precision", precision[i])
        mlflow.log_metric(f"{label_name}_recall", recall[i])
        mlflow.log_metric(f"{label_name}_f1", f1[i])
    
    # Log model
    mlflow.sklearn.log_model(
        pipeline,
        "model",
        registered_model_name="sentiment_analyzer"
    )
    
    # Log artifacts
    # Save classification report
    report = classification_report(y_test, y_pred_test, 
                                   target_names=['Negative', 'Neutral', 'Positive'])
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")
    
    run_id = mlflow.active_run().info.run_id
    print(f"  ✓ Logged to MLflow (run_id: {run_id})")

print("\n" + "=" * 60)
print("✓ Training Complete!")
print("=" * 60)
print(f"\nModel Performance:")
print(f"  - Test Accuracy: {test_accuracy:.2%}")
print(f"  - Macro F1: {macro_f1:.4f}")
print(f"\nMLflow:")
print(f"  - Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"  - Experiment: {MLFLOW_EXPERIMENT}")
print(f"  - Run ID: {run_id}")
print("\nNext steps:")
print("  1. View results: http://localhost:5000")
print("  2. Start serving: python poc/serve_sentiment_model.py")
print("  3. Test API: python poc/test_sentiment_predictions.py")
print("=" * 60)
