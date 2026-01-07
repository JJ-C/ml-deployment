import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report
import xgboost as xgb
from datetime import datetime
import os

print("=" * 60)
print("Training Fraud Detection Model (POC)")
print("=" * 60)

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("fraud_detection_poc")

print("\n[1/7] Loading data...")
data_file = "data/credit_card_fraud_10k.parquet"
df = pd.read_parquet(data_file)
print(f"✓ Loaded {len(df)} transactions from Parquet")
print(f"  - Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
print(f"  - Legitimate: {(~df['is_fraud'].astype(bool)).sum()} ({(1-df['is_fraud'].mean())*100:.2f}%)")

print("\n[2/7] Preparing features...")
feature_columns = [
    'amount', 'transaction_hour', 'merchant_category', 
    'foreign_transaction', 'location_mismatch', 
    'device_trust_score', 'velocity_last_24h', 'cardholder_age'
]

X = df[feature_columns].copy()
X['merchant_category'] = X['merchant_category'].astype('category').cat.codes
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Train set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

print("\n[3/7] Training XGBoost model...")
with mlflow.start_run(run_name=f"fraud_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'tree_method': 'hist'
    }
    
    mlflow.log_params(params)
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    
    print("✓ Model trained")
    
    print("\n[4/7] Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"  - AUC: {auc:.4f}")
    print(f"  - F1 Score: {f1:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    
    mlflow.log_metric("auc", auc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    print("\n[5/7] Logging model to MLflow...")
    mlflow.sklearn.log_model(
        model, 
        "model",
        signature=mlflow.models.infer_signature(X_train, y_pred_proba)
    )
    
    model_uri = mlflow.get_artifact_uri("model")
    print(f"✓ Model logged to: {model_uri}")
    
    print("\n[6/7] Registering model...")
    try:
        result = mlflow.register_model(model_uri, "fraud_detector")
        version = result.version
        print(f"✓ Model registered: fraud_detector version {version}")
        
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        client.update_model_version(
            name="fraud_detector",
            version=version,
            description=f"XGBoost fraud detection model - AUC: {auc:.4f}, F1: {f1:.4f}"
        )
        
        client.transition_model_version_stage(
            name="fraud_detector",
            version=version,
            stage="Production"
        )
        print(f"✓ Model promoted to Production stage")
        
    except Exception as e:
        print(f"⚠ Model registration: {e}")
    
    print("\n[7/7] Converting to ONNX for faster inference...")
    try:
        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import FloatTensorType
        
        initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
        onx = to_onnx(model, X_train[:1].values.astype(np.float32), target_opset=12)
        
        os.makedirs("poc/models", exist_ok=True)
        onnx_path = "poc/models/fraud_detector.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onx.SerializeToString())
        
        mlflow.log_artifact(onnx_path)
        print(f"✓ ONNX model saved to: {onnx_path}")
        
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        test_input = X_test[:1].values.astype(np.float32)
        onnx_pred = sess.run(None, {'float_input': test_input})
        print("✓ ONNX model validated")
        
    except Exception as e:
        print(f"⚠ ONNX conversion: {e}")

print("\n" + "=" * 60)
print("✓ Training Complete!")
print("=" * 60)
print("\nNext steps:")
print("1. View results in MLflow: http://localhost:5001")
print("2. Populate feature store: python poc/populate_features.py")
print("3. Start model server: python poc/serve_model.py")
print("=" * 60)
