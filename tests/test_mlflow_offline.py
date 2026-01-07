import mlflow
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def _make_xy(raw_df, feature_columns):
    X = raw_df[feature_columns].copy()
    X["merchant_category"] = X["merchant_category"].astype("category").cat.codes
    y = raw_df["is_fraud"].astype(int)
    return X, y


def test_mlflow_logs_run_and_metrics_offline(raw_df, feature_columns, mlflow_file_store):
    df = raw_df.sample(n=min(2000, len(raw_df)), random_state=42)
    X, y = _make_xy(df, feature_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("fraud_detection_poc_tests")

    with mlflow.start_run(run_name="pytest_offline_run") as run:
        model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=20,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            tree_method="hist",
        )
        model.fit(X_train, y_train, verbose=False)

        proba = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))

        mlflow.log_metric("auc", auc)
        mlflow.log_metric("proba_mean", float(np.mean(proba)))

    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run.info.run_id).data

    assert "auc" in data.metrics
    assert 0.5 <= data.metrics["auc"] <= 1.0

    # file store should have been created
    assert mlflow_file_store.exists()
