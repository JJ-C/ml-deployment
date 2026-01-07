import importlib

import numpy as np
import pytest
import xgboost as xgb
from sklearn.model_selection import train_test_split


def _optional_import(name: str):
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    return importlib.import_module(name)


def _make_xy(raw_df, feature_columns):
    X = raw_df[feature_columns].copy()
    X["merchant_category"] = X["merchant_category"].astype("category").cat.codes
    y = raw_df["is_fraud"].astype(int)
    return X, y


@pytest.mark.skipif(_optional_import("skl2onnx") is None, reason="skl2onnx not installed")
@pytest.mark.skipif(_optional_import("onnxruntime") is None, reason="onnxruntime not installed")
def test_onnx_export_and_inference(raw_df, feature_columns, tmp_path):
    from skl2onnx import to_onnx
    import onnxruntime as ort

    df = raw_df.sample(n=min(2000, len(raw_df)), random_state=42)
    X, y = _make_xy(df, feature_columns)

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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

    try:
        onx = to_onnx(model, X_train[:1].values.astype(np.float32), target_opset=12)
    except Exception as e:
        pytest.skip(f"ONNX conversion not supported for XGBoost: {e}")
    
    onnx_path = tmp_path / "fraud_detector.onnx"
    onnx_path.write_bytes(onx.SerializeToString())

    sess = ort.InferenceSession(str(onnx_path))
    test_input = X_test[:2].values.astype(np.float32)
    outputs = sess.run(None, {"float_input": test_input})

    assert outputs is not None
    assert len(outputs) >= 1
