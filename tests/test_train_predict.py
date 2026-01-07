import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def _make_xy(raw_df, feature_columns):
    X = raw_df[feature_columns].copy()
    X["merchant_category"] = X["merchant_category"].astype("category").cat.codes
    y = raw_df["is_fraud"].astype(int)
    return X, y


def test_train_predict_shapes_and_ranges(raw_df, feature_columns):
    df = raw_df.sample(n=min(2000, len(raw_df)), random_state=42)
    X, y = _make_xy(df, feature_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=50,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        tree_method="hist",
    )
    model.fit(X_train, y_train, verbose=False)

    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))

    auc = roc_auc_score(y_test, proba[:, 1])
    assert 0.5 <= auc <= 1.0


def test_stratified_split_preserves_rate(raw_df, feature_columns):
    df = raw_df.sample(n=min(5000, len(raw_df)), random_state=7)
    X, y = _make_xy(df, feature_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_rate = float(y_train.mean())
    test_rate = float(y_test.mean())
    assert abs(train_rate - test_rate) < 0.01

    assert set(X_train.index).isdisjoint(set(X_test.index))
