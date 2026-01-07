import numpy as np


def test_required_columns_present(raw_df, feature_columns):
    missing = [c for c in (feature_columns + ["is_fraud"]) if c not in raw_df.columns]
    assert missing == []


def test_label_binary(raw_df):
    vals = set(raw_df["is_fraud"].dropna().unique().tolist())
    assert vals.issubset({0, 1, False, True})


def test_no_nulls_in_features(raw_df, feature_columns):
    null_counts = raw_df[feature_columns].isna().sum()
    bad = null_counts[null_counts > 0]
    assert bad.empty, f"Nulls found in: {bad.to_dict()}"


def test_numeric_features_finite(raw_df):
    numeric_cols = [
        "amount",
        "transaction_hour",
        "device_trust_score",
        "velocity_last_24h",
        "cardholder_age",
    ]
    arr = raw_df[numeric_cols].to_numpy(dtype=float)
    assert np.isfinite(arr).all()
