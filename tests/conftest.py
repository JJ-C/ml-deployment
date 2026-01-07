import os
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_path(project_root: Path) -> Path:
    p = project_root / "data" / "credit_card_fraud_10k.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at: {p}")
    return p


@pytest.fixture(scope="session")
def raw_df(data_path: Path) -> pd.DataFrame:
    return pd.read_parquet(data_path)


@pytest.fixture(scope="session")
def feature_columns() -> list[str]:
    return [
        "amount",
        "transaction_hour",
        "merchant_category",
        "foreign_transaction",
        "location_mismatch",
        "device_trust_score",
        "velocity_last_24h",
        "cardholder_age",
    ]


@pytest.fixture()
def mlflow_file_store(tmp_path, monkeypatch):
    tracking_dir = tmp_path / "mlruns"
    tracking_uri = tracking_dir.as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    return tracking_dir
