from datetime import timedelta
from pathlib import Path
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

# Define the user entity
user = Entity(
    name="user",
    join_keys=["user_id"],
    description="User entity for fraud detection - features are keyed by user_id"
)

# Define the data source (Parquet file)
# Use absolute path from project root
project_root = Path(__file__).parent.parent
data_path = str(project_root / "data" / "credit_card_fraud_10k.parquet")

user_source = FileSource(
    path=data_path,
    timestamp_field="timestamp",
    created_timestamp_column="created_timestamp",
)

# Define feature view for user features
# These are user-level features that persist across transactions
user_features = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=365),
    schema=[
        # User profile features
        Field(name="cardholder_age", dtype=Int64),
        Field(name="device_trust_score", dtype=Float64),
        
        # User behavior features (aggregated from recent transactions)
        Field(name="velocity_last_24h", dtype=Int64),
        Field(name="foreign_transaction", dtype=Int64),
        Field(name="location_mismatch", dtype=Int64),
        
        # Latest transaction context (for reference)
        Field(name="amount", dtype=Float64),
        Field(name="transaction_hour", dtype=Int64),
        Field(name="merchant_category", dtype=String),
    ],
    online=True,
    source=user_source,
    tags={"team": "fraud_detection", "use_case": "fraud_model"},
)
