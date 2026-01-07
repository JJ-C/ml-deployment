import pandas as pd

df = pd.read_parquet('data/credit_card_fraud_10k.parquet')

feature_columns = [
    'amount', 'transaction_hour', 'merchant_category', 
    'foreign_transaction', 'location_mismatch', 
    'device_trust_score', 'velocity_last_24h', 'cardholder_age'
]

X = df[feature_columns].copy()
X['merchant_category'] = X['merchant_category'].astype('category').cat.codes

print("Data types after preprocessing (matching training):")
print(X.dtypes)
print("\nSample row:")
print(X.iloc[0])
