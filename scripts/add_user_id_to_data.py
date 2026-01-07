#!/usr/bin/env python3
"""
Add user_id column to existing training data.
Simulates realistic scenario where multiple transactions belong to same users.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def add_user_ids(input_file, output_file, num_users=1000, start_id=1000):
    """
    Add user_id column to transaction data.
    
    Args:
        input_file: Path to input parquet file
        output_file: Path to output parquet file
        num_users: Number of unique users to simulate (default 1000)
        start_id: Starting user ID (default 1000, creates IDs 1000-1999)
    """
    
    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file)
    
    print(f"Original data: {len(df)} transactions")
    
    # Assign user IDs
    # Use modulo to ensure multiple transactions per user
    # This simulates realistic behavior where users make multiple transactions
    np.random.seed(42)
    
    # Create user IDs with some clustering (some users more active than others)
    # Use power law distribution to simulate realistic user activity
    probs = np.random.power(0.5, num_users)
    probs = probs / probs.sum()  # Normalize to ensure sum is exactly 1.0
    
    user_ids = np.random.choice(
        range(start_id, start_id + num_users),
        size=len(df),
        replace=True,
        p=probs
    )
    
    df['user_id'] = [f"user_{uid:04d}" for uid in user_ids]
    
    # Reorder columns to put user_id near the front
    cols = df.columns.tolist()
    if 'transaction_id' in cols:
        # Put user_id right after transaction_id
        idx = cols.index('transaction_id')
        cols.insert(idx + 1, cols.pop(cols.index('user_id')))
    else:
        # Put user_id at the front
        cols = ['user_id'] + [c for c in cols if c != 'user_id']
    
    df = df[cols]
    
    # Save
    df.to_parquet(output_file, index=False)
    
    print(f"\n✓ Added user_id column")
    print(f"✓ Saved to: {output_file}")
    print(f"\nStatistics:")
    print(f"  - Total transactions: {len(df)}")
    print(f"  - Unique users: {df['user_id'].nunique()}")
    print(f"  - Avg transactions per user: {len(df) / df['user_id'].nunique():.1f}")
    print(f"  - Max transactions per user: {df['user_id'].value_counts().max()}")
    print(f"  - Min transactions per user: {df['user_id'].value_counts().min()}")
    
    print(f"\nTop 5 most active users:")
    print(df['user_id'].value_counts().head())
    
    print(f"\nSample data:")
    print(df[['transaction_id', 'user_id', 'amount', 'cardholder_age']].head(10))
    
    return df

if __name__ == "__main__":
    print("=" * 60)
    print("Adding user_id to Training Data")
    print("=" * 60)
    
    # Add user_id to training data
    # Note: This will overwrite the original file. Run only once!
    df = add_user_ids(
        input_file="data/credit_card_fraud_10k.parquet",
        output_file="data/credit_card_fraud_10k.parquet",
        num_users=1000,
        start_id=1001
    )
    
    print("\n" + "=" * 60)
    print("✓ Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Update Feast feature definitions to use user entity")
    print("  2. Run: python scripts/feast_setup.py")
    print("  3. Run: python scripts/feast_materialize.py")
    print("=" * 60)
