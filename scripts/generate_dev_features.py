#!/usr/bin/env python3
"""
Generate synthetic feature data for development/testing with user_id as entity key.
This simulates real-time user features without labels.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid

def generate_user_features(num_users=100, output_file="data/features_dev.parquet"):
    """
    Generate synthetic user-level features for development.
    
    Each user has persistent features that would be updated in real-time:
    - Profile data (age, device trust)
    - Behavioral patterns (velocity, foreign transactions)
    - Latest transaction context
    """
    
    print(f"Generating features for {num_users} users...")
    
    np.random.seed(42)
    
    # Generate user features
    data = {
        'user_id': [f"user_{i:04d}" for i in range(1, num_users + 1)],
        
        # User profile features (relatively static)
        'cardholder_age': np.random.randint(18, 80, num_users),
        'device_trust_score': np.random.randint(20, 100, num_users),
        
        # User behavior features (updated from recent transactions)
        'velocity_last_24h': np.random.randint(0, 15, num_users),
        'foreign_transaction': np.random.choice([0, 1], num_users, p=[0.85, 0.15]),
        'location_mismatch': np.random.choice([0, 1], num_users, p=[0.9, 0.1]),
        
        # Latest transaction context (for reference)
        'amount': np.random.lognormal(4, 1.5, num_users).round(2),
        'transaction_hour': np.random.randint(0, 24, num_users),
        'merchant_category': np.random.choice(
            ['Electronics', 'Travel', 'Grocery', 'Dining', 'Retail', 
             'Gas', 'Entertainment', 'Healthcare', 'Other'],
            num_users
        ),
    }
    
    # Add timestamps (recent updates)
    base_time = datetime.now() - timedelta(hours=24)
    data['timestamp'] = [
        base_time + timedelta(hours=np.random.randint(0, 24))
        for _ in range(num_users)
    ]
    data['created_timestamp'] = datetime.now()
    
    df = pd.DataFrame(data)
    
    # Add some test scenarios
    print("\nAdding test scenarios:")
    
    # High-risk users (for testing fraud detection)
    high_risk_users = [0, 1, 2, 3, 4]  # First 5 users
    df.loc[high_risk_users, 'device_trust_score'] = np.random.randint(10, 40, len(high_risk_users))
    df.loc[high_risk_users, 'velocity_last_24h'] = np.random.randint(8, 15, len(high_risk_users))
    df.loc[high_risk_users, 'foreign_transaction'] = 1
    df.loc[high_risk_users, 'location_mismatch'] = 1
    df.loc[high_risk_users, 'amount'] = np.random.uniform(2000, 5000, len(high_risk_users)).round(2)
    print(f"  - {len(high_risk_users)} high-risk users")
    
    # Low-risk users (for testing normal behavior)
    low_risk_users = [5, 6, 7, 8, 9]  # Next 5 users
    df.loc[low_risk_users, 'device_trust_score'] = np.random.randint(70, 100, len(low_risk_users))
    df.loc[low_risk_users, 'velocity_last_24h'] = np.random.randint(0, 3, len(low_risk_users))
    df.loc[low_risk_users, 'foreign_transaction'] = 0
    df.loc[low_risk_users, 'location_mismatch'] = 0
    df.loc[low_risk_users, 'amount'] = np.random.uniform(10, 100, len(low_risk_users)).round(2)
    print(f"  - {len(low_risk_users)} low-risk users")
    
    # Save to parquet
    df.to_parquet(output_file, index=False)
    
    print(f"\n✓ Generated features for {len(df)} users")
    print(f"✓ Saved to: {output_file}")
    
    print(f"\nSample user IDs:")
    for user_id in df['user_id'].head(5):
        print(f"  - {user_id}")
    
    print(f"\nFeature statistics:")
    print(df[['cardholder_age', 'device_trust_score', 'velocity_last_24h']].describe())
    
    return df

def generate_test_users(output_file="data/features_test_users.parquet"):
    """
    Generate specific test users with known user_ids for testing.
    """
    
    print("\nGenerating known test users...")
    
    test_users = [
        {
            'user_id': 'test_user_normal',
            'cardholder_age': 35,
            'device_trust_score': 85,
            'velocity_last_24h': 2,
            'foreign_transaction': 0,
            'location_mismatch': 0,
            'amount': 45.99,
            'transaction_hour': 14,
            'merchant_category': 'Grocery',
            'description': 'Normal user with good history'
        },
        {
            'user_id': 'test_user_suspicious',
            'cardholder_age': 28,
            'device_trust_score': 25,
            'velocity_last_24h': 10,
            'foreign_transaction': 1,
            'location_mismatch': 1,
            'amount': 2500.00,
            'transaction_hour': 3,
            'merchant_category': 'Electronics',
            'description': 'Suspicious user with high-risk patterns'
        },
        {
            'user_id': 'test_user_high_value',
            'cardholder_age': 45,
            'device_trust_score': 90,
            'velocity_last_24h': 1,
            'foreign_transaction': 0,
            'location_mismatch': 0,
            'amount': 5000.00,
            'transaction_hour': 10,
            'merchant_category': 'Travel',
            'description': 'High-value but legitimate user'
        },
        {
            'user_id': 'test_user_velocity',
            'cardholder_age': 25,
            'device_trust_score': 60,
            'velocity_last_24h': 15,
            'foreign_transaction': 0,
            'location_mismatch': 0,
            'amount': 150.00,
            'transaction_hour': 18,
            'merchant_category': 'Retail',
            'description': 'High velocity user (many transactions)'
        },
        {
            'user_id': 'test_user_new',
            'cardholder_age': 22,
            'device_trust_score': 50,
            'velocity_last_24h': 0,
            'foreign_transaction': 0,
            'location_mismatch': 0,
            'amount': 30.00,
            'transaction_hour': 12,
            'merchant_category': 'Dining',
            'description': 'New user with no history'
        },
    ]
    
    df = pd.DataFrame(test_users)
    
    # Add timestamps
    base_time = datetime.now() - timedelta(hours=1)
    df['timestamp'] = [base_time + timedelta(minutes=i*10) for i in range(len(df))]
    df['created_timestamp'] = datetime.now()
    
    # Save
    df.to_parquet(output_file, index=False)
    
    print(f"✓ Generated {len(df)} test users")
    print(f"✓ Saved to: {output_file}")
    print("\nTest users:")
    for _, row in df.iterrows():
        print(f"  - {row['user_id']}: {row['description']}")
    
    return df

if __name__ == "__main__":
    print("=" * 60)
    print("Development Feature Data Generator (User-based)")
    print("=" * 60)
    
    # Generate main dev feature dataset
    dev_df = generate_user_features(num_users=100)
    
    # Generate known test users
    test_df = generate_test_users()
    
    # Combine for Feast
    combined_df = pd.concat([dev_df, test_df], ignore_index=True)
    combined_df.to_parquet("data/features_combined.parquet", index=False)
    
    print("\n" + "=" * 60)
    print("✓ Feature Data Generation Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - data/features_dev.parquet (100 users)")
    print("  - data/features_test_users.parquet (5 known test users)")
    print("  - data/features_combined.parquet (all combined)")
    print("\nNext steps:")
    print("  1. Update Feast to use features_combined.parquet")
    print("  2. Run: python scripts/feast_setup.py")
    print("  3. Run: python scripts/feast_materialize.py")
    print("  4. Test with known user_ids from test users")
    print("\nExample test request:")
    print('  curl -X POST http://localhost:8000/predict \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"transaction_id": "txn_123", "user_id": "test_user_normal"}\'')
    print("=" * 60)
