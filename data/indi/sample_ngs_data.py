#!/usr/bin/env python3
"""
Script to randomly sample data from ngs_train_500k.csv
Creates two CSV files:
- ngs_train_1k.csv: 1,000 randomly sampled rows
- ngs_train_100k.csv: 100,000 randomly sampled rows
"""

import pandas as pd
import numpy as np
import os

def sample_ngs_data():
    # Read the original CSV file
    print("Reading ngs_train_500k.csv...")
    df = pd.read_csv('ngs_train_500k.csv')
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Sample 1k rows
    print("\nSampling 1,000 rows...")
    df_1k = df.sample(n=1000, random_state=42)
    
    # Sample 100k rows
    print("Sampling 100,000 rows...")
    df_100k = df.sample(n=100000, random_state=42)
    
    # Save the sampled datasets
    print("\nSaving sampled datasets...")
    df_1k.to_csv('ngs_train_1k.csv', index=False)
    df_100k.to_csv('ngs_train_100k.csv', index=False)
    
    print(f"Saved ngs_train_1k.csv with {len(df_1k)} rows")
    print(f"Saved ngs_train_100k.csv with {len(df_100k)} rows")
    
    # Display some statistics
    print("\nDataset Statistics:")
    print(f"1k sample - Length range: {df_1k['Length'].min()} to {df_1k['Length'].max()}")
    print(f"100k sample - Length range: {df_100k['Length'].min()} to {df_100k['Length'].max()}")
    
    print(f"\n1k sample - DB distribution:")
    print(df_1k['DB'].value_counts())
    
    print(f"\n100k sample - DB distribution:")
    print(df_100k['DB'].value_counts())

if __name__ == "__main__":
    sample_ngs_data() 