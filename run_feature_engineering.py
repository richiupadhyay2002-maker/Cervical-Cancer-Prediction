#!/usr/bin/env python3
"""
Direct Feature Engineering Script
Creates processed_data/*.npy files from raw data
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

print("="*70)
print("FEATURE ENGINEERING PIPELINE")
print("="*70)

# Paths
DATA_PATH = Path("data/raw/kag_risk_factors_cervical_cancer.csv")
OUTPUT_DIR = Path("processed_data")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\n✓ Output directory: {OUTPUT_DIR}")

# Step 1: Load and clean data
print("\n" + "="*70)
print("STEP 1: LOAD AND CLEAN DATA")
print("="*70)

df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# Clean column names
df.columns = df.columns.str.strip()

# Replace '?' with NaN
df = df.replace('?', np.nan)

# Convert columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"✓ Data cleaned: {df.shape[0]} rows × {df.shape[1]} columns")
print("✓ Missing values retained for fold-local imputation during model training")

# Step 2: Feature selection
print("\n" + "="*70)
print("STEP 2: FEATURE SELECTION")
print("="*70)

target_col = 'Biopsy'
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"✓ Features selected: {X.shape[1]} features")
print(f"✓ Target variable: {target_col}")
print(f"  - Class 0 (Negative): {(y == 0).sum()} samples")
print(f"  - Class 1 (Positive): {(y == 1).sum()} samples")
print(f"  - Imbalance ratio: {(y == 0).sum() / (y == 1).sum():.2f}:1")

# Step 3: Train-validation-test split
print("\n" + "="*70)
print("STEP 3: TRAIN/VALIDATION/TEST SPLIT")
print("="*70)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"✓ Train set: {X_train.shape[0]} samples (70%)")
print(f"✓ Validation set: {X_val.shape[0]} samples (15%)")
print(f"✓ Test set: {X_test.shape[0]} samples (15%)")
print(f"\n📊 Original Training Set Distribution:")
print(f"  - Class 0 (Negative): {(y_train == 0).sum()} samples")
print(f"  - Class 1 (Positive): {(y_train == 1).sum()} samples")

# Step 4: Defer learned preprocessing
print("\n" + "="*70)
print("STEP 4: DEFER PREPROCESSING TO MODEL TRAINING")
print("="*70)
print("✓ Imputation, scaling, and SMOTE are fitted inside each CV training fold")

# Step 5: Save raw split data
print("\n" + "="*70)
print("STEP 5: SAVE RAW SPLIT DATA")
print("="*70)

np.save(OUTPUT_DIR / 'X_train.npy', X_train.to_numpy(dtype=np.float64))
np.save(OUTPUT_DIR / 'X_val.npy', X_val.to_numpy(dtype=np.float64))
np.save(OUTPUT_DIR / 'X_test.npy', X_test.to_numpy(dtype=np.float64))
np.save(OUTPUT_DIR / 'y_train.npy', y_train.to_numpy(dtype=np.int64))
np.save(OUTPUT_DIR / 'y_val.npy', y_val.to_numpy(dtype=np.int64))
np.save(OUTPUT_DIR / 'y_test.npy', y_test.to_numpy(dtype=np.int64))

with open(OUTPUT_DIR / 'feature_columns.json', 'w') as f:
    json.dump(list(X.columns), f, indent=2)

print(f"✓ Saved processed data:")
print(f"  - {OUTPUT_DIR / 'X_train.npy'}: {X_train.shape}")
print(f"  - {OUTPUT_DIR / 'X_val.npy'}: {X_val.shape}")
print(f"  - {OUTPUT_DIR / 'X_test.npy'}: {X_test.shape}")
print(f"  - {OUTPUT_DIR / 'y_train.npy'}: {y_train.shape}")
print(f"  - {OUTPUT_DIR / 'y_val.npy'}: {y_val.shape}")
print(f"  - {OUTPUT_DIR / 'y_test.npy'}: {y_test.shape}")
print(f"  - {OUTPUT_DIR / 'feature_columns.json'}: {len(X.columns)} features")

# Summary
print("\n" + "="*70)
print("FEATURE ENGINEERING SUMMARY")
print("="*70)

print(f"\n📊 Final Dataset:")
print(f"   - Training samples: {X_train.shape[0]}")
print(f"   - Validation samples: {X_val.shape[0]}")
print(f"   - Test samples: {X_test.shape[0]}")
print(f"   - Features: {X_train.shape[1]}")

print(f"\n🎯 Target Distribution:")
print(f"   - Training: {(y_train == 0).sum()} negative, {(y_train == 1).sum()} positive")
print(f"   - Validation: {(y_val == 0).sum()} negative, {(y_val == 1).sum()} positive")
print(f"   - Test (original): {(y_test == 0).sum()} negative, {(y_test == 1).sum()} positive (IMBALANCED)")

print(f"\n⚠️  IMPORTANT NOTE:")
print(f"   - All saved splits retain their original class distributions")
print(f"   - Test data remains in its original distribution for realistic evaluation")
print(f"   - SMOTE, when used, is applied only inside CV training folds")

print(f"\n✅ Preprocessing Steps Completed:")
print(f"   1. Data cleaning (missing values retained)")
print(f"   2. Feature selection ({X_train.shape[1]} features)")
print(f"   3. Train/validation/test split (70/15/15)")
print(f"   4. Learned preprocessing deferred to model training pipelines")
print(f"   5. SMOTE deferred to CV training folds")

print("\n" + "="*70)
print("✓ FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("="*70)
print("\nNext step: Run 03_model_training.ipynb")