#!/usr/bin/env python3
"""
Preprocess SUPPORT2 dataset for FeatureWind tmap generation.
  - Label: hospdead
  - Categorical columns factorized: sex, dzgroup, dzclass, income, race, ca, dnr
  - Missing values imputed: median for numeric, mode for categorical
  - All feature columns normalized to [0, 1]
"""
import os
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

CATEGORICAL_COLS = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca', 'dnr']
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'support2_hospdead.csv')

print("Fetching SUPPORT2 from UCI (id=880)…")
ds = fetch_ucirepo(id=880)
X = pd.DataFrame(ds.data.features)
y = pd.DataFrame(ds.data.targets)

label = y['hospdead'].astype(int)
print(f"Label (hospdead) distribution:\n{label.value_counts().to_string()}")
print(f"Dataset shape: {X.shape}")

# ── Factorize categorical columns ─────────────────────────────────────────────
for col in CATEGORICAL_COLS:
    if col in X.columns:
        # Impute missing with most frequent before factorizing
        mode_val = X[col].mode(dropna=True)
        if len(mode_val):
            X[col] = X[col].fillna(mode_val[0])
        codes, _ = pd.factorize(X[col], sort=True)
        X[col] = codes.astype(float)
        print(f"  Factorized '{col}'")

# ── Impute remaining numeric columns with median ───────────────────────────────
for col in X.columns:
    if X[col].isna().any():
        med = X[col].median()
        X[col] = X[col].fillna(med if pd.notna(med) else 0.0)

X = X.astype(float)

# ── Normalize all features to [0, 1] ──────────────────────────────────────────
for col in X.columns:
    vmin, vmax = X[col].min(), X[col].max()
    if vmax > vmin:
        X[col] = (X[col] - vmin) / (vmax - vmin)
    else:
        X[col] = 0.0

# ── Save ───────────────────────────────────────────────────────────────────────
out = pd.concat([label.rename('label'), X], axis=1)
out.to_csv(OUT_PATH, index=False)
print(f"\n✓ Saved {len(out)} rows × {len(out.columns)} cols to {OUT_PATH}")
print(f"  NaN remaining: {out.isna().sum().sum()}")
