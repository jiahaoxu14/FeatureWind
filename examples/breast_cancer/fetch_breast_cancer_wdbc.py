#!/usr/bin/env python3
"""
Fetch the UCI Breast Cancer Wisconsin (Diagnostic) dataset (ID=17) using ucimlrepo
and write a CSV containing only a numeric 'label' column and all feature columns.

Usage:
  python examples/breast_cancer/fetch_breast_cancer_wdbc.py

Dependencies:
  pip install ucimlrepo pandas
"""

import os
import sys


def main():
    try:
        import pandas as pd
    except Exception:
        print("Error: pandas is required. Install with: pip install pandas", file=sys.stderr)
        sys.exit(1)

    try:
        from ucimlrepo import fetch_ucirepo
    except Exception:
        print("Error: ucimlrepo is required. Install with: pip install ucimlrepo", file=sys.stderr)
        sys.exit(1)

    # Fetch dataset (UCI ID 17: Breast Cancer Wisconsin (Diagnostic))
    ds = fetch_ucirepo(id=17)
    X = ds.data.features
    y = ds.data.targets

    # Ensure pandas DataFrame/Series
    X_df = pd.DataFrame(X)
    if isinstance(y, pd.DataFrame):
        # Use the first column as the target if multiple present
        y_series = y.iloc[:, 0]
    else:
        y_series = pd.Series(y)

    # Normalize column names: ensure simple, csv-friendly names
    X_df.columns = [str(c) for c in X_df.columns]

    # Build numeric label column named 'label'
    # If y is non-numeric (e.g., 'B'/'M'), map to integer codes consistently
    if not pd.api.types.is_numeric_dtype(y_series):
        y_codes = pd.Categorical(y_series)
        label_series = pd.Series(y_codes.codes, name='label')
    else:
        label_series = pd.Series(pd.to_numeric(y_series, errors='coerce').astype('Int64')).fillna(0).astype(int)
        label_series.name = 'label'

    # Normalize all feature columns to [0, 1] (min-max), leave label untouched
    # Ensure numeric dtype for features first
    X_num = X_df.apply(pd.to_numeric, errors='coerce')
    col_min = X_num.min(axis=0)
    col_max = X_num.max(axis=0)
    denom = (col_max - col_min).replace(0, 1.0)  # avoid divide-by-zero
    X_norm = (X_num - col_min) / denom
    # Concatenate label + normalized features
    out_df = pd.concat([label_series, X_norm], axis=1)

    # Output path: alongside this script (ensures predictable location regardless of CWD)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, 'breast_cancer_wdbc.csv')

    out_df.to_csv(out_path, index=False)
    print(f"✓ Wrote {len(out_df)} rows to {out_path}")


if __name__ == '__main__':
    main()
