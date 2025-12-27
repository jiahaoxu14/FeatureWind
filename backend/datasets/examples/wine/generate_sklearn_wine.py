#!/usr/bin/env python3
"""
Generate a Wine dataset from scikit-learn's load_wine.

Output: CSV with a leading `label` column (target class 0/1/2) and all
feature columns min–max normalized to [0, 1].

Usage:
  python examples/wine/generate_sklearn_wine.py
  python examples/wine/generate_sklearn_wine.py --out examples/wine/wine_sklearn_normalized.csv
  python examples/wine/generate_sklearn_wine.py --no-normalize --out examples/wine/wine_sklearn_raw.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine


def minmax01_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric, impute NaNs with column mean, then scale each feature to [0,1]."""
    df_num = df.apply(pd.to_numeric, errors='coerce')
    df_num = df_num.replace([np.inf, -np.inf], np.nan)
    means = df_num.mean(numeric_only=True)
    df_num = df_num.fillna(means)
    mins = df_num.min(numeric_only=True)
    maxs = df_num.max(numeric_only=True)
    ranges = (maxs - mins).replace(0.0, 1.0)
    for col in df_num.columns:
        mn = float(mins.get(col, 0.0))
        rg = float(ranges.get(col, 1.0))
        if rg == 0.0:
            rg = 1.0
        df_num[col] = (df_num[col] - mn) / rg
    return df_num


def main():
    ap = argparse.ArgumentParser(description='Export scikit-learn load_wine dataset to CSV')
    ap.add_argument('-o', '--out', type=str, default='', help='Output CSV path (default: examples/wine/wine_sklearn_normalized.csv)')
    ap.add_argument('--no-normalize', action='store_true', help='Do not min–max normalize features')
    args = ap.parse_args()

    data = load_wine(as_frame=True)
    X = data.data  # pandas DataFrame
    y = data.target  # pandas Series

    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    label = y.rename('label')

    if args.no_normalize:
        X_out = X
    else:
        X_out = minmax01_df(X)

    out_df = pd.concat([label.reset_index(drop=True), X_out.reset_index(drop=True)], axis=1)

    out_dir = Path('examples/wine')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / 'wine_sklearn_normalized.csv'

    out_df.to_csv(out_path, index=False)
    print(f'Saved scikit-learn Wine dataset to: {out_path}')
    print(f'Shape: {out_df.shape}')
    print('Columns:', list(out_df.columns)[:10], '...')


if __name__ == '__main__':
    main()

