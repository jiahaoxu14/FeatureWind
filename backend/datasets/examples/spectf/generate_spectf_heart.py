#!/usr/bin/env python3
"""
Fetch and prepare the UCI SPECTF Heart dataset (id=96) via ucimlrepo.

Output: a CSV with a leading `label` column (target) and all feature columns
min–max normalized to [0, 1].

Usage:
  python examples/spectf/generate_spectf_heart.py
  python examples/spectf/generate_spectf_heart.py --out examples/spectf/spectf_heart.csv

Notes:
- Requires `ucimlrepo` package. Install with: pip install ucimlrepo
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def minmax01_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to numeric, impute NaNs with column mean, then scale each
    numeric feature to [0, 1] using min–max (constant columns -> 0.0)."""
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
    ap = argparse.ArgumentParser(description='Fetch UCI SPECTF Heart (id=96) dataset')
    ap.add_argument('-o', '--out', type=str, default='', help='Output CSV path (default: examples/spectf/spectf_heart.csv)')
    args = ap.parse_args()

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print('Error: ucimlrepo not installed. Install with `pip install ucimlrepo`.', file=sys.stderr)
        sys.exit(1)

    # Fetch dataset
    spectf_heart = fetch_ucirepo(id=96)
    X = spectf_heart.data.features  # pandas DataFrame
    y = spectf_heart.data.targets   # pandas DataFrame

    # Determine label series
    if y is None or y.shape[1] == 0:
        print('Warning: No targets found; creating a dummy label column of zeros.', file=sys.stderr)
        label = pd.Series([0] * len(X), name='label')
    else:
        if y.shape[1] == 1:
            label = y.iloc[:, 0].rename('label')
        else:
            print(f'Warning: multiple target columns found {list(y.columns)}; using the first as label.', file=sys.stderr)
            label = y.iloc[:, 0].rename('label')

    # Normalize features to [0,1]
    X_norm = minmax01_df(X)

    # Assemble output: label first, then normalized features
    out_df = pd.concat([label.reset_index(drop=True), X_norm.reset_index(drop=True)], axis=1)

    # Output path
    out_dir = Path('examples/spectf')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / 'spectf_heart.csv'

    out_df.to_csv(out_path, index=False)
    print(f'Saved SPECTF Heart dataset to: {out_path}')
    print(f'Shape: {out_df.shape}')
    print('Columns:', list(out_df.columns)[:10], '...')


if __name__ == '__main__':
    main()
