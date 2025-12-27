#!/usr/bin/env python3
"""
Fetch and prepare the UCI Wine Quality dataset (id=186) via ucimlrepo.

Output: a CSV with a leading `label` column (target quality) and all other
normalized feature columns scaled to [0, 1] (min-max). The original target column name(s)
are not preserved; they are consolidated into `label`.

Usage:
  python examples/winequality/generate_wine_quality.py
  python examples/winequality/generate_wine_quality.py --out examples/winequality/wine_quality_norm.csv

Notes:
- Requires `ucimlrepo` package. Install with: pip install ucimlrepo
- Only numeric features are normalized. Non-numeric (if any) are dropped.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def minmax01_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to numeric, impute NaNs with column mean, then scale each
    numeric feature to [0, 1] using min-max (constant columns -> 0.0)."""
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
    ap = argparse.ArgumentParser(description='Fetch and normalize UCI Wine Quality (id=186) dataset')
    ap.add_argument('-o', '--out', type=str, default='', help='Output CSV path (default: examples/winequality/wine_quality_normalized.csv)')
    ap.add_argument('-n', '--n-points', type=int, default=300, help='Downsample to this many rows (approx when stratified). Use 0 to keep all (default: 300)')
    ap.add_argument('--no-stratify', action='store_true', help='Disable stratified sampling by label')
    ap.add_argument('--seed', type=int, default=7, help='Random seed for sampling (default: 7)')
    args = ap.parse_args()

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print('Error: ucimlrepo not installed. Install with `pip install ucimlrepo`.', file=sys.stderr)
        sys.exit(1)

    # Fetch dataset
    wine_quality = fetch_ucirepo(id=186)
    X = wine_quality.data.features  # pandas DataFrame
    y = wine_quality.data.targets   # pandas DataFrame (quality score)

    # Determine label series
    if y is None or y.shape[1] == 0:
        print('Warning: No targets found; using zeros as label.', file=sys.stderr)
        label = pd.Series(np.zeros(len(X), dtype=int), name='label')
    else:
        if y.shape[1] == 1:
            label = y.iloc[:, 0].rename('label')
        else:
            # If multiple targets exist, take the first as label
            print(f'Warning: multiple target columns found {list(y.columns)}; using the first as label.', file=sys.stderr)
            label = y.iloc[:, 0].rename('label')

    # Normalize numeric features to [0, 1]
    X_norm = minmax01_df(X)

    # Assemble output DataFrame (label first)
    out_df = pd.concat([label.reset_index(drop=True), X_norm.reset_index(drop=True)], axis=1)

    # Optional downsampling to ~N rows
    if args.n_points and args.n_points > 0 and args.n_points < len(out_df):
        if args.no_stratify:
            out_df = out_df.sample(n=args.n_points, random_state=args.seed).reset_index(drop=True)
        else:
            # Stratified approximate sampling by label
            total = len(out_df)
            target = args.n_points
            parts = []
            rng = np.random.default_rng(args.seed)
            for lab, g in out_df.groupby('label'):
                take = int(round(len(g) * target / total))
                take = max(1, min(len(g), take))
                parts.append(g.sample(n=take, random_state=int(rng.integers(0, 2**31-1))))
            out_df = pd.concat(parts, axis=0)
            # If rounding overshoots, trim to exact target
            if len(out_df) > target:
                out_df = out_df.sample(n=target, random_state=args.seed)
            out_df = out_df.reset_index(drop=True)

    # Output path
    out_dir = Path('examples/winequality')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / 'wine_quality_normalized.csv'

    out_df.to_csv(out_path, index=False)
    print(f'Saved normalized Wine Quality dataset to: {out_path}')
    print(f'Shape: {out_df.shape}')
    print('Columns:', list(out_df.columns))


if __name__ == '__main__':
    main()
