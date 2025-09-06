#!/usr/bin/env python3
"""
Fetch the UCI SUPPORT2 dataset (ID=880) using ucimlrepo and write a CSV
containing only a numeric 'label' column and all feature columns.

Usage:
  python examples/support2/fetch_support2.py

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

    # Fetch dataset (UCI ID 880: SUPPORT2)
    ds = fetch_ucirepo(id=880)
    X = ds.data.features
    y = ds.data.targets

    # Ensure pandas DataFrame/Series
    X_df = pd.DataFrame(X)
    if isinstance(y, pd.DataFrame):
        # Use the first column as the target if multiple present
        y_series = y.iloc[:, 0]
    else:
        y_series = pd.Series(y)

    # Normalize/clean column names to strings
    X_df.columns = [str(c) for c in X_df.columns]

    # Build numeric label column named 'label'
    # If y is non-numeric (strings), map to integer codes deterministically
    if not pd.api.types.is_numeric_dtype(y_series):
        y_codes = pd.Categorical(y_series)
        label_series = pd.Series(y_codes.codes, name='label')
    else:
        label_series = pd.Series(pd.to_numeric(y_series, errors='coerce').astype('Int64')).fillna(0).astype(int)
        label_series.name = 'label'

    # Concatenate label + raw features (no normalization here; dataset may contain NaNs)
    out_df = pd.concat([label_series, X_df], axis=1)

    # Output path: alongside this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(script_dir, exist_ok=True)
    out_path = os.path.join(script_dir, 'support2.csv')

    out_df.to_csv(out_path, index=False)
    print(f"✓ Wrote {len(out_df)} rows to {out_path}")


if __name__ == '__main__':
    main()

