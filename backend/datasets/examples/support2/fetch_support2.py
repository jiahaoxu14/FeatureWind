#!/usr/bin/env python3
"""
Fetch the UCI SUPPORT2 dataset (ID=880) and write a CSV where
every feature column is converted to a continuous numeric value and
normalized to [0, 1], while leaving the 'label' column unnormalized.

Rules per column (excluding 'label'):
- Numeric columns → min‑max scale to [0,1]; constant columns become 0.0
- Non‑numeric (categorical/text) → stable integer codes (0..K-1),
  then divide by (K-1) to map into [0,1] (K=1 → 0.0)
- Missing values are imputed (median for numeric; 'MISSING' category for non‑numeric)

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

    def make_continuous_and_normalize(df):
        out = df.copy()
        for col in out.columns:
            s = out[col]
            # Numeric branch
            if pd.api.types.is_numeric_dtype(s):
                s = pd.to_numeric(s, errors='coerce')
                # Impute NaNs with median
                if s.isna().any():
                    med = s.median()
                    s = s.fillna(med if pd.notna(med) else 0.0)
                s = s.astype(float)
                vmin, vmax = float(s.min()), float(s.max())
                if vmax > vmin:
                    s = (s - vmin) / (vmax - vmin)
                else:
                    s = s * 0.0
                out[col] = s
            else:
                # Treat as categorical/text
                s = s.astype(object).where(s.notna(), 'MISSING')
                cat = pd.Categorical(s)
                codes = pd.Series(cat.codes, index=s.index).astype(float)
                k = len(cat.categories)
                if k > 1:
                    s_norm = codes / float(k - 1)
                else:
                    s_norm = codes * 0.0
                out[col] = s_norm
        # Ensure float dtype for all features
        return out.astype('float32')

    # Process features (exclude label)
    X_proc = make_continuous_and_normalize(X_df)

    # Concatenate label + processed features
    out_df = pd.concat([label_series, X_proc], axis=1)

    # Output path: alongside this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(script_dir, exist_ok=True)
    out_path = os.path.join(script_dir, 'support2.csv')

    out_df.to_csv(out_path, index=False)
    print(f"✓ Wrote {len(out_df)} rows to {out_path}")
    print(f"  Columns: label + {len(X_proc.columns)} features (all normalized to [0,1])")


if __name__ == '__main__':
    main()
