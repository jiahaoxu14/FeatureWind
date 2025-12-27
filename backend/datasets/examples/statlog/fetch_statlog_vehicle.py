#!/usr/bin/env python3
"""
Fetch the UCI Statlog (Vehicle Silhouettes) dataset (ID=149) using ucimlrepo,
print basic info, and write a CSV with numeric 'label' + normalized features.

Usage:
  python examples/statlog/fetch_statlog_vehicle.py

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

    # fetch dataset (UCI ID 149: Statlog (Vehicle Silhouettes))
    ds = fetch_ucirepo(id=149)

    # data (as pandas dataframes)
    X = ds.data.features
    y = ds.data.targets

    # Ensure DataFrame/Series
    X_df = import_pandas_dataframe(X)
    y_series = extract_first_target_series(y)

    # Normalize/clean column names to strings
    X_df.columns = [str(c) for c in X_df.columns]

    # Build numeric label column 'label'
    label_series = to_numeric_label(y_series)

    # Make all features continuous in [0,1]
    X_proc = make_continuous_and_normalize(X_df)

    # Concatenate and save
    out_df = pd.concat([label_series, X_proc], axis=1)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(script_dir, exist_ok=True)
    out_path = os.path.join(script_dir, 'statlog_vehicle.csv')
    out_df.to_csv(out_path, index=False)

    # Print summary + metadata
    print(f"✓ Wrote {len(out_df)} rows to {out_path}")
    print(f"  Columns: label + {len(X_proc.columns)} features (normalized to [0,1])")
    print("\nMetadata:")
    print(ds.metadata)
    print("\nVariables:")
    print(ds.variables)


def import_pandas_dataframe(X):
    import pandas as pd
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X)


def extract_first_target_series(y):
    import pandas as pd
    if isinstance(y, pd.DataFrame):
        return y.iloc[:, 0]
    return pd.Series(y)


def to_numeric_label(y_series):
    import pandas as pd
    if not pd.api.types.is_numeric_dtype(y_series):
        cats = pd.Categorical(y_series)
        return pd.Series(cats.codes, name='label')
    s = pd.to_numeric(y_series, errors='coerce').astype('Int64').fillna(0).astype(int)
    s.name = 'label'
    return s


def make_continuous_and_normalize(df):
    import pandas as pd
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s, errors='coerce')
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
            s = s.astype(object).where(s.notna(), 'MISSING')
            cat = pd.Categorical(s)
            codes = pd.Series(cat.codes, index=s.index).astype(float)
            k = len(cat.categories)
            out[col] = (codes / float(k - 1)) if k > 1 else codes * 0.0
    return out.astype('float32')


if __name__ == '__main__':
    main()
