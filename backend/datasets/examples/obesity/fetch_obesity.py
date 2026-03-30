#!/usr/bin/env python3
"""
Fetch the UCI obesity-levels dataset (ID=544) and write a normalized CSV with
real class labels plus continuous feature columns.

Usage:
  python fetch_obesity.py
  python fetch_obesity.py --out obesity.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DATASET_ID = 544
OUTPUT_FILENAME = "obesity.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and normalize the UCI obesity-levels dataset")
    parser.add_argument(
        "-o",
        "--out",
        default="",
        help="Output CSV path (default: backend/datasets/examples/obesity/obesity.csv)",
    )
    return parser.parse_args()


def _normalize_numeric_series(series):
    import pandas as pd

    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    if numeric.isna().any():
        median = numeric.median()
        numeric = numeric.fillna(median if pd.notna(median) else 0.0)
    vmin = float(numeric.min())
    vmax = float(numeric.max())
    if vmax > vmin:
        return (numeric - vmin) / (vmax - vmin)
    return numeric * 0.0


def _encode_categorical_series(series):
    import pandas as pd

    cleaned = series.astype(str).where(series.notna(), "MISSING")
    categories = sorted(pd.unique(cleaned).tolist())
    categorical = pd.Categorical(cleaned, categories=categories, ordered=True)
    codes = pd.Series(categorical.codes, index=cleaned.index, dtype="float64")
    if len(categories) > 1:
        return codes / float(len(categories) - 1)
    return codes * 0.0


def _normalize_feature_frame(df):
    import pandas as pd

    out = df.copy()
    for column in out.columns:
        series = out[column]
        if pd.api.types.is_numeric_dtype(series):
            out[column] = _normalize_numeric_series(series)
        else:
            out[column] = _encode_categorical_series(series)
    return out.astype("float32")


def main() -> None:
    args = parse_args()

    try:
        import pandas as pd
    except Exception:
        print("Error: pandas is required. Install with: pip install pandas", file=sys.stderr)
        raise SystemExit(1)

    try:
        from ucimlrepo import fetch_ucirepo
    except Exception:
        print("Error: ucimlrepo is required. Install with: pip install ucimlrepo", file=sys.stderr)
        raise SystemExit(1)

    dataset = fetch_ucirepo(id=DATASET_ID)
    feature_df = pd.DataFrame(dataset.data.features)
    target_df = pd.DataFrame(dataset.data.targets)

    if target_df.shape[1] < 1:
        raise ValueError("Dataset does not contain a target column.")

    label_series = target_df.iloc[:, 0].astype(str)
    label_series.name = "label"

    feature_df.columns = [str(column) for column in feature_df.columns]
    normalized_features = _normalize_feature_frame(feature_df)
    out_df = pd.concat([label_series, normalized_features], axis=1)

    script_dir = Path(__file__).resolve().parent
    out_path = Path(args.out) if args.out else (script_dir / OUTPUT_FILENAME)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote {len(out_df)} rows to {out_path}")
    print(f"Columns: label + {len(feature_df.columns)} normalized features")
    print(f"Label counts: {label_series.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
