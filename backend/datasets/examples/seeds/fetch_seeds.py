#!/usr/bin/env python3
"""
Fetch the UCI Seeds dataset (ID=236) directly from the UCI archive and write a
CSV with real class names plus normalized numeric features.

Usage:
  python fetch_seeds.py
  python fetch_seeds.py --out seeds.csv
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path


UCI_SEEDS_ZIP_URL = "https://archive.ics.uci.edu/static/public/236/seeds.zip"
FEATURE_COLUMNS = [
    "area",
    "perimeter",
    "compactness",
    "kernel_length",
    "kernel_width",
    "asymmetry_coefficient",
    "groove_length",
]
LABEL_MAP = {
    1: "Kama",
    2: "Rosa",
    3: "Canadian",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and normalize the UCI Seeds dataset")
    parser.add_argument(
        "-o",
        "--out",
        default="",
        help="Output CSV path (default: backend/datasets/examples/seeds/seeds.csv)",
    )
    return parser.parse_args()


def _download_dataset_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        with archive.open("seeds_dataset.txt") as seeds_file:
            return seeds_file.read().decode("utf-8")


def _normalize_frame(df):
    import pandas as pd

    out = df.copy()
    for column in out.columns:
        series = pd.to_numeric(out[column], errors="coerce").astype(float)
        if series.isna().any():
            median = series.median()
            series = series.fillna(median if pd.notna(median) else 0.0)
        vmin = float(series.min())
        vmax = float(series.max())
        if vmax > vmin:
            out[column] = (series - vmin) / (vmax - vmin)
        else:
            out[column] = series * 0.0
    return out.astype("float32")


def main() -> None:
    args = parse_args()

    try:
        import pandas as pd
    except Exception:
        print("Error: pandas is required. Install with: pip install pandas", file=sys.stderr)
        raise SystemExit(1)

    raw_text = _download_dataset_text(UCI_SEEDS_ZIP_URL)
    column_names = FEATURE_COLUMNS + ["label"]
    df = pd.read_csv(io.StringIO(raw_text), sep=r"\s+", header=None, names=column_names)

    if df["label"].isna().any():
        raise ValueError("Seeds dataset contains missing class labels.")

    feature_df = _normalize_frame(df[FEATURE_COLUMNS])
    label_series = df["label"].astype(int).map(LABEL_MAP)
    if label_series.isna().any():
        unknown = sorted(df.loc[label_series.isna(), "label"].astype(int).unique().tolist())
        raise ValueError(f"Unexpected Seeds class ids: {unknown}")
    label_series.name = "label"

    out_df = pd.concat([label_series, feature_df], axis=1)

    script_dir = Path(__file__).resolve().parent
    out_path = Path(args.out) if args.out else (script_dir / "seeds.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote {len(out_df)} rows to {out_path}")
    print(f"Columns: label + {len(FEATURE_COLUMNS)} normalized features")
    print(f"Label counts: {label_series.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
