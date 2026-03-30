#!/usr/bin/env python3
"""
Generate a tiny 2-feature dataset for vector-field demos.

Output schema:
    label,horizontal_signal,vertical_signal

The points form a sparse 4x5 lattice with a single constant label so the
dataset is easy to inspect and does not imply any cluster structure.
"""

from __future__ import annotations

import csv
from pathlib import Path


X_VALUES = [0.10, 0.30, 0.50, 0.70, 0.90]
Y_VALUES = [0.10, 0.35, 0.60, 0.85]
LABEL = "example"


def main() -> None:
    out_path = Path(__file__).resolve().parent / "simple2d.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "horizontal_signal", "vertical_signal"])
        for y in Y_VALUES:
            for x in X_VALUES:
                writer.writerow([LABEL, x, y])

    print(f"Saved simple 2D example dataset to: {out_path}")
    print(f"Rows: {len(X_VALUES) * len(Y_VALUES)}")


if __name__ == "__main__":
    main()
