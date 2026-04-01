#!/usr/bin/env python3
"""
Generate a tiny 4-feature dataset for vector-field demos.

Output schema:
    label,horizontal_signal,vertical_signal,random_signal_a,random_signal_b

The points form a sparse 4x5 lattice with a single constant label so the
dataset is easy to inspect and does not imply any cluster structure. The
first two features follow the lattice coordinates, while the last two are
deterministic synthetic signals used for random-direction trail examples.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


X_VALUES = [0.10, 0.30, 0.50, 0.70, 0.90]
Y_VALUES = [0.10, 0.35, 0.60, 0.85]
LABEL = "example"


def synthetic_random_signals(x: float, y: float) -> tuple[float, float]:
    """Return deterministic pseudo-random feature values in [0, 1]."""
    signal_a = 0.5 + 0.5 * math.sin((17.0 * x) + (11.0 * y) + 0.3)
    signal_b = 0.5 + 0.5 * math.cos((13.0 * x) - (19.0 * y) + 1.1)
    return float(signal_a), float(signal_b)


def main() -> None:
    out_path = Path(__file__).resolve().parent / "simple2d.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "horizontal_signal",
                "vertical_signal",
                "random_signal_a",
                "random_signal_b",
            ]
        )
        for y in Y_VALUES:
            for x in X_VALUES:
                random_a, random_b = synthetic_random_signals(x, y)
                writer.writerow(
                    [
                        LABEL,
                        f"{x:.2f}",
                        f"{y:.2f}",
                        f"{random_a:.6f}",
                        f"{random_b:.6f}",
                    ]
                )

    print(f"Saved simple 2D example dataset to: {out_path}")
    print(f"Rows: {len(X_VALUES) * len(Y_VALUES)}")


if __name__ == "__main__":
    main()
