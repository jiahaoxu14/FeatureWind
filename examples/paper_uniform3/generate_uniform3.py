#!/usr/bin/env python3
"""
Generate a uniformly distributed synthetic dataset with three features.

Output CSV schema:
    label,f1,f2,f3

- Features f1,f2,f3 ~ Uniform(0,1)
- Label in {0,1,2} = argmax([f1,f2,f3]) for convenience (can be ignored)

Usage:
    python examples/paper_uniform3/generate_uniform3.py \
        --n 1000 --seed 42 --out examples/paper_uniform3/uniform3.csv
"""

import argparse
from pathlib import Path
import numpy as np
import csv


def main():
    ap = argparse.ArgumentParser(description='Generate a uniform D-feature dataset for paper figures')
    ap.add_argument('--n', type=int, default=200, help='number of samples')
    ap.add_argument('--features', '--d', type=int, default=100, help='number of features (columns)')
    ap.add_argument('--seed', type=int, default=42, help='random seed')
    ap.add_argument('--out', type=str, default='examples/paper_uniform3/uniform3.csv', help='output CSV path')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Draw features uniformly in [0, 1]
    d = int(args.features)
    if d < 1:
        d = 1
    X = rng.random((args.n, d))  # columns: f1..fD

    # Use a single constant label for all rows (e.g., class 0)
    y = np.zeros(args.n, dtype=int)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open('w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'f{j+1}' for j in range(d)]
        writer.writerow(header)
        for i in range(args.n):
            row = [int(y[i])] + [float(val) for val in X[i, :]]
            writer.writerow(row)

    print(f"Saved uniform-{d} dataset: {out_path} (n={args.n}, seed={args.seed})")


if __name__ == '__main__':
    main()
