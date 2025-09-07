# Uniform-D Features Synthetic Dataset

This folder contains a tiny generator to create a uniformly distributed
synthetic dataset with a configurable number of numeric features (default 100). It is intended as a
simple, reproducible example for figures in the paper.

- Features: `f1..fD` sampled i.i.d. from Uniform(0, 1) (default D=100)
- Label: single constant class `0` for all rows (kept for pipelines that expect a label column)
- Output format: CSV with a leading `label` column followed by `f1..fD`

## Generate

```
python examples/paper_uniform3/generate_uniform3.py --n 200 --features 100 --seed 42 --out examples/paper_uniform3/uniform100.csv
```

Arguments:
- `--n` (int): number of rows (default 200)
- `--features`/`--d` (int): number of features (default 100)
- `--seed` (int): RNG seed for reproducibility (default 42)
- `--out` (str): output CSV path (default `examples/paper_uniform3/uniform3.csv`)

## Notes
- The label is provided for convenience (e.g., to vary marker shapes in plots).
  Pipelines that ignore labels can still use this dataset: simply specify no
  target in preprocessing.
