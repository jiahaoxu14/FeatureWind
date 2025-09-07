# Uniform-3 Features Synthetic Dataset

This folder contains a tiny generator to create a uniformly distributed
synthetic dataset with exactly three numeric features. It is intended as a
simple, reproducible example for figures in the paper.

- Features: `f1, f2, f3` sampled i.i.d. from Uniform(0, 1)
- Label: single constant class `0` for all rows (kept for pipelines that expect a label column)
- Output format: CSV with a leading `label` column followed by `f1,f2,f3`

## Generate

```
python examples/paper_uniform3/generate_uniform3.py --n 200 --seed 42 --out examples/paper_uniform3/uniform3.csv
```

Arguments:
- `--n` (int): number of rows (default 200)
- `--seed` (int): RNG seed for reproducibility (default 42)
- `--out` (str): output CSV path (default `examples/paper_uniform3/uniform3.csv`)

## Notes
- The label is provided for convenience (e.g., to vary marker shapes in plots).
  Pipelines that ignore labels can still use this dataset: simply specify no
  target in preprocessing.
