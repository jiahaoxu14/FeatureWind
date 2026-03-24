# Global Oracle Trail Fidelity Experiment

## Goal

This is the only retained backend experiment path.

It evaluates whether a static trail generated from interpolated local feature vectors follows a fixed nonlinear dimensionality-reduction oracle trail better than baseline conditions.

## Compared Paths

- Oracle transform trail from a fitted DR model
- Static trail from the interpolated feature vector field
- Straight local linear baseline
- Anchored global nonlinear axis baseline

## Oracle Definition

For a selected point and feature:

1. Fit a UMAP embedding once on the synthetic dataset.
2. Perturb that single feature over multiple fixed increments in feature space.
3. Pass each perturbed point through the fitted model's `transform(...)`.
4. Use the resulting 2D sequence as the oracle trail.

This oracle is a fixed-model reference trajectory, not ground truth.

## Entry Point

From `backend/`:

```bash
PYTHONPATH=src python scripts/run_trail_global_fidelity.py \
  --output-dir var/output/trail_global_fidelity/latest
```

## Main Implementation

- `backend/src/featurewind/eval/trail_global_fidelity.py`
- `backend/src/featurewind/eval/synthetic_global_cases.py`
- `backend/scripts/run_trail_global_fidelity.py`

## Outputs

The run writes:

- `summary_metrics.csv`
- `per_case.json`
- `step_endpoint_error.png`
- `step_path_deviation.png`
- `step_winrate_plot.png`
- `representative_paths.png`
- `paper_summary.md`
