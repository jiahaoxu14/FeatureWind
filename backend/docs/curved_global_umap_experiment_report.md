# Curved Global UMAP Experiment Report

## 1. Scope and Provenance

This report covers the `curved_global_umap` experiment implemented in:

- `backend/src/featurewind/eval/synthetic_global_cases.py`
- `backend/src/featurewind/eval/trail_global_fidelity.py`
- `backend/src/featurewind/eval/metrics.py`
- `backend/src/featurewind/eval/advection.py`
- `backend/scripts/run_trail_global_fidelity.py`

The quantitative results in this report are taken from the saved run:

- `backend/var/output/trail_global_fidelity/curved_global_umap_rerun_20260323/summary_metrics.csv`
- `backend/var/output/trail_global_fidelity/curved_global_umap_rerun_20260323/per_case.json`
- `backend/var/output/trail_global_fidelity/curved_global_umap_rerun_20260323/paper_summary.md`
- `backend/var/output/trail_global_fidelity/curved_global_umap_rerun_20260323/representative_paths.png`

This is a paper-style technical report derived from the saved artifacts and source code. It does not add claims beyond what those artifacts support.

## 2. Abstract

The curved experiment evaluates whether FeatureWind's static trail, obtained by advecting a point through an interpolated local-vector field, tracks a fixed nonlinear dimensionality-reduction oracle better than two baselines: a straight local linear trail and an anchored global nonlinear axis. The experiment uses a deterministic arc-shaped synthetic dataset in six standardized features and fits a single UMAP model to define the oracle transform trail. Across 715 attempted point-feature cases and 583 valid cases, the static trail consistently outperformed the straight linear baseline as transport length increased. At the final step, the static trail reduced mean path deviation from `0.9893` to `0.6906` and mean endpoint error from `1.9492` to `1.2198`, corresponding to relative reductions of `30.2%` and `37.4%`. The anchored global nonlinear axis was a stronger baseline than the straight line, especially at early steps, but the static trail overtook it on mean error after step 2 and finished with lower mean path deviation (`0.6906` vs `0.9184`) and endpoint error (`1.2198` vs `1.7373`). Distributionally, however, the anchored axis remained competitive: the static trail beat it in only `47.3%` of cases on final path deviation and `50.8%` on final endpoint error. The main conclusion is that the interpolated local-vector field captures point-specific curved transport better than a purely local straight-line approximation, while a shared global curve remains a meaningful and sometimes stronger alternative in individual cases.

## 3. Research Question and Hypothesis

### Research question

When a fixed nonlinear DR model produces a visibly curved oracle trajectory under feature transport, can FeatureWind's static trail follow that trajectory more faithfully than simpler baselines?

### Tested hypothesis

1. The static trail should outperform the straight local linear baseline, especially at longer transport horizons where curvature accumulates.
2. The anchored global nonlinear axis should be a stronger baseline than the straight line, because it encodes shared global curvature.
3. Even against that stronger baseline, the static trail should gain an advantage when point-specific transport differs from the shared global curve.

## 4. Experimental Design

### 4.1 Synthetic dataset

The curved case is deterministic and uses a latent grid of `55 x 13 = 715` points over:

- `theta in [-2.3, 2.3]`
- `radial in [-0.32, 0.32]`

The six raw features are:

```text
feature_0 = theta
feature_1 = radial
feature_2 = cos(theta) * (1.0 + 0.30 * radial)
feature_3 = sin(theta) * (1.0 + 0.30 * radial)
feature_4 = 0.45 * theta + 0.12 * radial
feature_5 = 0.20 * sin(2.0 * theta) + 0.06 * radial
```

All columns are then standardized, so each feature has `std = 1.0` in the fitted dataset. The experiment evaluates only `feature_0`.

### 4.2 Nonlinear DR oracle

The oracle is not ground-truth geometry; it is a fixed-model reference path produced by a single fitted UMAP model. The UMAP configuration is:

- `n_components = 2`
- `n_neighbors = 24`
- `min_dist = 0.06`
- `metric = euclidean`
- `random_state = 0`
- `transform_seed = 0`
- `init = spectral`
- `low_memory = True`

For each point `x_i`, the oracle path is:

1. Step 0: the fitted embedding location `Y[i]`.
2. Steps 1..10: transform the perturbed points `x_i + t * delta * e_0` through the fitted UMAP model, where `t in {1, ..., 10}`.

Because the data are standardized and `feature_0` has standard deviation `1.0`, the feature step size is:

- `delta = 0.1 * std(feature_0) = 0.1`

### 4.3 Local vector estimation

Local vectors are estimated only once at the original training points with centered finite differences:

```text
grad_k(x) approx [f(x + epsilon * e_k) - f(x - epsilon * e_k)] / (2 * epsilon)
```

with:

- `epsilon = 0.02 * std(feature_0) = 0.02`

### 4.4 Static trail field construction

The experiment constructs a `48 x 48` grid over a bounding box that covers both the fitted embedding and all oracle trails, with `15%` padding. The field is built from the pointwise local vectors using the project grid pipeline:

1. interpolate vectors to grid cell centers with linear interpolation;
2. fill points outside the interpolation hull with nearest-neighbor values;
3. define a weak-flow threshold as `max(1e-6, 0.015 * p99(field magnitude))`;
4. advect trajectories with RK2 integration.

The rollout uses:

- `steps = 10`
- `rollout_substeps = 4`
- per-substep integration distance `delta / 4`

Reported static-trail positions are subsampled back to the oracle step index.

### 4.5 Baselines

The experiment compares the static trail to two baselines.

#### Baseline A: straight local linear trail

This baseline uses only the initial local vector at the oracle start point and moves in a straight line:

```text
path[t] = start_xy + t * delta * start_vector
```

#### Baseline B: anchored global nonlinear axis

This baseline builds a shared 2D curve for the evaluated feature:

1. z-score `feature_0`;
2. sort points by feature value;
3. split into bins;
4. compute median embedding position per bin;
5. interpolate a smooth shared curve through those medians;
6. anchor the curve to the actual start point;
7. rescale so the first step does not exceed the initial local linear step magnitude.

This makes the anchored axis deliberately conservative. In the saved run:

- mean axis scale = `0.3911`
- median axis scale = `0.3231`
- min axis scale = `0.0092`
- max axis scale = `1.0`
- fraction of cases with scale `< 1.0` = `91.25%`

## 5. Metrics

The evaluation uses step-aligned metrics computed against the oracle path.

### 5.1 Endpoint error

Euclidean distance between predicted and oracle endpoints at each step.

### 5.2 Path deviation

For each prefix, average pointwise deviation from oracle over all points in that prefix.

### 5.3 Direction agreement

Cosine agreement between predicted and oracle segment directions.

### 5.4 Reference curvature

The report also stores `reference_turning_deg`, defined as the sum of absolute turn angles along the oracle path. This is cumulative turning, not net angular displacement, so values can exceed `360 deg`.

## 6. Validity Filtering

Each of the 715 dataset points defines one attempted point-feature case because only `feature_0` is evaluated.

Results:

- attempted cases: `715`
- valid cases: `583`
- invalid cases: `132`
- valid rate: `81.5%`

Invalid reasons:

- `midpoint-below-weak-threshold`: `61`
- `end-below-weak-threshold`: `54`
- `midpoint-out-of-bbox`: `10`
- `end-out-of-bbox`: `5`
- `start-below-weak-threshold`: `2`

This means invalidity was driven mainly by weak-flow regions and, secondarily, by advection leaving the evaluation bounding box.

## 7. Curvature Regime

All valid cases fall into the code's `high` curvature bin (`> 45 deg`). The actual cumulative turning is much larger:

- mean turning: `1008.5748 deg`
- median turning: `1014.7748 deg`
- min turning: `465.0455 deg`
- max turning: `1446.2740 deg`
- 10th percentile: `810.8289 deg`
- 25th percentile: `913.2060 deg`
- 50th percentile: `1014.7748 deg`
- 75th percentile: `1111.1535 deg`
- 90th percentile: `1194.1779 deg`

This confirms that the curved case is not merely mildly nonlinear; it is an aggressively curved transport benchmark.

## 8. Main Quantitative Results

### 8.1 Mean path deviation and endpoint error by step

| Step | Static path dev | Axis path dev | Linear path dev | Static endpoint | Axis endpoint | Linear endpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1042 | 0.1005 | 0.1086 | 0.2083 | 0.2010 | 0.2172 |
| 2 | 0.1899 | 0.1926 | 0.2087 | 0.3613 | 0.3768 | 0.4088 |
| 3 | 0.2663 | 0.2778 | 0.3074 | 0.4954 | 0.5334 | 0.6036 |
| 4 | 0.3363 | 0.3599 | 0.4064 | 0.6162 | 0.6883 | 0.8023 |
| 5 | 0.4017 | 0.4509 | 0.5048 | 0.7292 | 0.9061 | 0.9970 |
| 6 | 0.4647 | 0.5478 | 0.6026 | 0.8422 | 1.1290 | 1.1893 |
| 7 | 0.5246 | 0.6523 | 0.6988 | 0.9441 | 1.3835 | 1.3723 |
| 8 | 0.5826 | 0.7482 | 0.7964 | 1.0463 | 1.5153 | 1.5772 |
| 9 | 0.6377 | 0.8365 | 0.8933 | 1.1335 | 1.6314 | 1.7652 |
| 10 | 0.6906 | 0.9184 | 0.9893 | 1.2198 | 1.7373 | 1.9492 |

### 8.2 Stepwise improvement of static trail over baselines

Relative reductions are computed against each baseline's mean error.

| Step | Path vs linear | Endpoint vs linear | Path vs axis | Endpoint vs axis |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4.1% | 4.1% | -3.6% | -3.6% |
| 2 | 9.0% | 11.6% | 1.4% | 4.1% |
| 3 | 13.4% | 17.9% | 4.2% | 7.1% |
| 4 | 17.3% | 23.2% | 6.6% | 10.5% |
| 5 | 20.4% | 26.9% | 10.9% | 19.5% |
| 6 | 22.9% | 29.2% | 15.2% | 25.4% |
| 7 | 24.9% | 31.2% | 19.6% | 31.8% |
| 8 | 26.8% | 33.7% | 22.1% | 31.0% |
| 9 | 28.6% | 35.8% | 23.8% | 30.5% |
| 10 | 30.2% | 37.4% | 24.8% | 29.8% |

Interpretation:

- Against the straight baseline, the static trail is already slightly better at step 1 and the advantage grows monotonically.
- Against the anchored global axis, the static trail is slightly worse at step 1, but overtakes it from step 2 onward on mean error.

### 8.3 Win rates by step

Win rate is the fraction of valid cases in which the static trail has lower error than the baseline.

| Step | Path win vs linear | Endpoint win vs linear | Path win vs axis | Endpoint win vs axis |
| --- | ---: | ---: | ---: | ---: |
| 1 | 50.8% | 50.8% | 47.5% | 47.5% |
| 2 | 54.7% | 55.4% | 40.3% | 40.5% |
| 3 | 57.6% | 60.4% | 39.6% | 40.5% |
| 4 | 61.7% | 65.2% | 39.8% | 42.5% |
| 5 | 64.7% | 67.2% | 41.7% | 44.8% |
| 6 | 66.0% | 69.5% | 43.9% | 46.3% |
| 7 | 67.4% | 70.8% | 47.2% | 46.7% |
| 8 | 68.1% | 71.0% | 47.3% | 47.7% |
| 9 | 69.3% | 73.6% | 46.8% | 48.2% |
| 10 | 70.2% | 74.1% | 47.3% | 50.8% |

Interpretation:

- The static trail beats the linear baseline on a growing majority of cases.
- Against the anchored axis, the aggregate mean favors the static trail after step 2, but the per-case win rate stays near parity, implying a heterogeneous distribution with some large static wins and many competitive axis cases.

## 9. Distributional Results at Final Step

### 9.1 Final-step error distributions

#### Path deviation

- static median / IQR = `0.5337` / `[0.3660, 0.9203]`
- axis median / IQR = `0.4686` / `[0.2096, 1.2818]`
- linear median / IQR = `0.8770` / `[0.5084, 1.2526]`

#### Endpoint error

- static median / IQR = `0.9312` / `[0.6149, 1.7885]`
- axis median / IQR = `0.8883` / `[0.2642, 2.5456]`
- linear median / IQR = `1.6803` / `[0.9759, 2.4468]`

The anchored axis has a lower median final error than the static trail, but a much wider upper tail. That matches the mean-vs-win-rate split: the static trail does not win most cases against the axis, but when the axis fails, it can fail badly.

### 9.2 Distribution of static-trail improvements

Against the linear baseline:

- final path improvement mean = `0.2987`
- final path improvement median = `0.2216`
- final path improvement min/max = `-1.0115 / 1.9647`
- positive path-improvement fraction = `70.15%`
- final endpoint improvement mean = `0.7294`
- final endpoint improvement median = `0.5697`
- final endpoint improvement min/max = `-3.1220 / 4.1137`
- positive endpoint-improvement fraction = `74.10%`

Against the anchored axis:

- final path improvement mean = `0.2278`
- final path improvement median = `-0.0277`
- final path improvement min/max = `-1.2823 / 4.4323`
- positive path-improvement fraction = `47.34%`
- final endpoint improvement mean = `0.5175`
- final endpoint improvement median = `0.0195`
- final endpoint improvement min/max = `-2.8634 / 6.8676`
- positive endpoint-improvement fraction = `50.77%`

This is the central distributional finding of the experiment:

- static clearly dominates the straight line on both mean and win rate;
- static improves the mean over the anchored axis because some axis failures are extreme;
- but static does not dominate the anchored axis on a typical-case basis.

## 10. Representative and Extreme Cases

### 10.1 Best representative static-oracle matches

The code chooses representatives by best overall static-oracle agreement. The top five are:

| Point | Turning deg | Static final path dev | Axis final path dev | Linear final path dev | Static final endpoint |
| --- | ---: | ---: | ---: | ---: | ---: |
| 29 | 1400.542 | 0.0775 | 0.0762 | 0.1345 | 0.0850 |
| 290 | 999.141 | 0.0968 | 0.5830 | 0.2728 | 0.1503 |
| 669 | 993.465 | 0.1383 | 0.1124 | 0.1627 | 0.2783 |
| 511 | 894.054 | 0.1962 | 1.0673 | 0.6613 | 0.6140 |
| 291 | 838.256 | 0.1256 | 0.5846 | 0.5653 | 0.3482 |

Qualitatively, the saved figure shows that the static trail usually bends toward the oracle while the linear baseline overshoots along a straight extrapolation. The anchored axis often captures shared curvature but can miss point-specific offsets or local turns.

### 10.2 Largest static gains vs linear

| Point | Path improvement vs linear | Turning deg | Static final | Linear final | Axis final |
| --- | ---: | ---: | ---: | ---: | ---: |
| 491 | 1.9647 | 1264.43 | 0.8298 | 2.7945 | 0.0852 |
| 590 | 1.7293 | 1066.17 | 0.5433 | 2.2725 | 1.4975 |
| 434 | 1.6267 | 948.63 | 0.9411 | 2.5678 | 0.4042 |
| 369 | 1.5872 | 717.39 | 0.3217 | 1.9089 | 1.6190 |
| 408 | 1.5623 | 1092.71 | 1.0394 | 2.6016 | 2.1588 |

### 10.3 Largest static losses vs linear

| Point | Path improvement vs linear | Turning deg | Static final | Linear final | Axis final |
| --- | ---: | ---: | ---: | ---: | ---: |
| 292 | -1.0115 | 863.50 | 1.2582 | 0.2466 | 0.3657 |
| 470 | -0.9669 | 1139.20 | 1.5575 | 0.5906 | 0.2752 |
| 305 | -0.9264 | 1144.64 | 1.3160 | 0.3895 | 0.1634 |
| 246 | -0.9028 | 929.13 | 2.0549 | 1.1521 | 1.7148 |
| 191 | -0.8625 | 927.32 | 2.1580 | 1.2956 | 1.8747 |

These failures show that even in a globally curved regime, a point-specific local rollout can still underperform a straight extrapolation in some regions.

### 10.4 Largest static gains and losses vs anchored axis

Best static gains vs axis:

- point `353`: `+4.4323`
- point `242`: `+3.8966`
- point `352`: `+3.8841`
- point `297`: `+3.8598`
- point `407`: `+3.8247`

Worst static losses vs axis:

- point `470`: `-1.2823`
- point `416`: `-1.2012`
- point `305`: `-1.1525`
- point `106`: `-1.0066`
- point `215`: `-0.9263`

This confirms that the anchored shared curve is a genuinely strong comparator, not a strawman baseline.

## 11. Interpretation

### 11.1 What the experiment shows

The curved experiment supports three statements.

1. A straight local linear extrapolation is inadequate once transport accumulates through a strongly curved oracle geometry.
2. A shared global curved baseline captures more structure than the straight line and is often competitive.
3. The interpolated local-vector field adds value because it can adapt to point-specific transport, producing large gains in cases where the shared global curve is misaligned.

### 11.2 What the experiment does not show

The experiment does not prove that the static trail recovers true manifold geodesics or true causal feature transport. The oracle is defined by the fitted UMAP transform, so the experiment evaluates agreement with a specific nonlinear DR model, not with external ground truth.

## 12. Limitations and Threats to Validity

1. The oracle is model-relative. It is the trajectory of a fitted UMAP transform, not a true physical or semantic transport path.
2. The dataset is synthetic and deterministic. It isolates geometry cleanly but does not establish behavior on real data.
3. Only one feature is evaluated in this curved case: `feature_0`.
4. The experiment uses descriptive statistics only. No confidence intervals or hypothesis tests are implemented in the saved artifact.
5. Validity filtering removes weak-flow and out-of-bbox trajectories, so the reported aggregate metrics cover the supported subset of cases.
6. The anchored axis baseline is conservative because its first step is rescaled not to exceed the local linear step magnitude. This makes it stronger than a naive global curve, but still leaves design choices that could be varied.

## 13. Reproducibility Notes

The run configuration stored in `per_case.json` is:

```json
{
  "output_dir": "var/output/trail_global_fidelity/curved_global_umap_rerun_20260323",
  "steps": 10,
  "delta_frac": 0.1,
  "fd_epsilon_frac": 0.02,
  "seed": 0,
  "datasets": ["curved_global_umap"],
  "feature_indices": null,
  "dr_backend": "umap",
  "grid_res": 48,
  "rollout_substeps": 4
}
```

The project requirement file pins:

- `umap-learn==0.5.7`

At the time of writing this report, the current shell environment does not have `umap` installed, so the saved run can be analyzed but not re-fit from this shell without restoring dependencies.

## 14. Paper-Ready Takeaway

A concise paper-ready conclusion is:

> In the highly curved `curved_global_umap` setting, FeatureWind's static trail substantially improves fidelity to a fixed nonlinear DR oracle over a straight local baseline, with gains that grow monotonically across transport steps. A shared anchored global nonlinear axis is a much stronger comparator and remains competitive on many individual cases, but the static trail achieves lower mean error by adapting to point-specific transport that a single shared curve cannot represent.

