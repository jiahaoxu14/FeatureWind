# FeatureWind

FeatureWind generates and visualizes “feature wind” fields over 2D embeddings. It computes per‑feature tangent vectors (gradients) of a dimensionality reduction (DR) mapping (t‑SNE or MDS in PyTorch), then renders interactive/static views to help you understand which features drive local movement on the map.


## Quick Start

- Install (Python 3.9+ recommended):

  ```bash
  pip install numpy pandas matplotlib scipy scikit-learn torch
  # Optional (dataset fetcher for the example):
  pip install ucimlrepo
  ```

- Generate a tangent map from a CSV (auto‑detects label column; uses features only for DR):

  ```bash
  # From repo root
  python generate_tangent_map.py examples/iris/iris.csv tsne iris_tsne
  python generate_tangent_map.py examples/iris/iris.csv mds  iris_mds
  ```

  This writes a .tmap JSON alongside the input CSV (or to the provided output name).

- Visualize in the Feature Wind Map UI:

  ```bash
  python createwind.py --tangent-map examples/iris/iris_tsne.tmap --top-k 5
  # or a single feature by name (substring match)
  python createwind.py --tangent-map examples/iris/iris_tsne.tmap --feature "sepal"
  # list all feature names in the .tmap
  python createwind.py --tangent-map examples/iris/iris_tsne.tmap --list-features
  ```


## Data → Tangent Map

- Script: `generate_tangent_map.py` (repo root)
  - Usage: `python generate_tangent_map.py <dataset.csv> <tsne|mds> [output_name]`
  - The CSV can include a label column (e.g., `label`, `class`, `target`, or any last non‑numeric column). The generator:
    1) extracts labels,
    2) runs DR only on feature columns to 2D with gradients via PyTorch autograd,
    3) computes per‑feature Jacobians per point,
    4) saves a `.tmap` JSON with positions, gradients, and labels re‑attached.

- Under the hood: `featurewind/core/tangent_map.py` + `featurewind/core/dim_reader.py` call a differentiable DR:
  - t‑SNE: `featurewind/core/tsne.py`
  - MDS: `featurewind/core/mds_torch.py`

  Acceleration: If a compatible GPU is available, the pipeline runs fully on GPU (CUDA on NVIDIA, or MPS on Apple Silicon). It automatically falls back to CPU when no GPU is detected.

- .tmap format (simplified):
  - `tmap`: list of entries with `domain` (original normalized feature values), `range` (2D position), `tangent` (2×D gradients), `class` (label).
  - `Col_labels`: feature names.


## Visualization (Wind Map + Wind Vane)

- Entry: `createwind.py`
  - `--top-k N | all` to show the N strongest features (by mean gradient magnitude) or all.
  - `--feature NAME` to show a single feature (substring match).
  - `--name-filter REGEX` to filter the feature set before selection (e.g., `'3$'` to keep columns ending with 3).
  - `--list-features` to print all feature names from the `.tmap`.


## Examples
- Breast Cancer (WDBC)
  - Dataset: `examples/breast_cancer/breast_cancer_wdbc.csv` (provided). Label codes: 0 = benign, 1 = malignant.
  - Generate + visualize:
    ```bash
    python generate_tangent_map.py examples/breast_cancer/breast_cancer_wdbc.csv tsne breast_tsne
    python createwind.py --tangent-map examples/breast_cancer/breast_tsne.tmap --top-k all
    # or select features ending with "3"
    python createwind.py --tangent-map examples/breast_cancer/breast_tsne.tmap --name-filter '3$' --top-k all
    ```


## Tips and Troubleshooting

- “Color by feature value” not showing? Ensure the name matches `Col_labels` in the `.tmap`. Use `--list-features` or set by integer index. Substring matching is case‑insensitive.
- Marker shapes by label are based on sorted unique labels and `MARKER_STYLES`. With labels `[0,1]`, 0→circle `'o'`, 1→square `'s'` by default.
- t‑SNE vs. MDS:
  - Both are implemented in PyTorch to allow autograd through the 2D coordinates so gradients (tangents) propagate back to original features.
  - MDS currently runs a 1‑iteration differentiable pass for gradient extraction; no warm‑start from the base layout.
- Performance: Large datasets may take time in the DR step and when computing per‑point Jacobians.


## Project Structure (selected)

- `generate_tangent_map.py` — CLI to create `.tmap` files from CSVs.
- `createwind.py` — Main visualization entry point (Wind Map + Wind Vane).
- `featurewind/core/`
  - `tangent_map.py` — Computes tangent maps from points using DimReader.
  - `dim_reader.py` — Runs differentiable DR (t‑SNE/MDS), collects Jacobians.
  - `tsne.py`, `mds_torch.py` — PyTorch DR implementations.
- `featurewind/preprocessing/` — Data loading and `.tmap` post‑processing utilities.
- `featurewind/visualization/visualization_core.py` — Plotting and UI logic.
- `featurewind/visualization/color_system.py` — Palettes and styling helpers.
- `featurewind/analysis/feature_clustering.py` — Clusters features into families by directional similarity.
- `examples/` — Example datasets and analysis scripts.


## License

This repository contains research/visualization code. See source files for details. If you plan to redistribute or publish derivatives, please include appropriate attribution.
