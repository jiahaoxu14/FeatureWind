# FeatureWind

FeatureWind generates and visualizes “feature wind” fields over 2D embeddings. It computes per‑feature tangent vectors (gradients) of a dimensionality reduction (DR) mapping (t‑SNE or MDS in PyTorch), then serves those fields to the web UI for interactive inspection.

## Monorepo Layout

- `backend/` — Python core, tangent-map generation, datasets, docs, and Flask API.
  - `src/featurewind/` core library
  - `api/` Flask service
  - `datasets/` sample CSVs and `.tmap` files
  - `var/` run artifacts (outputs/uploads; gitignored)
- `frontend/` — React UI.

> Tip: run backend commands from `backend/` with `PYTHONPATH=src` (or install editable). Frontend runs from `frontend/`.


## Quick Start

- Install (Python 3.9+ recommended):

  ```bash
  pip install numpy pandas matplotlib scipy scikit-learn torch
  # Optional (dataset fetcher for the example):
  pip install ucimlrepo
  ```

- Generate a tangent map from a CSV (auto‑detects label column; uses features only for DR). Run from repo root:

  ```bash
  ./.venv/bin/python backend/src/generate_tangent_map.py backend/datasets/examples/iris/iris.csv tsne iris_tsne
  ./.venv/bin/python backend/src/generate_tangent_map.py backend/datasets/examples/iris/iris.csv mds iris_mds
  ```

  This writes a .tmap JSON alongside the input CSV (or to the provided output name).

- Run the backend API and frontend UI:

  ```bash
  cd backend
  PYTHONPATH=src python app.py

  cd ../frontend
  npm run dev
  ```


## Data → Tangent Map

- Script: `backend/src/generate_tangent_map.py`
  - Usage: `python backend/src/generate_tangent_map.py <dataset.csv> <tsne|mds> [output_name]`
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


## Visualization

- Backend API entry: `backend/app.py`
- Frontend UI entry: `frontend/src/App.jsx`
- Main compute route: `POST /api/compute` in `backend/routes.py`


## Examples
- Breast Cancer (WDBC)
  - Dataset: `backend/datasets/examples/breast_cancer/breast_cancer_wdbc.csv` (provided). Label codes: 0 = benign, 1 = malignant.
  - Generate a tangent map:
    ```bash
    ./.venv/bin/python backend/src/generate_tangent_map.py backend/datasets/examples/breast_cancer/breast_cancer_wdbc.csv tsne breast_tsne
    ```


## Tips and Troubleshooting

- “Color by feature value” not showing? Ensure the name matches `Col_labels` in the `.tmap`.
- Marker shapes by label are based on sorted unique labels and `MARKER_STYLES`. With labels `[0,1]`, 0→circle `'o'`, 1→square `'s'` by default.
- t‑SNE vs. MDS:
  - Both are implemented in PyTorch to allow autograd through the 2D coordinates so gradients (tangents) propagate back to original features.
  - MDS currently runs a 1‑iteration differentiable pass for gradient extraction; no warm‑start from the base layout.
- Performance: Large datasets may take time in the DR step and when computing per‑point Jacobians.


## Project Structure (selected)

- `backend/src/generate_tangent_map.py` — Tangent-map generator for CSVs.
- `backend/src/featurewind/core/`
  - `tangent_map.py` — Computes tangent maps from points using DimReader.
  - `dim_reader.py` — Runs differentiable DR (t‑SNE/MDS), collects Jacobians.
  - `tsne.py`, `mds_torch.py` — PyTorch DR implementations.
- `backend/src/featurewind/preprocessing/` — Data loading and `.tmap` post‑processing utilities.
- `backend/src/featurewind/visualization/color_system.py` — Palettes and styling helpers.
- `backend/src/featurewind/analysis/feature_clustering.py` — Clusters features into families by directional similarity.
- `backend/datasets/` — Example datasets and analysis scripts.


## License

This repository contains research/visualization code. See source files for details. If you plan to redistribute or publish derivatives, please include appropriate attribution.
