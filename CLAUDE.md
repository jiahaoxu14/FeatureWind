# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FeatureWind is a Python package for visualizing feature flows in high-dimensional data using animated particle flow visualizations from tangent map data. The project creates interactive animations showing how gradients in feature space influence particle movement across a 2D projection.

## Architecture

The codebase consists of several key components:

- **Core library** (`src/featurewind/`): Main Python package containing utilities for tangent map generation, scalar field reconstruction, and visualization components
- **Examples** (`examples/`): Scripts demonstrating usage, including `generate_tangent_map.py` for preprocessing datasets and `test.py` for creating visualizations
- **Tangent maps** (`tangentmaps/`): Pre-generated `.tmap` files containing processed tangent map data for various datasets
- **Output** (`output/`): Generated visualization files and CSV exports

### Key Modules

- `TangentMap.py`: Generates tangent maps using dimensionality reduction (t-SNE, UMAP, etc.) with gradient computation
- `ScalarField.py`: Reconstructs scalar fields from point gradients using finite element methods
- `TangentPoint.py` & `TangentPointSet.py`: Data structures for managing tangent map points and their properties
- `DimReader.py`: Handles dimensionality reduction with gradient computation using PyTorch autograd
- `test.py`: Main visualization engine creating interactive particle flow animations

## Common Development Commands

Since there are no standard build/test configuration files, use these Python commands:

**Generate tangent maps from datasets:**
```bash
python examples/generate_tangent_map.py <dataset.csv> tsne --target <label_column> --output <output.tmap>
```

**Run interactive visualization:**
```bash
python examples/test.py
```

**Generate tangent maps directly (legacy method):**
```bash
python src/featurewind/TangentMap.py <input.csv> tsne
```

## Data Flow

1. **Input**: CSV datasets with numeric features and optional target labels
2. **Processing**: `generate_tangent_map.py` normalizes data and calls `TangentMap.py` to compute gradients via t-SNE
3. **Output**: `.tmap` files containing tangent vectors, 2D projections, and metadata
4. **Visualization**: `test.py` loads `.tmap` files and creates real-time particle animations with interactive controls

## Dependencies

The project uses:
- PyTorch for automatic differentiation in gradient computation
- NumPy/SciPy for numerical operations and interpolation
- Matplotlib for visualization and animation
- Scikit-learn for data preprocessing
- Optional: colorcet for enhanced color palettes

## File Formats

- **Input**: CSV files with numeric columns (features) and optional categorical target column
- **Output**: `.tmap` JSON files containing `tmap` (array of points with domain, range, tangent vectors) and `Col_labels` (feature names)
- **Visualization output**: PNG frames and interactive matplotlib animations