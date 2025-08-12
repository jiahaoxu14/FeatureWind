# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FeatureWind is a Python-based scientific visualization tool for creating animated particle flow visualizations from high-dimensional data. The system transforms tangent map data (representing gradients in feature space) into dynamic 2D particle animations that show how features flow across the projection space.

## Core Architecture

### Main Components

- **src/featurewind/**: Core package modules with mathematical and data processing functions
- **examples/test.py**: Advanced procedural implementation with dual-panel visualization and interactive controls
- **tangentmaps/**: Input data files in .tmap JSON format
- **output/**: Generated visualizations, frames, and analysis data

### Key Modules

- **TangentMap.py**: Calculates tangent maps from input datasets using DimReader projections
- **TangentPoint.py**: Represents individual data points with gradient vectors and geometric properties (anisotropy, convex hull, ellipse fitting)
- **ScalarField.py**: Reconstructs scalar fields from gradient data using finite element methods
- **DimReader.py**: Handles dimensionality reduction and projection methods

### Visualization Systems

The codebase supports grid-based interpolation for particle dynamics:

- Uses scipy's RegularGridInterpolator for smooth velocity fields
- Gaussian smoothing for feature dominance computation
- Grid masking based on kdtree_scale distance threshold
- Individual feature grids (grid_u_feats, grid_v_feats) for decomposed velocity fields

### Data Flow

1. Load tangent map data (.tmap JSON files) containing projected positions and gradient vectors
2. Select top-k features based on average gradient magnitude across all points
3. Build interpolation grids with distance masking using kdtree_scale
4. Apply Gaussian smoothing to feature magnitude grids for dominance computation
5. Initialize particle system with lifecycle management
6. Animate particles following the interpolated vector fields
7. Apply color-coding based on dominant features with transparency based on particle speed

## Development Commands

Install dependencies:
```bash
pip install -r requirements.txt
```

Install the package in development mode:
```bash
pip install -e .
```

Run the main advanced visualization:
```bash
python examples/test.py
```

Process data to create tangent maps:
```bash
# Using the convenient wrapper script (recommended)
python examples/generate_tangent_map.py data.csv tsne --target label_column

# Using TangentMap.py directly (advanced)
python src/featurewind/TangentMap.py [input_file] [projection_method]
```

## Key Parameters

### Visualization Control
- **k** (number_of_features): How many top features to visualize simultaneously
- **num_particles**: Particle density in the animation (typically 2000-3000)
- **velocity_scale**: Controls particle movement speed (0.04-0.1)

### Grid System
- **grid_res**: Resolution of interpolation grid (affects smoothness vs performance, typically 15)
- **kdtree_scale**: Distance threshold for masking interpolated regions far from data (0.01-0.1)
- **grid_res_scale**: Scaling factor for automatic grid resolution (0.15)

### Particle System
- **max_lifetime**: Maximum age before particle reinitialization (400 frames)
- **tail_gap**: Length of particle trail history (10 segments)

## Data Dependencies

- Requires .tmap files in `tangentmaps/` directory containing JSON data with:
  - `tmap`: Array of data points with domain, range, and tangent vectors
  - `Col_labels`: Feature names corresponding to gradient dimensions
- Example: `tangentmaps/breast_cancer.tmap`

## Tangent Map Generation

The `generate_tangent_map.py` script converts datasets into tangent map files:

### Basic Usage
```bash
# Generate from CSV with automatic preprocessing
python examples/generate_tangent_map.py data.csv tsne --target label_column --output my_data.tmap
```

### Processing Steps
1. **Data Preprocessing**: Load CSV, handle missing values, extract target column
2. **Normalization**: Scale features to [0, 1] range (optional with --no-normalize)
3. **Tangent Map Generation**: Run dimensionality reduction (t-SNE, UMAP, PCA, MDS)
4. **Post-processing**: Add class labels and feature metadata to create final .tmap file

### Supported Methods
- **tsne**: t-Distributed Stochastic Neighbor Embedding (default)
- **umap**: Uniform Manifold Approximation and Projection
- **pca**: Principal Component Analysis
- **mds**: Multidimensional Scaling

### Output Format
Generated .tmap files contain:
- `tmap`: Array of points with `domain`, `range`, `tangent`, and `class` fields
- `Col_labels`: List of original feature names for gradient vector interpretation

## Interactive Features

The main visualization (`examples/test.py`) includes:

### Dual Panel Layout
- **Left Panel**: Feature Wind Map with particle animation and grid visualization
- **Right Panel**: Wind Vane showing aggregate gradient analysis for brush-selected regions

### Interactive Controls
- **Top-k Feature Slider**: Dynamic selection of number of features to visualize
- **Brush Tool**: Square selection tool for regional analysis
- **Brush Size Slider**: Adjustable brush dimensions
- **Mouse Interaction**: Click-drag to move brush, click to relocate

### Wind Vane Visualization
- Aggregate point at center representing selected region
- Gradient vectors for each feature with directional arrows
- Convex hull boundary showing feature space extent
- Covariance ellipse indicating gradient distribution
- Support vectors (black) vs non-support vectors (gray) distinction

## Grid Visualization Features

### Vector Field Display
- Grid lines showing interpolation structure (gray dashed lines)
- Vector arrows at grid points showing velocity field
- Proper layering: grid lines (background) → particles → vectors → data points

### Wind Vane Aggregation
- Samples grid points within brush area (5x5 grid by default)
- Aggregates individual feature contributions from grid_u_feats and grid_v_feats arrays
- Handles ConvexHull computation errors when insufficient unique points
- Debugging output for grid sampling and feature aggregation

## Key Implementation Details

### Grid-Based Vector Fields
- Individual feature grids stored separately for decomposed analysis
- Distance masking prevents interpolation beyond kdtree_scale threshold
- RegularGridInterpolator provides efficient queries for particle updates
- Gaussian smoothing applied to feature magnitude grids before dominance computation

### Color Management
- ColorCET Glasbey palette for maximum color distinctness (fallback to Tableau)
- Consistent color mapping across all visualizations
- Transparency based on particle speed and age
- Feature-specific coloring with intensity based on local prevalence

### Performance Optimizations
- LineCollection for efficient particle trail rendering
- Batch processing of particle updates
- Pre-computed interpolation grids
- Selective particle reinitialization (5% random + boundary/age)
- Error handling for geometric computations (ConvexHull, covariance ellipses)