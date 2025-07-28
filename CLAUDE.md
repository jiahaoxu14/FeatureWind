# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FeatureWind is a Python-based scientific visualization tool for creating animated particle flow visualizations from high-dimensional data. The system transforms tangent map data (representing gradients in feature space) into dynamic 2D particle animations that show how features flow across the projection space.

## Core Architecture

### Main Components

- **feature_wind.py**: Object-oriented implementation with `FeatureWind` class that handles the complete pipeline
- **featurewind.py**: Procedural implementation with global variables and functional approach
- **funcs/**: Core utility modules containing mathematical and data processing functions

### Key Modules

- **TangentMap.py**: Calculates tangent maps from input datasets using DimReader projections
- **TangentPoint.py**: Represents individual data points with gradient vectors and geometric properties (anisotropy, convex hull, ellipse fitting)
- **ScalarField.py**: Reconstructs scalar fields from gradient data using finite element methods
- **DimReader.py**: Handles dimensionality reduction and projection methods

### Data Flow

1. Load tangent map data (.tmap JSON files) containing projected positions and gradient vectors
2. Select top-k features based on gradient magnitude
3. Build interpolation grids for velocity fields using scipy griddata
4. Initialize particle system with random positions
5. Animate particles following the interpolated vector field
6. Apply color-coding based on dominant features and particle properties

## Development Commands

Install the package in development mode:
```bash
pip install -e .
```

Run examples:
```bash
# Basic class-based example
python examples/basic_example.py

# Procedural implementation example
python examples/procedural_example.py

# Run main class directly
python src/featurewind/feature_wind.py

# Process data to create tangent maps
python src/featurewind/TangentMap.py [input_file] [projection_method]
```

## Key Parameters

- **grid_size**: Resolution of interpolation grid (affects smoothness vs performance)
- **velocity_scale**: Controls particle movement speed
- **kdtree_scale**: Distance threshold for masking interpolated regions far from data
- **number_of_features**: How many top features to visualize simultaneously
- **number_of_particles**: Particle density in the animation

## Data Dependencies

- Requires .tmap files in `tangentmaps/` directory containing JSON data with:
  - `tmap`: Array of data points with domain, range, and tangent vectors
  - `Col_labels`: Feature names corresponding to gradient dimensions

## Key Implementation Details

- Uses scipy's RegularGridInterpolator for efficient vector field queries
- Particle lifecycle management with automatic reinitization for out-of-bounds or aged particles
- Feature dominance computed using argmax of smoothed gradient magnitudes
- Color coding maps features to Tableau color palette with transparency based on particle speed
- LineCollection used for efficient rendering of particle trails with fading effects