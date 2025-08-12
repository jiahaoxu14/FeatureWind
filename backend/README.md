# FeatureWind

A Python package for creating animated particle flow visualizations from high-dimensional data using tangent map analysis.

## Project Structure

```
FeatureWind/
├── src/featurewind/          # Main package source code
│   ├── __init__.py          # Package initialization
│   ├── feature_wind.py      # Main FeatureWind class
│   ├── TangentMap.py        # Tangent map computation
│   ├── TangentPoint.py      # Point representation with gradients
│   ├── ScalarField.py       # Scalar field reconstruction
│   └── ...                  # Other utility modules
├── examples/                 # Example scripts
│   ├── basic_example.py     # Simple class-based example
│   └── procedural_example.py # Procedural implementation example
├── archive/                  # Archived development files
├── tangentmaps/             # Data files (.tmap format)
├── output/                  # Generated visualizations and data
├── requirements.txt         # Python dependencies
├── setup.py                # Package installation configuration
├── CLAUDE.md               # Development documentation
└── README.md               # This file
```

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

For development installation:
```bash
pip install -e .
```

## Quick Start

### Using the FeatureWind Class

```python
from featurewind import FeatureWind
import matplotlib.pyplot as plt

# Create visualization
fw = FeatureWind(
    tangentmap_path="tangentmaps/tworings.tmap",
    number_of_features=3,
    grid_size=20,
    number_of_particles=2000
)

# Animate
anim = fw.animate(frames=1000, interval=30)
plt.show()
```

### Running Examples

All examples automatically save generated files to the `output/` directory to keep the repository clean.

```bash
# Basic class-based example
python examples/basic_example.py

# Procedural implementation example  
python examples/procedural_example.py
```

## Key Parameters

- **grid_size**: Resolution of interpolation grid (affects smoothness vs performance)
- **velocity_scale**: Controls particle movement speed
- **kdtree_scale**: Distance threshold for masking interpolated regions
- **number_of_features**: How many top features to visualize simultaneously
- **number_of_particles**: Particle density in the animation

## Data Format

Requires .tmap files in JSON format containing:
- `tmap`: Array of data points with domain, range, and tangent vectors
- `Col_labels`: Feature names corresponding to gradient dimensions