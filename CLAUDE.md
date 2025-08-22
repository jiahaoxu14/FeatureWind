# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Starting the Application
- **Start Backend**: `./start_backend.sh` (installs Python deps and runs Flask on port 5001)
- **Start Frontend**: `./start_frontend.sh` (installs Node deps and runs React dev server on port 3000)
- **Access**: Open http://localhost:3000 after both servers are running

### Backend Commands (from `backend/` directory)
- **Install dependencies**: `pip install -r requirements.txt`
- **Run Flask server**: `python app.py` (starts on port 5001 with debug mode)
- **Install package for development**: `pip install -e .`

### Frontend Commands (from `frontend/` directory)
- **Install dependencies**: `npm install`
- **Start development server**: `npm start`
- **Build for production**: `npm run build`
- **Run tests**: `npm run test`

## Architecture Overview

This is an integrated React + Flask application for interactive wind field visualization of high-dimensional data using tangent map analysis.

### Backend Architecture (Flask + Python)
- **Main API**: `backend/app.py` - Flask server with CORS enabled, serves REST endpoints
- **Core Processor**: `FeatureWindProcessor` class handles data loading, grid computation, and velocity field interpolation
- **Data Processing Pipeline**:
  1. Load `.tmap` files containing tangent vectors and positions
  2. Create `TangentPoint` objects with gradient vectors
  3. Select top-k features by average magnitude
  4. Generate velocity grids using scipy griddata interpolation
  5. Compute dominant features per grid cell using magnitude comparison
  6. Create RegularGridInterpolators for real-time velocity queries

### Core Data Structures
- **TangentPoint** (`src/featurewind/TangentPoint.py`): Represents points with gradient vectors, computes anisotropy and geometric properties
- **FeatureWindProcessor**: Stateful processor managing loaded data, grids, and interpolators
- **Grid System**: Regular grid with velocity fields for each feature, dominant feature mapping per cell

### Frontend Architecture (React + D3.js)
- **App.js**: Main application state management, coordinates data loading and visualization setup
- **WindMapVisualization**: Particle animation system with D3.js, handles grid cell selection and particle physics
- **WindVane**: Vector analysis component showing individual feature vectors and resultants for selected grid cells
- **DataControls**: Form interface for file selection and parameter configuration

### API Endpoints
- `POST /api/load_data` - Load tangent map file
- `POST /api/setup_visualization` - Initialize grids with k, grid_res, kdtree_scale parameters
- `GET /api/get_data_points` - Retrieve original data points
- `POST /api/get_velocity_field` - Get velocity vectors for particle positions (batched)
- `POST /api/get_wind_vane_data` - Get detailed vector analysis for specific grid cell
- `GET /api/get_grid_data` - Get grid visualization data (lines and cell dominant features)
- `GET /api/available_files` - List available .tmap files in tangentmaps directory

### Data Flow
1. Frontend loads .tmap file via DataControls → backend processes tangent vectors
2. Frontend requests visualization setup with parameters → backend computes grids and interpolators  
3. Frontend renders particles and requests velocity updates in real-time → backend interpolates velocities
4. Grid cell selection triggers wind vane analysis → backend samples feature vectors at cell center

## Key Implementation Details

### Performance Optimizations
- **Batched Velocity Queries**: Particle updates sent in batches to minimize API calls
- **RegularGridInterpolator**: Fast scipy interpolation for real-time velocity field sampling
- **KDTree Masking**: Distance-based masking prevents extrapolation in empty regions
- **Grid Caching**: Computed grids stored in processor for multiple queries

### Coordinate Systems
- **Data Space**: Original tangent map positions (arbitrary units)
- **Grid Space**: Regular grid with configurable resolution (default 40x40)
- **Frontend Space**: SVG coordinates mapped from data bounding box with square aspect ratio

### Feature Selection Logic
- Features ranked by average gradient magnitude across all points
- Top-k features used for visualization (configurable via DataControls)
- Dominant feature per grid cell determined by maximum magnitude among ALL features (not just top-k)

## File Organization

```
backend/
├── app.py                 # Flask API server
├── src/featurewind/       # Core computation modules
│   ├── TangentPoint.py   # Point representation with gradients
│   ├── TangentMap.py     # Tangent map utilities
│   └── ScalarField.py    # Field reconstruction tools
├── tangentmaps/          # .tmap data files
├── examples/             # Usage examples and notebooks
└── requirements.txt      # Python dependencies

frontend/
├── src/
│   ├── App.js           # Main React application
│   └── components/
│       ├── WindMapVisualization.js  # D3.js particle system
│       ├── WindVane.js             # Vector analysis display
│       └── DataControls.js         # Parameter controls
└── package.json         # Node.js dependencies and scripts
```

## Data Format Requirements

### .tmap Files (JSON format)
```json
{
  "tmap": [
    {
      "range": [x, y, ...],     // Position coordinates (first 2 used)
      "class": "label",         // Point label/class
      "tangent": [[...], [...]] // 2×N array of gradient vectors
    }
  ],
  "Col_labels": ["feature1", "feature2", ...] // Feature names
}
```

## Common Development Scenarios

### Adding New Visualization Parameters
1. Add parameter to `visualizationParams` state in App.js
2. Update DataControls form to include new input
3. Pass parameter to `/api/setup_visualization` endpoint
4. Handle parameter in `FeatureWindProcessor.create_grids()` method

### Modifying Particle Behavior
- Particle physics in `WindMapVisualization.js` `updateParticles()` function
- Velocity scaling and boundary handling configured in component state
- Velocity field sampling occurs via `/api/get_velocity_field` batched requests

### Adding New Feature Selection Methods
1. Implement selection logic in `FeatureWindProcessor.select_top_features()`
2. Update API response to include new metadata
3. Modify frontend DataControls to display new selection criteria

### Backend Testing
Run individual example scripts from `backend/examples/` directory to test core functionality without frontend dependencies.