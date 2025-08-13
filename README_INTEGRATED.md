# FeatureWind Integrated System

An integrated React + Flask application for interactive wind field visualization based on the original `test_clean.py` structure.

## Architecture

### Backend (Flask + Python)
- **Location**: `backend/app.py`
- **Purpose**: Data processing, grid computation, and API endpoints
- **Key Features**:
  - Loads and processes tangent map data
  - Computes velocity grids and dominant features
  - Provides REST API endpoints for frontend communication
  - Handles particle system velocity calculations

### Frontend (React + JavaScript)
- **Location**: `frontend/src/`
- **Purpose**: Interactive visualization and user interface
- **Key Components**:
  - `WindMapVisualization`: Main particle animation and grid display
  - `WindVane`: Vector analysis for selected grid cells
  - `DataControls`: File loading and parameter configuration

## Quick Start

### Prerequisites
- Python 3.8+ with pip
- Node.js 16+ with npm

### Running the Application

1. **Start Backend Server**:
   ```bash
   ./start_backend.sh
   ```
   This will:
   - Install Python dependencies
   - Start Flask server on http://localhost:5000

2. **Start Frontend Server** (in a new terminal):
   ```bash
   ./start_frontend.sh
   ```
   This will:
   - Install Node.js dependencies
   - Start React development server on http://localhost:3000

3. **Access the Application**:
   Open http://localhost:3000 in your web browser

## Features

### Data Management
- Load different tangent map files from the `tangentmaps/` directory
- Configure visualization parameters (k features, grid resolution, kdtree scale)
- Real-time parameter updates

### Wind Map Visualization
- Interactive particle animation showing feature flow
- Grid overlay showing cell boundaries and dominant features
- Click on grid cells to analyze specific regions
- Toggle grid visibility and animation controls

### Wind Vane Analysis
- Detailed vector analysis for selected grid cells
- Displays individual feature vectors and resultant
- Highlights dominant features with different colors
- Shows magnitude and direction information

### Interactive Controls
- **Grid Cell Selection**: Click on grid cells to update wind vane
- **Animation Toggle**: Start/stop particle animation
- **Grid Toggle**: Show/hide grid overlay
- **Parameter Controls**: Adjust k, grid resolution, and kdtree scale

## API Endpoints

The Flask backend provides the following REST API endpoints:

- `POST /api/load_data`: Load tangent map data
- `POST /api/setup_visualization`: Initialize visualization with parameters
- `GET /api/get_data_points`: Retrieve data points for plotting
- `POST /api/get_velocity_field`: Get velocity vectors for particle positions
- `POST /api/get_wind_vane_data`: Get vector analysis for a specific grid cell
- `GET /api/get_grid_data`: Get grid visualization data
- `GET /api/available_files`: List available tangent map files

## Data Flow

1. **Frontend** requests data loading via `DataControls`
2. **Backend** processes tangent map and returns metadata
3. **Frontend** requests visualization setup with parameters
4. **Backend** computes grids, interpolators, and dominant features
5. **Frontend** renders initial visualization and starts particle animation
6. **Real-time Updates**:
   - Particles request velocity updates from backend
   - Grid cell selection triggers wind vane data requests
   - All updates flow through REST API calls

## Differences from Original

### Separation of Concerns
- **Original**: Single Python file with matplotlib visualization
- **Integrated**: Backend handles computation, frontend handles visualization

### Technology Stack
- **Original**: Pure Python with matplotlib and scipy
- **Integrated**: Python backend + React frontend with D3.js

### Interactivity
- **Original**: Mouse hover and keyboard shortcuts
- **Integrated**: Click-based selection, form controls, and real-time updates

### Data Communication
- **Original**: Direct function calls and shared variables
- **Integrated**: REST API communication between frontend and backend

## File Structure

```
FeatureWind/
├── backend/
│   ├── app.py                 # Flask application
│   ├── requirements.txt       # Python dependencies
│   ├── src/featurewind/      # Core computation modules
│   └── tangentmaps/          # Data files
├── frontend/
│   ├── package.json          # Node.js dependencies
│   ├── src/
│   │   ├── App.js           # Main application component
│   │   └── components/      # React components
│   └── public/              # Static assets
├── start_backend.sh          # Backend startup script
└── start_frontend.sh         # Frontend startup script
```

## Development Notes

### Backend Development
- Flask runs in debug mode for development
- CORS enabled for frontend communication
- Error handling for all API endpoints
- Modular processor class for data management

### Frontend Development
- React development server with hot reload
- D3.js for SVG-based visualizations
- Axios for API communication
- Component-based architecture for maintainability

### Performance Considerations
- Particle updates are batched for efficiency
- Grid computations cached on backend
- Animation frame limiting for smooth performance
- Lazy loading of visualization components