# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FeatureWind is a Python package for visualizing feature flows in high-dimensional data using animated particle flow visualizations from tangent map data. The project creates interactive animations showing how gradients in feature space influence particle movement across a 2D projection, revealing hidden structures and relationships in complex datasets.

## Architecture

The codebase has been modularized into focused, maintainable components:

### **Core Library** (`src/featurewind/`)
Main Python package containing utilities for tangent map generation and data structures:
- `TangentMap.py`: Generates tangent maps using dimensionality reduction (t-SNE, UMAP, etc.) with gradient computation
- `ScalarField.py`: Reconstructs scalar fields from point gradients using finite element methods
- `TangentPoint.py` & `TangentPointSet.py`: Data structures for managing tangent map points and their properties
- `DimReader.py`: Handles dimensionality reduction with gradient computation using PyTorch autograd

### **Examples Directory** (`examples/`)
Contains both legacy and modular visualization implementations:

#### **Modular Implementation** (Recommended)
- `main_modular.py`: **Main entry point** - orchestrates all components
- `config.py`: Configuration constants and global parameters
- `data_processing.py`: Data loading, validation, and feature selection
- `grid_computation.py`: Spatial discretization and velocity field construction
- `particle_system.py`: Physics simulation with adaptive RK4 integration
- `visualization_core.py`: Plotting, figure management, and rendering
- `ui_controls.py`: Interactive widgets and user interface management

#### **Recent Advanced Features**
- `color_system.py`: Paul Tol's colorblind-safe palette with LCH color space
- `event_manager.py`: Centralized event handling for reliable mouse/keyboard interaction
- `feature_clustering.py`: Hierarchical clustering for feature family grouping
- `legend_manager.py`: Dynamic legend management with family-based organization

#### **Legacy Files**
- `main.py`: Original monolithic implementation (92,385+ characters)
- `generate_tangent_map.py`: Preprocessing script for creating `.tmap` files

### **Data Directories**
- `tangentmaps/`: Pre-generated `.tmap` files containing processed tangent map data
- `output/`: Generated visualization files, animation frames, and CSV exports

## Common Development Commands

**Run modular visualization (recommended):**
```bash
python examples/main_modular.py
```

**Run legacy visualization:**
```bash
python examples/main.py
```

**Generate tangent maps from datasets:**
```bash
python examples/generate_tangent_map.py <dataset.csv> tsne --target <label_column> --output <output.tmap>
```

**Generate tangent maps directly (legacy method):**
```bash
python src/featurewind/TangentMap.py <input.csv> tsne
```

## Complete Application Workflow

### **Phase 1: Data Preprocessing** (Optional - if starting from raw CSV)

#### Step 1.1: Data Loading (`generate_tangent_map.py`)
```python
# Load CSV dataset with numeric features
data = pd.read_csv("dataset.csv")
# Extract features and optional target labels
features = data.drop(columns=['target'])  # Numeric features only
targets = data['target']  # Optional categorical labels
```

#### Step 1.2: Data Normalization
```python
# Normalize features to [0,1] range for stable gradient computation
normalized_features = (features - features.min()) / (features.max() - features.min())
```

#### Step 1.3: Tangent Map Generation (`TangentMap.py`)
```python
# Apply dimensionality reduction with gradient tracking
# Uses PyTorch autograd to compute gradients through t-SNE/UMAP
embedding_2d, gradients = compute_embedding_with_gradients(normalized_features)
# Save as .tmap file: {"tmap": [...], "Col_labels": [...]}
```

**Output**: `.tmap` files containing 2D projections + gradient vectors for each point

### **Phase 2: Visualization Initialization** (`main_modular.py`)

#### Step 2.1: Configuration Setup (`config.py`)
```python
# Initialize global parameters
velocity_scale = 0.04          # Controls flow speed
grid_res = 40                  # Spatial discretization resolution
num_particles = 800            # Number of flow particles
particle_lifetime = 200        # Frames before particle respawn
```

#### Step 2.2: Data Loading (`data_processing.py`)
```python
def preprocess_tangent_map(tangent_map_path):
    # Load .tmap file and create TangentPoint objects
    with open(tangent_map_path, 'r') as f:
        data = json.load(f)
    
    # Extract components
    tmap_entries = data['tmap']           # Point data with gradients
    feature_labels = data['Col_labels']   # Feature names
    
    # Create validated TangentPoint objects
    points = [TangentPoint(entry, 1.0, feature_labels) for entry in tmap_entries]
    valid_points = [p for p in points if p.valid]
    
    # Extract arrays for processing
    positions = np.array([p.position for p in valid_points])              # (N, 2)
    gradient_vectors = np.array([p.gradient_vectors for p in valid_points]) # (N, M, 2)
    
    return valid_points, gradient_vectors, positions, feature_labels
```

#### Step 2.3: Feature Selection (`data_processing.py`)
```python
def pick_top_k_features(all_grad_vectors, k):
    # Compute average gradient magnitude across all points for each feature
    feature_magnitudes = np.linalg.norm(all_grad_vectors, axis=2)  # (N, M)
    avg_magnitudes = feature_magnitudes.mean(axis=0)               # (M,)
    
    # Select top k features by average magnitude
    top_k_indices = np.argsort(-avg_magnitudes)[:k]
    return top_k_indices, avg_magnitudes
```

**Key Insight**: Features with higher average gradient magnitudes have more influence on the flow field.

### **Phase 3: Spatial Discretization** (`grid_computation.py`)

#### Step 3.1: Grid Setup
```python
def create_grid_coordinates(grid_res):
    # Create cell-center grid (not corner-based) for better interpolation
    xmin, xmax, ymin, ymax = bounding_box
    cell_centers_x = np.linspace(xmin + dx/2, xmax - dx/2, grid_res)
    cell_centers_y = np.linspace(ymin + dy/2, ymax - dy/2, grid_res)
    grid_x, grid_y = np.meshgrid(cell_centers_x, cell_centers_y)
    return grid_x, grid_y, cell_centers_x, cell_centers_y
```

#### Step 3.2: Adaptive Masking
```python
def compute_adaptive_threshold(positions, kdtree):
    # Determine distance threshold based on local point density
    k = min(5, len(positions))  # Use k-nearest neighbors
    distances, _ = kdtree.query(positions, k=k+1)  # k+1 because first is self
    local_densities = distances[:, 1:].mean(axis=1)  # Average k-NN distance
    
    # Use 75th percentile as threshold - areas beyond this are masked
    threshold = np.percentile(local_densities, 75)
    return threshold
```

**Purpose**: Prevents spurious flow in regions far from actual data points.

#### Step 3.3: Velocity Field Construction
```python
def interpolate_feature_onto_grid(positions, vectors, grid_x, grid_y, threshold, dist_grid):
    # Linear interpolation of gradient vectors onto regular grid
    grid_u = griddata(positions, vectors[:, 0], (grid_x, grid_y), method='linear', fill_value=0.0)
    grid_v = griddata(positions, vectors[:, 1], (grid_x, grid_y), method='linear', fill_value=0.0)
    
    # Apply adaptive masking with smooth boundaries
    magnitude_grid = np.sqrt(grid_u**2 + grid_v**2)
    smooth_magnitude = gaussian_filter(magnitude_grid, sigma=0.3)
    
    # Create composite mask: distance-based + magnitude-based
    distance_mask = dist_grid > threshold
    smooth_mask = binary_closing(distance_mask, structure=np.ones((3,3)))
    magnitude_threshold = np.percentile(smooth_magnitude[~smooth_mask], 10)
    final_mask = smooth_mask | (smooth_magnitude < magnitude_threshold)
    
    # Zero out masked regions
    grid_u[final_mask] = 0.0
    grid_v[final_mask] = 0.0
    return grid_u, grid_v
```

#### Step 3.4: Feature Dominance Analysis
```python
def compute_soft_dominance(cell_mags, temperature=2.0):
    # Compute probabilistic feature dominance using softmax
    cell_mags_safe = cell_mags + 1e-8  # Avoid division by zero
    softmax_scores = np.exp(cell_mags_safe / temperature)
    softmax_probs = softmax_scores / np.sum(softmax_scores)
    dominant_idx = np.argmax(cell_mags)
    return softmax_probs, dominant_idx
```

**Key Insight**: Soft dominance reveals uncertainty - high entropy indicates competing features.

### **Phase 4: Particle System Initialization** (`particle_system.py`)

#### Step 4.1: Particle Creation
```python
def create_particles(num_particles=800):
    # Random initialization throughout domain
    xmin, xmax, ymin, ymax = bounding_box
    positions = np.column_stack((
        np.random.uniform(xmin, xmax, size=num_particles),
        np.random.uniform(ymin, ymax, size=num_particles)
    ))
    
    # Initialize trajectory history for trail visualization
    tail_length = 10
    histories = np.full((num_particles, tail_length + 1, 2), np.nan)
    histories[:, :] = positions[:, None, :]  # Initialize all history to current position
    
    lifetimes = np.zeros(num_particles, dtype=int)
    return {'positions': positions, 'histories': histories, 'lifetimes': lifetimes}
```

### **Phase 5: Physics Simulation Loop** (`particle_system.py`)

#### Step 5.1: Velocity Interpolation
```python
def get_velocity_at_positions(positions, system, interp_u_sum, interp_v_sum):
    # Bilinear interpolation using RegularGridInterpolator
    U = interp_u_sum(positions)  # Query velocity at particle positions
    V = interp_v_sum(positions)
    return np.column_stack((U, V)) * velocity_scale
```

#### Step 5.2: Adaptive RK4 Integration
```python
def adaptive_rk4_step(pos, target_dt, get_vel_func, grid_res):
    # Calculate current velocity and speed
    vel = get_vel_func(pos)
    speed = np.linalg.norm(vel, axis=1)
    
    # CFL condition for stability: dt ≤ CFL_number * (cell_size / |velocity|)
    cell_size = min(bounding_box[1] - bounding_box[0], 
                    bounding_box[3] - bounding_box[2]) / grid_res
    cfl_number = 0.5  # Conservative choice
    max_speed = np.maximum(speed, 1e-6)
    cfl_dt = cfl_number * cell_size / max_speed
    dt = max(min(target_dt, np.min(cfl_dt)), 1e-4)  # Ensure minimum time step
    
    # Classical RK4 steps
    k1 = get_vel_func(pos)
    k2 = get_vel_func(pos + 0.5 * dt * k1)
    k3 = get_vel_func(pos + 0.5 * dt * k2) 
    k4 = get_vel_func(pos + dt * k3)
    rk4_result = pos + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    # Embedded Heun method for error estimation (RK2)
    heun_k1 = k1
    heun_k2 = get_vel_func(pos + dt * heun_k1)
    heun_result = pos + dt * (heun_k1 + heun_k2) / 2.0
    
    # Estimate truncation error
    error = np.linalg.norm(rk4_result - heun_result, axis=1)
    max_error = np.max(error)
    
    return rk4_result, dt, max_error, (k1 + 2*k2 + 2*k3 + k4) / 6.0
```

**Mathematical Foundation**: 
- **RK4**: 4th-order accuracy, excellent stability for smooth flows
- **CFL Condition**: Ensures particles don't "jump" across multiple grid cells
- **Error Control**: Embedded method provides error estimates for adaptive stepping

#### Step 5.3: Particle Lifecycle Management
```python
def update_particles(system, interp_u_sum, interp_v_sum, grid_res):
    # Adaptive integration with error control
    total_time, target_total_time = 0.0, 1.0
    current_pos = system['positions'].copy()
    
    while total_time < target_total_time and step_count < 10:  # Max 10 sub-steps
        new_pos, dt_used, error_est, vel_est = adaptive_rk4_step(...)
        
        if error_est < 0.01 or dt_used < 1e-3:  # Accept step
            current_pos = new_pos
            total_time += dt_used
        else:  # Reduce time step and retry
            target_dt *= 0.5
    
    # Update positions and history
    system['positions'][:] = current_pos
    system['histories'][:, :-1, :] = system['histories'][:, 1:, :]  # Shift history
    system['histories'][:, -1, :] = current_pos  # Add new position
    
    # Reinitialize out-of-bounds or over-age particles
    reinitialize_particles(system)
    
    # Density-aware reseeding for temporal coherence
    density_aware_reseed(system, grid_res)
```

#### Step 5.4: Smart Reseeding Algorithm
```python
def density_aware_reseed(system, grid_res):
    # Create coarse density grid for efficiency
    density_res = max(8, grid_res // 4)
    density_grid = np.zeros((density_res, density_res))
    
    # Compute particle density
    for pos in system['positions']:
        grid_x = int((pos[0] - xmin) / (xmax - xmin) * (density_res - 1))
        grid_y = int((pos[1] - ymin) / (ymax - ymin) * (density_res - 1))
        density_grid[grid_y, grid_x] += 1
    
    # Compute flow divergence: div = ∂u/∂x + ∂v/∂y
    divergence_grid = np.zeros((density_res, density_res))
    for i in range(1, density_res - 1):
        for j in range(1, density_res - 1):
            h = min((xmax - xmin) / density_res, (ymax - ymin) / density_res)
            u_right = interp_u_sum([[x + h, y]])[0]
            u_left = interp_u_sum([[x - h, y]])[0]
            v_up = interp_v_sum([[x, y + h]])[0]
            v_down = interp_v_sum([[x, y - h]])[0]
            divergence_grid[i, j] = (u_right - u_left)/(2*h) + (v_up - v_down)/(2*h)
    
    # Priority reseeding: low density + high divergence regions
    target_density = len(system['positions']) / (density_res * density_res)
    reseed_candidates = []
    for i in range(density_res):
        for j in range(density_res):
            current_density = density_grid[i, j]
            div_factor = max(0, divergence_grid[i, j])  # Only divergent regions
            if current_density < target_density * 0.7:
                weight = (target_density - current_density) * (1 + div_factor * 0.5)
                reseed_candidates.append((i, j, weight))
    
    # Limit reseeding to 2% per frame for temporal coherence
    max_reseed = int(0.02 * len(system['positions']))
    # ... respawn particles in high-priority cells
```

**Key Insight**: Divergent regions (sources) need more particles, convergent regions (sinks) need fewer.

### **Phase 6: Interactive Visualization** (`visualization_core.py` & `ui_controls.py`)

#### Step 6.1: Figure Layout Setup
```python
def setup_figure_layout():
    # Create figure with main plot (2/3 width) + wind vane (1/3 width)
    fig = plt.figure(figsize=(16, 8))
    ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)  # Main plot
    ax2 = plt.subplot2grid((2, 3), (0, 2), rowspan=2)             # Wind vane
    return fig, ax1, ax2
```

#### Step 6.2: Main Plot Preparation
```python
def prepare_figure(ax, valid_points, col_labels, feature_colors, lc):
    # Set equal aspect ratio and remove ticks for clean look
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Plot data points with different markers for each class
    unique_labels = sorted(set(p.tmap_label for p in valid_points))
    markers = ["o", "s", "^", "D", "v", "<", ">"]
    
    for i, label in enumerate(unique_labels):
        positions = np.array([p.position for p in valid_points if p.tmap_label == label])
        ax.scatter(positions[:, 0], positions[:, 1], 
                  marker=markers[i % len(markers)], color="gray", s=10, 
                  label=f"Label {label}", zorder=4)
    
    # Add particle line collection for flow visualization
    ax.add_collection(lc)
```

#### Step 6.3: Wind Vane Visualization
```python
def update_wind_vane(ax2, mouse_data, system, col_labels, selected_features, feature_colors):
    # Show feature vectors at current grid cell under mouse cursor
    cell_i, cell_j = mouse_data['grid_cell']
    
    # Extract all feature vectors for this grid cell
    vectors_all = []
    for feat_idx in range(len(col_labels)):
        u_val = system['grid_u_all_feats'][feat_idx, cell_i, cell_j]
        v_val = system['grid_v_all_feats'][feat_idx, cell_i, cell_j]
        mag = np.sqrt(u_val**2 + v_val**2)
        vectors_all.append({'vector': [u_val, v_val], 'mag': mag, 'feat_idx': feat_idx})
    
    # Scale vectors to fit in unit circle
    max_mag = max(info['mag'] for info in vectors_all)
    scale_factor = 0.45 / max_mag if max_mag > 0 else 1.0
    
    # Draw vectors as arrows from center
    for info in vectors_all:
        if info['mag'] > max_mag * 0.1:  # Only show significant vectors
            scaled_vector = np.array(info['vector']) * scale_factor
            color = feature_colors[info['feat_idx']] if info['feat_idx'] < len(feature_colors) else 'black'
            
            # Draw arrow from center to scaled endpoint
            ax2.annotate('', xy=scaled_vector, xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    # Draw unit circle for reference
    circle = plt.Circle((0, 0), 0.5, fill=False, color='lightgray', linewidth=1)
    ax2.add_patch(circle)
```

#### Step 6.4: Interactive Controls (`ui_controls.py`)
```python
class UIController:
    def setup_ui_controls(self):
        # Mode selection: Top-K vs Direction-Conditioned
        ax_mode = self.fig.add_axes([0.05, 0.02, 0.25, 0.06])
        self.mode_radio = RadioButtons(ax_mode, ('Top-K Mode', 'Direction-Conditioned Mode'))
        
        # Top-K mode: slider for number of features
        ax_k = self.fig.add_axes([0.35, 0.02, 0.30, 0.03])
        self.k_slider = Slider(ax_k, 'Top k Features', 1, len(self.col_labels), valinit=5)
        
        # Direction-Conditioned mode: angle and magnitude sliders
        ax_angle = self.fig.add_axes([0.70, 0.06, 0.25, 0.03])
        self.angle_slider = Slider(ax_angle, 'Direction (°)', 0, 360, valinit=0)
        
        ax_magnitude = self.fig.add_axes([0.70, 0.02, 0.25, 0.03])
        self.magnitude_slider = Slider(ax_magnitude, 'Magnitude', 0.1, 2.0, valinit=1.0)
```

#### Step 6.5: Direction-Conditioned Mode
```python
def optimize_features_for_constraints(self):
    # User clicks grid cells and specifies desired flow direction
    # Algorithm finds best feature combination to match constraints
    
    for step in range(max_features):
        best_candidate = None
        best_score = -float('inf')
        
        # Greedy forward selection: try adding each remaining feature
        for feat_idx in remaining_features:
            candidate_features = current_features + [feat_idx]
            score = self.evaluate_feature_combination(candidate_features)
            if score > best_score:
                best_score = score
                best_candidate = feat_idx
        
        if best_candidate and best_score > current_best:
            current_features.append(best_candidate)
        else:
            break  # No improvement
    
    # Apply optimized feature combination
    self.apply_feature_combination(current_features)
```

### **Phase 7: Advanced Features** (Recent Additions)

#### Feature Family Clustering (`feature_clustering.py`)
```python
def cluster_features_hierarchically(gradient_vectors, n_families=6):
    # Compute feature similarity using cosine distance
    feature_sims = compute_feature_similarity_matrix(gradient_vectors)
    
    # Hierarchical clustering with Ward linkage
    linkage_matrix = hierarchy.linkage(1 - feature_sims, method='ward')
    
    # Cut dendrogram to get n_families clusters
    family_assignments = hierarchy.fcluster(linkage_matrix, n_families, criterion='maxclust')
    
    return family_assignments, linkage_matrix
```

**Purpose**: Groups related features into families for coordinated visualization and analysis.

#### Color System (`color_system.py`)
```python
# Paul Tol's colorblind-safe palette
PAUL_TOL_FAMILIES = [
    "#4477AA",  # Blue - Family 0
    "#EE6677",  # Red - Family 1  
    "#228833",  # Green - Family 2
    "#CCBB44",  # Yellow - Family 3
    "#66CCEE",  # Cyan - Family 4
    "#AA3377"   # Purple - Family 5
]

def encode_magnitude_in_lightness(base_color, magnitude, max_magnitude):
    # Convert to LCH color space for perceptually uniform lightness
    lch = rgb_to_lch(base_color)
    # Map magnitude to lightness (L* in [30, 90])
    lch[0] = 30 + (magnitude / max_magnitude) * 60
    return lch_to_rgb(lch)
```

**Key Insight**: LCH color space provides perceptually uniform transitions for data visualization.

#### Event Manager (`event_manager.py`)
```python
class EventManager:
    def __init__(self):
        self.mouse_handlers = []
        self.key_handlers = []
        self.update_handlers = []
        
    def on_mouse_move(self, event):
        # Centralized mouse event handling with error recovery
        for handler in self.mouse_handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Handler error: {e}")
                continue
```

**Purpose**: Provides robust event handling with error isolation to prevent UI freezes.

### **Phase 8: Animation Loop** (`main_modular.py`)

#### Step 8.1: Frame Update Function
```python
def update_frame(frame):
    # Called once per animation frame (typically 30 FPS)
    return particle_system.update_particles(
        system, interp_u_sum, interp_v_sum, grid_u_sum, grid_v_sum, grid_res)

# Create matplotlib animation
anim = FuncAnimation(fig, update_frame, frames=1000, interval=30, blit=False)
```

#### Step 8.2: Trail Visualization Update
```python
def update_particle_visualization(system, velocity):
    # Build line segments connecting trajectory points
    segments = np.zeros((n_particles * tail_length, 2, 2))
    colors_rgba = np.zeros((n_particles * tail_length, 4))
    
    speeds = np.linalg.norm(velocity, axis=1)
    max_speed = speeds.max() + 1e-9
    
    for i in range(n_particles):
        alpha_part = speeds[i] / max_speed  # Speed-based opacity
        
        for t in range(tail_length):
            seg_idx = i * tail_length + t
            # Connect consecutive history points
            segments[seg_idx, 0, :] = histories[i, t, :]
            segments[seg_idx, 1, :] = histories[i, t + 1, :]
            
            # Fade older segments
            age_factor = (t+1) / tail_length
            alpha_final = max(0.15, alpha_part * age_factor * 0.7)
            colors_rgba[seg_idx] = [0, 0, 0, alpha_final]  # Black with alpha
    
    # Update line collection for efficient rendering
    line_collection.set_segments(segments)
    line_collection.set_colors(colors_rgba)
```

## **Key Insights and Convergence Analysis**

### **What Trail Convergence Reveals**
1. **Point Attractors**: Multiple trails → same region = **data clusters**
2. **Line Attractors**: Trails → ridges/lines = **manifold structure** 
3. **Slow Convergence**: Spiraling patterns = **flat regions**, **noise**
4. **Divergence**: Spreading trails = **saddle points**, **boundaries**
5. **Feature Dominance**: **Wind vane** shows which features drive convergence

### **Multi-Scale Analysis**
- **velocity_scale = 0.1**: Fast convergence → major attractors
- **velocity_scale = 0.01**: Slow convergence → fine structure  
- **velocity_scale = 0.001**: Very slow → local neighborhoods

### **Practical Applications**
1. **Cluster Discovery**: Convergence points = natural groupings
2. **Feature Relationships**: Parallel flows = cooperating features
3. **Outlier Detection**: Isolated convergence = anomalies
4. **Quality Assessment**: Smooth patterns = good dimensionality reduction

## **File Formats and Data Structures**

### **Input Files**
- **CSV datasets**: Numeric features + optional categorical target column
- **Normalization**: Features scaled to [0,1] for gradient stability

### **Tangent Map Format** (`.tmap` JSON files)
```json
{
  "tmap": [
    {
      "domain": [val1, val2, ..., valM],     // Original feature values
      "range": [x, y],                       // 2D projection coordinates  
      "tangent": [[dx1/dt, dy1/dt], [dx2/dt, dy2/dt], ...],  // Gradients per feature
      "label": "class_name"                  // Optional class label
    },
    ...
  ],
  "Col_labels": ["feature1", "feature2", ...] // Feature names
}
```

### **Output Files**
- **PNG images**: Static frames and final visualization
- **CSV files**: Grid data, dominant features, cell classifications
- **Interactive animation**: Real-time matplotlib visualization

## **Dependencies and Requirements**

The project requires:
- **PyTorch**: Automatic differentiation for gradient computation through dimensionality reduction
- **NumPy/SciPy**: Numerical operations, interpolation, spatial data structures
- **Matplotlib**: Visualization, animation, interactive widgets
- **Scikit-learn**: Data preprocessing, normalization
- **Optional: colorcet**: Enhanced color palettes for feature visualization

## **Performance Considerations**

### **Computational Complexity**
- **Grid resolution**: O(grid_res²) memory, O(grid_res² × M) computation for M features  
- **Particle count**: O(num_particles) per frame
- **RK4 integration**: O(num_particles × sub_steps) per frame
- **Reseeding analysis**: O(density_res²) per frame

### **Optimization Strategies**
1. **Adaptive thresholding**: Mask irrelevant regions
2. **Cell-center convention**: Better interpolation stability
3. **Coarse density grid**: Efficient reseeding analysis
4. **CFL condition**: Prevents numerical instability
5. **Limited reseeding**: Maximum 2% per frame for temporal coherence

## **Usage Examples and Best Practices**

### **Basic Usage**
```bash
# Generate tangent map from CSV data
python examples/generate_tangent_map.py data.csv tsne --target class --output data.tmap

# Run interactive visualization
python examples/main_modular.py
```

### **Advanced Analysis**
```python
# Adjust parameters for different insights
config.velocity_scale = 0.01  # Slow, detailed flow for fine structure
config.PARTICLE_LIFETIME = 500  # Longer observation period
config.DEFAULT_NUM_PARTICLES = 1200  # More particles for denser visualization
```

### **Interpretation Guidelines**
1. **Convergence Patterns**: Look for regions where trails consistently end
2. **Flow Strength**: Brighter/faster particles indicate stronger gradients
3. **Feature Competition**: Wind vane shows competing influences at boundaries
4. **Stability**: Consistent patterns indicate reliable dimensionality reduction
5. **Outliers**: Isolated flow patterns may indicate data anomalies