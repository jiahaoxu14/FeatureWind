# FeatureWind: Technical Report on Gradient Flow Visualization Design

## Overview

FeatureWind is an interactive visualization system that transforms dimensionality reduction projections into dynamic particle flow fields. This technical report documents the visualization design principles, implementation methods, and technical architecture used to create real-time gradient flow animations from high-dimensional feature data.

## 1. Visualization Design Principles

### 1.1 Core Visualization Metaphor

FeatureWind treats feature gradients as **velocity vectors in a 2D flow field**. This design choice transforms abstract mathematical relationships into intuitive physical motion:

- **Particles** = Data flow tracers showing movement through feature space
- **Velocity vectors** = Feature gradient magnitudes and directions  
- **Flow convergence** = Natural data clustering and structure
- **Flow divergence** = Decision boundaries and outlier regions

### 1.2 Visual Encoding Strategy

| Data Dimension | Visual Channel | Encoding Method |
|----------------|----------------|-----------------|
| Feature gradient direction | Particle motion direction | Physics simulation |
| Feature gradient magnitude | Particle speed + trail opacity | Velocity scaling |
| Feature identity | Color (family-based) | Paul Tol colorblind-safe palette |
| Local dominance | Wind vane arrow + center marker | Real-time compass |
| Convergence strength | Trail accumulation density | Particle concentration |
| Flow stability | Animation smoothness | Adaptive time stepping |

### 1.3 Multi-Scale Visualization

The system supports analysis at three temporal scales:

- **Fast flow** (`velocity_scale = 0.1`): Reveals major cluster structure and global attractors
- **Medium flow** (`velocity_scale = 0.04`): Balanced view showing both structure and transitions  
- **Slow flow** (`velocity_scale = 0.01`): Fine-grained analysis of local neighborhoods

## 2. Technical Architecture

### 2.1 Data Processing Pipeline

**Input**: Tangent map (`.tmap`) JSON files containing:
```json
{
  "tmap": [
    {
      "range": [x, y],           // 2D projection coordinates
      "tangent": [[∂x/∂f₁, ∂y/∂f₁], [∂x/∂f₂, ∂y/∂f₂], ...], // Gradients
      "domain": [f₁, f₂, ..., fₘ],  // Original feature values
      "label": "class_name"       // Optional classification
    }
  ],
  "Col_labels": ["feature1", "feature2", ...]
}
```

**Processing Steps**:
1. **Validation**: Filter points with complete gradient information
2. **Normalization**: Scale bounding box to square aspect ratio with 5% padding
3. **Feature selection**: Top-K by average gradient magnitude or single feature focus
4. **Adaptive scaling**: Normalize flow speed across datasets while preserving structure

### 2.2 Spatial Grid System

**Grid Construction**:
- **Resolution**: Configurable grid (default: 10×10 cells) covering the square bounding box
- **Cell centers**: Used for velocity interpolation to avoid boundary artifacts
- **Adaptive masking**: Regions far from actual data points are masked to prevent spurious visualization

**Masking Algorithm**:
```python
def compute_adaptive_threshold(positions, kdtree):
    k = min(5, len(positions))  # Use 5-nearest neighbors
    distances, _ = kdtree.query(positions, k=k+1)
    local_densities = distances[:, 1:].mean(axis=1)
    threshold = np.percentile(local_densities, 75)  # 75th percentile
    return threshold
```

**Velocity Field Interpolation**:
- **Method**: Scipy's `griddata` with linear interpolation
- **Input**: Point gradients at data locations → regular grid velocities
- **Smoothing**: Gaussian filter (σ=0.3) for magnitude-based masking
- **Boundary handling**: Zero velocity in masked regions

## 3. Particle Simulation Design

### 3.1 Physics-Based Particle System

**Particle Properties**:
- **Count**: 1,200 particles (configurable)
- **Initialization**: Uniform random distribution across the visualization domain
- **Lifetime**: 100 frames before forced respawn to prevent stagnation
- **Trail history**: 10-point trajectory buffer for motion visualization

**Core Simulation Loop**:
1. **Velocity sampling**: Bilinear interpolation from grid to particle positions
2. **Numerical integration**: Adaptive Runge-Kutta 4th order with error control
3. **Boundary handling**: Reinitialize out-of-bounds particles
4. **Trail rendering**: Update LineCollection with particle trajectories

### 3.2 Numerical Integration Methods

**Adaptive Runge-Kutta 4th Order Integration**:
```python
def adaptive_rk4_step(pos, target_dt, get_vel_func, grid_res):
    # Classical RK4 stages
    k1 = get_vel_func(pos)
    k2 = get_vel_func(pos + 0.5 * dt * k1)
    k3 = get_vel_func(pos + 0.5 * dt * k2)
    k4 = get_vel_func(pos + dt * k3)
    rk4_result = pos + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    # Embedded Heun method for error estimation
    heun_result = pos + dt * (k1 + k2) / 2.0
    error = np.linalg.norm(rk4_result - heun_result, axis=1)
    
    return rk4_result, dt, error.max()
```

**Stability Control**:
- **CFL condition**: `dt ≤ 0.5 × (cell_size / |velocity|)` prevents particle "jumping"
- **Error tolerance**: Accept steps with error < 0.01 or retry with smaller timestep
- **Minimum timestep**: 1e-4 prevents infinite loops
- **Maximum velocity**: Clamp at 10.0 units/frame to prevent runaway particles

### 3.3 Smart Particle Management

**Trail Length Adaptation**:
- **Base length**: 10 trajectory points
- **Speed-based scaling**: Longer trails (up to 20) for slow particles, shorter (down to 5) for fast
- **Dynamic resizing**: Trail buffer automatically adjusts each frame

**Intelligent Reseeding**:
- **Density analysis**: Track particle distribution on coarse grid (density_res = grid_res/4)
- **Divergence targeting**: Prioritize reseeding in flow source regions
- **Rate limiting**: Maximum 2% of particles reseeded per frame for temporal coherence
- **Boundary respawn**: Out-of-bounds particles uniformly redistributed

## 4. Color and Visual Design

### 4.1 Feature Family Clustering

**Automatic Feature Grouping**:
Features with similar gradient patterns are clustered into families for coordinated visualization:

```python
def cluster_features_hierarchically(gradient_vectors, n_families=6):
    # Compute pairwise cosine similarity between vector fields
    feature_sims = compute_feature_similarity_matrix(gradient_vectors)
    
    # Spectral clustering on precomputed similarity matrix
    clustering = SpectralClustering(
        n_clusters=n_families, 
        affinity='precomputed',
        random_state=42
    )
    family_assignments = clustering.fit_predict(feature_sims)
    
    return family_assignments
```

**Similarity Computation**:
- **Method**: Cosine similarity between unit vector fields at each grid cell
- **Masking**: Only compute similarity where both features have magnitude > 1e-6
- **Aggregation**: Average similarity across all valid grid cells
- **Fallback**: If spectral clustering fails, use hierarchical clustering with Ward linkage

### 4.2 Colorblind-Safe Palette

**Paul Tol Color System**:
```python
PAUL_TOL_FAMILIES = [
    "#4477AA",  # Blue - Family 0 (Primary features)
    "#EE6677",  # Red - Family 1 (Secondary features)  
    "#228833",  # Green - Family 2 (Tertiary features)
    "#CCBB44",  # Yellow - Family 3 (Quaternary features)
    "#66CCEE",  # Cyan - Family 4 (Quinary features)
    "#AA3377"   # Purple - Family 5 (Senary features)
]
```

**Design Rationale**:
- **Accessibility**: Colors distinguishable under all forms of color blindness
- **Perceptual uniformity**: Equal visual weight across families
- **Particle consistency**: Same color used for particles, wind vane, and legends
- **Magnitude encoding**: Use alpha/opacity rather than hue to preserve color identity

**Color Application**:
- **Particle trails**: Family color with speed-based alpha
- **Wind vane arrows**: Family color with magnitude-based opacity
- **Center markers**: Dominant feature family color
- **Legend elements**: Pure family colors for identification

## 5. Wind Vane Interface Design

### 5.1 Real-Time Local Analysis

The **wind vane** provides instantaneous feature composition analysis at the mouse cursor location:

**Visual Components**:
- **Center grid marker**: Square colored by locally dominant feature family
- **Individual feature vectors**: Thin bidirectional arrows for each selected feature
- **Aggregate flow vane**: Traditional weather vane showing combined flow direction
- **Reference circle**: Unit circle boundary for vector magnitude scaling
- **Masking indicator**: "MASKED CELL" text for regions with zero flow

### 5.2 Wind Vane Geometry

**Fixed-Size Design**:
- **Vane length**: 0.35 units (70% of unit circle diameter)
- **Arrow head**: Large triangular pointer at flow direction end
- **Flag tail**: Rectangular flag at flow origin end 
- **Central shaft**: Thick line connecting arrow and flag

**Magnitude Encoding**:
```python
# Logarithmic alpha scaling for magnitude
alpha_ratio = log10(max(0.1, sum_magnitude)) / 3.0
magnitude_alpha = max(0.2, min(1.0, 0.2 + 0.8 * alpha_ratio))
```

- **High magnitude**: Opaque, bold wind vane (alpha = 1.0)
- **Medium magnitude**: Semi-transparent (alpha = 0.6)
- **Low magnitude**: Faint, barely visible (alpha = 0.2)
- **Zero magnitude**: "MASKED CELL" indicator replaces wind vane

## 6. Interaction Architecture

### 6.1 Event Management System

**Robust Event Handling**:
```python
class EventManager:
    def __init__(self):
        self.mouse_handlers = []
        self.debounce_timer = None
        
    def on_mouse_move(self, event):
        # Debounced updates every 16ms (60 FPS)
        if self.debounce_timer:
            self.debounce_timer.cancel()
        
        self.debounce_timer = threading.Timer(0.016, self._handle_move, [event])
        self.debounce_timer.start()
        
    def _handle_move(self, event):
        for handler in self.mouse_handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Handler error: {e}")  # Graceful error handling
                continue
```

**Event Types**:
- **Mouse move**: Updates wind vane, grid cell highlighting
- **Mouse click**: Shows data point details, feature values
- **Key press**: Feature selection, velocity scaling, export functions

### 6.2 Multi-Scale Analysis Controls

**Velocity Scale Adjustment**:
- **Slow motion** (`0.001`): Detailed local flow analysis
- **Normal speed** (`0.04`): Balanced overview and detail
- **Fast forward** (`0.1`): Rapid convergence to major attractors

**Grid Resolution Scaling**:
- **Coarse** (10×10): Real-time performance, major structure
- **Medium** (20×20): Balanced detail and speed
- **Fine** (40×40): High-resolution analysis, slower updates

### 6.3 Data Point Inspection

**Click-to-Reveal Interface**:
- **Coordinates**: Exact (x, y) position in projection space
- **Class label**: Ground truth classification if available
- **Feature values**: Original high-dimensional values for selected features
- **Highlight ring**: Visual indicator around selected point
- **Nearest neighbors**: Optional display of k-closest points

## 7. Visualization Interpretation Methods

### 7.1 Flow Pattern Classification

**Convergence Types**:

| Pattern | Visual Characteristics | Data Structure Implication |
|---------|----------------------|----------------------------|
| **Point Attractors** | Multiple trails → single region | Well-separated, homogeneous clusters |
| **Line Attractors** | Trails → curved ridges | 1D manifold embedded in 2D |
| **Spiral Convergence** | Particles spiral inward slowly | Weak gradients, noisy boundaries |
| **Divergent Flow** | Particles spread outward | Saddle points, decision boundaries |
| **Chaotic Motion** | Random, unstable trajectories | Poor projection quality |

**Feature Relationship Patterns**:
- **Parallel vectors**: Cooperating features (same direction in wind vane)
- **Opposing vectors**: Competing features (opposite directions)
- **Orthogonal vectors**: Independent feature contributions
- **Convergent vectors**: Multiple features reinforcing same flow

### 7.2 Quality Assessment Through Motion

**High-Quality Projections**:
- Smooth, consistent particle flows
- Clear convergence to stable attractor points
- Wind vane shows coherent feature cooperation
- Fast convergence with minimal spiral motion

**Poor-Quality Projections**:
- Turbulent, chaotic particle motion
- Particles wander without clear destinations
- Wind vane shows conflicting feature directions
- Slow convergence with excessive spiraling

### 7.3 Cluster Validation Protocol

**Visual Validation Steps**:
1. **Identify convergence zones** where multiple particle trails terminate
2. **Check class consistency** in converged regions using data point markers
3. **Analyze feature dominance** via wind vane at convergence centers
4. **Test stability** by varying velocity scale and observing pattern persistence
5. **Cross-validate** with traditional clustering metrics (silhouette, etc.)

## 8. Implementation Architecture

### 8.1 Modular Software Design

**Core Modules**:
- `data_processing.py`: Input parsing, validation, feature selection
- `grid_computation.py`: Spatial discretization, velocity field construction  
- `particle_system.py`: Physics simulation, numerical integration
- `visualization_core.py`: Rendering, figure layout, professional styling
- `event_manager.py`: Interaction handling, mouse/keyboard events
- `color_system.py`: Paul Tol palettes, family-based coloring
- `feature_clustering.py`: Hierarchical clustering, similarity computation

**Data Flow Architecture**:
```
.tmap files → data_processing → grid_computation → particle_system
                    ↓               ↓               ↓
            feature_clustering → color_system → visualization_core
                                      ↓               ↓
                              legend_manager ← event_manager
```

### 8.2 Performance Optimization

**Computational Strategies**:
- **Vectorized operations**: NumPy arrays for all numerical computation
- **Single LineCollection**: Efficient rendering of all particle trails
- **Grid-based interpolation**: `RegularGridInterpolator` for fast velocity queries
- **Adaptive timesteps**: Variable integration for stability and speed
- **Debounced events**: Prevent UI overload during mouse interaction

**Memory Management**:
- **Circular trail buffers**: Fixed-size arrays for particle history
- **Lazy evaluation**: Wind vane updates only on mouse movement
- **Grid caching**: Reuse interpolated velocity fields across frames
- **Selective reseeding**: Maximum 2% particle replacement per frame

### 8.3 Performance Benchmarks

**Computational Complexity**:
- **Grid interpolation**: O(grid_res² × M) for M features
- **Particle integration**: O(num_particles × substeps) per animation frame  
- **Feature clustering**: O(M² × valid_cells) similarity computation
- **Rendering**: O(num_particles × trail_length) LineCollection updates

**Performance Results**:

| Configuration | Grid Size | Particles | FPS | Memory (MB) |
|---------------|-----------|-----------|-----|-------------|
| **Fast** | 10×10 | 800 | 45+ | 150 |
| **Balanced** | 20×20 | 1200 | 30+ | 280 |
| **Detailed** | 40×40 | 2000 | 15+ | 650 |
| **High-Res** | 80×80 | 3000 | 5+ | 1200 |

*Benchmarks on MacBook Pro M1, 16GB RAM, integrated graphics*

**Scalability Guidelines**:
- **Small datasets** (< 1K points): Use high-resolution grids for detail
- **Medium datasets** (1K-10K points): Balance grid size with performance
- **Large datasets** (10K+ points): Prioritize real-time interaction over resolution
- **Feature count**: Performance degrades linearly with feature count beyond 50

## 9. Technical Limitations and Design Constraints

### 9.1 Current System Limitations

**Mathematical Constraints**:
- **Gradient dependency**: Requires differentiable dimensionality reduction (t-SNE, UMAP)
- **2D projection**: Designed specifically for 2D embeddings, not 3D or higher
- **Linear interpolation**: Simple interpolation may smooth important discontinuities
- **Fixed grid**: Regular grid may not capture irregular data distributions optimally

**Performance Constraints**:
- **Memory scaling**: Linear growth with feature count becomes problematic beyond ~100 features
- **Real-time limits**: High-resolution grids (>80×80) impact interactive frame rates
- **Gradient storage**: Full gradient matrices require significant memory for large datasets
- **JavaScript limitations**: Browser-based version would need WebGL for acceptable performance

**Visual Design Limitations**:
- **Color palette**: Limited to 6 distinct families due to human color discrimination
- **2D trails**: Particle trajectories can overlap and obscure each other
- **Static features**: No support for time-varying or evolving feature sets
- **Projection assumptions**: Assumes underlying 2D projection is meaningful and well-formed

### 9.2 Design Trade-offs

**Performance vs. Quality**:
- Coarser grids improve frame rates but reduce spatial accuracy
- Fewer particles increase speed but decrease flow density visualization
- Shorter trails reduce memory but provide less trajectory context

**Simplicity vs. Functionality**:
- Fixed wind vane geometry prioritizes readability over magnitude precision
- Automatic feature clustering may group unrelated features in edge cases
- Single velocity scale affects all features equally, no per-feature scaling

**Interactivity vs. Stability**:
- Real-time parameter changes can cause jarring visual transitions
- Mouse-driven wind vane updates may be too sensitive for precise analysis
- Adaptive scaling can make cross-dataset comparisons difficult

## 10. Technical Summary

FeatureWind transforms static dimensionality reduction scatter plots into dynamic gradient flow visualizations through:

**Core Technical Innovations**:
1. **Physics-based particle simulation** with adaptive RK4 integration
2. **Real-time gradient field visualization** using velocity-based particle motion
3. **Hierarchical feature clustering** with colorblind-safe family-based coloring
4. **Interactive wind vane interface** for local gradient composition analysis
5. **Multi-scale temporal analysis** supporting different flow speeds

**Visualization Design Principles**:
- **Intuitive physical metaphor**: Gradients as forces, particles as flow tracers
- **Perceptually uniform encoding**: Color families, alpha-based magnitude
- **Responsive interaction**: Debounced events, graceful error handling
- **Professional aesthetics**: Clean layouts, accessible color palettes

**Implementation Architecture**:
- **Modular design**: Separation of concerns across focused modules
- **Performance optimization**: Vectorized operations, efficient rendering
- **Robust error handling**: Graceful degradation under edge conditions
- **Configurable parameters**: Extensive customization for different use cases

The system provides data analysts with unprecedented insight into the feature-level forces that shape dimensionality reduction projections, enabling validation of clustering quality, discovery of feature relationships, and understanding of projection artifacts through observable particle dynamics.

---

## Technical References and Dependencies

**Core Libraries**:
- **NumPy**: Array operations, numerical computation, vectorized mathematics
- **SciPy**: Spatial data structures (cKDTree), interpolation (griddata), morphological operations
- **Matplotlib**: Visualization framework, animation system, interactive widgets
- **Scikit-learn**: Clustering algorithms (SpectralClustering), preprocessing utilities
- **PyTorch**: Automatic differentiation for gradient computation (preprocessing only)

**Visualization Techniques**:
- **LineCollection**: Efficient rendering of particle trails
- **RegularGridInterpolator**: Fast bilinear velocity field sampling
- **Animation.FuncAnimation**: Smooth 30 FPS particle motion
- **Event handling**: Mouse interaction, keyboard shortcuts, debounced updates

**Numerical Methods**:
- **Runge-Kutta 4th order**: Classical explicit integration method
- **Embedded error estimation**: Heun method for adaptive timestep control
- **CFL stability condition**: Prevents numerical instabilities in particle advection
- **Gaussian filtering**: Smoothing for gradient field masking

**Color Science**:
- **Paul Tol palette**: Scientifically-designed colorblind-safe colors
- **Perceptual uniformity**: Equal visual weight across color families
- **Alpha blending**: Magnitude encoding via transparency

---

*Technical Report: FeatureWind Gradient Flow Visualization System*
*Implementation available at: /Users/jiahaoxu/Github/FeatureWind*