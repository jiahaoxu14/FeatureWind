"""
Configuration module for FeatureWind visualization.

This module contains global constants, default parameters, and configuration
settings used throughout the FeatureWind visualization system.
"""

import numpy as np

# Global configuration variables
velocity_scale = 1.2  # Original stable value
k = None  # Number of top features (will be set dynamically)
bounding_box = None  # Will be computed dynamically
real_feature_rgba = {}  # Feature to RGBA mapping for particles

# Default grid resolution
DEFAULT_GRID_RES = 25

# Particle system parameters
DEFAULT_NUM_PARTICLES = 1000  # Default particle count for trails
PARTICLE_LIFETIME = 25  # frames - increased to reduce respawn frequency
TAIL_LENGTH = 10  # number of position history points
SHOW_PARTICLES = True  # When False, hide particles and disable particle animation

# Animation parameters
ANIMATION_FRAMES = 200
ANIMATION_INTERVAL = 30  # milliseconds between frames (33 FPS - optimized for performance)

# Window title
WINDOW_TITLE = "FeatureWind Visualization"

# Color palettes and visualization constants
# Color palettes for feature visualization
GLASBEY_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948",
    "#AF7AA1", "#FF9D9A", "#9C755F", "#BAB0AC", "#1f77b4", "#ff7f0e",
    "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf"
]

# Grid computation parameters
MASK_BUFFER_FACTOR = 0.2    # Buffer size around data points for interpolation (in cell units)
# Unified masking threshold for summed magnitude checks
MASK_THRESHOLD = 1e-6

# Particle physics parameters
MAX_SAFE_VELOCITY = 10.0  # Maximum velocity magnitude per frame to prevent runaway particles

# UI and visualization parameters
MARKER_STYLES = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
DPI = 300  # For saved figures

TEMPERATURE_SOFTMAX = 2.0  # Temperature for soft dominance computation


# Feature selection limits
MAX_FEATURE_FAMILIES = 4      # Maximum number of feature families (reduced to 4)

# Visualization mode
# Supported: 'feature_wind_map' (default), 'dimreader'
VIS_MODE = 'feature_wind_map'

# Single feature color override used when --feature is specified
SINGLE_FEATURE_COLOR = "#EE6677"

# Feature Clock mode: show a second wind vane that displays
# only the top-N strongest feature vectors with distinct colors.
FEATURE_CLOCK_ENABLED = False
FEATURE_CLOCK_TOP_N = 4

# Coloring behavior
# When visualizing a small number of features, use distinct per-feature colors
# to avoid visual merging by family hue (helps interpretability for 2–10 features).
COLOR_BY_FEATURE_WHEN_FEW = True
# Threshold for switching to per-feature colors (count of selected features)
FEATURE_COLOR_DISTINCT_THRESHOLD = 5

# Always use distinct per-feature colors (no families)
USE_PER_FEATURE_COLORS = False

# Map overlay (gray background for unmasked cells)
# Toggle to show or hide the gray cell overlay in the main map view.
SHOW_UNMASKED_OVERLAY = False

# Grid lines toggle (main map)
# Draw interpolation cell boundary grid lines in the main map when True
SHOW_GRID_LINES = True

# Data points style (in main map view)
# Draw data points as solid markers with lower opacity for subtlety
SHOW_DATA_POINTS = True
HOLLOW_DATA_POINTS = False
# Alpha for solid data points (lower = more transparent)
DATA_POINT_ALPHA = 0.30
# Optional: color data points by a specific feature's value
# Accepts: None (disabled), an integer feature index, or a case-insensitive substring
# of the feature name to match the first hit.
DATA_POINT_COLOR_BY_FEATURE = None
# Colormap for value-to-color mapping. Options:
#  - 'grayscale' (default)
#  - any Matplotlib colormap name (e.g., 'viridis', 'magma', 'cividis', 'plasma', 'turbo')
DATA_POINT_COLOR_MAP = 'viridis'
# If True, invert mapping:
#  - grayscale: higher values become lighter (default: higher→darker)
#  - other colormaps: reverse the colormap (high→low end)
DATA_POINT_COLOR_INVERT = False
# Edge width for hollow markers
DATA_POINT_EDGEWIDTH = 0.6
# Marker size for data points in the main map (Matplotlib scatter 's' in points^2)
DATA_POINT_SIZE = 30

# Z-order for data points to ensure they render on top
DATA_POINT_ZORDER = 20

# Initial particle spawn markers (for visualization)
SHOW_INITIAL_SPAWNS = False
INITIAL_SPAWN_SIZE = 4           # matplotlib scatter 's' in points^2
INITIAL_SPAWN_COLOR = '#222222'   # solid fill color
INITIAL_SPAWN_ALPHA = 0.6         # 0..1
INITIAL_SPAWN_ZORDER = 26
SHOW_INITIAL_SPAWN_VECTORS = True
INITIAL_SPAWN_VEC_REL_SCALE = 0.03
INITIAL_SPAWN_VEC_ABS_SCALE = 0.0
INITIAL_SPAWN_VEC_ALPHA = 0.9
INITIAL_SPAWN_VEC_ZORDER = 27
INITIAL_SPAWN_VEC_WIDTH = 0.003

# Wind-vane style
# Use 'ring_dot' to draw a circle that encloses all feature vectors
# and a small circle on its edge to indicate flow direction.
WIND_VANE_STYLE = 'ring_dot'  # options: 'ring_dot' (recommended), 'needle'
WIND_VANE_CIRCLE_RADIUS = 0.055  # radius of the direction dot (axes units)
WIND_VANE_CIRCLE_GUIDE = False   # draw a faint center→dot guide line
WIND_VANE_RING_SCALE = 1.04      # multiply max feature-vector radius to pad the ring
WIND_VANE_RING_MAX_R = 0.66      # clamp ring radius (reference ring is 0.7)
WIND_VANE_RING_COLOR = '#999999' # ring line color
WIND_VANE_USE_CONVEX_HULL = True  # when False, do not de-emphasize non-hull vectors
WIND_VANE_SHOW_HULL = False       # when True and convex hull is used, draw the hull boundary
WIND_VANE_HULL_COLOR = '#727272'  # face color for hull fill
WIND_VANE_HULL_ALPHA = 0.3       # face alpha for hull fill
WIND_VANE_HULL_EDGE_COLOR = "#727272"
WIND_VANE_HULL_EDGE_WIDTH = 2.0
WIND_VANE_HULL_ZORDER = 30

# Axes titles
# When False, hide titles on all canvases (Wind Map, Wind Vane, Feature Clock)
SHOW_TITLES = False

# Vector labels
# When False, hide feature/vector name annotations in Wind Vane and Feature Clock
SHOW_VECTOR_LABELS = True
# Font sizes for labels in wind-vane views
# Default label size for wind vane feature labels
WIND_VANE_LABEL_FONTSIZE = 11
# Label size in Feature Clock mode
FEATURE_CLOCK_LABEL_FONTSIZE = 9
# Font size for informational texts (e.g., masked cell/selection)
WIND_VANE_INFO_FONTSIZE = 10

# Wind-Map data-point vector overlay (for paper figures)
# When True, draw per-feature gradient vectors at each data point on the Wind-Map.
SHOW_DATA_VECTORS_ON_MAP = False
# Relative target length of the 95th percentile vector (fraction of plot width)
DATA_VECTOR_REL_SCALE = 0.06
# Fixed absolute scale in data units (if > 0, overrides relative scaling)
DATA_VECTOR_ABS_SCALE = 0.0
# Choose which feature's vectors to show on the map overlay.
# Accepts: None (use selected features), an integer index, a list/tuple of
# integer indices (e.g., [1,3,5]), a comma-separated string of indices
# (e.g., '1,3,5'), or a case-insensitive substring of the feature name to
# match the first hit.
DATA_VECTOR_FEATURE = 1
# Appearance
DATA_VECTOR_ALPHA = 1.0
DATA_VECTOR_ZORDER = 25

# Wind-Map grid (interpolated) vector overlay
# When True, draw vectors at each grid cell after interpolation.
SHOW_GRID_VECTORS_ON_MAP = False
# Choose grid vectors to draw: None (use selected features),
# 'sum' (use summed field), an integer index, a list/tuple (e.g., [2,4]),
# a comma-separated string of indices (e.g., '2,4'), or a substring match.
GRID_VECTOR_FEATURE = 'sum'
# Scaling for grid vectors (same semantics as data-point overlay)
GRID_VECTOR_REL_SCALE = 0.05
GRID_VECTOR_ABS_SCALE = 0.0
GRID_VECTOR_ALPHA = 0.50
GRID_VECTOR_ZORDER = 24

# Mask-awareness for vector overlays
# When True, data-point and grid-cell vector overlays respect the final mask
# (i.e., vectors are shown only for unmasked cells based on the summed field).
SHOW_VECTOR_OVERLAY_RESPECT_MASK = True

# Cell dominance coloring
# When True, color each grid cell by its dominant feature's color.
SHOW_CELL_DOMINANCE = False
# Opacity for the cell dominance color overlay (0..1)
CELL_DOM_ALPHA = 0.18
# Z-order for the cell dominance overlay
CELL_DOM_ZORDER = 3

# UI hotkeys
# Press this key while the figure is focused to save snapshots of
# Wind Map, Wind Vane, and Feature Clock panes into the output directory.
SNAPSHOT_HOTKEY = 'a'

# Clear all selected grid cells
CLEAR_SELECTION_HOTKEY = 'c'

# Particle trail opacity fade (older segments more transparent)
# Alpha per segment = base_alpha * (TRAIL_TAIL_MIN_FACTOR + (1-TRAIL_TAIL_MIN_FACTOR) * ((t+1)/TAIL_LENGTH)**TRAIL_TAIL_EXP)
TRAIL_TAIL_MIN_FACTOR = 0.10  # 0..1, alpha factor for the oldest segment
TRAIL_TAIL_EXP = 2.0          # >1 for stronger decay near the tail

# Auto snapshots
# When enabled, the UI will capture snapshots automatically every K frames
AUTO_SNAPSHOT_ENABLED = False
AUTO_SNAPSHOT_EVERY_K_FRAMES = 3  # capture interval in animation frames

# Particle dynamics mode
# When True, particle velocities have constant magnitude and follow only the
# direction of the vector field; "velocity_scale" sets that constant speed.
CONSISTENT_PARTICLE_SPEED = True

# Trail opacity mapping
# When True, particle trail opacity is derived from the local field strength
# (magnitude of the summed vector field) rather than particle speed.
TRAIL_ALPHA_USES_FIELD_STRENGTH = True

# Ring-dot opacity mode for wind vane
# 'speed'  -> alpha tied to particle speed (legacy behavior)
# 'field'  -> alpha tied to summed field magnitude relative to ring radius
RING_DOT_ALPHA_MODE = 'field'


def initialize_global_state():
    """Initialize global state variables that need dynamic computation."""
    global k, bounding_box, real_feature_rgba
    k = None
    bounding_box = None
    real_feature_rgba = {}

def set_bounding_box(positions):
    """
    Compute and set the global bounding box from position data.
    
    Args:
        positions (np.ndarray): Array of 2D positions, shape (N, 2)
    """
    global bounding_box
    
    xmin, xmax = positions[:,0].min(), positions[:,0].max()
    ymin, ymax = positions[:,1].min(), positions[:,1].max()
    
    # Add padding
    x_padding = (xmax - xmin) * 0.05
    y_padding = (ymax - ymin) * 0.05
    xmin -= x_padding
    xmax += x_padding
    ymin -= y_padding
    ymax += y_padding
    
    # Make the bounding box square by expanding the smaller dimension
    x_range = xmax - xmin
    y_range = ymax - ymin
    
    if x_range > y_range:
        # Expand y range to match x range
        y_center = (ymin + ymax) / 2
        ymin = y_center - x_range / 2
        ymax = y_center + x_range / 2
    else:
        # Expand x range to match y range
        x_center = (xmin + xmax) / 2
        xmin = x_center - y_range / 2
        xmax = x_center + y_range / 2
    
    bounding_box = [xmin, xmax, ymin, ymax]
