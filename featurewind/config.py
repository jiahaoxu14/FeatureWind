"""
Configuration module for FeatureWind visualization.

This module contains global constants, default parameters, and configuration
settings used throughout the FeatureWind visualization system.
"""

import numpy as np

# Global configuration variables
velocity_scale = 0.4  # Original stable value
grid_res_scale = 0.15
k = None  # Number of top features (will be set dynamically)
bounding_box = None  # Will be computed dynamically
real_feature_rgba = {}  # Feature to RGBA mapping for particles

# Default grid resolution
DEFAULT_GRID_RES = 30

# Particle system parameters
DEFAULT_NUM_PARTICLES = 1500  # Increased to ensure visible particle density
PARTICLE_LIFETIME = 150  # frames - increased to reduce respawn frequency
TAIL_LENGTH = 10  # number of position history points

# Adaptive trail parameters
ADAPTIVE_TRAIL_LENGTH = True
MIN_TRAIL_LENGTH = 5
MAX_TRAIL_LENGTH = 20
BASE_TRAIL_LENGTH = 10

# Animation parameters
ANIMATION_FRAMES = 1000
ANIMATION_INTERVAL = 30  # milliseconds between frames (33 FPS - optimized for performance)

# Window title
WINDOW_TITLE = "FeatureWind Visualization"

# Color palettes and visualization constants
try:
    import colorcet as cc
    COLORCET_AVAILABLE = True
    # Use ColorCET Glasbey for maximum distinctness
    GLASBEY_COLORS = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948",
        "#AF7AA1", "#FF9D9A", "#9C755F", "#BAB0AC", "#1f77b4", "#ff7f0e",
        "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]
except ImportError:
    COLORCET_AVAILABLE = False
    pass  # Silently use fallback colors
    # Fallback to expanded Tableau-like palette
    GLASBEY_COLORS = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948",
        "#AF7AA1", "#FF9D9A", "#9C755F", "#BAB0AC", "#1f77b4", "#ff7f0e", 
        "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]

# Grid computation parameters
MASK_BUFFER_FACTOR = 0    # Buffer size around data points for interpolation (in cell units)

# Particle physics parameters
MIN_TIME_STEP = 1e-4
MAX_TIME_STEP = 0.1
CFL_NUMBER = 0.5  # For adaptive time stepping
ERROR_TOLERANCE = 1e-3  # For RK4 error control
MAX_RESEED_RATE = 0.01  # Maximum 1% particles reseeded per frame - reduced for stability
MAX_SAFE_VELOCITY = 10.0  # Maximum velocity magnitude per frame to prevent runaway particles

# UI and visualization parameters
MARKER_STYLES = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
DPI = 300  # For saved figures
ALPHA_FADE_FACTOR = 1.0  # For particle trail alpha (set to 1.0 for better visibility)

# Direction-conditioned mode parameters
DEFAULT_ANGLE = 0.0  # degrees
DEFAULT_MAGNITUDE = 1.0
TEMPERATURE_SOFTMAX = 2.0  # Temperature for soft dominance computation

# File paths (relative to project root)
DEFAULT_TANGENT_MAP = 'data/tangentmaps/breast_cancer.tmap'
DEFAULT_OUTPUT_DIR = 'output'

# Feature selection limits
MAX_FEATURES_WITH_COLORS = 6  # Only top 6 features get distinct colors
MAX_FEATURE_FAMILIES = 6      # Maximum number of feature families (Paul Tol palette limit)

def get_feature_color(feature_idx, total_features):
    """
    Get color for a specific feature index.
    
    Args:
        feature_idx (int): Index of the feature
        total_features (int): Total number of features
        
    Returns:
        str: Hex color string
    """
    if feature_idx < len(GLASBEY_COLORS):
        return GLASBEY_COLORS[feature_idx]
    else:
        # Cycle through colors if we need more than available
        return GLASBEY_COLORS[feature_idx % len(GLASBEY_COLORS)]

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