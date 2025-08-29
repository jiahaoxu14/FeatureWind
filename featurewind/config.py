"""
Configuration module for FeatureWind visualization.

This module contains global constants, default parameters, and configuration
settings used throughout the FeatureWind visualization system.
"""

import numpy as np

# Global configuration variables
velocity_scale = 0.4  # Original stable value
k = None  # Number of top features (will be set dynamically)
bounding_box = None  # Will be computed dynamically
real_feature_rgba = {}  # Feature to RGBA mapping for particles

# Default grid resolution
DEFAULT_GRID_RES = 30

# Particle system parameters
DEFAULT_NUM_PARTICLES = 1500  # Increased to ensure visible particle density
PARTICLE_LIFETIME = 50  # frames - increased to reduce respawn frequency
TAIL_LENGTH = 10  # number of position history points

# Animation parameters
ANIMATION_FRAMES = 1000
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

# Particle physics parameters
MAX_SAFE_VELOCITY = 10.0  # Maximum velocity magnitude per frame to prevent runaway particles

# UI and visualization parameters
MARKER_STYLES = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
DPI = 300  # For saved figures

TEMPERATURE_SOFTMAX = 2.0  # Temperature for soft dominance computation


# Feature selection limits
MAX_FEATURE_FAMILIES = 6      # Maximum number of feature families (Paul Tol palette limit)


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