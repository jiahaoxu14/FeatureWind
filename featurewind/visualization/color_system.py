"""
Color system module for FeatureWind visualization.

Implements Paul Tol's colorblind-safe palette with perceptually uniform
LCH color space encoding for magnitude, dominance, and family assignment.
"""

import numpy as np
import colorsys


# Paul Tol's colorblind-safe qualitative palette (6 colors)
# https://personal.sron.nl/~pault/
PAUL_TOL_FAMILIES = [
    "#4477AA",  # Blue - Family 0
    "#EE6677",  # Red - Family 1  
    "#228833",  # Green - Family 2
    "#CCBB44",  # Yellow - Family 3
    "#66CCEE",  # Cyan - Family 4
    "#AA3377"   # Purple - Family 5
]

# Extended palette for >6 families (adds neutral tones)
PAUL_TOL_EXTENDED = PAUL_TOL_FAMILIES + [
    "#BBBBBB",  # Gray - Family 6
    "#88CCEE",  # Light Blue - Family 7
    "#44AA99",  # Teal - Family 8
    "#99DDFF",  # Pale Blue - Family 9
]

# Professional styling colors
BACKGROUND_COLOR = "#F7F7F7"  # Off-white background
GRID_COLOR = "#E0E0E0"        # Light gray for grid lines
TEXT_COLOR = "#333333"        # Dark gray for text
LOW_DOMINANCE_ALPHA = 0.2     # For uncertain/non-dominant features
HIGH_DOMINANCE_ALPHA = 0.9    # For dominant features


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_hex(rgb_tuple):
    """Convert RGB tuple (0-1 range) to hex string."""
    rgb_int = tuple(int(c * 255) for c in rgb_tuple)
    return '#{:02x}{:02x}{:02x}'.format(*rgb_int)


def rgb_to_hls(rgb_tuple):
    """Convert RGB to HLS color space."""
    return colorsys.rgb_to_hls(*rgb_tuple)


def hls_to_rgb(hls_tuple):
    """Convert HLS to RGB color space."""
    return colorsys.hls_to_rgb(*hls_tuple)


def create_lightness_ramp(base_hex, magnitude, max_magnitude, 
                         min_lightness=0.2, max_lightness=0.9):
    """
    Create lightness-modulated color for magnitude encoding using HLS color space.
    
    This is a simplified version that doesn't require colorspacious dependency.
    Uses HLS as an approximation to LCH for perceptual uniformity.
    
    Args:
        base_hex: str, base family color (e.g., "#4477AA")
        magnitude: float, current gradient magnitude  
        max_magnitude: float, maximum magnitude for normalization
        min_lightness: float, minimum lightness value (0-1, darker)
        max_lightness: float, maximum lightness value (0-1, lighter)
    
    Returns:
        str: hex color string with adjusted lightness
    """
    if max_magnitude <= 0:
        return base_hex
    
    # Convert base color to HLS
    rgb = hex_to_rgb(base_hex)
    h, l, s = rgb_to_hls(rgb)
    
    # Map magnitude to lightness range [min_lightness, max_lightness]
    magnitude_norm = np.clip(magnitude / max_magnitude, 0.0, 1.0)
    target_lightness = min_lightness + (max_lightness - min_lightness) * magnitude_norm
    
    # Keep hue and saturation constant, only modify lightness
    new_rgb = hls_to_rgb((h, target_lightness, s))
    
    # Clamp RGB values to [0, 1]
    new_rgb = tuple(np.clip(c, 0.0, 1.0) for c in new_rgb)
    
    return rgb_to_hex(new_rgb)


def create_alpha_for_dominance(dominance, min_alpha=0.3, max_alpha=1.0):
    """
    Create alpha value based on dominance/uncertainty.
    
    Args:
        dominance: float, dominance value (0-1, higher = more dominant)
        min_alpha: float, minimum alpha for uncertain features
        max_alpha: float, maximum alpha for dominant features
    
    Returns:
        float: alpha value in range [min_alpha, max_alpha]
    """
    dominance_norm = np.clip(dominance, 0.0, 1.0)
    return min_alpha + (max_alpha - min_alpha) * dominance_norm


def assign_family_colors(family_assignments, use_extended_palette=False):
    """
    Map family IDs to Paul Tol colors.
    
    Args:
        family_assignments: numpy array of family IDs for each feature
        use_extended_palette: bool, whether to use extended palette for >6 families
    
    Returns:
        list: hex color strings for each feature
    """
    palette = PAUL_TOL_EXTENDED if use_extended_palette else PAUL_TOL_FAMILIES
    
    feature_colors = []
    for family_id in family_assignments:
        color_idx = family_id % len(palette)
        feature_colors.append(palette[color_idx])
    
    return feature_colors


def get_family_color_palette(n_families):
    """
    Get appropriate color palette for given number of families.
    
    Args:
        n_families: int, number of families to color
    
    Returns:
        list: hex color strings for families
    """
    if n_families <= len(PAUL_TOL_FAMILIES):
        return PAUL_TOL_FAMILIES[:n_families]
    else:
        return PAUL_TOL_EXTENDED[:min(n_families, len(PAUL_TOL_EXTENDED))]


def create_magnitude_colormap(base_hex, n_steps=256):
    """
    Create a matplotlib-compatible colormap for magnitude encoding.
    
    Args:
        base_hex: str, base color for the colormap
        n_steps: int, number of discrete steps in colormap
    
    Returns:
        matplotlib.colors.ListedColormap: colormap object
    """
    try:
        from matplotlib.colors import ListedColormap
        
        # Create lightness ramp
        colors = []
        for i in range(n_steps):
            magnitude = i / (n_steps - 1)  # Normalize to [0, 1]
            color_hex = create_lightness_ramp(base_hex, magnitude, 1.0)
            colors.append(hex_to_rgb(color_hex))
        
        return ListedColormap(colors, name=f'magnitude_{base_hex}')
        
    except ImportError:
        print("Warning: matplotlib not available, cannot create colormap")
        return None


def apply_professional_styling(fig, ax_list):
    """
    Apply professional color styling to figure and axes.
    
    Args:
        fig: matplotlib figure object
        ax_list: list of matplotlib axes objects
    """
    # Set figure background
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    for ax in ax_list:
        # Set axes background
        ax.set_facecolor(BACKGROUND_COLOR)
        
        # Style grid lines
        ax.grid(True, alpha=0.3, linewidth=0.5, color=GRID_COLOR, zorder=0)
        
        # Style spines
        for spine in ax.spines.values():
            spine.set_color(TEXT_COLOR)
            spine.set_linewidth(0.5)
        
        # Style tick labels
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        
        # Style axis labels
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)


def create_family_color_legend_data(family_assignments, col_labels, family_colors):
    """
    Create data structure for family color legend.
    
    Args:
        family_assignments: numpy array of family IDs
        col_labels: list of feature names  
        family_colors: list of hex colors for families
    
    Returns:
        list: list of dicts with legend information
    """
    unique_families = np.unique(family_assignments)
    legend_data = []
    
    for family_id in unique_families:
        family_mask = (family_assignments == family_id)
        family_features = [col_labels[i] for i, is_in_family in enumerate(family_mask) if is_in_family]
        
        # Get representative color (first feature's color)
        feature_indices = np.where(family_mask)[0]
        if len(feature_indices) > 0:
            color = family_colors[feature_indices[0]]
        else:
            color = PAUL_TOL_FAMILIES[family_id % len(PAUL_TOL_FAMILIES)]
        
        # Create short family name
        if len(family_features) == 1:
            family_name = family_features[0][:15]
        else:
            family_name = f"Family {family_id+1} ({len(family_features)})"
        
        legend_data.append({
            'family_id': family_id,
            'color': color,
            'name': family_name,
            'n_features': len(family_features),
            'sample_features': family_features[:3]  # Show first 3 as examples
        })
    
    return legend_data


def compute_perceptual_distance(color1_hex, color2_hex):
    """
    Compute approximate perceptual distance between two colors using Delta E in RGB space.
    
    This is a simplified version that doesn't require colorspacious.
    Uses Euclidean distance in RGB space as approximation.
    
    Args:
        color1_hex: str, first color in hex format
        color2_hex: str, second color in hex format
    
    Returns:
        float: perceptual distance (higher = more different)
    """
    rgb1 = hex_to_rgb(color1_hex)
    rgb2 = hex_to_rgb(color2_hex)
    
    # Simple Euclidean distance in RGB space
    # This is not as accurate as true Delta E, but works as approximation
    distance = np.sqrt(sum((c1 - c2)**2 for c1, c2 in zip(rgb1, rgb2)))
    
    return distance


def validate_color_accessibility(colors, min_distance=0.3):
    """
    Validate that colors are sufficiently distinct for accessibility.
    
    Args:
        colors: list of hex color strings
        min_distance: float, minimum perceptual distance required
    
    Returns:
        dict: validation results with any problematic pairs
    """
    n_colors = len(colors)
    problematic_pairs = []
    
    for i in range(n_colors):
        for j in range(i + 1, n_colors):
            distance = compute_perceptual_distance(colors[i], colors[j])
            if distance < min_distance:
                problematic_pairs.append({
                    'color1': colors[i],
                    'color2': colors[j], 
                    'indices': (i, j),
                    'distance': distance
                })
    
    return {
        'is_accessible': len(problematic_pairs) == 0,
        'problematic_pairs': problematic_pairs,
        'min_distance_found': min([pair['distance'] for pair in problematic_pairs]) if problematic_pairs else 1.0
    }


if __name__ == "__main__":
    # Test color system functionality
    print("Testing Paul Tol color system...")
    
    # Test family color assignment
    test_families = np.array([0, 1, 2, 0, 1, 2, 3, 4, 5])
    family_colors = assign_family_colors(test_families)
    print(f"Family assignments: {test_families}")
    print(f"Assigned colors: {family_colors}")
    
    # Test lightness ramping
    base_color = PAUL_TOL_FAMILIES[0]  # Blue
    print(f"\nTesting lightness ramp for {base_color}:")
    
    for mag in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ramped_color = create_lightness_ramp(base_color, mag, 1.0)
        print(f"  Magnitude {mag}: {ramped_color}")
    
    # Test color accessibility
    print(f"\nTesting color accessibility:")
    accessibility = validate_color_accessibility(PAUL_TOL_FAMILIES[:6])
    print(f"Colors are accessible: {accessibility['is_accessible']}")
    print(f"Minimum distance found: {accessibility['min_distance_found']:.3f}")
    
    # Test alpha for dominance
    print(f"\nTesting dominance alpha mapping:")
    for dom in [0.0, 0.25, 0.5, 0.75, 1.0]:
        alpha = create_alpha_for_dominance(dom)
        print(f"  Dominance {dom}: alpha {alpha:.3f}")
    
    print("\n✓ Color system tests completed!")