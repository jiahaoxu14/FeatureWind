"""
Legend manager module for FeatureWind visualization.

Creates compact, professional legends showing family colors, magnitude encoding,
and dominance/uncertainty visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


def create_family_legend(fig, family_assignments, col_labels, feature_colors, 
                        legend_position='upper_left'):
    """
    Create a compact family color legend.
    
    Args:
        fig: matplotlib figure object
        family_assignments: numpy array of family IDs for each feature
        col_labels: list of feature names
        feature_colors: list of hex colors for features
        legend_position: str, position for legend ('upper_left', 'upper_right', etc.)
        
    Returns:
        matplotlib.axes.Axes: legend axes object
    """
    # Define legend positions
    # Larger legend area to accommodate all features
    positions = {
        'upper_left': [0.01, 0.15, 0.22, 0.80],  # Taller legend on the left
        'upper_right': [0.77, 0.15, 0.22, 0.80], 
        'lower_left': [0.01, 0.05, 0.22, 0.40],
        'lower_right': [0.77, 0.05, 0.22, 0.40]
    }
    
    pos = positions.get(legend_position, positions['upper_left'])
    ax_families = fig.add_axes(pos)
    
    # Get unique families and their info
    unique_families = np.unique(family_assignments)
    family_info = []
    
    for family_id in unique_families:
        family_mask = (family_assignments == family_id)
        family_feature_indices = np.where(family_mask)[0]
        family_features = [col_labels[i] for i in family_feature_indices]
        
        # Get representative color (first feature's color in this family)
        if len(family_feature_indices) > 0:
            representative_color = feature_colors[family_feature_indices[0]]
        else:
            from .color_system import PAUL_TOL_FAMILIES
            representative_color = PAUL_TOL_FAMILIES[family_id % len(PAUL_TOL_FAMILIES)]
        
        # Create descriptive family name
        if len(family_features) == 1:
            family_name = family_features[0][:12] + "..." if len(family_features[0]) > 12 else family_features[0]
        else:
            # Try to find common patterns in feature names
            family_name = _create_family_name(family_features, family_id)
        
        family_info.append({
            'id': family_id,
            'name': family_name,
            'color': representative_color,
            'size': len(family_features),
            'all_features': family_features  # Show all features
        })
    
    # Sort families by size (largest first) for better visual hierarchy
    family_info.sort(key=lambda x: x['size'], reverse=True)
    
    # Draw family legend with all features
    current_y = 0.95  # Start from top
    
    for info in family_info:
        # Color patch
        color_patch = Rectangle((0.02, current_y - 0.04), 0.03, 0.03, 
                               facecolor=info['color'], alpha=0.8, edgecolor='black', linewidth=0.5)
        ax_families.add_patch(color_patch)
        
        # Family header
        ax_families.text(0.06, current_y - 0.025, f"Family {info['id']} ({info['size']} features):", 
                        fontsize=7, va='center', weight='bold')
        current_y -= 0.05
        
        # List all features in this family
        for feature_name in info['all_features']:
            # Show full feature name without truncation
            ax_families.text(0.08, current_y, f"• {feature_name}", 
                           fontsize=6, va='top', alpha=0.8)
            current_y -= 0.025
        
        # Add spacing between families
        current_y -= 0.02
    
    # Adjust y-limits to fit content
    min_y = min(current_y - 0.05, 0)
    ax_families.set_xlim(0, 1)
    ax_families.set_ylim(min_y, 1)
    ax_families.axis('off')
    ax_families.set_title("Feature Families", fontsize=9, fontweight='bold', pad=5)
    
    return ax_families


def create_magnitude_legend(fig, sample_family_color=None, legend_position='magnitude'):
    """
    Create a magnitude encoding legend showing lightness ramp.
    
    Args:
        fig: matplotlib figure object
        sample_family_color: hex color to use for demonstration
        legend_position: position identifier
        
    Returns:
        matplotlib.axes.Axes: legend axes object
    """
    if sample_family_color is None:
        from .color_system import PAUL_TOL_FAMILIES
        sample_family_color = PAUL_TOL_FAMILIES[0]  # Use blue as default
    
    # Position below family legend
    ax_magnitude = fig.add_axes([0.02, 0.64, 0.18, 0.06])
    
    try:
        from .color_system import create_lightness_ramp
        
        # Create gradient from dark to light
        n_steps = 50
        magnitude_colors = []
        for i in range(n_steps):
            magnitude = i / (n_steps - 1)  # Normalize to [0, 1]
            color_hex = create_lightness_ramp(sample_family_color, magnitude, 1.0)
            magnitude_colors.append(color_hex)
        
        # Draw color ramp
        for i, color in enumerate(magnitude_colors):
            rect = Rectangle((i/n_steps, 0.3), 1/n_steps, 0.4, 
                           facecolor=color, edgecolor='none')
            ax_magnitude.add_patch(rect)
            
    except ImportError:
        # Fallback: simple grayscale ramp
        for i in range(50):
            intensity = i / 49
            rect = Rectangle((i/50, 0.3), 1/50, 0.4, 
                           facecolor=(intensity, intensity, intensity), edgecolor='none')
            ax_magnitude.add_patch(rect)
    
    # Add border
    border = Rectangle((0, 0.3), 1, 0.4, facecolor='none', 
                      edgecolor='black', linewidth=0.5)
    ax_magnitude.add_patch(border)
    
    # Labels
    ax_magnitude.text(0.5, 0.8, "Gradient Magnitude", ha='center', fontsize=8, weight='bold')
    ax_magnitude.text(0, 0.1, "weak", ha='left', fontsize=6, alpha=0.7)
    ax_magnitude.text(1, 0.1, "strong", ha='right', fontsize=6, alpha=0.7)
    
    ax_magnitude.set_xlim(0, 1)
    ax_magnitude.set_ylim(0, 1)
    ax_magnitude.axis('off')
    
    return ax_magnitude


def create_dominance_legend(fig, legend_position='dominance'):
    """
    Create a dominance/uncertainty legend showing alpha progression.
    
    Args:
        fig: matplotlib figure object
        legend_position: position identifier
        
    Returns:
        matplotlib.axes.Axes: legend axes object
    """
    # Position below magnitude legend
    ax_alpha = fig.add_axes([0.02, 0.56, 0.18, 0.06])
    
    # Create alpha progression
    n_steps = 20
    alphas = np.linspace(0.3, 1.0, n_steps)
    
    for i, alpha in enumerate(alphas):
        rect = Rectangle((i/n_steps, 0.3), 1/n_steps, 0.4, 
                        facecolor='gray', alpha=alpha, edgecolor='none')
        ax_alpha.add_patch(rect)
    
    # Add border
    border = Rectangle((0, 0.3), 1, 0.4, facecolor='none', 
                      edgecolor='black', linewidth=0.5)
    ax_alpha.add_patch(border)
    
    # Labels
    ax_alpha.text(0.5, 0.8, "Feature Dominance", ha='center', fontsize=8, weight='bold')
    ax_alpha.text(0, 0.1, "uncertain", ha='left', fontsize=6, alpha=0.7)
    ax_alpha.text(1, 0.1, "dominant", ha='right', fontsize=6, alpha=0.7)
    
    ax_alpha.set_xlim(0, 1)
    ax_alpha.set_ylim(0, 1)
    ax_alpha.axis('off')
    
    return ax_alpha


def create_comprehensive_legend(fig, family_assignments, col_labels, feature_colors,
                              legend_position='upper_left', show_instructions=True):
    """
    Create a comprehensive legend with all encoding channels.
    
    Args:
        fig: matplotlib figure object
        family_assignments: numpy array of family IDs
        col_labels: list of feature names
        feature_colors: list of hex colors
        legend_position: position for legend
        show_instructions: whether to show usage instructions
        
    Returns:
        dict: dictionary of legend axes
    """
    legend_axes = {}
    
    # Create family legend
    legend_axes['families'] = create_family_legend(
        fig, family_assignments, col_labels, feature_colors, legend_position
    )
    
    # Magnitude and dominance legends removed per user request
    # sample_color = feature_colors[0] if feature_colors else None
    # legend_axes['magnitude'] = create_magnitude_legend(fig, sample_color)
    # legend_axes['dominance'] = create_dominance_legend(fig)
    
    # Add usage instructions if requested
    if show_instructions:
        legend_axes['instructions'] = create_usage_instructions(fig)
    
    return legend_axes


def create_usage_instructions(fig):
    """
    Create a small instruction box explaining the color encoding.
    
    Args:
        fig: matplotlib figure object
        
    Returns:
        matplotlib.axes.Axes: instruction axes object
    """
    ax_instructions = fig.add_axes([0.02, 0.45, 0.18, 0.08])
    
    instructions = [
        "Color Encoding:",
        "• Hue = Feature family", 
        "• Lightness = Magnitude",
        "• Alpha = Dominance"
    ]
    
    for i, instruction in enumerate(instructions):
        y_pos = 0.9 - i * 0.2
        weight = 'bold' if i == 0 else 'normal'
        fontsize = 7 if i == 0 else 6
        ax_instructions.text(0.02, y_pos, instruction, fontsize=fontsize, 
                           weight=weight, va='top')
    
    ax_instructions.set_xlim(0, 1)
    ax_instructions.set_ylim(0, 1)
    ax_instructions.axis('off')
    
    return ax_instructions


def _create_family_name(feature_names, family_id, max_length=12):
    """
    Create a descriptive name for a family based on feature names.
    
    Args:
        feature_names: list of feature names in this family
        family_id: numeric family ID
        max_length: maximum length for the name
        
    Returns:
        str: descriptive family name
    """
    if not feature_names:
        return f"Family {family_id + 1}"
    
    if len(feature_names) == 1:
        name = feature_names[0]
        return name[:max_length] + "..." if len(name) > max_length else name
    
    # Look for common prefixes or words
    words_sets = [set(name.lower().replace('_', ' ').split()) for name in feature_names]
    
    if len(words_sets) > 1:
        common_words = set.intersection(*words_sets)
        
        if common_words:
            # Use most meaningful common word
            meaningful_words = [w for w in common_words 
                              if len(w) > 2 and w not in ['the', 'and', 'for', 'with']]
            if meaningful_words:
                name = max(meaningful_words, key=len).title()
                return (name[:max_length-2] + "..") if len(name) > max_length else name
    
    # Look for common prefixes
    if len(feature_names) > 1:
        # Find longest common prefix
        sorted_names = sorted(feature_names)
        prefix = ""
        for i, char in enumerate(sorted_names[0]):
            if i < len(sorted_names[-1]) and char == sorted_names[-1][i]:
                prefix += char
            else:
                break
        
        # Clean up prefix (remove trailing underscores/spaces)
        prefix = prefix.rstrip('_ ').strip()
        
        if len(prefix) >= 3:
            name = prefix.title()
            return (name[:max_length-2] + "..") if len(name) > max_length else name
    
    # Fallback: generic family name
    return f"Family {family_id + 1}"


def update_legend_visibility(legend_axes, visible=True):
    """
    Show or hide all legend components.
    
    Args:
        legend_axes: dictionary of legend axes from create_comprehensive_legend
        visible: whether legends should be visible
    """
    for ax in legend_axes.values():
        ax.set_visible(visible)


def create_colorbar_style_legend(fig, family_assignments, feature_colors, 
                                position='right', orientation='vertical'):
    """
    Create a colorbar-style legend for families (alternative compact style).
    
    Args:
        fig: matplotlib figure object
        family_assignments: numpy array of family IDs
        feature_colors: list of hex colors
        position: 'right', 'bottom', etc.
        orientation: 'vertical' or 'horizontal'
        
    Returns:
        matplotlib.axes.Axes: colorbar legend axes
    """
    unique_families = np.unique(family_assignments)
    n_families = len(unique_families)
    
    if position == 'right':
        if orientation == 'vertical':
            ax_colorbar = fig.add_axes([0.92, 0.3, 0.02, 0.4])
            
            # Draw color segments
            for i, family_id in enumerate(unique_families):
                family_mask = (family_assignments == family_id)
                representative_idx = np.where(family_mask)[0][0]
                color = feature_colors[representative_idx]
                
                y_bottom = i / n_families
                y_height = 1 / n_families
                
                rect = Rectangle((0, y_bottom), 1, y_height, 
                               facecolor=color, edgecolor='white', linewidth=0.5)
                ax_colorbar.add_patch(rect)
                
                # Add family label
                ax_colorbar.text(1.1, y_bottom + y_height/2, f"F{family_id+1}", 
                               fontsize=6, va='center')
            
            ax_colorbar.set_xlim(0, 1)
            ax_colorbar.set_ylim(0, 1)
            ax_colorbar.set_title("Families", fontsize=8, pad=5)
    
    ax_colorbar.axis('off')
    return ax_colorbar


if __name__ == "__main__":
    # Test legend creation
    print("Testing legend system...")
    
    # Create test data
    n_features = 12
    family_assignments = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 5])
    col_labels = [f"feature_{i}" for i in range(n_features)]
    
    # Test with sample colors
    from .color_system import PAUL_TOL_FAMILIES, assign_family_colors
    feature_colors = assign_family_colors(family_assignments)
    
    # Create test figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title("Legend System Test")
    
    # Create comprehensive legend
    legend_axes = create_comprehensive_legend(
        fig, family_assignments, col_labels, feature_colors
    )
    
    print(f"Created {len(legend_axes)} legend components:")
    for key in legend_axes.keys():
        print(f"  - {key}")
    
    plt.savefig("legend_test.png", dpi=150, bbox_inches='tight')
    print("✓ Legend test saved as legend_test.png")
