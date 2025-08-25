"""
Wind strength indicator module for FeatureWind.

This module provides visual indicators showing actual wind strength
and display scaling for adaptive particle visualization.
"""

import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch


class WindStrengthIndicator:
    """Visual indicator for wind strength and scaling."""
    
    def __init__(self, ax, position=(0.02, 0.95)):
        """
        Initialize wind strength indicator.
        
        Args:
            ax: Matplotlib axes
            position: (x, y) position in axes coordinates
        """
        self.ax = ax
        self.position = position
        self.elements = []
        
    def update(self, actual_magnitude, scale_factor):
        """
        Update wind strength indicator.
        
        Args:
            actual_magnitude: Actual flow magnitude
            scale_factor: Display scaling factor
        """
        # Clear previous elements
        for elem in self.elements:
            try:
                elem.remove()
            except:
                pass
        self.elements.clear()
        
        # Determine actual wind strength category
        if actual_magnitude < 0.01:
            actual_strength = "Calm"
            actual_color = '#E8E8E8'
        elif actual_magnitude < 0.05:
            actual_strength = "Light"
            actual_color = '#B3D9FF'
        elif actual_magnitude < 0.1:
            actual_strength = "Moderate"
            actual_color = '#66B2FF'
        elif actual_magnitude < 0.2:
            actual_strength = "Fresh"
            actual_color = '#0080FF'
        else:
            actual_strength = "Strong"
            actual_color = '#0040FF'
        
        # Create main indicator box
        main_text = self.ax.text(
            self.position[0], self.position[1],
            f"Wind: {actual_strength}\n"
            f"Actual: {actual_magnitude:.3f}\n"
            f"Display: {scale_factor:.1f}×",
            transform=self.ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', 
                     facecolor=actual_color, 
                     alpha=0.3,
                     edgecolor='black',
                     linewidth=1)
        )
        self.elements.append(main_text)
        
        # Add scale indicator thermometer
        self.add_scale_thermometer(scale_factor)
    
    def add_scale_thermometer(self, scale_factor):
        """
        Add visual thermometer showing scaling amount.
        
        Args:
            scale_factor: Display scaling factor
        """
        x, y = self.position[0] + 0.12, self.position[1] - 0.03
        
        # Background
        bg = FancyBboxPatch((x, y - 0.08), 0.01, 0.08,
                            boxstyle="round,pad=0.001",
                            transform=self.ax.transAxes,
                            facecolor='lightgray',
                            edgecolor='black',
                            alpha=0.3,
                            linewidth=0.5)
        self.ax.add_patch(bg)
        self.elements.append(bg)
        
        # Fill based on scale factor (log scale)
        # 1x = bottom, 10x = top
        log_scale = np.log10(np.clip(scale_factor, 1, 10))
        fill_height = 0.08 * (log_scale / 1.0)  # log10(10) = 1
        
        # Color based on scaling
        if scale_factor >= 5:
            fill_color = 'red'  # Heavy boost
        elif scale_factor >= 2:
            fill_color = 'orange'  # Moderate boost
        elif scale_factor >= 1.5:
            fill_color = 'yellow'  # Light boost
        else:
            fill_color = 'green'  # No/minimal boost
        
        fill = Rectangle((x, y - 0.08), 0.01, fill_height,
                        transform=self.ax.transAxes,
                        facecolor=fill_color,
                        alpha=0.7)
        self.ax.add_patch(fill)
        self.elements.append(fill)
        
        # Add scale labels
        if scale_factor >= 2.0:
            label = self.ax.text(
                x + 0.015, y - 0.04,
                f"{scale_factor:.0f}×",
                transform=self.ax.transAxes,
                fontsize=7,
                verticalalignment='center'
            )
            self.elements.append(label)
    
    def clear(self):
        """Clear all indicator elements."""
        for elem in self.elements:
            try:
                elem.remove()
            except:
                pass
        self.elements.clear()


def add_static_flow_indicators(ax, grid_u, grid_v, grid_x, grid_y, threshold=0.001):
    """
    Add static arrow indicators when flow is too weak for particles.
    
    Args:
        ax: Matplotlib axes
        grid_u, grid_v: Velocity field components
        grid_x, grid_y: Grid coordinates
        threshold: Magnitude threshold for showing arrows
        
    Returns:
        Quiver object or None
    """
    magnitude = np.sqrt(grid_u**2 + grid_v**2)
    max_magnitude = magnitude.max()
    
    if max_magnitude < threshold:
        # Flow is extremely weak - add arrow field
        skip = max(grid_u.shape[0] // 10, 1)
        
        # Subsample grids
        X_sub = grid_x[::skip, ::skip]
        Y_sub = grid_y[::skip, ::skip]
        U_sub = grid_u[::skip, ::skip]
        V_sub = grid_v[::skip, ::skip]
        
        # Normalize arrows for visibility
        M_sub = np.sqrt(U_sub**2 + V_sub**2)
        mask = M_sub > 0
        U_norm = np.zeros_like(U_sub)
        V_norm = np.zeros_like(V_sub)
        U_norm[mask] = U_sub[mask] / M_sub[mask]
        V_norm[mask] = V_sub[mask] / M_sub[mask]
        
        # Draw with fixed length arrows
        quiver = ax.quiver(X_sub, Y_sub, U_norm, V_norm,
                          M_sub,  # Color by magnitude
                          cmap='Blues',
                          alpha=0.3,
                          scale=30,
                          width=0.002,
                          headwidth=3,
                          headlength=4)
        
        # Add annotation
        ax.text(0.5, 0.98, "Flow too weak - showing direction field",
               transform=ax.transAxes,
               fontsize=8,
               ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        return quiver
    return None