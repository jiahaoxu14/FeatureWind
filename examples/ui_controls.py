"""
UI controls module for FeatureWind.

Minimal UI for CLI-driven visualization. Provides an optional single-feature
label or a hidden placeholder to keep layout consistent.
"""

import config


class UIController:
    """Main UI controller for managing interactive controls."""
    
    def __init__(self, fig, ax1, ax2, system, grid_data, col_labels):
        """
        Initialize UI controller.
        
        Args:
            fig: Matplotlib figure
            ax1: Main plot axes
            ax2: Wind vane axes
            system: Particle system dictionary
            grid_data: Grid computation data
            col_labels: Feature column labels
        """
        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        self.system = system
        self.grid_data = grid_data
        self.col_labels = col_labels
        
        # Extract grid data (kept for compatibility with consumers)
        (self.interp_u_sum, self.interp_v_sum, self.interp_argmax,
         self.grid_x, self.grid_y, self.grid_u_feats, self.grid_v_feats,
         self.cell_dominant_features, self.grid_u_all_feats, self.grid_v_all_feats,
         self.cell_centers_x, self.cell_centers_y, self.cell_soft_dominance) = grid_data

        # Current state (top_k only; direction-conditioned removed)
        self.current_mode = {'mode': 'top_k'}
        self.grad_indices = list(range(min(config.k or len(col_labels), len(col_labels))))
        
        # Setup UI components
        self.setup_ui_controls()
        
    def setup_ui_controls(self):
        """Setup minimal UI elements (no legacy/direction-conditioned controls)."""
        single_feature_mode = getattr(self, 'single_feature_mode', False)

        if single_feature_mode:
            # Display the selected feature label
            ax_feature_label = self.fig.add_axes([0.35, 0.02, 0.30, 0.03])
            ax_feature_label.axis('off')
            feature_name = getattr(self, 'single_feature_name', 'Unknown Feature')
            ax_feature_label.text(
                0.5, 0.5,
                f'Single Feature: {feature_name}',
                ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
            )
            self.ax_k = ax_feature_label
        else:
            # Invisible placeholder to maintain consistent layout
            ax_k = self.fig.add_axes([0.35, 0.02, 0.30, 0.03])
            ax_k.set_visible(False)
            self.ax_k = ax_k
        # No further controls; interactions are handled via event_manager.
