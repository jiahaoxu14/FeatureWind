"""
UI controls module for FeatureWind.

Minimal UI for CLI-driven visualization. Provides an optional single-feature
label or a hidden placeholder to keep layout consistent, plus snapshot controls.
"""

from .. import config
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


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

        # Current state (top_k mode only)
        self.current_mode = {'mode': 'top_k'}
        self.grad_indices = list(range(min(config.k or len(col_labels), len(col_labels))))
        
        # Setup UI components
        self.setup_ui_controls()
        
    def setup_ui_controls(self):
        """Setup minimal UI elements (CLI-driven approach)."""
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
        # Add a snapshot button to save Wind Map, Wind Vane, and Feature Clock views
        try:
            btn_ax = self.fig.add_axes([0.02, 0.02, 0.14, 0.045])
            self._snapshot_button = Button(btn_ax, 'Save Snapshots')
            self._snapshot_button.on_clicked(self._on_snapshot_clicked)
        except Exception:
            pass
        # No further controls; interactions are handled via event_manager.

    def save_snapshots(self):
        """Save snapshots of the Wind Map, Wind Vane, and Feature Clock as PNGs."""
        try:
            # Ensure the renderer is ready for tight bbox
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()

            # Resolve output directory
            out_dir = 'output'
            try:
                # If createwind.py created an output dir at repo root, prefer it
                root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                candidate = os.path.join(root, 'output')
                if os.path.isdir(candidate):
                    out_dir = candidate
            except Exception:
                pass
            os.makedirs(out_dir, exist_ok=True)

            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Create a per-timestamp subfolder (e.g., output/20250101_235959)
            ts_dir = os.path.join(out_dir, ts)
            os.makedirs(ts_dir, exist_ok=True)

            def save_ax(ax, fname):
                if ax is None:
                    return
                try:
                    renderer = self.fig.canvas.get_renderer()
                except Exception:
                    renderer = None
                if renderer is not None:
                    bbox = ax.get_tightbbox(renderer).transformed(self.fig.dpi_scale_trans.inverted())
                    self.fig.savefig(os.path.join(ts_dir, fname), dpi=config.DPI, bbox_inches=bbox)
                else:
                    # Fallback: save full figure with suffix
                    self.fig.savefig(os.path.join(ts_dir, fname), dpi=config.DPI)

            # Use stored axes references instead of titles (titles may be hidden)
            ax_map = self.ax1
            ax_vane = None
            ax_clock = None
            if isinstance(self.ax2, (list, tuple)) and len(self.ax2) == 2:
                ax_vane, ax_clock = self.ax2[0], self.ax2[1]
            else:
                ax_vane = self.ax2

            # Save files inside the timestamped folder with stable names
            save_ax(ax_map, 'wind_map.png')
            save_ax(ax_vane, 'wind_vane.png')
            save_ax(ax_clock, 'feature_clock.png')
        except Exception:
            pass

    def _on_snapshot_clicked(self, event):
        self.save_snapshots()
