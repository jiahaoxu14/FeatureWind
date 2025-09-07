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
from matplotlib.patches import Rectangle


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
        # Auto-snapshot session state (single folder per run)
        self._auto_snapshot_dir = None
        self._auto_snapshot_count = 0
        # Selection state for multi-cell selection on the wind-map
        self.selected_cells = set()  # set of (i, j)
        self._selected_patches = []  # drawn rectangles on ax1
        # Determine grid resolution from system or grid data
        try:
            if system.get('unmasked_cells') is not None:
                self.grid_res = int(system['unmasked_cells'].shape[0])
            elif system.get('grid_u_sum') is not None:
                self.grid_res = int(system['grid_u_sum'].shape[0])
            else:
                # fallback to cell centers length if available
                self.grid_res = int(len(grid_data[10])) if grid_data and len(grid_data) > 10 else int(getattr(config, 'DEFAULT_GRID_RES', 40))
        except Exception:
            self.grid_res = int(getattr(config, 'DEFAULT_GRID_RES', 40))
        
        # Extract grid data (kept for compatibility with consumers)
        (self.interp_u_sum, self.interp_v_sum, self.interp_argmax,
         self.grid_x, self.grid_y, self.grid_u_feats, self.grid_v_feats,
         self.cell_dominant_features, self.grid_u_all_feats, self.grid_v_all_feats,
         self.cell_centers_x, self.cell_centers_y, self.final_mask) = grid_data

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

    # EventManager will call this on mouse clicks
    def handle_mouse_click(self, event):
        try:
            # Only handle clicks on the main wind-map axes
            if event.inaxes != self.ax1 or event.xdata is None or event.ydata is None:
                return
            from .. import config as _cfg
            xmin, xmax, ymin, ymax = _cfg.bounding_box
            # Outside bounds
            if not (xmin <= event.xdata <= xmax and ymin <= event.ydata <= ymax):
                return
            # Convert to grid cell indices (clamped)
            j = int((event.xdata - xmin) / (xmax - xmin) * self.grid_res)
            i = int((event.ydata - ymin) / (ymax - ymin) * self.grid_res)
            i = max(0, min(self.grid_res - 1, i))
            j = max(0, min(self.grid_res - 1, j))

            # Left click: toggle selection; Right click (3): clear all
            if int(getattr(event, 'button', 1)) == 3:
                # Clear selection
                self.selected_cells.clear()
                self._redraw_selection_overlays()
            else:
                cell = (i, j)
                if cell in self.selected_cells:
                    self.selected_cells.remove(cell)
                else:
                    self.selected_cells.add(cell)
                self._redraw_selection_overlays()

            # Propagate selection to event manager's mouse_data and update wind vane
            try:
                # Event manager is attached by the caller after creation
                if hasattr(self, 'event_manager') and self.event_manager is not None:
                    self.event_manager.mouse_data['selected_cells'] = sorted(list(self.selected_cells))
                    # keep grid_res consistent
                    self.event_manager.mouse_data['grid_res'] = self.grid_res
                    # Trigger wind vane update (uses either hovered cell or selection)
                    if hasattr(self.event_manager, 'wind_vane_callback') and self.event_manager.wind_vane_callback:
                        self.event_manager.wind_vane_callback(self.event_manager.mouse_data)
                        # Safe canvas update
                        if hasattr(self.event_manager, '_safe_canvas_update'):
                            self.event_manager._safe_canvas_update()
            except Exception:
                pass
        except Exception:
            pass

    def _redraw_selection_overlays(self):
        """Draw or update semi-transparent overlays for selected cells on ax1."""
        try:
            # Remove existing patches
            for p in self._selected_patches:
                try:
                    p.remove()
                except Exception:
                    pass
            self._selected_patches.clear()

            if not self.selected_cells:
                # Force redraw to clear
                try:
                    self.fig.canvas.draw_idle()
                except Exception:
                    pass
                return

            from .. import config as _cfg
            xmin, xmax, ymin, ymax = _cfg.bounding_box
            dx = (xmax - xmin) / self.grid_res
            dy = (ymax - ymin) / self.grid_res
            # Draw a semi-transparent rectangle for each selected cell
            for (i, j) in self.selected_cells:
                x_left = xmin + j * dx
                y_bottom = ymin + i * dy
                # Draw only the border in black, no fill
                patch = Rectangle(
                    (x_left, y_bottom), dx, dy,
                    fill=False,
                    facecolor='none',
                    edgecolor='black',
                    linewidth=1.5,
                    zorder=30,
                )
                try:
                    self.ax1.add_patch(patch)
                    self._selected_patches.append(patch)
                except Exception:
                    pass

            # Request a redraw
            try:
                self.fig.canvas.draw_idle()
            except Exception:
                pass
        except Exception:
            pass

    def clear_selection(self):
        """Clear all selected cells and update views."""
        try:
            if not self.selected_cells:
                return
            self.selected_cells.clear()
            self._redraw_selection_overlays()
            # Propagate to event manager and refresh wind vane to show hover-only state
            if hasattr(self, 'event_manager') and self.event_manager is not None:
                self.event_manager.mouse_data['selected_cells'] = []
                if hasattr(self.event_manager, 'wind_vane_callback') and self.event_manager.wind_vane_callback:
                    self.event_manager.wind_vane_callback(self.event_manager.mouse_data)
                if hasattr(self.event_manager, '_safe_canvas_update'):
                    self.event_manager._safe_canvas_update()
        except Exception:
            pass

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

    def save_auto_snapshot(self, frame_index=None):
        """Save ONLY the Wind Map (ax1) into a single folder for this run.

        Args:
            frame_index (int, optional): Frame number to include in filename.
        """
        try:
            # Resolve base output directory
            out_dir = 'output'
            try:
                root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                candidate = os.path.join(root, 'output')
                if os.path.isdir(candidate):
                    out_dir = candidate
            except Exception:
                pass

            # Initialize per-run auto snapshot folder once
            if self._auto_snapshot_dir is None:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                self._auto_snapshot_dir = os.path.join(out_dir, f'auto_{ts}')
                os.makedirs(self._auto_snapshot_dir, exist_ok=True)
                self._auto_snapshot_count = 0

            # Prepare renderer
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()

            # Build filename
            self._auto_snapshot_count += 1
            if isinstance(frame_index, int):
                fname = f'wind_map_frame_{frame_index:06d}.png'
            else:
                fname = f'wind_map_{self._auto_snapshot_count:06d}.png'

            # Save only ax1
            try:
                renderer = self.fig.canvas.get_renderer()
            except Exception:
                renderer = None
            if renderer is not None:
                bbox = self.ax1.get_tightbbox(renderer).transformed(self.fig.dpi_scale_trans.inverted())
                self.fig.savefig(os.path.join(self._auto_snapshot_dir, fname), dpi=config.DPI, bbox_inches=bbox)
            else:
                self.fig.savefig(os.path.join(self._auto_snapshot_dir, fname), dpi=config.DPI)
        except Exception:
            pass
