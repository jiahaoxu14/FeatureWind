"""
UI controls module for FeatureWind.

This module handles interactive widgets, mode switching, and user interface
components for controlling the visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.patches import Rectangle
from scipy.interpolate import RegularGridInterpolator
import config
import grid_computation


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
        
        # Extract grid data
        (self.interp_u_sum, self.interp_v_sum, self.interp_argmax, 
         self.grid_x, self.grid_y, self.grid_u_feats, self.grid_v_feats, 
         self.cell_dominant_features, self.grid_u_all_feats, self.grid_v_all_feats, 
         self.cell_centers_x, self.cell_centers_y, self.cell_soft_dominance) = grid_data
        
        # Current state
        self.current_mode = {'mode': 'top_k'}  # 'top_k' or 'direction_conditioned'
        self.grad_indices = list(range(min(config.k or len(col_labels), len(col_labels))))
        
        # Direction-conditioned mode data
        self.selected_cells = set()  # Set of (i, j) grid cell indices
        self.user_constraints = {}   # {(i, j): {"direction": (dx, dy), "weight": float}}
        self.constraint_arrows = []  # Visual arrows for constraints
        self.cell_highlight_patches = []  # Visual highlighting for selected cells
        
        # Setup UI components
        self.setup_ui_controls()
        
    def setup_ui_controls(self):
        """Setup all UI controls and widgets."""
        # Hide mode selection - only Top-K Mode available
        # ax_mode = self.fig.add_axes([0.05, 0.02, 0.25, 0.06])
        # self.mode_radio = RadioButtons(ax_mode, ('Top-K Mode', 'Direction-Conditioned Mode'))
        # self.mode_radio.set_active(0)  # Start with Top-K mode
        # self.mode_radio.on_clicked(self.switch_mode)
        
        # Top-K Mode Controls
        # Slider for selecting k in Top k mode
        ax_k = self.fig.add_axes([0.35, 0.02, 0.30, 0.03])
        initial_k = len(self.grad_indices)
        self.k_slider = Slider(ax_k, 'Top k Features', 1, len(self.col_labels), 
                              valinit=initial_k, valfmt='%d', facecolor='lightgreen', alpha=0.7)
        self.k_slider.on_changed(self.update_top_k_features)
        
        # Direction-Conditioned Mode Controls (hidden)
        # ax_angle = self.fig.add_axes([0.70, 0.06, 0.25, 0.03])
        # self.angle_slider = Slider(ax_angle, 'Direction (°)', 0, 360, valinit=config.DEFAULT_ANGLE, 
        #                           valfmt='%.0f°', facecolor='lightblue', alpha=0.7)
        # self.angle_slider.on_changed(self.update_direction_constraints)
        
        # ax_magnitude = self.fig.add_axes([0.70, 0.02, 0.25, 0.03])
        # self.magnitude_slider = Slider(ax_magnitude, 'Magnitude', 0.1, 2.0, valinit=config.DEFAULT_MAGNITUDE, 
        #                               valfmt='%.1f', facecolor='lightcoral', alpha=0.7)
        # self.magnitude_slider.on_changed(self.update_direction_constraints)
        
        # self.ax_status = self.fig.add_axes([0.35, 0.06, 0.30, 0.03])
        # self.ax_status.text(0.5, 0.5, 'Select grid cells and specify direction', 
        #                    ha='center', va='center', fontsize=10, 
        #                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        # self.ax_status.set_xlim(0, 1)
        # self.ax_status.set_ylim(0, 1)
        # self.ax_status.axis('off')
        
        # Store axes for visibility control
        self.ax_k = ax_k
        
    def switch_mode(self, label):
        """Handle mode switching between Top-K and Direction-Conditioned modes."""
        if label == 'Top-K Mode':
            self.current_mode['mode'] = 'top_k'
            # Show top-k controls, hide direction controls
            self.ax_k.set_visible(True)
            self.ax_angle.set_visible(False)
            self.ax_magnitude.set_visible(False)
            self.ax_status.set_visible(False)
            # Clear any selected cells and constraints
            self.selected_cells.clear()
            self.user_constraints.clear()
            self.clear_constraint_visuals()
        else:  # Direction-Conditioned Mode
            self.current_mode['mode'] = 'direction_conditioned'
            # Hide top-k controls, show direction controls
            self.ax_k.set_visible(False)
            self.ax_angle.set_visible(True)
            self.ax_magnitude.set_visible(True)
            self.ax_status.set_visible(True)
            # Reset to no features selected (no wind)
            self.reset_to_no_features()
            self.update_status_text("Select grid cells and specify direction")
        
        # Redraw the interface
        self.fig.canvas.draw_idle()
    
    def clear_constraint_visuals(self):
        """Remove all constraint arrows and highlights from the visualization."""
        for arrow in self.constraint_arrows:
            try:
                arrow.remove()
            except (ValueError, AttributeError):
                pass
        self.constraint_arrows.clear()
        
        # Clear cell highlighting patches
        for patch in self.cell_highlight_patches:
            try:
                patch.remove()
            except (ValueError, AttributeError):
                pass
        self.cell_highlight_patches.clear()
    
    def update_cell_highlighting(self):
        """Update visual highlighting for selected grid cells."""
        # Clear existing highlights
        for patch in self.cell_highlight_patches:
            try:
                patch.remove()
            except (ValueError, AttributeError):
                pass
        self.cell_highlight_patches.clear()
        
        # Add highlights for all selected cells
        xmin, xmax, ymin, ymax = config.bounding_box
        grid_res = len(self.cell_centers_x)
        
        for cell_i, cell_j in self.selected_cells:
            # Calculate cell boundaries
            cell_xmin = xmin + cell_j * (xmax - xmin) / grid_res
            cell_xmax = xmin + (cell_j + 1) * (xmax - xmin) / grid_res
            cell_ymin = ymin + cell_i * (ymax - ymin) / grid_res
            cell_ymax = ymin + (cell_i + 1) * (ymax - ymin) / grid_res
            
            # Create highlighting rectangle
            highlight = Rectangle((cell_xmin, cell_ymin), 
                                cell_xmax - cell_xmin, cell_ymax - cell_ymin,
                                facecolor='yellow', alpha=0.3, 
                                edgecolor='orange', linewidth=2, zorder=5)
            self.ax1.add_patch(highlight)
            self.cell_highlight_patches.append(highlight)
    
    def reset_to_no_features(self):
        """Reset system to have no features selected (no particle flow)."""
        self.grad_indices = []  # No features selected
        
        # Create zero grids
        grid_res = len(self.cell_centers_x)
        grid_u_sum = np.zeros((grid_res, grid_res))
        grid_v_sum = np.zeros((grid_res, grid_res))
        
        # Update system grids
        self.system['grid_u_sum'] = grid_u_sum
        self.system['grid_v_sum'] = grid_v_sum
        
        # Create zero interpolators for smooth motion
        self.system['interp_u_sum'] = RegularGridInterpolator(
            (self.cell_centers_y, self.cell_centers_x),
            grid_u_sum, bounds_error=False, fill_value=0.0)
        self.system['interp_v_sum'] = RegularGridInterpolator(
            (self.cell_centers_y, self.cell_centers_x),
            grid_v_sum, bounds_error=False, fill_value=0.0)
    
    def update_status_text(self, message):
        """Update the status text in direction-conditioned mode."""
        self.ax_status.clear()
        self.ax_status.text(0.5, 0.5, message, ha='center', va='center', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        self.ax_status.set_xlim(0, 1)
        self.ax_status.set_ylim(0, 1)
        self.ax_status.axis('off')
    
    def update_direction_constraints(self, val=None):
        """Update direction constraints for selected cells based on slider values."""
        if self.current_mode['mode'] != 'direction_conditioned' or not self.selected_cells:
            return
            
        # Get current angle and magnitude from sliders
        angle_deg = self.angle_slider.val
        magnitude = self.magnitude_slider.val
        
        # Convert angle to direction vector
        angle_rad = np.radians(angle_deg)
        direction_x = np.cos(angle_rad) * magnitude
        direction_y = np.sin(angle_rad) * magnitude
        
        # Update constraints for all selected cells
        for cell in self.selected_cells:
            self.user_constraints[cell] = {
                "direction": (direction_x, direction_y),
                "weight": 1.0,
                "enabled": True
            }
        
        # Update constraint visualization
        self.update_constraint_arrows()
        
        # Trigger feature optimization
        self.optimize_features_for_constraints()
        
        # Update status text with current constraint info
        num_cells = len(self.selected_cells)
        self.update_status_text(f"{num_cells} cell(s) constrained: {angle_deg:.0f}°, mag={magnitude:.1f}")
        
        print(f"Updated constraints: angle={angle_deg:.0f}°, magnitude={magnitude:.1f}")
        self.fig.canvas.draw_idle()
    
    def update_constraint_arrows(self):
        """Update visual arrows showing user constraints on selected cells."""
        # Clear existing constraint arrows
        for arrow in self.constraint_arrows:
            try:
                arrow.remove()
            except (ValueError, AttributeError):
                pass
        self.constraint_arrows.clear()
        
        # Draw arrows for each constraint
        xmin, xmax, ymin, ymax = config.bounding_box
        grid_res = len(self.cell_centers_x)
        
        for (cell_i, cell_j), constraint in self.user_constraints.items():
            if not constraint.get("enabled", True):
                continue
                
            # Calculate cell center
            cell_center_x = xmin + (cell_j + 0.5) * (xmax - xmin) / grid_res
            cell_center_y = ymin + (cell_i + 0.5) * (ymax - ymin) / grid_res
            
            # Get direction vector
            dx, dy = constraint["direction"]
            
            # Scale arrow for visibility
            arrow_scale = 0.8 * min((xmax - xmin) / grid_res, (ymax - ymin) / grid_res)
            arrow_dx = dx * arrow_scale
            arrow_dy = dy * arrow_scale
            
            # Create arrow
            arrow = self.ax1.arrow(cell_center_x, cell_center_y, arrow_dx, arrow_dy,
                                  head_width=arrow_scale*0.2, head_length=arrow_scale*0.2,
                                  fc='red', ec='red', linewidth=2, alpha=0.8, zorder=6)
            self.constraint_arrows.append(arrow)
    
    def optimize_features_for_constraints(self):
        """Find the best combination of features that match user constraints."""
        if not self.user_constraints:
            # No constraints, reset to no features
            self.reset_to_no_features()
            return
        
        print(f"Optimizing features for {len(self.user_constraints)} constraints...")
        
        # Show constraints being optimized for
        for (i, j), constraint in self.user_constraints.items():
            dx, dy = constraint["direction"]
            angle = np.degrees(np.arctan2(dy, dx))
            mag = np.sqrt(dx**2 + dy**2)
            print(f"  Cell ({i},{j}): angle={angle:.0f}°, magnitude={mag:.2f}")
        
        # Parameters for optimization
        max_features = min(10, len(self.col_labels))  # Limit search space
        best_score = -float('inf')
        best_features = []
        
        # Try different feature combinations using greedy forward selection
        current_features = []
        remaining_features = list(range(len(self.col_labels)))
        
        for step in range(max_features):
            best_candidate = None
            best_candidate_score = -float('inf')
            
            # Try adding each remaining feature
            for feat_idx in remaining_features:
                candidate_features = current_features + [feat_idx]
                score = self.evaluate_feature_combination(candidate_features)
                
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = feat_idx
            
            # If adding this feature improves the score, add it
            if best_candidate is not None and best_candidate_score > best_score:
                current_features.append(best_candidate)
                remaining_features.remove(best_candidate)
                best_score = best_candidate_score
                best_features = current_features.copy()
                print(f"Added feature {best_candidate} ({self.col_labels[best_candidate]}), score: {best_score:.3f}")
            else:
                # No improvement, stop search
                break
        
        # Apply the best feature combination
        self.apply_feature_combination(best_features)
        
        # Update status with optimization results
        if best_features:
            feature_names = [self.col_labels[i] for i in best_features[:3]]  # Show first 3 feature names
            if len(best_features) > 3:
                status_msg = f"Using {len(best_features)} features: {', '.join(feature_names)}..."
            else:
                status_msg = f"Using {len(best_features)} features: {', '.join(feature_names)}"
            self.update_status_text(status_msg)
        else:
            self.update_status_text("No suitable features found")
        
        print(f"Final optimization: {len(best_features)} features, score: {best_score:.3f}")
    
    def evaluate_feature_combination(self, feature_indices):
        """Evaluate how well a feature combination matches user constraints."""
        if not feature_indices or not self.user_constraints:
            return -1.0
        
        # Create combined velocity field for these features
        grid_res = len(self.cell_centers_x)
        combined_u = np.zeros((grid_res, grid_res))
        combined_v = np.zeros((grid_res, grid_res))
        
        for feat_idx in feature_indices:
            if feat_idx < len(self.grid_u_all_feats):
                combined_u += self.grid_u_all_feats[feat_idx]
                combined_v += self.grid_v_all_feats[feat_idx]
        
        total_score = 0.0
        total_weight = 0.0
        
        # Evaluate alignment at each constrained cell
        for (cell_i, cell_j), constraint in self.user_constraints.items():
            if not constraint.get("enabled", True):
                continue
            
            # Get desired direction
            desired_dx, desired_dy = constraint["direction"]
            desired_magnitude = np.sqrt(desired_dx**2 + desired_dy**2)
            
            if desired_magnitude < 1e-6:
                continue  # Skip zero constraints
            
            # Get actual velocity directly from cell center
            actual_u = combined_u[cell_i, cell_j]
            actual_v = combined_v[cell_i, cell_j]
            
            actual_magnitude = np.sqrt(actual_u**2 + actual_v**2)
            
            if actual_magnitude < 1e-6:
                # No flow at this location
                alignment = 0.0
            else:
                # Compute cosine similarity (dot product of normalized vectors)
                alignment = (actual_u * desired_dx + actual_v * desired_dy) / (actual_magnitude * desired_magnitude)
                # Bonus for magnitude matching
                magnitude_ratio = min(actual_magnitude / desired_magnitude, desired_magnitude / actual_magnitude)
                alignment = alignment * (0.7 + 0.3 * magnitude_ratio)  # Weight direction more than magnitude
            
            weight = constraint.get("weight", 1.0)
            total_score += alignment * weight
            total_weight += weight
        
        return total_score / max(total_weight, 1e-6)
    
    def apply_feature_combination(self, feature_indices):
        """Apply the optimized feature combination to the system."""
        self.grad_indices = feature_indices
        
        grid_res = len(self.cell_centers_x)
        
        if not feature_indices:
            # No features selected
            grid_u_sum = np.zeros((grid_res, grid_res))
            grid_v_sum = np.zeros((grid_res, grid_res))
        else:
            # Sum the selected features
            grid_u_sum = np.zeros((grid_res, grid_res))
            grid_v_sum = np.zeros((grid_res, grid_res))
            
            for feat_idx in feature_indices:
                if feat_idx < len(self.grid_u_all_feats):
                    grid_u_sum += self.grid_u_all_feats[feat_idx]
                    grid_v_sum += self.grid_v_all_feats[feat_idx]
        
        # Update system grids for particle animation
        self.system['grid_u_sum'] = grid_u_sum
        self.system['grid_v_sum'] = grid_v_sum
        
        # Create RegularGridInterpolators for smooth bilinear interpolation
        self.system['interp_u_sum'] = RegularGridInterpolator(
            (self.cell_centers_y, self.cell_centers_x),
            grid_u_sum, bounds_error=False, fill_value=0.0)
        self.system['interp_v_sum'] = RegularGridInterpolator(
            (self.cell_centers_y, self.cell_centers_x),
            grid_v_sum, bounds_error=False, fill_value=0.0)
    
    def update_top_k_features(self, val):
        """Callback to update selected features based on Top k slider."""
        if self.current_mode['mode'] != 'top_k':
            return
            
        k_new = int(self.k_slider.val)
        
        # Get top k feature indices
        from data_processing import pick_top_k_features
        # Create dummy all_grad_vectors for feature selection
        # In practice, this should come from the main data
        # For now, we'll use the existing grad_indices as a fallback
        if hasattr(self, 'all_grad_vectors'):
            top_k_indices, _ = pick_top_k_features(self.all_grad_vectors, k_new)
        else:
            # Fallback: use first k features
            top_k_indices = list(range(min(k_new, len(self.col_labels))))
        
        self.grad_indices = top_k_indices
        
        # Rebuild grids for new top-k selection
        grid_u_sum = np.sum(self.grid_u_feats[top_k_indices], axis=0)
        grid_v_sum = np.sum(self.grid_v_feats[top_k_indices], axis=0)
        
        # Update system with new grids that animation will use
        self.system['grid_u_sum'] = grid_u_sum
        self.system['grid_v_sum'] = grid_v_sum
        
        # Update interpolators for consistent smooth motion
        self.system['interp_u_sum'] = RegularGridInterpolator(
            (self.cell_centers_y, self.cell_centers_x),
            grid_u_sum, bounds_error=False, fill_value=0.0)
        self.system['interp_v_sum'] = RegularGridInterpolator(
            (self.cell_centers_y, self.cell_centers_x),
            grid_v_sum, bounds_error=False, fill_value=0.0)
        
        print(f"Updated to top {k_new} features")
    
    def handle_mouse_click(self, event):
        """Handle mouse clicks for grid cell selection in direction-conditioned mode."""
        if (self.current_mode['mode'] != 'direction_conditioned' or 
            event.inaxes != self.ax1 or event.button != 1):
            return
        
        # Convert mouse position to grid cell
        xmin, xmax, ymin, ymax = config.bounding_box
        grid_res = len(self.cell_centers_x)
        
        if not (xmin <= event.xdata <= xmax and ymin <= event.ydata <= ymax):
            return
        
        cell_j = int((event.xdata - xmin) / (xmax - xmin) * grid_res)
        cell_i = int((event.ydata - ymin) / (ymax - ymin) * grid_res)
        
        # Clamp to grid bounds
        cell_i = max(0, min(grid_res - 1, cell_i))
        cell_j = max(0, min(grid_res - 1, cell_j))
        
        # Toggle cell selection
        cell = (cell_i, cell_j)
        if cell in self.selected_cells:
            self.selected_cells.remove(cell)
            if cell in self.user_constraints:
                del self.user_constraints[cell]
        else:
            self.selected_cells.add(cell)
        
        # Update visualization
        self.update_cell_highlighting()
        self.update_direction_constraints()
        self.fig.canvas.draw_idle()
        
        print(f"Selected cells: {len(self.selected_cells)}")


def setup_mouse_interactions(ui_controller):
    """
    Setup mouse interaction handlers.
    
    Note: This function is now deprecated in favor of the centralized
    event_manager system to prevent conflicts. It's kept for compatibility
    but no longer connects duplicate event handlers.
    """
    print("Note: Mouse interactions are now handled by event_manager.py")
    print("UI controller will receive click events through the centralized system")