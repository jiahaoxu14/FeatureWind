"""
Event manager module for reliable mouse interaction handling.

This module provides centralized event management to prevent conflicts
and ensure reliable mouse interactions in the FeatureWind visualization.
"""

import time
import numpy as np
from threading import Lock


class EventManager:
    """
    Centralized event manager to handle mouse interactions reliably.
    
    Features:
    - Debounced event handling to prevent conflicts
    - Thread-safe canvas updates
    - Error recovery for failed interactions
    - Performance monitoring and diagnostics
    """
    
    def __init__(self, fig, ax1, ax2, ui_controller=None):
        """Initialize the event manager."""
        self.fig = fig
        self.ax1 = ax1  # Main plot axis
        self.ax2 = ax2  # Wind vane axis
        self.ui_controller = ui_controller
        
        # Event handling state
        self.last_mouse_update = 0
        self.last_click_update = 0
        self.update_lock = Lock()
        self.mouse_data = {'grid_cell': None, 'grid_res': 40}
        
        # Performance monitoring
        self.event_count = 0
        self.failed_updates = 0
        self.last_performance_report = time.time()
        
        # Debouncing parameters - reduced for better responsiveness
        self.MOUSE_DEBOUNCE_MS = 16   # Minimum time between mouse updates (60 FPS)
        self.CLICK_DEBOUNCE_MS = 50   # Minimum time between click updates
        self.CANVAS_UPDATE_MS = 16    # Target 60 FPS for canvas updates
        
        # Connected event handlers
        self.connected_handlers = []
        
    def connect_events(self):
        """Connect all event handlers in a coordinated way."""
        # Disconnect any existing handlers first
        self.disconnect_events()
        
        # Clear all existing callbacks to avoid conflicts  
        try:
            # Clear motion and button press callbacks specifically
            if hasattr(self.fig.canvas.callbacks, 'callbacks'):
                self.fig.canvas.callbacks.callbacks.get('motion_notify_event', []).clear()
                self.fig.canvas.callbacks.callbacks.get('button_press_event', []).clear()
        except (AttributeError, KeyError):
            pass  # Fallback if callback structure is different
        
        # Connect our centralized handlers
        try:
            motion_cid = self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
            click_cid = self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_click)
            
            self.connected_handlers = [motion_cid, click_cid]
            print("✓ Event handlers connected successfully")
            
        except Exception as e:
            print(f"✗ Failed to connect event handlers: {e}")
            self.failed_updates += 1
    
    def disconnect_events(self):
        """Safely disconnect all event handlers."""
        for cid in self.connected_handlers:
            try:
                self.fig.canvas.mpl_disconnect(cid)
            except Exception as e:
                print(f"Warning: Failed to disconnect handler {cid}: {e}")
        self.connected_handlers.clear()
    
    def _on_mouse_move(self, event):
        """Centralized mouse movement handler with debouncing."""
        current_time = time.time() * 1000  # Convert to milliseconds
        
        # Debounce rapid mouse movements
        if current_time - self.last_mouse_update < self.MOUSE_DEBOUNCE_MS:
            return
        
        try:
            self._handle_mouse_move_safe(event)
            self.last_mouse_update = current_time
            self.event_count += 1
            
        except Exception as e:
            self.failed_updates += 1
            pass  # Silently handle mouse move error
            # Don't propagate exceptions to avoid breaking the event loop
    
    def _on_mouse_click(self, event):
        """Centralized mouse click handler with debouncing."""
        current_time = time.time() * 1000  # Convert to milliseconds
        
        # Debounce rapid clicks
        if current_time - self.last_click_update < self.CLICK_DEBOUNCE_MS:
            return
        
        try:
            self._handle_mouse_click_safe(event)
            self.last_click_update = current_time
            self.event_count += 1
            
        except Exception as e:
            self.failed_updates += 1
            pass  # Silently handle mouse click error
    
    def _handle_mouse_move_safe(self, event):
        """Safe mouse movement handling with error recovery."""
        if not (event.inaxes == self.ax1 and event.xdata is not None and event.ydata is not None):
            return
        
        # Import config here to avoid circular imports
        import config
        
        xmin, xmax, ymin, ymax = config.bounding_box
        grid_res = self.mouse_data['grid_res']
        
        # Bounds checking
        if not (xmin <= event.xdata <= xmax and ymin <= event.ydata <= ymax):
            return
        
        # Calculate grid cell with bounds clamping
        cell_j = int((event.xdata - xmin) / (xmax - xmin) * grid_res)
        cell_i = int((event.ydata - ymin) / (ymax - ymin) * grid_res)
        
        cell_i = max(0, min(grid_res - 1, cell_i))
        cell_j = max(0, min(grid_res - 1, cell_j))
        
        # Only update if cell changed (reduce unnecessary updates)
        new_cell = (cell_i, cell_j)
        if self.mouse_data['grid_cell'] != new_cell:
            self.mouse_data['grid_cell'] = new_cell
            
            # Update wind vane through callback system
            if hasattr(self, 'wind_vane_callback') and self.wind_vane_callback:
                try:
                    self.wind_vane_callback(self.mouse_data)
                except Exception as e:
                    pass  # Silently handle wind vane update error
            
            # Thread-safe canvas update with timing control
            self._safe_canvas_update()
    
    def _handle_mouse_click_safe(self, event):
        """Safe mouse click handling."""
        if self.ui_controller and hasattr(self.ui_controller, 'handle_mouse_click'):
            try:
                self.ui_controller.handle_mouse_click(event)
            except Exception as e:
                pass  # Silently handle UI controller click error
                self.failed_updates += 1
    
    def _safe_canvas_update(self):
        """Thread-safe canvas update with timing control."""
        current_time = time.time() * 1000
        
        # Rate limit canvas updates to prevent overwhelming the GUI thread
        if current_time - getattr(self, '_last_canvas_update', 0) < self.CANVAS_UPDATE_MS:
            return
        
        try:
            with self.update_lock:
                # Use draw_idle() for non-blocking updates
                self.fig.canvas.draw_idle()
                self._last_canvas_update = current_time
                
        except Exception as e:
            pass  # Silently handle canvas update error
            self.failed_updates += 1
    
    def set_wind_vane_callback(self, callback):
        """Set the callback function for wind vane updates."""
        self.wind_vane_callback = callback
    
    def set_grid_resolution(self, grid_res):
        """Update grid resolution for coordinate calculations."""
        self.mouse_data['grid_res'] = grid_res
    
    def get_performance_stats(self):
        """Get performance statistics for diagnostics."""
        current_time = time.time()
        elapsed = current_time - self.last_performance_report
        
        if elapsed > 5.0:  # Report every 5 seconds
            events_per_sec = self.event_count / elapsed if elapsed > 0 else 0
            failure_rate = (self.failed_updates / max(self.event_count, 1)) * 100
            
            stats = {
                'events_per_second': events_per_sec,
                'failure_rate_percent': failure_rate,
                'total_events': self.event_count,
                'total_failures': self.failed_updates
            }
            
            print(f"Event Performance: {events_per_sec:.1f} events/sec, {failure_rate:.1f}% failure rate")
            
            # Reset counters
            self.event_count = 0
            self.failed_updates = 0
            self.last_performance_report = current_time
            
            return stats
        
        return None
    
    def force_refresh(self):
        """Force a complete refresh of the visualization."""
        try:
            with self.update_lock:
                self.fig.canvas.draw()  # Synchronous draw for immediate update
                print("✓ Forced canvas refresh completed")
        except Exception as e:
            print(f"✗ Force refresh failed: {e}")
    
    def reset_state(self):
        """Reset event manager state (useful for troubleshooting)."""
        with self.update_lock:
            self.mouse_data['grid_cell'] = None
            self.last_mouse_update = 0
            self.last_click_update = 0
            self._last_canvas_update = 0
            print("✓ Event manager state reset")


def create_reliable_event_system(fig, ax1, ax2, ui_controller, system, col_labels, 
                                grad_indices, feature_colors, grid_res):
    """
    Create a reliable event handling system for FeatureWind.
    
    This function sets up the enhanced event manager and connects all
    necessary callbacks for robust mouse interaction handling.
    
    Returns:
        EventManager: Configured event manager instance
    """
    # Create the event manager
    event_manager = EventManager(fig, ax1, ax2, ui_controller)
    event_manager.set_grid_resolution(grid_res)
    
    # Create wind vane update callback with family support
    def wind_vane_update_callback(mouse_data):
        """Callback for updating wind vane with family-based colors."""
        try:
            import visualization_core
            # Check if family assignments are stored in system
            family_assignments = system.get('family_assignments', None)
            
            visualization_core.update_wind_vane(ax2, mouse_data, system, col_labels, 
                                              grad_indices, feature_colors, family_assignments)
        except Exception as e:
            pass  # Silently handle wind vane callback error
    
    # Set the callback and connect events
    event_manager.set_wind_vane_callback(wind_vane_update_callback)
    event_manager.connect_events()
    
    
    # Performance monitoring setup
    def periodic_performance_check():
        """Check performance periodically."""
        stats = event_manager.get_performance_stats()
        if stats and stats['failure_rate_percent'] > 5.0:
            pass  # Silently handle high failure rate
    
    # You can call periodic_performance_check() from your animation loop if desired
    
    return event_manager