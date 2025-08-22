"""
IBFV (Image-Based Flow Visualization) Renderer
Implements texture-based flow visualization using PyTorch for GPU acceleration.
Supports M1 Mac MPS backend for high-performance texture advection.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import base64
import io
from PIL import Image


class IBFVRenderer:
    """
    Image-Based Flow Visualization renderer using GPU-accelerated texture advection.
    """
    
    def __init__(self, texture_size: int = 512, use_gpu: bool = True):
        """
        Initialize IBFV renderer.
        
        Args:
            texture_size: Size of the flow texture (texture_size x texture_size)
            use_gpu: Whether to use GPU acceleration (MPS on M1 Macs)
        """
        self.texture_size = texture_size
        self.use_gpu = use_gpu
        
        # Setup device (MPS for M1 Macs, CPU fallback)
        if use_gpu and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("Using MPS (Metal Performance Shaders) for GPU acceleration")
        elif use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("Using CUDA for GPU acceleration")
        else:
            self.device = torch.device('cpu')
            print("Using CPU for computation")
        
        # Initialize noise texture for injection
        self.noise_texture = self._generate_noise_texture()
        
        # Flow texture state
        self.flow_texture = None
        self.velocity_field_u = None
        self.velocity_field_v = None
        
        # IBFV parameters
        self.injection_rate = 0.05  # Rate of noise injection
        self.decay_rate = 0.98      # Texture decay factor
        self.advection_steps = 1    # Number of advection steps per frame
        
    def _generate_noise_texture(self) -> torch.Tensor:
        """Generate random noise texture for injection."""
        noise = torch.rand(1, 1, self.texture_size, self.texture_size, device=self.device)
        return noise
    
    def set_velocity_field(self, 
                          velocity_u: np.ndarray, 
                          velocity_v: np.ndarray,
                          bounding_box: list):
        """
        Set the velocity field for flow visualization.
        
        Args:
            velocity_u: U component of velocity field (grid)
            velocity_v: V component of velocity field (grid)
            bounding_box: [xmin, xmax, ymin, ymax] of the field
        """
        # Convert to tensors and resize to texture size
        u_tensor = torch.from_numpy(velocity_u.astype(np.float32))
        v_tensor = torch.from_numpy(velocity_v.astype(np.float32))
        
        # Resize to texture size using bilinear interpolation
        u_resized = F.interpolate(
            u_tensor.unsqueeze(0).unsqueeze(0), 
            size=(self.texture_size, self.texture_size), 
            mode='bilinear', 
            align_corners=False
        )
        v_resized = F.interpolate(
            v_tensor.unsqueeze(0).unsqueeze(0), 
            size=(self.texture_size, self.texture_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        self.velocity_field_u = u_resized.to(self.device)
        self.velocity_field_v = v_resized.to(self.device)
        self.bounding_box = bounding_box
        
        # Initialize flow texture if not exists
        if self.flow_texture is None:
            self.flow_texture = torch.zeros(
                1, 1, self.texture_size, self.texture_size, 
                device=self.device
            )
    
    def _advect_texture(self, texture: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Advect texture using semi-Lagrangian advection.
        
        Args:
            texture: Input texture to advect
            dt: Time step for advection
            
        Returns:
            Advected texture
        """
        if self.velocity_field_u is None or self.velocity_field_v is None:
            return texture
        
        # Create coordinate grids
        h, w = self.texture_size, self.texture_size
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h, device=self.device),
            torch.linspace(-1, 1, w, device=self.device),
            indexing='ij'
        )
        
        # Current grid coordinates
        grid_coords = torch.stack([x_coords, y_coords], dim=0)  # [2, H, W]
        
        # Get velocity at current positions
        u_vel = self.velocity_field_u.squeeze()  # [H, W]
        v_vel = self.velocity_field_v.squeeze()  # [H, W]
        
        # Normalize velocity for texture space (-1 to 1)
        velocity_scale = 2.0 / self.texture_size  # Convert to normalized coordinates
        u_normalized = u_vel * velocity_scale * dt
        v_normalized = v_vel * velocity_scale * dt
        
        # Compute source positions (semi-Lagrangian: where did this pixel come from?)
        source_x = x_coords - u_normalized
        source_y = y_coords - v_normalized
        
        # Clamp to valid range
        source_x = torch.clamp(source_x, -1, 1)
        source_y = torch.clamp(source_y, -1, 1)
        
        # Create sampling grid for F.grid_sample
        sampling_grid = torch.stack([source_x, source_y], dim=-1)  # [H, W, 2]
        sampling_grid = sampling_grid.unsqueeze(0)  # [1, H, W, 2]
        
        # Sample texture at source positions
        # Note: MPS backend doesn't support 'border' padding, use 'zeros' instead
        padding_mode = 'zeros' if self.device.type == 'mps' else 'border'
        advected = F.grid_sample(
            texture,
            sampling_grid,
            mode='bilinear',
            padding_mode=padding_mode,
            align_corners=False
        )
        
        return advected
    
    def update_flow_texture(self) -> torch.Tensor:
        """
        Update flow texture with one IBFV iteration.
        
        Returns:
            Updated flow texture
        """
        if self.flow_texture is None:
            self.flow_texture = torch.zeros(
                1, 1, self.texture_size, self.texture_size,
                device=self.device
            )
        
        # Advect current texture
        for _ in range(self.advection_steps):
            self.flow_texture = self._advect_texture(self.flow_texture)
        
        # Apply decay
        self.flow_texture *= self.decay_rate
        
        # Inject noise
        noise_contribution = self.noise_texture * self.injection_rate
        self.flow_texture = torch.clamp(
            self.flow_texture + noise_contribution, 
            0.0, 1.0
        )
        
        return self.flow_texture
    
    def render_frame(self, 
                    velocity_u: Optional[np.ndarray] = None,
                    velocity_v: Optional[np.ndarray] = None,
                    bounding_box: Optional[list] = None,
                    steps: int = 1) -> np.ndarray:
        """
        Render one frame of IBFV visualization.
        
        Args:
            velocity_u: U component of velocity field (optional, uses existing if None)
            velocity_v: V component of velocity field (optional, uses existing if None)
            bounding_box: Bounding box (optional, uses existing if None)
            steps: Number of advection steps to perform
            
        Returns:
            Rendered frame as numpy array [H, W] in range [0, 1]
        """
        # Update velocity field if provided
        if velocity_u is not None and velocity_v is not None:
            self.set_velocity_field(velocity_u, velocity_v, bounding_box or self.bounding_box)
        
        # Perform multiple advection steps
        for _ in range(steps):
            self.update_flow_texture()
        
        # Convert to numpy for output
        frame = self.flow_texture.squeeze().cpu().numpy()
        return frame
    
    def render_to_base64(self, 
                        velocity_u: Optional[np.ndarray] = None,
                        velocity_v: Optional[np.ndarray] = None,
                        bounding_box: Optional[list] = None,
                        steps: int = 1) -> str:
        """
        Render frame and encode as base64 PNG for web transmission.
        
        Returns:
            Base64-encoded PNG image string
        """
        frame = self.render_frame(velocity_u, velocity_v, bounding_box, steps)
        
        # Convert to 8-bit image
        frame_8bit = (frame * 255).astype(np.uint8)
        
        # Create PIL image
        img = Image.fromarray(frame_8bit, mode='L')  # Grayscale
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_b64}"
    
    def set_parameters(self, 
                      injection_rate: Optional[float] = None,
                      decay_rate: Optional[float] = None,
                      advection_steps: Optional[int] = None):
        """Update IBFV parameters."""
        if injection_rate is not None:
            self.injection_rate = max(0.0, min(1.0, injection_rate))
        if decay_rate is not None:
            self.decay_rate = max(0.0, min(1.0, decay_rate))
        if advection_steps is not None:
            self.advection_steps = max(1, advection_steps)
    
    def reset_texture(self):
        """Reset flow texture to initial state."""
        if self.flow_texture is not None:
            self.flow_texture.zero_()
    
    def get_parameters(self) -> dict:
        """Get current IBFV parameters."""
        return {
            'injection_rate': self.injection_rate,
            'decay_rate': self.decay_rate,
            'advection_steps': self.advection_steps,
            'texture_size': self.texture_size,
            'device': str(self.device)
        }


class IBFVAnimator:
    """
    Animator for continuous IBFV visualization updates.
    Manages frame generation and parameter evolution over time.
    """
    
    def __init__(self, renderer: IBFVRenderer):
        self.renderer = renderer
        self.frame_count = 0
        self.target_fps = 30
        
    def generate_animation_frames(self, 
                                 velocity_u: np.ndarray,
                                 velocity_v: np.ndarray,
                                 bounding_box: list,
                                 num_frames: int = 60) -> list:
        """
        Generate sequence of IBFV animation frames.
        
        Args:
            velocity_u, velocity_v: Velocity field components
            bounding_box: Field bounding box
            num_frames: Number of frames to generate
            
        Returns:
            List of base64-encoded frame images
        """
        self.renderer.set_velocity_field(velocity_u, velocity_v, bounding_box)
        
        frames = []
        for i in range(num_frames):
            # Vary injection rate for dynamic visualization
            injection_rate = 0.03 + 0.02 * np.sin(i * 0.2)
            self.renderer.set_parameters(injection_rate=injection_rate)
            
            frame_b64 = self.renderer.render_to_base64(steps=1)
            frames.append(frame_b64)
            
        return frames
    
    def get_realtime_frame(self, 
                          velocity_u: np.ndarray,
                          velocity_v: np.ndarray,
                          bounding_box: list) -> str:
        """Get single frame for real-time visualization."""
        self.frame_count += 1
        
        # Dynamic parameters for visual interest
        time_factor = self.frame_count * 0.05
        injection_rate = 0.04 + 0.01 * np.sin(time_factor)
        
        self.renderer.set_parameters(injection_rate=injection_rate)
        
        return self.renderer.render_to_base64(velocity_u, velocity_v, bounding_box)