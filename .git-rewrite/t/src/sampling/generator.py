"""
Data generation for CVD-PINN training.

This module provides classes for generating training data for CVD simulations.
"""
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import tensorflow as tf

from src.config import DomainConfig, SamplingConfig


class CVDDataGenerator:
    """
    Data generator for CVD simulation.
    """
    
    def __init__(self, 
                domain_config: DomainConfig, 
                sampling_config: Optional[SamplingConfig] = None):
        """
        Initialize the data generator.
        
        Args:
            domain_config: Domain configuration
            sampling_config: Sampling configuration
        """
        self.domain = domain_config
        
        # Use default sampling configuration if not provided
        if sampling_config is None:
            sampling_config = SamplingConfig()
        
        self.sampling = sampling_config
    
    def generate_collocation_points(self, n_points: Optional[int] = None) -> np.ndarray:
        """
        Generate random collocation points for PDE residuals.
        
        Args:
            n_points: Number of points to generate
            
        Returns:
            Array of points, shape (n_points, input_dim)
        """
        if n_points is None:
            n_points = self.sampling.n_collocation_points
        
        # Generate random points within domain bounds
        if self.domain.dimension == 2:  # 2D + time
            x = np.random.uniform(self.domain.x_min, self.domain.x_max, n_points)
            y = np.random.uniform(self.domain.y_min, self.domain.y_max, n_points)
            t = np.random.uniform(self.domain.t_min, self.domain.t_max, n_points)
            
            # Stack coordinates
            points = np.stack([x, y, t], axis=1)
        else:  # 3D + time
            x = np.random.uniform(self.domain.x_min, self.domain.x_max, n_points)
            y = np.random.uniform(self.domain.y_min, self.domain.y_max, n_points)
            z = np.random.uniform(self.domain.z_min, self.domain.z_max, n_points)
            t = np.random.uniform(self.domain.t_min, self.domain.t_max, n_points)
            
            # Stack coordinates
            points = np.stack([x, y, z, t], axis=1)
        
        return points
    
    def generate_boundary_points(self, n_points_per_boundary: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate boundary points for boundary conditions.
        
        Args:
            n_points_per_boundary: Number of points to generate per boundary
            
        Returns:
            Dictionary with boundary points for each boundary
        """
        if n_points_per_boundary is None:
            n_points_per_boundary = self.sampling.n_boundary_points
        
        # Time points - used for all boundaries
        t = np.random.uniform(self.domain.t_min, self.domain.t_max, n_points_per_boundary)
        
        if self.domain.dimension == 2:  # 2D + time
            # Generate boundary points for each boundary
            boundary_points = {}
            
            # Lower boundary (y = y_min) - Inlet
            x_lower = np.random.uniform(self.domain.x_min, self.domain.x_max, n_points_per_boundary)
            y_lower = np.ones_like(x_lower) * self.domain.y_min
            boundary_points['inlet'] = np.stack([x_lower, y_lower, t], axis=1)
            
            # Upper boundary (y = y_max) - Substrate
            x_upper = np.random.uniform(self.domain.x_min, self.domain.x_max, n_points_per_boundary)
            y_upper = np.ones_like(x_upper) * self.domain.y_max
            boundary_points['substrate'] = np.stack([x_upper, y_upper, t], axis=1)
            
            # Left boundary (x = x_min) - Wall
            y_left = np.random.uniform(self.domain.y_min, self.domain.y_max, n_points_per_boundary)
            x_left = np.ones_like(y_left) * self.domain.x_min
            boundary_points['left_wall'] = np.stack([x_left, y_left, t], axis=1)
            
            # Right boundary (x = x_max) - Wall
            y_right = np.random.uniform(self.domain.y_min, self.domain.y_max, n_points_per_boundary)
            x_right = np.ones_like(y_right) * self.domain.x_max
            boundary_points['right_wall'] = np.stack([x_right, y_right, t], axis=1)
        
        else:  # 3D + time
            # Generate boundary points for 3D domain
            boundary_points = {}
            
            # Bottom boundary (z = z_min) - Inlet
            x_bottom = np.random.uniform(self.domain.x_min, self.domain.x_max, n_points_per_boundary)
            y_bottom = np.random.uniform(self.domain.y_min, self.domain.y_max, n_points_per_boundary)
            z_bottom = np.ones_like(x_bottom) * self.domain.z_min
            boundary_points['inlet'] = np.stack([x_bottom, y_bottom, z_bottom, t], axis=1)
            
            # Top boundary (z = z_max) - Substrate
            x_top = np.random.uniform(self.domain.x_min, self.domain.x_max, n_points_per_boundary)
            y_top = np.random.uniform(self.domain.y_min, self.domain.y_max, n_points_per_boundary)
            z_top = np.ones_like(x_top) * self.domain.z_max
            boundary_points['substrate'] = np.stack([x_top, y_top, z_top, t], axis=1)
            
            # Add other boundaries for 3D...
            # (left, right, front, back walls)
        
        return boundary_points
    
    def generate_initial_points(self, n_points: Optional[int] = None) -> np.ndarray:
        """
        Generate initial condition points at t = t_min.
        
        Args:
            n_points: Number of points to generate
            
        Returns:
            Array of points, shape (n_points, input_dim)
        """
        if n_points is None:
            n_points = self.sampling.n_initial_points
        
        # Generate random spatial points
        if self.domain.dimension == 2:  # 2D + time
            x = np.random.uniform(self.domain.x_min, self.domain.x_max, n_points)
            y = np.random.uniform(self.domain.y_min, self.domain.y_max, n_points)
            t = np.ones_like(x) * self.domain.t_min
            
            # Stack coordinates
            points = np.stack([x, y, t], axis=1)
        else:  # 3D + time
            x = np.random.uniform(self.domain.x_min, self.domain.x_max, n_points)
            y = np.random.uniform(self.domain.y_min, self.domain.y_max, n_points)
            z = np.random.uniform(self.domain.z_min, self.domain.z_max, n_points)
            t = np.ones_like(x) * self.domain.t_min
            
            # Stack coordinates
            points = np.stack([x, y, z, t], axis=1)
        
        return points
    
    def generate_uniform_grid(self, nx: int, ny: int, nt: int, nz: Optional[int] = None) -> Tuple[np.ndarray, Tuple]:
        """
        Generate uniform grid for visualization and testing.
        
        Args:
            nx: Number of points in x direction
            ny: Number of points in y direction
            nt: Number of points in t direction
            nz: Number of points in z direction (for 3D)
            
        Returns:
            Tuple of (grid_points, grid_shape)
        """
        # Generate grid points
        x = np.linspace(self.domain.x_min, self.domain.x_max, nx)
        y = np.linspace(self.domain.y_min, self.domain.y_max, ny)
        t = np.linspace(self.domain.t_min, self.domain.t_max, nt)
        
        if self.domain.dimension == 2:  # 2D + time
            # Create meshgrid
            X, Y, T = np.meshgrid(x, y, t, indexing='ij')
            
            # Stack coordinates
            grid_points = np.stack([X.flatten(), Y.flatten(), T.flatten()], axis=1)
            grid_shape = (nx, ny, nt)
        else:  # 3D + time
            if nz is None:
                nz = min(nx, ny)  # Default value if not provided
            
            z = np.linspace(self.domain.z_min, self.domain.z_max, nz)
            
            # Create meshgrid
            X, Y, Z, T = np.meshgrid(x, y, z, t, indexing='ij')
            
            # Stack coordinates
            grid_points = np.stack([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()], axis=1)
            grid_shape = (nx, ny, nz, nt)
        
        return grid_points, grid_shape
    
    def convert_to_tensor(self, points: np.ndarray) -> tf.Tensor:
        """
        Convert numpy array to TensorFlow tensor.
        
        Args:
            points: Numpy array of points
            
        Returns:
            TensorFlow tensor
        """
        return tf.convert_to_tensor(points, dtype=tf.float32)