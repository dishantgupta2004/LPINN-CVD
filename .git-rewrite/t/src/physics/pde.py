"""
PDE residual calculations for CVD modeling.

This module computes the residuals of the partial differential equations 
governing the Chemical Vapor Deposition (CVD) process.
"""
from typing import List, Dict, Tuple, Optional, Union
import tensorflow as tf
import numpy as np

from src.config import PhysicalConfig


class CVDPDE:
    """
    Class to compute PDE residuals for CVD simulation.
    """
    
    def __init__(self, physical_config: PhysicalConfig):
        """
        Initialize with physical parameters.
        
        Args:
            physical_config: Physical parameters for CVD simulation
        """
        self.config = physical_config
    
    def compute_reaction_rates(self, 
                              concentrations: List[tf.Tensor], 
                              temperature: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute reaction rates based on Arrhenius equation.
        
        Args:
            concentrations: List of concentration tensors [SiH4, Si, H2, SiH2]
            temperature: Temperature tensor
            
        Returns:
            Tuple of reaction rate tensors (R1, R2, R3)
        """
        # Unpack concentrations
        SiH4, Si, H2, SiH2 = concentrations
        
        # Compute reaction rates using Arrhenius equation
        # R_j = A_j * exp(-E_j/(R*T)) * product(C_i^n_ij)
        
        # Reaction 1: SiH4 -> Si + 2H2
        R1 = self.config.A1 * tf.exp(-self.config.E1 / (self.config.R * temperature)) * SiH4
        
        # Reaction 2: SiH4 + H2 -> SiH2 + 2H2
        R2 = self.config.A2 * tf.exp(-self.config.E2 / (self.config.R * temperature)) * SiH4 * H2
        
        # Reaction 3: SiH2 + SiH4 -> Si2H6
        R3 = self.config.A3 * tf.exp(-self.config.E3 / (self.config.R * temperature)) * SiH2 * SiH4
        
        return R1, R2, R3
    
    def compute_residuals(self, 
                          coordinates: tf.Tensor, 
                          predictions: tf.Tensor, 
                          derivatives: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute PDE residuals for the CVD system.
        
        Args:
            coordinates: Input coordinates tensor (x, y, t) or (x, y, z, t)
            predictions: Predictions tensor [SiH4, Si, H2, SiH2, T]
            derivatives: Dictionary of derivatives from PINN.get_gradients()
            
        Returns:
            Tuple of residual tensors for each equation (SiH4, Si, H2, SiH2, T)
        """
        # Extract predictions
        SiH4 = predictions[:, 0:1]
        Si = predictions[:, 1:2]
        H2 = predictions[:, 2:3]
        SiH2 = predictions[:, 3:4]
        T = predictions[:, 4:5]
        
        # Extract derivatives
        y_t = derivatives['y_t']    # Time derivatives
        y_xx = derivatives['y_xx']  # Second x-derivatives
        y_yy = derivatives['y_yy']  # Second y-derivatives
        
        if 'y_zz' in derivatives:  # 3D case
            y_zz = derivatives['y_zz']  # Second z-derivatives
        else:
            y_zz = 0.0  # 2D case
        
        # Extract individual species derivatives
        # Temporal derivatives
        SiH4_t = y_t[:, 0:1]
        Si_t = y_t[:, 1:2]
        H2_t = y_t[:, 2:3]
        SiH2_t = y_t[:, 3:4]
        T_t = y_t[:, 4:5]
        
        # Second spatial derivatives (Laplacian terms)
        SiH4_lap = y_xx[:, 0:1] + y_yy[:, 0:1]
        Si_lap = y_xx[:, 1:2] + y_yy[:, 1:2]
        H2_lap = y_xx[:, 2:3] + y_yy[:, 2:3]
        SiH2_lap = y_xx[:, 3:4] + y_yy[:, 3:4]
        T_lap = y_xx[:, 4:5] + y_yy[:, 4:5]
        
        if 'y_zz' in derivatives:  # 3D case
            SiH4_lap += y_zz[:, 0:1]
            Si_lap += y_zz[:, 1:2]
            H2_lap += y_zz[:, 2:3]
            SiH2_lap += y_zz[:, 3:4]
            T_lap += y_zz[:, 4:5]
        
        # Calculate reaction rates
        R1, R2, R3 = self.compute_reaction_rates([SiH4, Si, H2, SiH2], T)
        
        # Compute PDE residuals for each species
        
        # Residual for SiH4 (silane)
        # ∂SiH4/∂t = D_SiH4 ∇²SiH4 - R1 - R2 - R3
        res_SiH4 = SiH4_t - self.config.D_SiH4 * SiH4_lap + R1 + R2 + R3
        
        # Residual for Si (silicon)
        # ∂Si/∂t = D_Si ∇²Si + R1
        res_Si = Si_t - self.config.D_Si * Si_lap - R1
        
        # Residual for H2 (hydrogen)
        # ∂H2/∂t = D_H2 ∇²H2 + 2*R1 + 2*R2
        res_H2 = H2_t - self.config.D_H2 * H2_lap - 2.0 * R1 - 2.0 * R2
        
        # Residual for SiH2 (silylene)
        # ∂SiH2/∂t = D_SiH2 ∇²SiH2 + R2 - R3
        res_SiH2 = SiH2_t - self.config.D_SiH2 * SiH2_lap - R2 + R3
        
        # Energy equation (heat transfer)
        # ∂T/∂t = (k/(ρ*Cp)) ∇²T + Q/(ρ*Cp)
        # where Q is the heat source term from reactions
        
        # Heat source term (exothermic reactions release heat)
        # Simplified with artificial heat release coefficients
        Q = 1000.0 * (R1 + 500.0 * R2 + 300.0 * R3)
        
        # Thermal diffusivity
        thermal_diffusivity = self.config.thermal_conductivity / (self.config.density * self.config.specific_heat)
        
        # Residual for temperature
        res_T = T_t - thermal_diffusivity * T_lap - Q / (self.config.density * self.config.specific_heat)
        
        return res_SiH4, res_Si, res_H2, res_SiH2, res_T
    
    def compute_total_residual_norm(self, 
                                   coordinates: tf.Tensor, 
                                   predictions: tf.Tensor, 
                                   derivatives: Dict[str, tf.Tensor],
                                   weights: Optional[List[float]] = None) -> tf.Tensor:
        """
        Compute weighted norm of all PDE residuals.
        
        Args:
            coordinates: Input coordinates tensor
            predictions: Predictions tensor
            derivatives: Dictionary of derivatives
            weights: Optional weights for each equation
            
        Returns:
            Total residual norm
        """
        # Compute individual residuals
        residuals = self.compute_residuals(coordinates, predictions, derivatives)
        
        # Default weights if not provided
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        
        # Square each residual, apply weights, and sum
        total_residual = 0.0
        for i, residual in enumerate(residuals):
            total_residual += weights[i] * tf.reduce_mean(tf.square(residual))
        
        return total_residual


class AdaptivePDEWeights:
    """
    Adaptive weighting for PDE residuals based on gradient statistics.
    Helps balance multiple PDEs in the loss function.
    """
    
    def __init__(self, num_equations: int = 5, initial_weights: Optional[List[float]] = None):
        """
        Initialize with number of equations and initial weights.
        
        Args:
            num_equations: Number of PDE equations
            initial_weights: Initial weights for each equation
        """
        self.num_equations = num_equations
        
        # Default initial weights (all equal)
        if initial_weights is None:
            initial_weights = [1.0] * num_equations
        
        # Create TensorFlow variables for weights
        self.weights = [tf.Variable(w, dtype=tf.float32, trainable=False) for w in initial_weights]
        
        # Moving averages of gradient norms
        self.grad_norms = [tf.Variable(0.0, dtype=tf.float32, trainable=False) for _ in range(num_equations)]
        
        # Decay factor for moving average
        self.decay = 0.99
    
    def update_weights(self, gradients: List[tf.Tensor]):
        """
        Update weights based on gradient statistics.
        
        Args:
            gradients: List of gradients for each equation
        """
        # Compute gradient norms
        norms = [tf.norm(grad) for grad in gradients]
        
        # Update moving averages
        for i in range(self.num_equations):
            self.grad_norms[i].assign(self.decay * self.grad_norms[i] + (1.0 - self.decay) * norms[i])
        
        # Compute weights: inversely proportional to gradient norms
        norm_sum = tf.reduce_sum([1.0 / (norm + 1e-10) for norm in self.grad_norms])
        
        # Normalize weights to sum to num_equations
        for i in range(self.num_equations):
            new_weight = self.num_equations * (1.0 / (self.grad_norms[i] + 1e-10)) / norm_sum
            self.weights[i].assign(new_weight)
    
    def get_weights(self) -> List[float]:
        """
        Get current weights.
        
        Returns:
            List of current weights
        """
        return [w.numpy() for w in self.weights]