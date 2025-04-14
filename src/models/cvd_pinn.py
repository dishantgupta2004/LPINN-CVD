"""
CVD-specific PINN implementation.

This module extends the base PINN model for Chemical Vapor Deposition (CVD) modeling.
"""
from typing import List, Optional, Dict, Any, Tuple, Union, Callable
import tensorflow as tf
from tensorflow.keras.layers import Lambda

from .base import PINN
from src.config import ModelConfig, PhysicalConfig


class CVDPINN(PINN):
    """CVD-specific PINN model."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        physical_config: PhysicalConfig,
        dimension: int = 2,
        name: str = "cvd_pinn",
        **kwargs
    ):
        """
        Initialize the CVD-PINN model.
        
        Args:
            model_config: Model configuration
            physical_config: Physical parameters configuration
            dimension: Spatial dimension (2 or 3)
            name: Name of the model
            **kwargs: Additional arguments for the base class
        """
        super(CVDPINN, self).__init__(
            hidden_layers=model_config.hidden_layers,
            output_size=model_config.output_size,
            activation=model_config.activation,
            output_activation=model_config.output_activation,
            weight_initializer=model_config.weight_initializer,
            dropout_rate=model_config.dropout_rate,
            use_bias=model_config.use_bias,
            name=name,
            **kwargs
        )
        
        self.model_config = model_config
        self.physical_config = physical_config
        self.dimension = dimension
        
        # Species names for outputs
        self.species_names = ["SiH4", "Si", "H2", "SiH2", "Temperature"]
        
        # Additional layers for output scaling
        self.scaling_layer = None
        
        # Setup output scaling for better numerical stability
        self._setup_output_scaling()
    
    def _setup_output_scaling(self) -> None:
        """Setup output scaling for better numerical stability."""
        
        def scale_outputs(x):
            """Scale outputs to appropriate ranges."""
            # Extract outputs
            SiH4 = x[:, 0:1]       # Range: [0, 0.2]
            Si = x[:, 1:2]         # Range: [0, 0.2]
            H2 = x[:, 2:3]         # Range: [0, 0.3]
            SiH2 = x[:, 3:4]       # Range: [0, 0.1]
            Temperature = x[:, 4:5]  # Range: [300, 800]
            
            # Apply scaling
            # For concentrations: Apply sigmoid to ensure positive values
            SiH4_scaled = 0.2 * tf.sigmoid(SiH4)
            Si_scaled = 0.2 * tf.sigmoid(Si)
            H2_scaled = 0.3 * tf.sigmoid(H2)
            SiH2_scaled = 0.1 * tf.sigmoid(SiH2)
            
            # For temperature: Apply softplus and shift to ensure physical range
            T_scaled = 300.0 + 500.0 * tf.nn.softplus(Temperature) / (1.0 + tf.nn.softplus(Temperature))
            
            # Concatenate
            return tf.concat([SiH4_scaled, Si_scaled, H2_scaled, SiH2_scaled, T_scaled], axis=1)
        
        self.scaling_layer = Lambda(scale_outputs)
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass with output scaling.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Scaled output tensor
        """
        # Forward pass through base model
        x = super(CVDPINN, self).call(inputs, training=training)
        
        # Apply scaling
        if self.scaling_layer is not None:
            x = self.scaling_layer(x)
        
        return x
    
    def get_species_concentration(self, x: tf.Tensor, species: str) -> tf.Tensor:
        """
        Get concentration of a specific species.
        
        Args:
            x: Input tensor
            species: Species name
            
        Returns:
            Concentration tensor
        """
        y = self(x, training=False)
        
        if species not in self.species_names:
            raise ValueError(f"Unknown species: {species}")
        
        idx = self.species_names.index(species)
        return y[:, idx:idx+1]
    
    def get_temperature(self, x: tf.Tensor) -> tf.Tensor:
        """
        Get temperature.
        
        Args:
            x: Input tensor
            
        Returns:
            Temperature tensor
        """
        y = self(x, training=False)
        return y[:, 4:5]  # Temperature is the last output


class CVDPINNEnsemble:
    """Ensemble of CVD-PINN models for uncertainty quantification."""
    
    def __init__(
        self,
        models: List[CVDPINN],
        name: str = "cvd_pinn_ensemble"
    ):
        """
        Initialize the CVD-PINN ensemble.
        
        Args:
            models: List of CVD-PINN models
            name: Name of the ensemble
        """
        self.models = models
        self.name = name
        
        if not models:
            raise ValueError("Ensemble must contain at least one model")
        
        # All models should have the same configuration
        self.model_config = models[0].model_config
        self.physical_config = models[0].physical_config
        self.dimension = models[0].dimension
        self.species_names = models[0].species_names
    
    def predict(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Make predictions with uncertainty quantification.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean_prediction, std_prediction)
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x, training=False)
            predictions.append(pred)
        
        # Stack predictions along a new axis
        predictions = tf.stack(predictions, axis=0)
        
        # Compute mean and standard deviation
        mean_prediction = tf.reduce_mean(predictions, axis=0)
        std_prediction = tf.math.reduce_std(predictions, axis=0)
        
        return mean_prediction, std_prediction
    
    def get_species_concentration(self, x: tf.Tensor, species: str) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get concentration of a specific species with uncertainty.
        
        Args:
            x: Input tensor
            species: Species name
            
        Returns:
            Tuple of (mean_concentration, std_concentration)
        """
        if species not in self.species_names:
            raise ValueError(f"Unknown species: {species}")
        
        idx = self.species_names.index(species)
        
        # Get predictions from all models
        mean_pred, std_pred = self.predict(x)
        
        # Extract the specific species
        mean_concentration = mean_pred[:, idx:idx+1]
        std_concentration = std_pred[:, idx:idx+1]
        
        return mean_concentration, std_concentration
    
    def get_temperature(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get temperature with uncertainty.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean_temperature, std_temperature)
        """
        # Get predictions from all models
        mean_pred, std_pred = self.predict(x)
        
        # Temperature is the last output
        mean_temperature = mean_pred[:, 4:5]
        std_temperature = std_pred[:, 4:5]
        
        return mean_temperature, std_temperature
    
    def save(self, filepath: str, **kwargs) -> None:
        """
        Save all models in the ensemble.
        
        Args:
            filepath: Base path to save the models
            **kwargs: Additional arguments for save method
        """
        for i, model in enumerate(self.models):
            model_path = f"{filepath}_model_{i}"
            model.save(model_path, **kwargs)
    
    @classmethod
    def load(cls, filepath: str, num_models: int, model_config: ModelConfig, 
             physical_config: PhysicalConfig, dimension: int = 2, **kwargs) -> "CVDPINNEnsemble":
        """
        Load ensemble from saved models.
        
        Args:
            filepath: Base path to load the models from
            num_models: Number of models in the ensemble
            model_config: Model configuration
            physical_config: Physical parameters configuration
            dimension: Spatial dimension (2 or 3)
            **kwargs: Additional arguments for load method
            
        Returns:
            Loaded ensemble
        """
        models = []
        for i in range(num_models):
            model_path = f"{filepath}_model_{i}"
            
            # Create model with configuration
            model = CVDPINN(
                model_config=model_config,
                physical_config=physical_config,
                dimension=dimension
            )
            
            # Build the model with dummy input
            input_dim = 3 if dimension == 2 else 4  # 2D+time or 3D+time
            dummy_input = tf.zeros((1, input_dim))
            _ = model(dummy_input)
            
            # Load weights
            model.load_weights(model_path, **kwargs)
            models.append(model)
        
        return cls(models)