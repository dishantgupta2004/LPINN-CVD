"""
Base PINN model implementation.

This module provides the base class for Physics-Informed Neural Networks
that will be extended for CVD modeling.
"""
from typing import List, Optional, Dict, Any, Tuple, Union, Callable
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform
from tensorflow.keras.regularizers import l2


class PINN(tf.keras.Model):
    """Base Physics-Informed Neural Network model."""
    
    def __init__(
        self,
        hidden_layers: List[int],
        output_size: int = 1,
        activation: str = "tanh",
        output_activation: Optional[str] = None,
        weight_initializer: str = "glorot_normal",
        use_bias: bool = True,
        dropout_rate: float = 0.0,
        l2_regularization: float = 0.0,
        name: str = "pinn",
        **kwargs
    ):
        """
        Initialize the PINN model.
        
        Args:
            hidden_layers: List of neurons in each hidden layer
            output_size: Number of output variables
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
            weight_initializer: Weight initialization method
            use_bias: Whether to use bias terms
            dropout_rate: Dropout rate for regularization
            l2_regularization: L2 regularization factor
            name: Name of the model
            **kwargs: Additional arguments for the base class
        """
        super(PINN, self).__init__(name=name, **kwargs)
        
        # Store configuration
        self.hidden_layers_sizes = hidden_layers
        self.output_size = output_size
        self.activation_name = activation
        self.output_activation_name = output_activation
        self.weight_initializer_name = weight_initializer
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        
        # Resolve activation functions
        self.hidden_activation = self._get_activation(activation)
        self.output_activation = self._get_activation(output_activation)
        
        # Resolve weight initializer
        self.weight_initializer = self._get_initializer(weight_initializer)
        
        # L2 regularization
        self.regularizer = l2(l2_regularization) if l2_regularization > 0 else None
        
        # Build the network
        self.layers_list = []
        
        # Input layer is handled implicitly
        
        # Hidden layers
        for units in hidden_layers:
            self.layers_list.append(
                Dense(
                    units,
                    activation=self.hidden_activation,
                    kernel_initializer=self.weight_initializer,
                    use_bias=use_bias,
                    kernel_regularizer=self.regularizer,
                    bias_regularizer=self.regularizer
                )
            )
            
            if dropout_rate > 0:
                self.layers_list.append(Dropout(dropout_rate))
        
        # Output layer
        self.layers_list.append(
            Dense(
                output_size,
                activation=self.output_activation,
                kernel_initializer=self.weight_initializer,
                use_bias=use_bias,
                kernel_regularizer=self.regularizer,
                bias_regularizer=self.regularizer
            )
        )
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through the network.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        x = inputs
        for layer in self.layers_list:
            if isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x
    
    def _get_activation(self, activation_name: Optional[str]) -> Optional[Union[str, Callable]]:
        """
        Get activation function from name.
        
        Args:
            activation_name: Name of the activation function
            
        Returns:
            Activation function or None
        """
        if activation_name is None:
            return None
        
        # Handle custom activations
        if activation_name == "swish":
            return tf.nn.swish
        elif activation_name == "adaptive_tanh":
            # Adaptive tanh activation with learnable parameters
            # Implement if needed
            return tf.nn.tanh
        
        return activation_name
    
    def _get_initializer(self, initializer_name: str) -> tf.keras.initializers.Initializer:
        """
        Get initializer from name.
        
        Args:
            initializer_name: Name of the initializer
            
        Returns:
            Initializer object
        """
        if initializer_name == "glorot_normal":
            return GlorotNormal()
        elif initializer_name == "glorot_uniform":
            return GlorotUniform()
        elif initializer_name == "he_normal":
            return tf.keras.initializers.HeNormal()
        elif initializer_name == "he_uniform":
            return tf.keras.initializers.HeUniform()
        else:
            return GlorotNormal()
    
    @tf.function
    def get_gradients(self, x: tf.Tensor, y: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
        """
        Compute gradients of output with respect to input.
        
        Args:
            x: Input tensor, shape (batch_size, input_dim)
            y: Output tensor, shape (batch_size, output_dim)
                If None, a forward pass is performed
                
        Returns:
            Dictionary containing first and second derivatives
        """
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(x)
                
                # If y is not provided, perform a forward pass
                if y is None:
                    y = self(x, training=False)
                
            # First derivatives
            dy_dx = tape1.batch_jacobian(y, x)
            
            # Extract components for easier access
            # Assuming input is (x, y, t) or (x, y, z, t)
            input_dim = x.shape[-1]
            
            # For 2D + time
            if input_dim == 3:
                y_x = dy_dx[..., 0]  # derivatives w.r.t. x
                y_y = dy_dx[..., 1]  # derivatives w.r.t. y
                y_t = dy_dx[..., 2]  # derivatives w.r.t. t
            # For 3D + time
            elif input_dim == 4:
                y_x = dy_dx[..., 0]  # derivatives w.r.t. x
                y_y = dy_dx[..., 1]  # derivatives w.r.t. y
                y_z = dy_dx[..., 2]  # derivatives w.r.t. z
                y_t = dy_dx[..., 3]  # derivatives w.r.t. t
            
        # Second derivatives
        derivatives = {"dy_dx": dy_dx}
        
        if input_dim == 3:  # 2D + time
            dy_xx = tape2.batch_jacobian(y_x, x)[..., 0]  # d²y/dx²
            dy_yy = tape2.batch_jacobian(y_y, x)[..., 1]  # d²y/dy²
            
            derivatives.update({
                "y_x": y_x,
                "y_y": y_y,
                "y_t": y_t,
                "y_xx": dy_xx,
                "y_yy": dy_yy
            })
        elif input_dim == 4:  # 3D + time
            dy_xx = tape2.batch_jacobian(y_x, x)[..., 0]  # d²y/dx²
            dy_yy = tape2.batch_jacobian(y_y, x)[..., 1]  # d²y/dy²
            dy_zz = tape2.batch_jacobian(y_z, x)[..., 2]  # d²y/dz²
            
            derivatives.update({
                "y_x": y_x,
                "y_y": y_y,
                "y_z": y_z,
                "y_t": y_t,
                "y_xx": dy_xx,
                "y_yy": dy_yy,
                "y_zz": dy_zz
            })
        
        del tape1, tape2
        
        return derivatives
    
    def summary(self, *args, **kwargs) -> None:
        """Print model summary."""
        # Create dummy input for model summary
        input_shape = (None, 3)  # Default to 2D + time
        self.build(input_shape)
        super(PINN, self).summary(*args, **kwargs)
    
    def save(self, filepath: str, **kwargs) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
            **kwargs: Additional arguments for save method
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save the model weights
            self.save_weights(filepath, **kwargs)
            
            # Save model configuration as JSON
            config = {
                "hidden_layers": self.hidden_layers_sizes,
                "output_size": self.output_size,
                "activation": self.activation_name,
                "output_activation": self.output_activation_name,
                "weight_initializer": self.weight_initializer_name,
                "use_bias": self.use_bias,
                "dropout_rate": self.dropout_rate
            }
            
            with open(f"{filepath}_config.json", "w") as f:
                f.write(tf.keras.backend.json.dumps(config, indent=2))
                
        except Exception as e:
            raise IOError(f"Failed to save model: {str(e)}")
    
    @classmethod
    def load(cls, filepath: str, **kwargs) -> "PINN":
        """
        Load a model from a file.
        
        Args:
            filepath: Path to load the model from
            **kwargs: Additional arguments for load method
            
        Returns:
            Loaded model
        """
        try:
            # Load model configuration
            with open(f"{filepath}_config.json", "r") as f:
                config = tf.keras.backend.json.loads(f.read())
            
            # Create model with loaded configuration
            model = cls(
                hidden_layers=config["hidden_layers"],
                output_size=config["output_size"],
                activation=config["activation"],
                output_activation=config["output_activation"],
                weight_initializer=config["weight_initializer"],
                use_bias=config["use_bias"],
                dropout_rate=config["dropout_rate"]
            )
            
            # Build the model with dummy input
            dummy_input = tf.zeros((1, 3))  # Default to 2D + time
            _ = model(dummy_input)
            
            # Load the weights
            model.load_weights(filepath, **kwargs)
            
            return model
            
        except Exception as e:
            raise IOError(f"Failed to load model: {str(e)}")