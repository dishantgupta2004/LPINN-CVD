"""
Entropy-Langevin training for PINNs.

This module implements the Entropy-Langevin training algorithm for
Physics-Informed Neural Networks, as described in the paper.
"""
from typing import List, Dict, Tuple, Optional, Union, Callable
import os
import time
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.config import (
    CVDPinnConfig, EntropyLangevinConfig, TrainingConfig,
    DomainConfig, PhysicalConfig, ModelConfig, SamplingConfig
)
from src.models.cvd_pinn import CVDPINN, CVDPINNEnsemble
from src.physics.pde import CVDPDE, AdaptivePDEWeights
from src.sampling.generator import CVDDataGenerator
from src.utils.logger import get_logger


class EntropyRegularizedLoss:
    """
    Entropy-regularized loss for ensemble training.
    """
    
    def __init__(self, alpha: float = 0.1, beta: float = 10.0, name: str = "entropy_loss"):
        """
        Initialize entropy-regularized loss.
        
        Args:
            alpha: Weight of entropy term
            beta: Inverse temperature parameter
            name: Name of the loss function
        """
        self.alpha = tf.Variable(alpha, trainable=False, dtype=tf.float32, name=f"{name}_alpha")
        self.beta = tf.Variable(beta, trainable=False, dtype=tf.float32, name=f"{name}_beta")
        self.name = name
    
    def __call__(self, losses: tf.Tensor) -> tf.Tensor:
        """
        Compute entropy-regularized loss.
        
        Args:
            losses: Loss values for each model in the ensemble, shape (ensemble_size,)
            
        Returns:
            Entropy-regularized loss
        """
        # Mean loss
        mean_loss = tf.reduce_mean(losses)
        
        # Entropy term: -α * log(mean(exp(-β*L)))
        entropy_term = -self.alpha * tf.math.log(
            tf.reduce_mean(tf.exp(-self.beta * losses))
        ) / self.beta
        
        # Total loss
        total_loss = mean_loss + entropy_term
        
        return total_loss
    
    def compute_modified_gradients(self, 
                                  losses: tf.Tensor, 
                                  gradients: List[List[tf.Tensor]]) -> List[List[tf.Tensor]]:
        """
        Compute modified gradients with entropy feedback.
        
        Args:
            losses: Loss values for each model, shape (ensemble_size,)
            gradients: List of gradient lists for each model
            
        Returns:
            Modified gradients
        """
        ensemble_size = len(gradients)
        
        # Compute softmax weights
        weights = tf.nn.softmax(-self.beta * losses)
        
        # Compute weighted average gradients
        avg_gradients = []
        for i in range(len(gradients[0])):  # For each variable
            # Stack gradients from all models for this variable
            var_grads = tf.stack([gradients[j][i] for j in range(ensemble_size)], axis=0)
            
            # Compute weighted average
            avg_grad = tf.reduce_sum(tf.expand_dims(weights, -1) * var_grads, axis=0)
            avg_gradients.append(avg_grad)
        
        # Compute modified gradients for each model
        modified_gradients = []
        for i in range(ensemble_size):
            model_modified_gradients = []
            for j in range(len(gradients[0])):  # For each variable
                # Modified gradient: g - α*β*(g - E[g])
                modified_grad = gradients[i][j] - self.alpha * self.beta * (
                    gradients[i][j] - avg_gradients[j]
                )
                model_modified_gradients.append(modified_grad)
            modified_gradients.append(model_modified_gradients)
        
        return modified_gradients
    
    def update_parameters(self, progress: float, schedule: str = "linear") -> None:
        """
        Update alpha and beta parameters based on scheduling.
        
        Args:
            progress: Training progress from 0 to 1
            schedule: Scheduling type (linear, exponential, cosine)
        """
        if schedule == "linear":
            # Linear scheduling
            alpha_new = 0.1 * (1.0 - 0.9 * progress)  # 0.1 -> 0.01
            beta_new = 10.0 * (1.0 + 9.0 * progress)  # 10.0 -> 100.0
        
        elif schedule == "exponential":
            # Exponential scheduling
            alpha_new = 0.1 * np.exp(-2.3 * progress)  # 0.1 -> 0.01
            beta_new = 10.0 * np.exp(2.3 * progress)   # 10.0 -> 100.0
        
        elif schedule == "cosine":
            # Cosine scheduling
            alpha_new = 0.01 + 0.09 * (1.0 + np.cos(np.pi * progress)) / 2.0  # 0.1 -> 0.01
            beta_new = 10.0 + 90.0 * (1.0 - np.cos(np.pi * progress)) / 2.0   # 10.0 -> 100.0
        
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        # Update parameters
        self.alpha.assign(alpha_new)
        self.beta.assign(beta_new)


class EntropyLangevinTrainer:
    """
    Trainer implementing the Entropy-Langevin algorithm for ensemble of PINNs.
    """
    
    def __init__(self, config: CVDPinnConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__, log_level=config.log_level, log_dir=config.log_dir)
        
        # Set random seed
        tf.random.set_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Create data generator
        self.data_generator = CVDDataGenerator(config.domain, config.sampling)
        
        # Create PDE calculator
        self.pde_calculator = CVDPDE(config.physical)
        
        # Create models
        self.models = []
        for i in range(config.entropy_langevin.ensemble_size):
            model = CVDPINN(
                model_config=config.model,
                physical_config=config.physical,
                dimension=config.domain.dimension,
                name=f"cvd_pinn_{i}"
            )
            self.models.append(model)
        
        # Create optimizers
        self.optimizers = []
        for _ in range(config.entropy_langevin.ensemble_size):
            optimizer = self._create_optimizer(
                optimizer_name=config.training.optimizer,
                learning_rate=config.training.learning_rate
            )
            self.optimizers.append(optimizer)
        
        # Create entropy-regularized loss
        self.entropy_loss = EntropyRegularizedLoss(
            alpha=config.entropy_langevin.alpha_initial,
            beta=config.entropy_langevin.beta_initial
        )
        
        # Initialize loss history
        self.loss_history = {
            'total': [],
            'pde': [[] for _ in range(config.entropy_langevin.ensemble_size)],
            'bc': [[] for _ in range(config.entropy_langevin.ensemble_size)],
            'ic': [[] for _ in range(config.entropy_langevin.ensemble_size)],
            'entropy': []
        }
        
        # Initialize parameter statistics
        self.parameter_statistics = []
        
        # Create checkpoint directory
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        
        self.logger.info(f"Initialized Entropy-Langevin trainer with {len(self.models)} models")
    
    def _create_optimizer(self, optimizer_name: str, learning_rate: float) -> tf.keras.optimizers.Optimizer:
        """
        Create optimizer.
        
        Args:
            optimizer_name: Name of the optimizer
            learning_rate: Learning rate
            
        Returns:
            Optimizer object
        """
        if optimizer_name.lower() == "adam":
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name.lower() == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            self.logger.warning(f"Unknown optimizer: {optimizer_name}, using Adam")
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    def compute_pde_loss(self, model_idx: int, x_collocation: tf.Tensor) -> tf.Tensor:
        """
        Compute PDE residual loss for a specific model.
        
        Args:
            model_idx: Index of the model
            x_collocation: Collocation points tensor
            
        Returns:
            PDE loss tensor
        """
        # Get model
        model = self.models[model_idx]
        
        # Forward pass
        with tf.GradientTape() as tape:
            tape.watch(x_collocation)
            
            # Get model predictions
            y_pred = model(x_collocation, training=True)
            
            # Get derivatives
            derivatives = model.get_gradients(x_collocation, y_pred)
            
            # Compute PDE residuals
            residuals = self.pde_calculator.compute_residuals(
                x_collocation, y_pred, derivatives
            )
        
        # Compute mean squared residual for each equation
        losses = [tf.reduce_mean(tf.square(res)) for res in residuals]
        
        # Combine all residuals with equal weights
        # Could use adaptive weighting instead
        pde_loss = sum(losses)
        
        return pde_loss
    
    def compute_bc_loss(self, model_idx: int, boundary_points: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Compute boundary condition loss for a specific model.
        
        Args:
            model_idx: Index of the model
            boundary_points: Dictionary with boundary points
            
        Returns:
            Boundary condition loss tensor
        """
        # Get model
        model = self.models[model_idx]
        
        # Initialize boundary loss
        bc_loss = tf.constant(0.0, dtype=tf.float32)
        
        # Inlet boundary conditions (y = y_min)
        if 'inlet' in boundary_points:
            inlet_points = boundary_points['inlet']
            inlet_pred = model(inlet_points, training=True)
            
            # At inlet: SiH4 = 0.2, H2 = 0.1, T = 350
            inlet_SiH4_target = 0.2 * tf.ones_like(inlet_pred[:, 0:1])
            inlet_H2_target = 0.1 * tf.ones_like(inlet_pred[:, 2:3])
            inlet_T_target = 350.0 * tf.ones_like(inlet_pred[:, 4:5])
            
            inlet_loss = tf.reduce_mean(tf.square(inlet_pred[:, 0:1] - inlet_SiH4_target)) + \
                         tf.reduce_mean(tf.square(inlet_pred[:, 2:3] - inlet_H2_target)) + \
                         tf.reduce_mean(tf.square(inlet_pred[:, 4:5] - inlet_T_target))
            
            bc_loss += inlet_loss
        
        # Substrate boundary conditions (y = y_max)
        if 'substrate' in boundary_points:
            substrate_points = boundary_points['substrate']
            substrate_pred = model(substrate_points, training=True)
            
            # At substrate: T = 700
            substrate_T_target = 700.0 * tf.ones_like(substrate_pred[:, 4:5])
            
            substrate_loss = tf.reduce_mean(tf.square(substrate_pred[:, 4:5] - substrate_T_target))
            
            bc_loss += substrate_loss
        
        # Wall boundary conditions (no flux)
        # We can implement this using derivative conditions if needed
        
        return bc_loss
    
    def compute_ic_loss(self, model_idx: int, initial_points: tf.Tensor) -> tf.Tensor:
        """
        Compute initial condition loss for a specific model.
        
        Args:
            model_idx: Index of the model
            initial_points: Initial condition points tensor
            
        Returns:
            Initial condition loss tensor
        """
        # Get model
        model = self.models[model_idx]
        
        # Get model predictions at initial points
        initial_pred = model(initial_points, training=True)
        
        # Initial conditions:
        # SiH4 = 0.1 (uniform low concentration)
        # Si = 0.0 (no silicon initially)
        # H2 = 0.0 (no hydrogen initially)
        # SiH2 = 0.0 (no silylene initially)
        # T = 300.0 (room temperature)
        
        initial_targets = tf.concat([
            0.1 * tf.ones_like(initial_pred[:, 0:1]),  # SiH4
            0.0 * tf.ones_like(initial_pred[:, 1:2]),  # Si
            0.0 * tf.ones_like(initial_pred[:, 2:3]),  # H2
            0.0 * tf.ones_like(initial_pred[:, 3:4]),  # SiH2
            300.0 * tf.ones_like(initial_pred[:, 4:5])  # T
        ], axis=1)
        
        # Compute mean squared error
        ic_loss = tf.reduce_mean(tf.square(initial_pred - initial_targets))
        
        return ic_loss
    
    def add_langevin_noise(self, gradients: List[tf.Tensor], beta: float) -> List[tf.Tensor]:
        """
        Add Langevin noise to gradients.
        
        Args:
            gradients: List of gradient tensors
            beta: Inverse temperature parameter
            
        Returns:
            List of gradient tensors with added noise
        """
        # Get noise scale factor
        noise_scale_factor = self.config.entropy_langevin.noise_scale_factor
        
        # Add Langevin noise to each gradient
        noisy_gradients = []
        for grad in gradients:
            # Compute noise scale
            noise_scale = tf.sqrt(2.0 * self.config.training.learning_rate / beta) * noise_scale_factor
            
            # Generate noise
            noise = noise_scale * tf.random.normal(shape=grad.shape, dtype=grad.dtype)
            
            # Add noise to gradient
            noisy_grad = grad + noise
            
            noisy_gradients.append(noisy_grad)
        
        return noisy_gradients
    
    def train_step(self, epoch: int, x_collocation: tf.Tensor, 
                  boundary_points: Dict[str, tf.Tensor], initial_points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Perform one training step for all models in the ensemble.
        
        Args:
            epoch: Current epoch
            x_collocation: Collocation points tensor
            boundary_points: Dictionary with boundary points
            initial_points: Initial condition points tensor
            
        Returns:
            Tuple of (total_losses, pde_losses, bc_losses, ic_losses)
        """
        # Update entropy-Langevin parameters
        progress = epoch / self.config.training.n_epochs
        self.entropy_loss.update_parameters(
            progress, 
            schedule=self.config.entropy_langevin.alpha_schedule
        )
        
        # Initialize lists to store losses and gradients
        total_losses = []
        pde_losses = []
        bc_losses = []
        ic_losses = []
        all_gradients = []
        
        # Step 1: Compute losses and gradients for all models
        for i in range(len(self.models)):
            with tf.GradientTape() as tape:
                # Compute losses
                pde_loss = self.compute_pde_loss(i, x_collocation)
                bc_loss = self.compute_bc_loss(i, boundary_points)
                ic_loss = self.compute_ic_loss(i, initial_points)
                
                # Weight the losses
                # You may need to tune these weights for your specific problem
                weighted_pde_loss = self.config.training.loss_weights['pde'] * pde_loss
                weighted_bc_loss = self.config.training.loss_weights['bc'] * bc_loss
                weighted_ic_loss = self.config.training.loss_weights['ic'] * ic_loss
                
                # Compute total loss
                total_loss = weighted_pde_loss + weighted_bc_loss + weighted_ic_loss
            
            # Store losses
            total_losses.append(total_loss)
            pde_losses.append(pde_loss)
            bc_losses.append(bc_loss)
            ic_losses.append(ic_loss)
            
            # Compute gradients
            gradients = tape.gradient(total_loss, self.models[i].trainable_variables)
            
            # Handle NaN or Inf gradients
            gradients = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in gradients]
            
            # Clip gradients if specified
            if self.config.training.max_grad_norm is not None:
                gradients, _ = tf.clip_by_global_norm(gradients, self.config.training.max_grad_norm)
            
            all_gradients.append(gradients)
        
        # Step 2: Apply entropy-regularized loss to modify gradients
        total_losses_tensor = tf.stack(total_losses)
        
        # Compute modified gradients with entropy feedback
        modified_gradients = self.entropy_loss.compute_modified_gradients(
            total_losses_tensor, all_gradients
        )
        
        # Step 3: Apply modified gradients with Langevin dynamics
        for i in range(len(self.models)):
            # Add Langevin noise
            noisy_gradients = self.add_langevin_noise(
                modified_gradients[i], self.entropy_loss.beta
            )
            
            # Apply gradients
            self.optimizers[i].apply_gradients(
                zip(noisy_gradients, self.models[i].trainable_variables)
            )
        
        # Convert lists to tensors for return
        total_losses_tensor = tf.stack(total_losses)
        pde_losses_tensor = tf.stack(pde_losses)
        bc_losses_tensor = tf.stack(bc_losses)
        ic_losses_tensor = tf.stack(ic_losses)
        
        return total_losses_tensor, pde_losses_tensor, bc_losses_tensor, ic_losses_tensor
    
    def extract_parameters(self, model_idx: int) -> np.ndarray:
        """
        Extract flattened parameters from a model.
        
        Args:
            model_idx: Index of the model
            
        Returns:
            Flattened parameters
        """
        model = self.models[model_idx]
        params = []
        
        # Extract weights and biases from each layer
        for layer in model.layers_list:
            weights = layer.get_weights()
            for w in weights:
                params.append(w.flatten())
        
        # Concatenate all parameters
        return np.concatenate(params)
    
    def record_parameter_statistics(self) -> None:
        """Record parameter statistics for all models."""
        # Extract parameters from all models
        params = []
        for i in range(len(self.models)):
            params.append(self.extract_parameters(i))
        
        # Compute statistics
        params = np.array(params)
        stats = {
            'mean': np.mean(params, axis=0),
            'std': np.std(params, axis=0),
            'min': np.min(params, axis=0),
            'max': np.max(params, axis=0),
            'median': np.median(params, axis=0)
        }
        
        # Store statistics
        self.parameter_statistics.append(stats)
    
    def train(self, n_epochs: Optional[int] = None, print_frequency: Optional[int] = None) -> Dict:
        """
        Train the ensemble of PINN models using Entropy-Langevin algorithm.
        
        Args:
            n_epochs: Number of training epochs (overrides config)
            print_frequency: Frequency of printing results (overrides config)
            
        Returns:
            Loss history
        """
        # Use config values if not specified
        if n_epochs is None:
            n_epochs = self.config.training.n_epochs
        
        if print_frequency is None:
            print_frequency = self.config.training.print_frequency
        
        self.logger.info(f"Starting Entropy-Langevin training for {n_epochs} epochs...")
        start_time = time.time()
        
        # Generate training data once
        collocation_points = self.data_generator.generate_collocation_points()
        x_collocation = tf.convert_to_tensor(collocation_points, dtype=tf.float32)
        
        boundary_points_np = self.data_generator.generate_boundary_points()
        boundary_points = {}
        for key, points in boundary_points_np.items():
            boundary_points[key] = tf.convert_to_tensor(points, dtype=tf.float32)
        
        initial_points = self.data_generator.generate_initial_points()
        initial_points = tf.convert_to_tensor(initial_points, dtype=tf.float32)
        
        # Record initial parameter statistics
        self.record_parameter_statistics()
        
        # Training loop
        for epoch in range(n_epochs):
            # Perform one training step for all models
            total_losses, pde_losses, bc_losses, ic_losses = self.train_step(
                epoch, x_collocation, boundary_points, initial_points
            )
            
            # Compute average losses across ensemble
            avg_total_loss = tf.reduce_mean(total_losses)
            avg_pde_loss = tf.reduce_mean(pde_losses)
            avg_bc_loss = tf.reduce_mean(bc_losses)
            avg_ic_loss = tf.reduce_mean(ic_losses)
            
            # Update loss history
            self.loss_history['total'].append(avg_total_loss.numpy())
            for i in range(len(self.models)):
                self.loss_history['pde'][i].append(pde_losses[i].numpy())
                self.loss_history['bc'][i].append(bc_losses[i].numpy())
                self.loss_history['ic'][i].append(ic_losses[i].numpy())
            
            # Record parameter statistics periodically
            if (epoch + 1) % self.config.training.checkpoint_frequency == 0:
                self.record_parameter_statistics()
            
            # Save model periodically
            if (epoch + 1) % self.config.training.checkpoint_frequency == 0:
                model_path = os.path.join(
                    self.config.training.checkpoint_dir,
                    f"cvd_pinn_ensemble_epoch_{epoch+1}"
                )
                self.save_models(model_path)
            
            # Print progress
            if (epoch + 1) % print_frequency == 0:
                elapsed = time.time() - start_time
                self.logger.info(
                    f"Epoch {epoch+1}/{n_epochs}, "
                    f"Avg Loss: {avg_total_loss.numpy():.6e}, "
                    f"Avg PDE: {avg_pde_loss.numpy():.6e}, "
                    f"Avg BC: {avg_bc_loss.numpy():.6e}, "
                    f"Avg IC: {avg_ic_loss.numpy():.6e}, "
                    f"Alpha: {self.entropy_loss.alpha.numpy():.4f}, "
                    f"Beta: {self.entropy_loss.beta.numpy():.2f}, "
                    f"Time: {elapsed:.2f}s"
                )
                
                # Plot loss history if specified
                if (epoch + 1) % self.config.training.plot_frequency == 0:
                    self.plot_loss_history()
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds.")
        
        # Save final model
        final_model_path = os.path.join(
            self.config.training.checkpoint_dir,
            "cvd_pinn_ensemble_final"
        )
        self.save_models(final_model_path)
        
        # Save loss history
        if self.config.training.save_history:
            history_path = os.path.join(
                self.config.training.checkpoint_dir,
                "loss_history.pkl"
            )
            with open(history_path, 'wb') as f:
                pickle.dump(self.loss_history, f)
            
            stats_path = os.path.join(
                self.config.training.checkpoint_dir,
                "parameter_statistics.pkl"
            )
            with open(stats_path, 'wb') as f:
                pickle.dump(self.parameter_statistics, f)
        
        return self.loss_history
    
    def save_models(self, filepath: str) -> None:
        """
        Save all models in the ensemble.
        
        Args:
            filepath: Base path to save the models
        """
        ensemble = CVDPINNEnsemble(self.models)
        ensemble.save(filepath)
        
        # Save entropy-Langevin parameters
        config_path = f"{filepath}_entropy_params.npz"
        np.savez(
            config_path,
            alpha=self.entropy_loss.alpha.numpy(),
            beta=self.entropy_loss.beta.numpy()
        )
        
        self.logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str) -> None:
        """
        Load models from saved files.
        
        Args:
            filepath: Base path to load the models from
        """
        ensemble = CVDPINNEnsemble.load(
            filepath,
            num_models=len(self.models),
            model_config=self.config.model,
            physical_config=self.config.physical,
            dimension=self.config.domain.dimension
        )
        
        self.models = ensemble.models
        
        # Load entropy-Langevin parameters if available
        config_path = f"{filepath}_entropy_params.npz"
        if os.path.exists(config_path):
            params = np.load(config_path)
            self.entropy_loss.alpha.assign(params['alpha'])
            self.entropy_loss.beta.assign(params['beta'])
        
        self.logger.info(f"Models loaded from {filepath}")
    
    def predict(self, x: Union[np.ndarray, tf.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty quantification.
        
        Args:
            x: Input tensor or array
            
        Returns:
            Tuple of (mean_prediction, std_prediction)
        """
        # Convert to tensor if numpy array
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        
        # Create ensemble
        ensemble = CVDPINNEnsemble(self.models)
        
        # Make predictions
        mean_pred, std_pred = ensemble.predict(x)
        
        return mean_pred.numpy(), std_pred.numpy()
    
    def plot_loss_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot the loss history.
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        epochs = range(1, len(self.loss_history['total']) + 1)
        
        # Plot total loss
        plt.semilogy(epochs, self.loss_history['total'], 'k-', linewidth=2, label='Avg Total Loss')
        
        # Plot individual model PDE losses with transparency
        for i in range(len(self.models)):
            plt.semilogy(epochs, self.loss_history['pde'][i], 'r-', linewidth=0.5, alpha=0.3)
        
        # Plot average PDE loss
        avg_pde_loss = np.mean([self.loss_history['pde'][i] for i in range(len(self.models))], axis=0)
        plt.semilogy(epochs, avg_pde_loss, 'r-', linewidth=1.5, label='Avg PDE Loss')
        
        # Plot average BC loss
        avg_bc_loss = np.mean([self.loss_history['bc'][i] for i in range(len(self.models))], axis=0)
        plt.semilogy(epochs, avg_bc_loss, 'b-', linewidth=1.5, label='Avg BC Loss')
        
        # Plot average IC loss
        avg_ic_loss = np.mean([self.loss_history['ic'][i] for i in range(len(self.models))], axis=0)
        plt.semilogy(epochs, avg_ic_loss, 'g-', linewidth=1.5, label='Avg IC Loss')
        
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Loss History (Entropy-Langevin PINN)', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(
                self.config.training.checkpoint_dir,
                "loss_history.png"
            )
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_predictions(self, t_idx: int = 5, output_idx: int = 0, nx: int = 50, ny: int = 50, nt: int = 10,
                             save_path: Optional[str] = None) -> None:
        """
        Visualize predictions with uncertainty quantification.
        
        Args:
            t_idx: Time index to visualize
            output_idx: Output index to visualize (0: SiH4, 1: Si, 2: H2, 3: SiH2, 4: T)
            nx, ny, nt: Number of points in each dimension for visualization grid
            save_path: Path to save the plot
        """
        # Species names and titles
        species_names = ["SiH4", "Si", "H2", "SiH2", "Temperature"]
        
        # Generate uniform grid for visualization
        grid_points, grid_shape = self.data_generator.generate_uniform_grid(nx, ny, nt)
        
        # Make predictions
        mean_pred, std_pred = self.predict(grid_points)
        
        # Reshape predictions
        mean_pred = mean_pred.reshape(grid_shape[0], grid_shape[1], grid_shape[2], 5)
        std_pred = std_pred.reshape(grid_shape[0], grid_shape[1], grid_shape[2], 5)
        
        # Extract domain grid
        x = np.linspace(self.config.domain.x_min, self.config.domain.x_max, nx)
        y = np.linspace(self.config.domain.y_min, self.config.domain.y_max, ny)
        t = np.linspace(self.config.domain.t_min, self.config.domain.t_max, nt)
        
        # Get actual time value
        time_val = t[t_idx]
        
        # Extract mean and std predictions for the specified time and output
        mean_slice = mean_pred[:, :, t_idx, output_idx]
        std_slice = std_pred[:, :, t_idx, output_idx]
        
        # Create meshgrid for plotting
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Create a 3x1 grid of plots
        fig = plt.figure(figsize=(18, 6))
        gs = GridSpec(1, 3, figure=fig)
        
        # Plot mean prediction
        ax1 = fig.add_subplot(gs[0, 0])
        cf1 = ax1.contourf(X, Y, mean_slice, 50, cmap='viridis')
        plt.colorbar(cf1, ax=ax1, label=f"{species_names[output_idx]}")
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_title(f"Mean {species_names[output_idx]} at t = {time_val:.2f}s")
        
        # Plot standard deviation
        ax2 = fig.add_subplot(gs[0, 1])
        cf2 = ax2.contourf(X, Y, std_slice, 50, cmap='plasma')
        plt.colorbar(cf2, ax=ax2, label=f"Std Dev of {species_names[output_idx]}")
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        ax2.set_title(f"Uncertainty in {species_names[output_idx]} at t = {time_val:.2f}s")
        
        # Plot coefficient of variation (std/mean)
        ax3 = fig.add_subplot(gs[0, 2])
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10
        cv = std_slice / (np.abs(mean_slice) + epsilon)
        # Clip extremely high values for better visualization
        cv = np.clip(cv, 0, 0.5)
        cf3 = ax3.contourf(X, Y, cv, 50, cmap='hot')
        plt.colorbar(cf3, ax=ax3, label='Coefficient of Variation')
        ax3.set_xlabel('x (m)')
        ax3.set_ylabel('y (m)')
        ax3.set_title(f"Relative Uncertainty at t = {time_val:.2f}s")
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(
                self.config.training.checkpoint_dir,
                f"prediction_{species_names[output_idx]}_t{t_idx}.png"
            )
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()