"""
Configuration module for CVD-PINN.

This module contains configuration classes and default parameters
for the CVD simulation and PINN training.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import os
import yaml
import json
import numpy as np


@dataclass
class DomainConfig:
    """Configuration for the simulation domain."""
    x_min: float = 0.0
    x_max: float = 0.1
    y_min: float = 0.0
    y_max: float = 0.05
    z_min: float = 0.0  # For 3D simulations
    z_max: float = 0.0  # For 3D simulations
    t_min: float = 0.0
    t_max: float = 10.0
    dimension: int = 2  # 2D or 3D

    @property
    def domain_size(self) -> Tuple[float, float, float, float]:
        """Get the domain size for 2D simulations."""
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    @property
    def domain_size_3d(self) -> Tuple[float, float, float, float, float, float]:
        """Get the domain size for 3D simulations."""
        return (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

    @property
    def time_range(self) -> Tuple[float, float]:
        """Get the time range."""
        return (self.t_min, self.t_max)


@dataclass
class PhysicalConfig:
    """Configuration for physical parameters."""
    # Diffusion coefficients (m²/s)
    D_SiH4: float = 1.0e-5  # Silane
    D_Si: float = 5.0e-6    # Silicon
    D_H2: float = 4.0e-5    # Hydrogen
    D_SiH2: float = 1.5e-5  # Silylene
    
    # Thermal properties
    thermal_conductivity: float = 0.1  # W/(m·K)
    specific_heat: float = 700.0       # J/(kg·K)
    density: float = 1.0               # kg/m³
    
    # Reaction kinetics (Arrhenius parameters)
    # Reaction 1: SiH4 -> Si + 2H2
    A1: float = 1.0e6     # Pre-exponential factor
    E1: float = 1.5e5     # Activation energy (J/mol)
    
    # Reaction 2: SiH4 + H2 -> SiH2 + 2H2
    A2: float = 2.0e5     # Pre-exponential factor
    E2: float = 1.2e5     # Activation energy (J/mol)
    
    # Reaction 3: SiH2 + SiH4 -> Si2H6
    A3: float = 3.0e5     # Pre-exponential factor
    E3: float = 1.0e5     # Activation energy (J/mol)
    
    # Universal gas constant (J/(mol·K))
    R: float = 8.314


@dataclass
class ModelConfig:
    """Configuration for PINN model."""
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64, 64, 64, 64, 64])
    activation: str = "tanh"
    output_size: int = 5  # [SiH4, Si, H2, SiH2, T]
    weight_initializer: str = "glorot_normal"
    output_activation: Optional[str] = None
    dropout_rate: float = 0.0
    use_bias: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 1024
    learning_rate: float = 1e-3
    n_epochs: int = 10000
    validation_split: float = 0.1
    optimizer: str = "adam"
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "pde": 1.0,
        "bc": 10.0,
        "ic": 10.0
    })
    early_stopping: bool = True
    early_stopping_patience: int = 1000
    checkpoint_dir: str = "checkpoints"
    checkpoint_frequency: int = 100
    print_frequency: int = 100
    plot_frequency: int = 1000
    save_history: bool = True
    use_mixed_precision: bool = False
    use_tf_function: bool = True
    profile: bool = False
    max_grad_norm: Optional[float] = None  # For gradient clipping


@dataclass
class EntropyLangevinConfig:
    """Configuration for Entropy-Langevin training."""
    ensemble_size: int = 10
    alpha_initial: float = 0.1  # Initial entropy weight
    beta_initial: float = 10.0  # Initial inverse temperature
    alpha_final: float = 0.01   # Final entropy weight
    beta_final: float = 100.0   # Final inverse temperature
    alpha_schedule: str = "linear"  # Options: linear, exponential, cosine
    beta_schedule: str = "linear"   # Options: linear, exponential, cosine
    noise_type: str = "gaussian"    # Options: gaussian, uniform
    noise_scale_factor: float = 1.0  # Scale factor for Langevin noise
    consensus_update_frequency: int = 1  # Frequency of averaging gradients


@dataclass
class SamplingConfig:
    """Configuration for sampling strategies."""
    n_collocation_points: int = 5000
    n_boundary_points: int = 1000
    n_initial_points: int = 500
    n_validation_points: int = 1000
    use_adaptive_sampling: bool = False
    adaptive_sampling_frequency: int = 5
    adaptive_residual_threshold: float = 0.01
    adaptive_sample_ratio: float = 0.2
    refinement_radius: float = 0.01
    n_refinement_points: int = 1000
    sampling_method: str = "mixed"  # Options: uniform, residual_weighted, top_residual, mixed


@dataclass
class CVDPinnConfig:
    """Main configuration for CVD-PINN."""
    domain: DomainConfig = field(default_factory=DomainConfig)
    physical: PhysicalConfig = field(default_factory=PhysicalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    entropy_langevin: EntropyLangevinConfig = field(default_factory=EntropyLangevinConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    random_seed: int = 1234
    gpu_memory_limit: Optional[int] = None
    device: str = "cpu"  # Options: "cpu", "gpu", "auto"
    log_level: str = "INFO"
    log_dir: str = "logs"
    
    def save(self, filename: str) -> None:
        """Save configuration to a file."""
        config_dict = self.to_dict()
        
        if filename.endswith(".json"):
            with open(filename, "w") as f:
                json.dump(config_dict, f, indent=2)
        elif filename.endswith(".yaml") or filename.endswith(".yml"):
            with open(filename, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    
    @classmethod
    def load(cls, filename: str) -> "CVDPinnConfig":
        """Load configuration from a file."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Configuration file not found: {filename}")
        
        if filename.endswith(".json"):
            with open(filename, "r") as f:
                config_dict = json.load(f)
        elif filename.endswith(".yaml") or filename.endswith(".yml"):
            with open(filename, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict:
        """Convert configuration to a dictionary."""
        return {
            "domain": self.domain.__dict__,
            "physical": self.physical.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "entropy_langevin": self.entropy_langevin.__dict__,
            "sampling": self.sampling.__dict__,
            "random_seed": self.random_seed,
            "gpu_memory_limit": self.gpu_memory_limit,
            "device": self.device,
            "log_level": self.log_level,
            "log_dir": self.log_dir
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "CVDPinnConfig":
        """Create configuration from a dictionary."""
        domain_config = DomainConfig(**config_dict.get("domain", {}))
        physical_config = PhysicalConfig(**config_dict.get("physical", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        entropy_langevin_config = EntropyLangevinConfig(**config_dict.get("entropy_langevin", {}))
        sampling_config = SamplingConfig(**config_dict.get("sampling", {}))
        
        return cls(
            domain=domain_config,
            physical=physical_config,
            model=model_config,
            training=training_config,
            entropy_langevin=entropy_langevin_config,
            sampling=sampling_config,
            random_seed=config_dict.get("random_seed", 1234),
            gpu_memory_limit=config_dict.get("gpu_memory_limit", None),
            device=config_dict.get("device", "cpu"),
            log_level=config_dict.get("log_level", "INFO"),
            log_dir=config_dict.get("log_dir", "logs")
        )


# Create default configuration
DEFAULT_CONFIG = CVDPinnConfig()