"""
CVD-PINN: Physics-Informed Neural Networks for Chemical Vapor Deposition

A package for modeling Chemical Vapor Deposition using Physics-Informed Neural Networks
with Entropy-Langevin dynamics for improved training stability and uncertainty quantification.
"""

__version__ = "0.1.0"

from src.config import CVDPinnConfig, DEFAULT_CONFIG
from src.models.cvd_pinn import CVDPINN, CVDPINNEnsemble
from src.physics.pde import CVDPDE
from src.sampling.generator import CVDDataGenerator
from src.training.entropy_langevin import EntropyLangevinTrainer

__all__ = [
    'CVDPinnConfig',
    'DEFAULT_CONFIG',
    'CVDPINN',
    'CVDPINNEnsemble',
    'CVDPDE',
    'CVDDataGenerator',
    'EntropyLangevinTrainer'
]