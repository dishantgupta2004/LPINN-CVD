"""
Training modules for CVD-PINN package.
"""

from src.training.entropy_langevin import EntropyRegularizedLoss, EntropyLangevinTrainer

__all__ = ['EntropyRegularizedLoss', 'EntropyLangevinTrainer']