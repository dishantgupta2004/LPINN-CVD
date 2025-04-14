# CVD-PINN: Physics-Informed Neural Networks for Chemical Vapor Deposition

A package for modeling Chemical Vapor Deposition (CVD) processes using Physics-Informed Neural Networks with Entropy-Langevin dynamics for improved training stability and uncertainty quantification.

## Overview

CVD-PINN combines the power of Physics-Informed Neural Networks (PINNs) with an enhanced training approach based on Langevin dynamics to efficiently solve complex reaction-diffusion equations governing CVD processes. The package includes:

- A modular PINN framework designed specifically for CVD modeling
- Entropy-Langevin training algorithm for improved convergence and stability
- Ensemble methods for uncertainty quantification
- Adaptive sampling techniques to focus computational resources on regions of interest
- A Streamlit-based interactive visualization application

## Installation

Install the package using pip:

```bash
pip install -e .
```

Or directly from GitHub:

```bash
pip install git+https://github.com/dishantgupta2004/LPINN-CVD.git
```

## Project Structure

```
cvd-pinn-project/
│
├── src/                    # Main package directory
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration and hyperparameters
│   ├── constants.py             # Physical constants and default values
│   ├── models/                  # Neural network models
│   │   ├── __init__.py
│   │   ├── base.py              # Base PINN model
│   │   ├── cvd_pinn.py          # CVD-specific PINN implementation
│   │   └── ensemble.py          # Ensemble model implementation
│   ├── physics/                 # Physics modules
│   │   ├── __init__.py
│   │   ├── cvd.py               # CVD physics implementation
│   │   ├── pde.py               # PDE residual calculations
│   │   └── reactions.py         # Chemical reaction models
│   ├── training/                # Training functionality
│   │   ├── __init__.py
│   │   ├── trainer.py           # Base trainer class 
│   │   ├── traditional.py       # Traditional PINN trainer
│   │   ├── entropy_langevin.py  # Entropy-Langevin trainer
│   │   └── callbacks.py         # Training callbacks
│   ├── sampling/                # Sampling strategies
│   │   ├── __init__.py
│   │   ├── generator.py         # Data generation
│   │   ├── adaptive.py          # Adaptive sampling
│   │   └── residual.py          # Residual-based refinement
│   ├── analysis/                # Analysis tools
│   │   ├── __init__.py
│   │   ├── parameter_space.py   # Parameter space analysis
│   │   ├── fokker_planck.py     # Fokker-Planck analysis
│   │   └── visualizer.py        # Visualization utilities
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       ├── io.py                # Input/output utilities
│       ├── metrics.py           # Evaluation metrics
│       └── logger.py            # Logging functionality
│
├── examples/                    # Example scripts
│   ├── basic_cvd.py             # Basic CVD example
│   ├── adaptive_sampling.py     # Adaptive sampling example
│   └── uncertainty_analysis.py  # Uncertainty analysis example
│
├── scripts/                     # Utility scripts
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── visualize.py             # Visualization script
│
├── app/                         # Streamlit application
│   ├── app.py                   # Main Streamlit app
│   ├── components/              # UI components
│   │   ├── __init__.py
│   │   ├── sidebar.py           # Sidebar components
│   │   ├── plots.py             # Plotting components
│   │   └── simulation.py       # Simulation components
│       |__ plots.py
│   └── pages/                   # App pages
│       ├── __init__.py
│       ├── home.py              # Home page
│       ├── model_comparison.py  # Model comparison page
│       ├── simulation.py        # Simulation page
│       ├── parameter_analysis.py # Parameter analysis page
│       └── adaptive_sampling.py # Adaptive sampling page
│
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_models.py           # Tests for models
│   ├── test_physics.py          # Tests for physics
│   ├── test_training.py         # Tests for training
│   └── test_sampling.py         # Tests for sampling
│
├── setup.py                     # Package setup file
├── requirements.txt             # Package dependencies
├── README.md                    # Project documentation
└── LICENSE                      # License file
```

## Usage

### Basic Example

```python
import tensorflow as tf
import numpy as np
from cvd_pinn import CVDPinnConfig, CVDPINN, CVDDataGenerator, EntropyLangevinTrainer

# Create configuration
config = CVDPinnConfig()

# Customize configuration if needed
config.model.hidden_layers = [64, 64, 64, 64]
config.training.learning_rate = 1e-3
config.training.n_epochs = 5000

# Create trainer
trainer = EntropyLangevinTrainer(config)

# Train the model
trainer.train()

# Make predictions with uncertainty quantification
x_test = np.array([[0.05, 0.025, 5.0]])  # Example point (x, y, t)
mean_pred, std_pred = trainer.predict(x_test)

print(f"Mean prediction: {mean_pred}")
print(f"Std deviation: {std_pred}")
```

### Running the Streamlit App

To run the interactive visualization app:

```bash
cd app
streamlit run app.py
```

## Key Features

1. **Physics-Informed Training**: Incorporates physical laws directly into the neural network training process, ensuring physically consistent predictions even with limited data.

2. **Entropy-Langevin Dynamics**: A novel training approach that combines entropy-regularized objectives with stochastic Langevin dynamics for improved stability and convergence.

3. **Uncertainty Quantification**: Uses ensemble methods to quantify uncertainty in predictions, essential for reliable decision-making.

4. **Adaptive Sampling**: Intelligent sampling strategies that focus computational resources on regions with high uncertainty or error.

5. **Interactive Visualization**: A user-friendly Streamlit interface for exploring models, comparing different approaches, and analyzing parameter effects.

## Physics Equations

The CVD process is modeled by the following coupled reaction-diffusion PDEs:

```
∂Ci/∂t = ∇·(Di∇Ci) + Ri(C1, C2, ..., Cn, T)
ρCp·∂T/∂t + ∇·(Tu) = k∇²T + Q
```

Where:
- Ci represents species concentrations
- Di are diffusion coefficients
- Ri are reaction terms
- T is temperature
- u is the velocity field
- Q represents heat source/sink terms

Reaction rates follow Arrhenius kinetics:

```
Ri = Ai·e^(-Ei/RT)·∏j Cj^αij
```

## Citation

If you use CVD-PINN in your research, please cite:

```
@article{gupta2025langevin,
  title={A Langevin Dynamics Approach to Physics-Informed Neural Networks: Application to Chemical Vapor Deposition Modeling},
  author={Gupta, Dishant},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This work was inspired by research on Physics-Informed Neural Networks by Raissi et al.
- Special thanks to the TensorFlow team for their excellent deep learning framework