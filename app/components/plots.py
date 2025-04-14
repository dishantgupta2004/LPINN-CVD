"""
Plotting components for the Streamlit application.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Optional, Tuple, Dict


def plot_concentration_profile(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    species_name: str,
    time_value: float,
    cmap: str = 'viridis'
) -> plt.Figure:
    """
    Plot concentration profile for a given species at a specific time.
    
    Args:
        X: X-coordinates meshgrid
        Y: Y-coordinates meshgrid
        Z: Concentration values
        species_name: Name of the species
        time_value: Time value for the plot
        cmap: Colormap name
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create contour plot
    contour = ax.contourf(X, Y, Z, 50, cmap=cmap)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    
    # Set labels
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    
    # Set title based on species
    if species_name.lower() == 'temperature':
        ax.set_title(f'Temperature (K) at t = {time_value:.2f}s')
        cbar.set_label('Temperature (K)')
    else:
        ax.set_title(f'{species_name} Concentration at t = {time_value:.2f}s')
        cbar.set_label(f'{species_name} Concentration')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add substrate and inlet indicators
    ax.axhline(y=Y.min(), color='blue', linestyle='-', linewidth=2, alpha=0.7, label='Inlet')
    ax.axhline(y=Y.max(), color='red', linestyle='-', linewidth=2, alpha=0.7, label='Substrate')
    
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    return fig


def plot_uncertainty_visualization(
    X: np.ndarray,
    Y: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    species_name: str,
    time_value: float
) -> plt.Figure:
    """
    Create a 3-panel plot with mean prediction, standard deviation, and coefficient of variation.
    
    Args:
        X: X-coordinates meshgrid
        Y: Y-coordinates meshgrid
        mean: Mean prediction values
        std: Standard deviation values
        species_name: Name of the species
        time_value: Time value for the plot
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig)
    
    # Plot mean prediction
    ax1 = fig.add_subplot(gs[0, 0])
    cf1 = ax1.contourf(X, Y, mean, 50, cmap='viridis')
    plt.colorbar(cf1, ax=ax1, label=f"{species_name}")
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title(f"Mean {species_name} at t = {time_value:.2f}s")
    
    # Plot standard deviation
    ax2 = fig.add_subplot(gs[0, 1])
    cf2 = ax2.contourf(X, Y, std, 50, cmap='plasma')
    plt.colorbar(cf2, ax=ax2, label=f"Std Dev of {species_name}")
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title(f"Uncertainty in {species_name} at t = {time_value:.2f}s")
    
    # Plot coefficient of variation (std/mean)
    ax3 = fig.add_subplot(gs[0, 2])
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    cv = std / (np.abs(mean) + epsilon)
    # Clip extremely high values for better visualization
    cv = np.clip(cv, 0, 0.5)
    cf3 = ax3.contourf(X, Y, cv, 50, cmap='hot')
    plt.colorbar(cf3, ax=ax3, label='Coefficient of Variation')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.set_title(f"Relative Uncertainty at t = {time_value:.2f}s")
    
    plt.tight_layout()
    
    return fig


def plot_learning_curve(
    epochs: List[int],
    total_loss: List[float],
    pde_loss: Optional[List[float]] = None,
    bc_loss: Optional[List[float]] = None,
    ic_loss: Optional[List[float]] = None
) -> plt.Figure:
    """
    Plot learning curves showing loss evolution during training.
    
    Args:
        epochs: List of epoch numbers
        total_loss: List of total loss values
        pde_loss: Optional list of PDE loss values
        bc_loss: Optional list of boundary condition loss values
        ic_loss: Optional list of initial condition loss values
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot total loss
    ax.semilogy(epochs, total_loss, 'k-', linewidth=2, label='Total Loss')
    
    # Plot component losses if provided
    if pde_loss is not None:
        ax.semilogy(epochs, pde_loss, 'r-', linewidth=1.5, label='PDE Loss')
    
    if bc_loss is not None:
        ax.semilogy(epochs, bc_loss, 'b-', linewidth=1.5, label='BC Loss')
    
    if ic_loss is not None:
        ax.semilogy(epochs, ic_loss, 'g-', linewidth=1.5, label='IC Loss')
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training Loss History')
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    return fig


def plot_parametric_study(
    parameter_values: List[float],
    results: Dict[str, List[float]],
    parameter_name: str,
    parameter_unit: str = ''
) -> plt.Figure:
    """
    Plot the results of a parametric study.
    
    Args:
        parameter_values: List of parameter values
        results: Dictionary of results for each metric
        parameter_name: Name of the parameter being varied
        parameter_unit: Unit of the parameter (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for metric_name, metric_values in results.items():
        ax.plot(parameter_values, metric_values, 'o-', linewidth=2, label=metric_name)
    
    # Add labels
    param_label = f"{parameter_name}"
    if parameter_unit:
        param_label += f" ({parameter_unit})"
    
    ax.set_xlabel(param_label)
    ax.set_ylabel('Value')
    ax.set_title(f'Effect of {parameter_name} on CVD Process')
    ax.grid(True)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    return fig