"""
Model comparison page for the Streamlit application.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List


def show_model_comparison_page():
    """Display the model comparison page content."""
    st.title("Model Comparison")
    
    st.markdown(
        """
        This page lets you compare different PINN models and training approaches for CVD simulation.
        Select models to compare and view the comparative performance metrics and visualizations.
        """
    )
    
    # Dummy data for demonstration - in a real app, this would come from your actual models
    model_options = [
        "Standard PINN",
        "Entropy-Langevin PINN (3 models)",
        "Entropy-Langevin PINN (5 models)",
        "Entropy-Langevin PINN (10 models)",
        "Adaptive Sampling PINN"
    ]
    
    # Model selection
    selected_models = st.multiselect(
        "Select models to compare",
        options=model_options,
        default=["Standard PINN", "Entropy-Langevin PINN (5 models)"]
    )
    
    if not selected_models:
        st.warning("Please select at least one model to display comparisons.")
        return
    
    # Metric to compare
    metric_options = ["Accuracy", "Training Time", "PDE Residual", "Uncertainty"]
    selected_metric = st.selectbox("Select metric for comparison", options=metric_options)
    
    # Generate comparison
    if st.button("Generate Comparison"):
        # Performance metrics comparison
        st.subheader("Performance Metrics")
        
        # Create sample data for the comparison
        metrics_data = _generate_sample_metrics(selected_models)
        
        # Display metrics as a table
        st.dataframe(metrics_data)
        
        # Plot the selected metric
        fig = _plot_metric_comparison(metrics_data, selected_models, selected_metric)
        st.pyplot(fig)
        plt.close(fig)
        
        # Visual comparison
        st.subheader("Visual Comparison")
        
        # Generate sample visualization
        fig = _generate_sample_visualization(selected_models)
        st.pyplot(fig)
        plt.close(fig)
        
        # Training convergence comparison
        st.subheader("Training Convergence Comparison")
        
        # Generate convergence plot
        fig = _plot_convergence_comparison(selected_models)
        st.pyplot(fig)
        plt.close(fig)


def _generate_sample_metrics(models: List[str]) -> pd.DataFrame:
    """
    Generate sample performance metrics for the selected models.
    
    Args:
        models: List of selected model names
        
    Returns:
        DataFrame with model metrics
    """
    # Random seed for reproducibility
    np.random.seed(42)
    
    # Generate data
    data = {
        "Model": models,
        "Accuracy": np.random.uniform(0.8, 0.98, size=len(models)),
        "Training Time (min)": np.random.uniform(5, 120, size=len(models)),
        "PDE Residual": np.random.uniform(1e-4, 1e-2, size=len(models)),
        "Uncertainty": np.random.uniform(0.01, 0.2, size=len(models))
    }
    
    # Make Entropy-Langevin models generally better
    for i, model in enumerate(models):
        if "Entropy-Langevin" in model:
            # Better accuracy
            data["Accuracy"][i] *= 1.1
            data["Accuracy"][i] = min(data["Accuracy"][i], 0.99)
            
            # Smaller PDE residual
            data["PDE Residual"][i] *= 0.5
            
            # Lower uncertainty for larger ensembles
            if "10 models" in model:
                data["Uncertainty"][i] *= 0.6
            elif "5 models" in model:
                data["Uncertainty"][i] *= 0.8
            
            # But longer training time
            data["Training Time (min)"][i] *= 1.5
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df = df.set_index("Model")
    
    return df


def _plot_metric_comparison(data: pd.DataFrame, models: List[str], metric: str) -> plt.Figure:
    """
    Plot comparison of a specific metric across models.
    
    Args:
        data: DataFrame with model metrics
        models: List of selected model names
        metric: Metric to plot
        
    Returns:
        Matplotlib figure
    """
    # Get the column name
    column_name = metric
    if metric == "Accuracy":
        column_name = "Accuracy"
    elif metric == "Training Time":
        column_name = "Training Time (min)"
    elif metric == "PDE Residual":
        column_name = "PDE Residual"
    elif metric == "Uncertainty":
        column_name = "Uncertainty"
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot bars
    bars = ax.bar(models, data[column_name], color=colors[:len(models)])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height * 1.01,
            f'{height:.4f}',
            ha='center', 
            va='bottom', 
            rotation=0
        )
    
    # Set labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel(column_name)
    ax.set_title(f'Comparison of {column_name} across Models')
    
    # Adjust tick labels rotation for better visibility if needed
    plt.xticks(rotation=15, ha='right')
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def _generate_sample_visualization(models: List[str]) -> plt.Figure:
    """
    Generate sample visualization comparing different models.
    
    Args:
        models: List of selected model names
        
    Returns:
        Matplotlib figure with subplots
    """
    # Number of models to compare
    n_models = len(models)
    
    # Create figure with subplots
    fig, axs = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    
    # Handle case with only one model
    if n_models == 1:
        axs = [axs]
    
    # Create sample data
    x = np.linspace(0, 0.1, 50)
    y = np.linspace(0, 0.05, 50)
    X, Y = np.meshgrid(x, y)
    
    # Loop through models and create visualizations
    for i, (model, ax) in enumerate(zip(models, axs)):
        # Create different patterns for different models
        if "Standard PINN" in model:
            Z = 0.2 * np.exp(-((X - 0.05)**2 + (Y - 0.025)**2) / 0.001)
        elif "3 models" in model:
            Z = 0.2 * np.exp(-((X - 0.05)**2 + (Y - 0.025)**2) / 0.0012)
        elif "5 models" in model:
            Z = 0.2 * np.exp(-((X - 0.05)**2 + (Y - 0.025)**2) / 0.0008)
        elif "10 models" in model:
            Z = 0.2 * np.exp(-((X - 0.05)**2 + (Y - 0.025)**2) / 0.0005)
        else:
            Z = 0.2 * np.exp(-((X - 0.055)**2 + (Y - 0.02)**2) / 0.001)
        
        # Add some random noise
        np.random.seed(i)
        if "Entropy-Langevin" in model:
            # Less noise for Entropy-Langevin models
            noise_level = 0.02 / (0.5 + i)
        else:
            noise_level = 0.05
        
        Z += np.random.normal(0, noise_level, Z.shape)
        
        # Create contour plot
        cont = ax.contourf(X, Y, Z, 50, cmap='viridis')
        
        # Add colorbar
        plt.colorbar(cont, ax=ax)
        
        # Set title and labels
        ax.set_title(model)
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def _plot_convergence_comparison(models: List[str]) -> plt.Figure:
    """
    Plot convergence comparison for different models.
    
    Args:
        models: List of selected model names
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate epochs
    epochs = np.arange(0, 1000, 10)
    
    # Generate convergence curves for each model
    for i, model in enumerate(models):
        if "Standard PINN" in model:
            # Standard PINN converges slower
            loss = 0.5 * np.exp(-0.002 * epochs) + 0.05
        elif "Entropy-Langevin" in model:
            # Entropy-Langevin converges faster
            if "3 models" in model:
                loss = 0.5 * np.exp(-0.004 * epochs) + 0.03
            elif "5 models" in model:
                loss = 0.5 * np.exp(-0.005 * epochs) + 0.02
            elif "10 models" in model:
                loss = 0.5 * np.exp(-0.006 * epochs) + 0.01
            else:
                loss = 0.5 * np.exp(-0.003 * epochs) + 0.03
        else:
            # Adaptive sampling converges at a medium rate
            loss = 0.5 * np.exp(-0.003 * epochs) + 0.02
        
        # Add some noise to the curves
        np.random.seed(i)
        noise = np.random.normal(0, 0.02, len(epochs))
        smoothed_noise = np.convolve(noise, np.ones(10)/10, mode='same')
        loss = loss + smoothed_noise * loss
        
        # Plot the curve
        ax.semilogy(epochs, loss, linewidth=2, label=model)
    
    # Set labels and title
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Convergence Comparison')
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    
    # Add legend
    ax.legend()
    
    # Tight layout
    plt.tight_layout()
    
    return fig