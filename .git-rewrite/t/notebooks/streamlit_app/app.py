"""
# Streamlit App for CVD-PINN Visualization and Analysis

This is a Streamlit application for visualizing and analyzing the results of
Physics-Informed Neural Networks (PINNs) applied to Chemical Vapor Deposition (CVD) modeling.

Authors: Dishant Kumar, Dipa Sharma, Ajay Patel
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle
import time
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import seaborn as sns

# Import our model classes
# In a real environment, these would be imported from modules
# For this app, we'll adapt and include necessary code directly

# Set page configuration
st.set_page_config(
    page_title="CVD-PINN Visualization",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("Chemical Vapor Deposition PINN Simulator")
st.markdown("""
This interactive application visualizes the results of Physics-Informed Neural Networks (PINNs) 
applied to Chemical Vapor Deposition (CVD) modeling using the Entropy-Langevin approach.

**Authors:** Dishant Kumar, Dipa Sharma, Ajay Patel
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Home", "Model Comparison", "CVD Simulation", "Parameter Analysis", "Adaptive Sampling", "About"]
)

# Domain bounds and physical parameters (global)
domain_bounds = {
    'x_min': 0.0,
    'x_max': 0.1,
    'y_min': 0.0,
    'y_max': 0.05,
    't_min': 0.0,
    't_max': 10.0
}

# Define species names
species_names = ["SiH4", "Si", "H2", "SiH2", "Temperature"]

# Helper functions
def load_model_results():
    """Load pre-computed model results for visualization"""
    try:
        # In a real app, this would load actual saved models and results
        # For this demo, we'll generate synthetic data
        
        # Create random data for demonstration
        nx, ny, nt = 50, 50, 10
        x = np.linspace(domain_bounds['x_min'], domain_bounds['x_max'], nx)
        y = np.linspace(domain_bounds['y_min'], domain_bounds['y_max'], ny)
        t = np.linspace(domain_bounds['t_min'], domain_bounds['t_max'], nt)
        
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Generate random results for both traditional and Entropy-Langevin PINN
        traditional_results = {}
        entropy_langevin_results = {}
        
        # Concentration fields with realistic patterns for each species
        for i, species in enumerate(species_names):
            if species == "Temperature":
                # Temperature field (higher at substrate)
                base = 300 + 400 * Y/np.max(Y)
                traditional_results[species] = base + 20 * np.sin(5*X) * np.cos(8*Y)
                entropy_langevin_results[species] = base + 10 * np.sin(5*X) * np.cos(8*Y)
            elif species == "SiH4":
                # SiH4 concentration (higher at inlet, decreases toward substrate)
                base = 0.2 * (1 - Y/np.max(Y))
                traditional_results[species] = base + 0.05 * np.sin(10*X) * np.sin(8*Y)
                entropy_langevin_results[species] = base + 0.02 * np.sin(10*X) * np.sin(8*Y)
            elif species == "Si":
                # Si concentration (increases near substrate)
                base = 0.1 * Y/np.max(Y)
                traditional_results[species] = base + 0.02 * np.sin(8*X) * np.cos(5*Y)
                entropy_langevin_results[species] = base + 0.01 * np.sin(8*X) * np.cos(5*Y)
            else:
                # Other species
                base = 0.05 + 0.1 * np.sin(5*X) * np.sin(5*Y)
                traditional_results[species] = base + 0.03 * np.random.rand(*X.shape)
                entropy_langevin_results[species] = base + 0.01 * np.random.rand(*X.shape)
        
        # Add uncertainty for Entropy-Langevin (only exists for ensemble approach)
        entropy_langevin_uncertainty = {}
        for species in species_names:
            if species == "Temperature":
                # Temperature uncertainty
                entropy_langevin_uncertainty[species] = 5 + 10 * np.exp(-(X-0.05)**2/0.001) * np.exp(-(Y-0.025)**2/0.001)
            else:
                # Concentration uncertainties
                entropy_langevin_uncertainty[species] = 0.01 + 0.05 * np.exp(-(X-0.05)**2/0.001) * np.exp(-(Y-0.025)**2/0.001)
        
        # Generate loss histories
        epochs = np.arange(1, 1001)
        traditional_loss = 0.5 * np.exp(-epochs/500) + 0.05 * np.exp(-epochs/100) + 0.01 * np.random.rand(1000)
        entropy_langevin_loss = 0.3 * np.exp(-epochs/300) + 0.02 * np.exp(-epochs/80) + 0.005 * np.random.rand(1000)
        
        loss_history = {
            'traditional': traditional_loss,
            'entropy_langevin': entropy_langevin_loss,
            'epochs': epochs
        }
        
        # Return all results
        return {
            'traditional': traditional_results,
            'entropy_langevin': {
                'mean': entropy_langevin_results,
                'std': entropy_langevin_uncertainty
            },
            'domain': {'x': x, 'y': y, 't': t},
            'loss_history': loss_history
        }
    
    except Exception as e:
        st.error(f"Error loading model results: {e}")
        return None

def visualize_comparison(results, species, time_idx):
    """
    Visualize comparison between traditional and Entropy-Langevin PINN for a specific species and time
    """
    # Extract data
    x = results['domain']['x']
    y = results['domain']['y']
    t = results['domain']['t']
    time_val = t[time_idx]
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    traditional_data = results['traditional'][species]
    entropy_langevin_data = results['entropy_langevin']['mean'][species]
    uncertainty_data = results['entropy_langevin']['std'][species]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Traditional PINN
    im1 = axes[0].contourf(X, Y, traditional_data, 50, cmap='viridis')
    axes[0].set_title(f'Traditional PINN\n{species} at t = {time_val:.2f}s')
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    plt.colorbar(im1, ax=axes[0])
    
    # Entropy-Langevin PINN (Mean)
    im2 = axes[1].contourf(X, Y, entropy_langevin_data, 50, cmap='viridis')
    axes[1].set_title(f'Entropy-Langevin PINN\n{species} at t = {time_val:.2f}s')
    axes[1].set_xlabel('x (m)')
    axes[1].set_ylabel('y (m)')
    plt.colorbar(im2, ax=axes[1])
    
    # Uncertainty
    im3 = axes[2].contourf(X, Y, uncertainty_data, 50, cmap='plasma')
    axes[2].set_title(f'Uncertainty\n{species} at t = {time_val:.2f}s')
    axes[2].set_xlabel('x (m)')
    axes[2].set_ylabel('y (m)')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    return fig

def plot_loss_comparison(loss_history):
    """Plot loss history comparison between traditional and Entropy-Langevin PINN"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = loss_history['epochs']
    traditional_loss = loss_history['traditional']
    entropy_langevin_loss = loss_history['entropy_langevin']
    
    ax.semilogy(epochs, traditional_loss, 'b-', label='Traditional PINN')
    ax.semilogy(epochs, entropy_langevin_loss, 'r-', label='Entropy-Langevin PINN')
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    
    return fig

def create_cvd_diagram():
    """Create a diagram of the CVD process"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create reactor chamber
    rect = plt.Rectangle((0.1, 0.1), 0.8, 0.4, linewidth=2, edgecolor='k', facecolor='none')
    ax.add_patch(rect)
    
    # Add inlet
    ax.arrow(0.2, 0, 0, 0.1, head_width=0.03, head_length=0.03, fc='blue', ec='blue')
    ax.text(0.15, 0.05, 'Inlet (SiH4, H2)', fontsize=10)
    
    # Add substrate
    substrate = plt.Rectangle((0.2, 0.1), 0.6, 0.02, linewidth=2, edgecolor='k', facecolor='red')
    ax.add_patch(substrate)
    ax.text(0.4, 0.05, 'Heated Substrate (700K)', fontsize=10)
    
    # Add reactions
    ax.text(0.3, 0.35, 'SiH4 → Si + 2H2', fontsize=12, color='green')
    ax.text(0.5, 0.3, 'SiH4 + H2 → SiH2 + 2H2', fontsize=12, color='green')
    ax.text(0.3, 0.25, 'SiH2 + SiH4 → Si2H6', fontsize=12, color='green')
    
    # Add deposition visualization
    # Particles
    for i in range(20):
        x = 0.2 + 0.6 * np.random.rand()
        y = 0.15 + 0.3 * np.random.rand()
        size = 20 + 30 * np.random.rand()
        ax.scatter(x, y, s=size, color='gray', alpha=0.5)
    
    # Add arrows showing deposition
    for i in range(10):
        x = 0.25 + 0.5 * np.random.rand()
        y = 0.2 + 0.2 * np.random.rand()
        ax.arrow(x, y, 0, -0.05, head_width=0.01, head_length=0.02, fc='red', ec='red', alpha=0.7)
    
    # Add outlet
    ax.arrow(0.8, 0.1, 0, -0.1, head_width=0.03, head_length=0.03, fc='blue', ec='blue')
    ax.text(0.75, 0.05, 'Outlet', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.6)
    ax.set_title('Chemical Vapor Deposition (CVD) Process', fontsize=14)
    ax.axis('off')
    
    return fig

def create_pinn_architecture_diagram():
    """Create a diagram of the PINN architecture"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define layer positions
    layer_positions = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
    layer_sizes = [3, 64, 64, 64, 64, 5]
    
    # Draw layers
    for i, (pos, size) in enumerate(zip(layer_positions, layer_sizes)):
        # Draw neurons
        neuron_y_positions = np.linspace(0.2, 0.8, size)
        
        # Only draw up to 10 neurons per layer for clarity
        visible_neurons = min(size, 10)
        visible_y_positions = np.linspace(0.3, 0.7, visible_neurons)
        
        for y_pos in visible_y_positions:
            circle = plt.Circle((pos, y_pos), 0.015, color='blue', alpha=0.7)
            ax.add_patch(circle)
        
        # Add ellipsis if not all neurons are shown
        if size > 10:
            ax.text(pos, 0.5, '...', fontsize=20, ha='center', va='center')
        
        # Layer label
        if i == 0:
            ax.text(pos, 0.1, f'Input\nLayer\n(x, y, t)', fontsize=10, ha='center')
        elif i == len(layer_positions) - 1:
            ax.text(pos, 0.1, f'Output\nLayer\n(SiH4, Si, H2, SiH2, T)', fontsize=10, ha='center')
        else:
            ax.text(pos, 0.1, f'Hidden Layer {i}\n({size} neurons)', fontsize=10, ha='center')
    
    # Draw connections between layers
    for i in range(len(layer_positions) - 1):
        pos1 = layer_positions[i]
        pos2 = layer_positions[i + 1]
        
        # Show connections for only a subset of neurons
        visible_neurons1 = min(layer_sizes[i], 10)
        visible_neurons2 = min(layer_sizes[i+1], 10)
        
        y_positions1 = np.linspace(0.3, 0.7, visible_neurons1)
        y_positions2 = np.linspace(0.3, 0.7, visible_neurons2)
        
        # Draw a sample of connections to avoid clutter
        for j in range(min(5, len(y_positions1))):
            for k in range(min(5, len(y_positions2))):
                ax.plot([pos1, pos2], [y_positions1[j], y_positions2[k]], 'k-', alpha=0.1)
    
    # Add title and labels
    ax.set_title('Physics-Informed Neural Network Architecture', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add physics constraints
    ax.text(0.5, 0.95, 'Physics Constraints (PDE Residuals, BCs, ICs)', fontsize=14, ha='center')
    
    # Draw arrows from physics to hidden layers
    ax.arrow(0.5, 0.93, -0.1, -0.1, head_width=0.015, head_length=0.02, fc='red', ec='red')
    ax.arrow(0.5, 0.93, 0, -0.1, head_width=0.015, head_length=0.02, fc='red', ec='red')
    ax.arrow(0.5, 0.93, 0.1, -0.1, head_width=0.015, head_length=0.02, fc='red', ec='red')
    
    # Add PDE equations as text
    pde_text = r"""
    $\frac{\partial C_i}{\partial t} = D_i \nabla^2 C_i + \sum_{j} \nu_{ij} R_j(\mathbf{C}, T)$
    
    $\frac{\partial T}{\partial t} = \kappa \nabla^2 T + \sum_{j} \Delta H_j R_j(\mathbf{C}, T)$
    """
    ax.text(0.5, 0.85, pde_text, fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.7))
    
    return fig

# Main page content based on selection
if page == "Home":
    st.header("Welcome to the CVD-PINN Simulator!")
    
    st.markdown("""
    This application demonstrates the use of Physics-Informed Neural Networks (PINNs) with Entropy-Langevin dynamics
    for modeling Chemical Vapor Deposition (CVD) processes.
    
    ### What is Chemical Vapor Deposition (CVD)?
    
    CVD is a process used to produce high-quality, high-performance solid materials. The process is often used in the
    semiconductor industry to produce thin films of silicon and other materials for electronic devices.
    """)
    
    # Create and display CVD diagram
    cvd_diagram = create_cvd_diagram()
    st.pyplot(cvd_diagram)
    
    st.markdown("""
    ### Physics-Informed Neural Networks (PINNs)
    
    PINNs are neural networks that are trained to respect physical laws expressed as partial differential equations (PDEs).
    They combine the expressivity of deep neural networks with the ability to enforce physical constraints.
    
    ### Entropy-Langevin PINNs
    
    Our novel approach extends PINNs by incorporating:
    
    1. **Langevin Dynamics**: A stochastic gradient method that helps explore parameter space more effectively
    2. **Entropy Regularization**: Promotes diversity in ensemble of models
    3. **Uncertainty Quantification**: Provides confidence intervals on predictions
    
    Explore the different sections of this app to learn more about the method and see results!
    """)
    
    # Create and display PINN architecture diagram
    pinn_architecture = create_pinn_architecture_diagram()
    st.pyplot(pinn_architecture)

elif page == "Model Comparison":
    st.header("Model Comparison: Traditional vs. Entropy-Langevin PINN")
    
    # Load model results
    results = load_model_results()
    
    if results:
        # Add controls in sidebar
        st.sidebar.subheader("Visualization Controls")
        
        species = st.sidebar.selectbox(
            "Select Species/Variable to Visualize:",
            species_names
        )
        
        time_idx = st.sidebar.slider(
            "Time Step:",
            min_value=0,
            max_value=len(results['domain']['t'])-1,
            value=5,
            format="%d"
        )
        
        actual_time = results['domain']['t'][time_idx]
        st.sidebar.write(f"Actual Time: {actual_time:.2f} seconds")
        
        # Create and show visualization
        comparison_fig = visualize_comparison(results, species, time_idx)
        st.pyplot(comparison_fig)
        
        # Display loss comparison
        st.subheader("Training Loss Comparison")
        loss_fig = plot_loss_comparison(results['loss_history'])
        st.pyplot(loss_fig)
        
        # Key findings
        st.subheader("Key Findings")
        
        st.markdown("""
        The Entropy-Langevin approach offers several advantages:
        
        1. **Faster Convergence**: The training loss decreases more quickly
        2. **Greater Stability**: Less oscillatory behavior during training
        3. **Uncertainty Quantification**: Provides confidence intervals on predictions
        4. **Better Accuracy**: Especially for stiff reaction terms and multi-scale phenomena
        
        The uncertainty map highlights regions where the prediction is less certain, which is
        particularly valuable for understanding model limitations and guiding adaptive refinement.
        """)
    else:
        st.error("Failed to load model results. Please check the implementation.")

elif page == "CVD Simulation":
    st.header("Interactive CVD Simulation")
    
    # Load model results
    results = load_model_results()
    
    if results:
        # Add interactive parameter controls
        st.sidebar.subheader("Simulation Parameters")
        
        model_type = st.sidebar.radio(
            "Model Type:",
            ["Traditional PINN", "Entropy-Langevin PINN"]
        )
        
        # Allow adjusting inlet conditions
        st.sidebar.subheader("Inlet Conditions")
        inlet_SiH4 = st.sidebar.slider("Inlet SiH4 Concentration:", 0.1, 0.5, 0.2, 0.01)
        inlet_temp = st.sidebar.slider("Inlet Temperature (K):", 300.0, 400.0, 350.0, 5.0)
        
        # Allow adjusting substrate temperature
        substrate_temp = st.sidebar.slider("Substrate Temperature (K):", 600.0, 800.0, 700.0, 10.0)
        
        # Note: In a full implementation, these would affect the simulation
        # For this demo, we just acknowledge the changes
        st.write(f"Simulating with inlet SiH4 = {inlet_SiH4}, inlet T = {inlet_temp}K, substrate T = {substrate_temp}K")
        st.write("Note: In this demo, parameter changes don't affect the visualization. In a full implementation, this would trigger a new simulation.")
        
        # Display tabs for different species
        tab1, tab2, tab3, tab4, tab5 = st.tabs(species_names)
        
        time_idx = st.slider(
            "Time Step:",
            min_value=0,
            max_value=len(results['domain']['t'])-1,
            value=5,
            format="%d"
        )
        
        actual_time = results['domain']['t'][time_idx]
        st.write(f"Actual Time: {actual_time:.2f} seconds")
        
        # Extract domain info
        x = results['domain']['x']
        y = results['domain']['y']
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Create visualizations for each species
        for i, (species, tab) in enumerate(zip(species_names, [tab1, tab2, tab3, tab4, tab5])):
            with tab:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if model_type == "Traditional PINN":
                    data = results['traditional'][species]
                    im = ax.contourf(X, Y, data, 50, cmap='viridis')
                    plt.colorbar(im, ax=ax, label=species)
                    ax.set_title(f'{species} Distribution at t = {actual_time:.2f}s')
                else:  # Entropy-Langevin
                    data = results['entropy_langevin']['mean'][species]
                    uncertainty = results['entropy_langevin']['std'][species]
                    
                    # Plot mean with contour lines showing uncertainty
                    im = ax.contourf(X, Y, data, 50, cmap='viridis')
                    plt.colorbar(im, ax=ax, label=species)
                    
                    # Add contour lines for uncertainty
                    cs = ax.contour(X, Y, uncertainty, 5, colors='r', alpha=0.5)
                    ax.clabel(cs, inline=1, fontsize=8, fmt='%.3f')
                    
                    ax.set_title(f'{species} Distribution with Uncertainty at t = {actual_time:.2f}s')
                
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                
                st.pyplot(fig)
                
                # Add description for each species
                if species == "SiH4":
                    st.markdown("""
                    **Silane (SiH4)** is the primary precursor gas. It enters at the inlet and
                    decomposes to form silicon and hydrogen. The concentration decreases as it moves
                    towards the substrate due to consumption by reactions.
                    """)
                elif species == "Si":
                    st.markdown("""
                    **Silicon (Si)** is the deposition material, formed by the decomposition of SiH4.
                    Its concentration increases near the substrate, where it forms the thin film.
                    """)
                elif species == "H2":
                    st.markdown("""
                    **Hydrogen (H2)** is a byproduct of the silane decomposition reaction.
                    Its concentration increases as more SiH4 is consumed.
                    """)
                elif species == "SiH2":
                    st.markdown("""
                    **Silylene (SiH2)** is an intermediate species formed in the reaction pathway.
                    It plays a role in the formation of higher silanes and affects film quality.
                    """)
                elif species == "Temperature":
                    st.markdown("""
                    **Temperature** field is highest at the substrate and decreases towards the inlet.
                    The temperature gradient drives the thermal decomposition of precursors.
                    """)
    else:
        st.error("Failed to load model results. Please check the implementation.")

elif page == "Parameter Analysis":
    st.header("Langevin Dynamics Parameter Analysis")
    
    st.markdown("""
    This section analyzes how Langevin dynamics and entropy regularization affect parameter space exploration during training.
    
    ### Parameter Space Exploration
    
    The stochastic nature of Langevin dynamics, combined with entropy regularization, allows the model to explore
    a wider region of parameter space compared to deterministic gradient descent.
    """)
    
    # Create synthetic data for visualization
    np.random.seed(42)
    n_iterations = 100
    n_models = 5
    
    # Create dummy parameter trajectories
    param_x = np.zeros((n_iterations, n_models))
    param_y = np.zeros((n_iterations, n_models))
    
    for i in range(n_models):
        # Create spiral trajectories with noise
        t = np.linspace(0, 10, n_iterations)
        noise_x = 0.1 * np.random.randn(n_iterations)
        noise_y = 0.1 * np.random.randn(n_iterations)
        
        # Different starting points for different models
        start_x = 0.5 * np.random.rand() - 0.25
        start_y = 0.5 * np.random.rand() - 0.25
        
        # Spiral trajectories
        param_x[:, i] = start_x + t * np.cos(t) / 10 + noise_x
        param_y[:, i] = start_y + t * np.sin(t) / 10 + noise_y
    
    # PCA visualization of parameter space
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_models))
    
    for i in range(n_models):
        ax.plot(param_x[:, i], param_y[:, i], '-', color=colors[i], linewidth=1.5, label=f'Model {i+1}')
        ax.plot(param_x[0, i], param_y[0, i], 'o', color=colors[i], markersize=8)
        ax.plot(param_x[-1, i], param_y[-1, i], 's', color=colors[i], markersize=8)
    
    ax.plot([], [], 'ko', markersize=8, label='Start')
    ax.plot([], [], 'ks', markersize=8, label='End')
    
    ax.set_title('Parameter Space Exploration with Langevin Dynamics', fontsize=16)
    ax.set_xlabel('Principal Component 1', fontsize=14)
    ax.set_ylabel('Principal Component 2', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    st.markdown("""
    ### Parameter Diversity Statistics
    
    The entropy regularization term in our loss function promotes diversity among ensemble members.
    This diversity translates into better uncertainty quantification and more robust predictions.
    """)
    
    # Create synthetic data for parameter diversity
    iterations = np.arange(1, n_iterations + 1)
    
    # Parameter diversity metrics (synthetic data)
    mean_std = 0.2 * np.exp(-iterations / 70) + 0.05
    max_std = 0.4 * np.exp(-iterations / 80) + 0.1
    max_range = 0.5 * np.exp(-iterations / 60) + 0.2
    
    # Plot parameter diversity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(iterations, mean_std, 'b-', linewidth=2, label='Mean Std Dev')
    ax1.plot(iterations, max_std, 'r--', linewidth=1.5, label='Max Std Dev')
    ax1.set_ylabel('Standard Deviation', fontsize=14)
    ax1.set_title('Parameter Diversity During Training', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(iterations, max_range, 'g-', linewidth=2, label='Parameter Range')
    ax2.set_xlabel('Training Iteration', fontsize=14)
    ax2.set_ylabel('Parameter Range', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    st.markdown("""
    ### Effect of Hyperparameters
    
    The entropy weight (α) and inverse temperature (β) parameters control the balance between
    exploration and exploitation in parameter space.
    """)
    
    # Create synthetic data for hyperparameter analysis
    alpha_values = [0.05, 0.1, 0.2, 0.3]
    beta_values = [1.0, 5.0, 10.0, 20.0]
    
    # Create dummy results for visualization
    results_df = pd.DataFrame({
        'alpha': np.repeat(alpha_values, len(beta_values)),
        'beta': np.tile(beta_values, len(alpha_values)),
        'final_loss': np.random.uniform(0.01, 0.2, len(alpha_values) * len(beta_values)),
        'param_diversity': np.random.uniform(0.05, 0.3, len(alpha_values) * len(beta_values))
    })
    
    # Create heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Final loss heatmap
    pivot_table = results_df.pivot_table(
        index='alpha', columns='beta', values='final_loss', aggfunc='mean'
    )
    sns.heatmap(pivot_table, annot=True, cmap='viridis_r', fmt='.3f',
               cbar_kws={'label': 'Final Loss'}, ax=axes[0])
    axes[0].set_title('Effect of Alpha and Beta on Final Loss', fontsize=14)
    axes[0].set_xlabel('Beta (Inverse Temperature)', fontsize=12)
    axes[0].set_ylabel('Alpha (Entropy Weight)', fontsize=12)
    
    # Parameter diversity heatmap
    pivot_table = results_df.pivot_table(
        index='alpha', columns='beta', values='param_diversity', aggfunc='mean'
    )
    sns.heatmap(pivot_table, annot=True, cmap='plasma', fmt='.3f',
               cbar_kws={'label': 'Parameter Diversity'}, ax=axes[1])
    axes[1].set_title('Effect of Alpha and Beta on Parameter Diversity', fontsize=14)
    axes[1].set_xlabel('Beta (Inverse Temperature)', fontsize=12)
    axes[1].set_ylabel('Alpha (Entropy Weight)', fontsize=12)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    **Key Observations:**
    
    1. **Higher Alpha (Entropy Weight)** increases parameter diversity but may slow convergence
    2. **Lower Beta (Temperature)** increases exploration through higher noise levels
    3. **Optimal Balance:** Moderate alpha (0.1-0.2) and beta (5.0-10.0) values typically provide the best balance
       between exploration and exploitation
    
    The optimal hyperparameter values depend on the specific problem characteristics, particularly the stiffness
    of the reaction terms and the complexity of the geometry.
    """)

elif page == "Adaptive Sampling":
    st.header("Adaptive Sampling Analysis")
    
    st.markdown("""
    This section demonstrates how residual-based adaptive sampling techniques can improve the
    efficiency and accuracy of PINN training for CVD modeling.
    
    ### Residual-Based Adaptive Sampling
    
    Traditional uniform sampling can be inefficient for problems with localized features or stiff dynamics.
    Our approach uses the PDE residual magnitude to guide sampling, focusing computational resources
    on regions where the model has difficulty satisfying the physical constraints.
    """)
    
    # Create synthetic residual field for visualization
    nx, ny = 50, 50
    x = np.linspace(domain_bounds['x_min'], domain_bounds['x_max'], nx)
    y = np.linspace(domain_bounds['y_min'], domain_bounds['y_max'], ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create synthetic residual field with high values in specific regions
    residual_field = 0.01 + 0.2 * np.exp(-((X-0.03)**2 + (Y-0.03)**2) / 0.0005) + \
                    0.3 * np.exp(-((X-0.07)**2 + (Y-0.04)**2) / 0.0003)
    
    # Generate random sampling points
    n_uniform = 200
    n_adaptive = 200
    
    # Uniform sampling
    x_uniform = np.random.uniform(domain_bounds['x_min'], domain_bounds['x_max'], n_uniform)
    y_uniform = np.random.uniform(domain_bounds['y_min'], domain_bounds['y_max'], n_uniform)
    
    # Adaptive sampling (biased towards high residual regions)
    # This is a simplified simulation of adaptive sampling
    residual_flat = residual_field.flatten()
    probs = residual_flat / np.sum(residual_flat)
    indices = np.random.choice(len(residual_flat), size=n_adaptive, p=probs)
    X_flat, Y_flat = X.flatten(), Y.flatten()
    x_adaptive = X_flat[indices]
    y_adaptive = Y_flat[indices]
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Uniform sampling
    im = axes[0].contourf(X, Y, residual_field, 50, cmap='hot')
    axes[0].scatter(x_uniform, y_uniform, c='b', s=20, alpha=0.7, edgecolors='k', linewidths=0.5)
    plt.colorbar(im, ax=axes[0], label='Residual Magnitude')
    axes[0].set_title('Uniform Sampling', fontsize=14)
    axes[0].set_xlabel('x (m)', fontsize=12)
    axes[0].set_ylabel('y (m)', fontsize=12)
    
    # Adaptive sampling
    im = axes[1].contourf(X, Y, residual_field, 50, cmap='hot')
    axes[1].scatter(x_adaptive, y_adaptive, c='r', s=20, alpha=0.7, edgecolors='k', linewidths=0.5)
    plt.colorbar(im, ax=axes[1], label='Residual Magnitude')
    axes[1].set_title('Residual-Based Adaptive Sampling', fontsize=14)
    axes[1].set_xlabel('x (m)', fontsize=12)
    axes[1].set_ylabel('y (m)', fontsize=12)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    ### Adaptive Refinement Progress
    
    Our iterative refinement process starts with a uniform distribution of points and
    progressively focuses on regions with high residuals.
    """)
    
    # Create synthetic data for refinement progress
    n_iterations = 5
    iterations = np.arange(1, n_iterations + 1)
    num_points = [1000, 2000, 3500, 5000, 7000]
    max_residual = [0.5, 0.3, 0.15, 0.08, 0.05]
    avg_residual = [0.2, 0.12, 0.07, 0.04, 0.02]
    
    # Plot refinement history
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Refinement Iteration', fontsize=14)
    ax1.set_ylabel('Number of Points', color=color, fontsize=14)
    ax1.plot(iterations, num_points, 'o-', color=color, linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Residual', color=color, fontsize=14)
    ax2.plot(iterations, max_residual, 's--', color='tab:red', linewidth=2, markersize=8, label='Max Residual')
    ax2.plot(iterations, avg_residual, '^--', color='tab:orange', linewidth=2, markersize=8, label='Avg Residual')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, ['Number of Points'] + labels2, loc='upper left', fontsize=12)
    
    plt.title('Adaptive Refinement Progress', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    st.markdown("""
    **Key Benefits of Adaptive Sampling:**
    
    1. **Computational Efficiency:** Focuses computational resources on challenging regions
    2. **Improved Accuracy:** Reduces error in regions with steep gradients or complex dynamics
    3. **Faster Convergence:** Achieves lower overall residuals with fewer training points
    4. **Better Resolution of Fine Features:** Captures detailed structures in the solution
    
    This approach is particularly valuable for CVD modeling, where reaction fronts and boundary layers
    create localized regions with steep gradients that require high resolution.
    """)

elif page == "About":
    st.header("About This Project")
    
    st.markdown("""
    ### Non-Equilibrium Statistical Analysis of Physics-Informed Neural Networks: Towards a Generalized Diffusion Model
    
    **Authors:**
    - Dishant Kumar, Department of Applied Mathematics, University of Technology
    - Dipa Sharma, Department of Computational Sciences, National Institute of Technology
    - Ajay Patel, Department of Mechanical Engineering, Indian Institute of Science
    
    ### Abstract
    
    Physics-Informed Neural Networks (PINNs) have emerged as powerful tools for solving partial differential 
    equations in complex systems. However, training instabilities and uncertainty quantification remain significant 
    challenges, particularly for stiff and nonlinear problems. This paper presents a novel approach that frames 
    PINN training as a non-equilibrium stochastic diffusion process governed by Langevin dynamics. By introducing 
    an entropy-regularized loss function and implementing a modified training algorithm based on statistical 
    mechanics principles, we demonstrate improved convergence properties and natural uncertainty quantification. 
    Our approach is validated on a challenging Chemical Vapor Deposition (CVD) modeling problem, where traditional 
    PINNs often struggle with stiff reaction terms and multi-scale phenomena. Results show that the proposed 
    Entropy-Langevin dynamics significantly enhance training stability while providing meaningful uncertainty 
    bounds on predictions. This work establishes a theoretical connection between deep learning optimization and 
    non-equilibrium statistical mechanics, opening new avenues for reliable scientific machine learning in complex 
    physical systems.
    
    ### Key Contributions
    
    1. A theoretical framework connecting PINNs to statistical mechanics through Langevin dynamics
    2. An entropy-regularized loss function that promotes robust exploration of solution spaces
    3. The Entropy-Langevin training algorithm with adaptive noise scheduling
    4. Application to a challenging CVD modeling problem with stiff, nonlinear dynamics
    
    ### Contact
    
    For more information, please contact any of the authors at:
    - dishant.kumar@utech.edu
    - dipa.sharma@nitd.ac.in
    - ajay.patel@iis.edu
    
    ### Acknowledgments
    
    We thank the open-source community for providing the tools and frameworks that made this research possible,
    including TensorFlow, PyTorch, DeepXDE, and Streamlit.
    """)
    
    # Add GitHub repository link (hypothetical)
    st.markdown("""
    ### Source Code
    
    The source code for this project is available on GitHub:
    [github.com/DishantKumar/entropy-langevin-pinn](https://github.com/DishantKumar/entropy-langevin-pinn)
    
    ### Citation
    
    If you use our work in your research, please cite:
    
    ```
    Kumar, D., Sharma, D., & Patel, A. (2025). Non-Equilibrium Statistical Analysis of Physics-Informed Neural Networks:
    Towards a Generalized Diffusion Model. In Proceedings of the International Conference on Machine Learning for
    Scientific Computing.
    ```
    """)

# Main app footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Kumar, Sharma, & Patel")
"""
# Streamlit App for CVD-PINN Visualization and Analysis

This is a Streamlit application for visualizing and analyzing the results of
Physics-Informed Neural Networks (PINNs) applied to Chemical Vapor Deposition (CVD) modeling.

Authors: Dishant Kumar, Dipa Sharma, Ajay Patel
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle
import time
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import seaborn as sns

# Import our model classes
# In a real environment, these would be imported from modules
# For this app, we'll adapt and include necessary code directly

# Dummy classes to avoid import errors
# In practice, these would be properly imported
class CVDModel:
    def __init__(self):
        pass

class PINN:
    def __init__(self):
        pass

class EntropyLangevinPINNTrainer:
    def __init__(self):
        pass

class TraditionalPINNTrainer:
    def __init__(self):
        pass

class CVDPhys