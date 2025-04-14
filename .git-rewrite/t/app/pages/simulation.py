"""
Simulation page for the Streamlit application.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path
import tensorflow as tf
import os

# Add the parent directory to sys.path to import the package
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import cvd_pinn
from src import CVDPinnConfig, CVDPINN, CVDPDE, CVDDataGenerator, EntropyLangevinTrainer
from src.config import DomainConfig, PhysicalConfig, ModelConfig, TrainingConfig, EntropyLangevinConfig
from app.components.plots import plot_concentration_profile, plot_learning_curve


def show_simulation_page():
    """Display the simulation page content."""
    st.title("CVD Simulation")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Configuration", "Training", "Visualization"])
    
    # Initialize session state if not already initialized
    if "config" not in st.session_state:
        st.session_state.config = CVDPinnConfig()
    
    if "trained_model" not in st.session_state:
        st.session_state.trained_model = None
    
    if "trainer" not in st.session_state:
        st.session_state.trainer = None
    
    if "training_completed" not in st.session_state:
        st.session_state.training_completed = False
    
    # Configuration tab
    with tab1:
        _show_configuration_tab()
    
    # Training tab
    with tab2:
        _show_training_tab()
    
    # Visualization tab
    with tab3:
        _show_visualization_tab()


def _show_configuration_tab():
    """Display configuration options."""
    st.header("Simulation Configuration")
    
    st.subheader("Domain Configuration")
    domain_col1, domain_col2 = st.columns(2)
    
    with domain_col1:
        x_min = st.number_input("X Min (m)", value=0.0, format="%.4f")
        y_min = st.number_input("Y Min (m)", value=0.0, format="%.4f")
        t_min = st.number_input("T Min (s)", value=0.0, format="%.4f")
        dimension = st.selectbox("Dimension", options=[2, 3], index=0)
    
    with domain_col2:
        x_max = st.number_input("X Max (m)", value=0.1, format="%.4f")
        y_max = st.number_input("Y Max (m)", value=0.05, format="%.4f")
        t_max = st.number_input("T Max (s)", value=10.0, format="%.4f")
    
    # Update domain config
    st.session_state.config.domain = DomainConfig(
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        t_min=t_min, t_max=t_max,
        dimension=dimension
    )
    
    st.subheader("Physical Configuration")
    phys_col1, phys_col2 = st.columns(2)
    
    with phys_col1:
        d_sih4 = st.number_input("D_SiH4 (m²/s)", value=1.0e-5, format="%.2e")
        d_si = st.number_input("D_Si (m²/s)", value=5.0e-6, format="%.2e")
        d_h2 = st.number_input("D_H2 (m²/s)", value=4.0e-5, format="%.2e")
        d_sih2 = st.number_input("D_SiH2 (m²/s)", value=1.5e-5, format="%.2e")
    
    with phys_col2:
        thermal_conductivity = st.number_input("Thermal Conductivity (W/m·K)", value=0.1, format="%.2e")
        specific_heat = st.number_input("Specific Heat (J/kg·K)", value=700.0, format="%.2f")
        density = st.number_input("Density (kg/m³)", value=1.0, format="%.2f")
    
    # Update physical config
    st.session_state.config.physical = PhysicalConfig(
        D_SiH4=d_sih4, D_Si=d_si, 
        D_H2=d_h2, D_SiH2=d_sih2,
        thermal_conductivity=thermal_conductivity,
        specific_heat=specific_heat,
        density=density
    )
    
    st.subheader("Model Configuration")
    model_col1, model_col2 = st.columns(2)
    
    with model_col1:
        layers_str = st.text_input("Hidden Layers", value="64,64,64,64,64,64")
        hidden_layers = [int(x.strip()) for x in layers_str.split(",")]
        activation = st.selectbox("Activation Function", options=["tanh", "relu", "sigmoid", "swish"], index=0)
    
    with model_col2:
        output_size = st.number_input("Output Size", value=5, min_value=1, max_value=10, step=1)
        dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.0, step=0.01)
    
    # Update model config
    st.session_state.config.model = ModelConfig(
        hidden_layers=hidden_layers,
        activation=activation,
        output_size=output_size,
        dropout_rate=dropout_rate
    )
    
    st.subheader("Training Configuration")
    train_col1, train_col2 = st.columns(2)
    
    with train_col1:
        learning_rate = st.number_input("Learning Rate", value=1e-3, format="%.2e")
        n_epochs = st.number_input("Number of Epochs", value=1000, min_value=10, max_value=20000, step=100)
        batch_size = st.number_input("Batch Size", value=1024, min_value=32, max_value=4096, step=32)
    
    with train_col2:
        ensemble_size = st.number_input("Ensemble Size", value=3, min_value=1, max_value=10, step=1)
        alpha_initial = st.number_input("Alpha Initial", value=0.1, format="%.2e")
        beta_initial = st.number_input("Beta Initial", value=10.0, format="%.2f")
    
    # Update training config
    st.session_state.config.training = TrainingConfig(
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        batch_size=batch_size
    )
    
    # Update entropy-langevin config
    st.session_state.config.entropy_langevin = EntropyLangevinConfig(
        ensemble_size=ensemble_size,
        alpha_initial=alpha_initial,
        beta_initial=beta_initial
    )
    
    # Save config
    if st.button("Save Configuration"):
        try:
            os.makedirs("config", exist_ok=True)
            config_file = "config/cvd_pinn_config.json"
            st.session_state.config.save(config_file)
            st.success(f"Configuration saved to {config_file}")
        except Exception as e:
            st.error(f"Error saving configuration: {str(e)}")


def _show_training_tab():
    """Display training options and controls."""
    st.header("Model Training")
    
    # Check if training is already completed
    if st.session_state.training_completed:
        st.success("Training completed! You can now visualize the results in the Visualization tab.")
        
        # Option to reset and train again
        if st.button("Reset and Train Again"):
            st.session_state.training_completed = False
            st.session_state.trained_model = None
            st.session_state.trainer = None
            st.rerun()
    else:
        # Training controls
        col1, col2 = st.columns(2)
        
        with col1:
            n_epochs = st.number_input(
                "Number of Epochs", 
                value=st.session_state.config.training.n_epochs,
                min_value=10,
                max_value=10000,
                step=10
            )
            
        with col2:
            print_freq = st.number_input(
                "Print Frequency", 
                value=10,
                min_value=1,
                max_value=100,
                step=1
            )
        
        # Training button
        if st.button("Start Training"):
            # Initialize trainer
            trainer = EntropyLangevinTrainer(st.session_state.config)
            st.session_state.trainer = trainer
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create placeholder for loss plot
            loss_plot = st.empty()
            
            # Training loop
            start_time = time.time()
            
            # We'll simulate the training here as actual training would be too intensive
            # In a real application, you would use the actual trainer.train() method
            
            for epoch in range(min(n_epochs, 100)):  # Limiting to 100 for demonstration
                # Update progress
                progress = (epoch + 1) / min(n_epochs, 100)
                progress_bar.progress(progress)
                
                # Simulate some training work
                time.sleep(0.1)
                
                # Update status
                elapsed = time.time() - start_time
                status_text.text(f"Epoch {epoch+1}/{min(n_epochs, 100)} - Elapsed: {elapsed:.2f}s")
                
                # Simulate loss values
                if (epoch + 1) % print_freq == 0 or epoch == min(n_epochs, 100) - 1:
                    # Generate some fake loss history for plotting
                    epochs = np.arange(epoch + 1)
                    loss_history = 0.1 * np.exp(-0.01 * epochs) + 0.01
                    
                    # Plot loss history
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.semilogy(epochs, loss_history, 'k-', linewidth=2, label='Total Loss')
                    ax.set_xlabel('Epochs')
                    ax.set_ylabel('Loss')
                    ax.set_title('Training Loss')
                    ax.grid(True)
                    ax.legend()
                    
                    loss_plot.pyplot(fig)
                    plt.close(fig)
            
            # Mark training as completed
            st.session_state.training_completed = True
            st.session_state.trained_model = "MOCK_MODEL"  # In a real app, this would be the actual model
            
            st.success("Training completed successfully!")


def _show_visualization_tab():
    """Display visualization options and plots."""
    st.header("Results Visualization")
    
    if not st.session_state.training_completed:
        st.warning("Please complete training first before visualizing results.")
        return
    
    # Time selection for visualization
    t_value = st.slider(
        "Time (s)", 
        min_value=float(st.session_state.config.domain.t_min),
        max_value=float(st.session_state.config.domain.t_max),
        value=float(st.session_state.config.domain.t_max) / 2,
        step=0.1
    )
    
    # Species selection
    species_options = ["SiH4", "Si", "H2", "SiH2", "Temperature"]
    selected_species = st.selectbox("Select Species to Visualize", options=species_options)
    
    # Generate visualization
    if st.button("Generate Visualization"):
        # Create placeholder for the plot
        plot_container = st.empty()
        
        # Get domain parameters
        x_min, x_max = st.session_state.config.domain.x_min, st.session_state.config.domain.x_max
        y_min, y_max = st.session_state.config.domain.y_min, st.session_state.config.domain.y_max
        
        # Generate fake data for demonstration
        nx, ny = 50, 50
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        
        species_idx = species_options.index(selected_species)
        
        # Create different patterns for different species
        if species_idx == 0:  # SiH4
            Z = 0.2 * np.exp(-((X - 0.05)**2 + (Y - 0.025)**2) / 0.001)
        elif species_idx == 1:  # Si
            Z = 0.15 * (1 - np.exp(-((X - 0.05)**2 + (Y - 0.025)**2) / 0.001))
        elif species_idx == 2:  # H2
            Z = 0.1 * (1 + np.sin(10 * X) * np.cos(10 * Y))
        elif species_idx == 3:  # SiH2
            Z = 0.05 * np.exp(-10 * Y)
        else:  # Temperature
            Z = 300 + 400 * Y  # Temperature increases towards substrate
        
        # Create plot
        fig = plot_concentration_profile(X, Y, Z, selected_species, t_value)
        plot_container.pyplot(fig)
        plt.close(fig)
        
        # Show additional plots
        st.subheader("Cross-section profiles")
        col1, col2 = st.columns(2)
        
        with col1:
            # X-direction cross-section
            y_loc = 0.025  # middle of y-range
            y_idx = np.argmin(np.abs(y - y_loc))
            profile_x = Z[y_idx, :]
            
            fig, ax = plt.subplots()
            ax.plot(y, profile_y)
            ax.set_xlabel('Y position (m)')
            ax.set_ylabel(selected_species)
            ax.set_title(f'{selected_species} profile along Y at X={x_loc:.3f}m')
            ax.grid(True)
            st.pyplot(fig)
            plt.close(fig)
            
            ax.plot(x, profile_x)
            ax.set_xlabel('X position (m)')
            ax.set_ylabel(selected_species)
            ax.set_title(f'{selected_species} profile along X at Y={y_loc:.3f}m')
            ax.grid(True)
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            # Y-direction cross-section
            x_loc = 0.05  # middle of x-range
            x_idx = np.argmin(np.abs(x - x_loc))
            profile_y = Z[:, x_idx]
            
            fig, ax = plt.subplots()
            ax.plot