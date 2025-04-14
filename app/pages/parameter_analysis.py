"""
Parameter analysis page for the Streamlit application.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple

from app.components.plots import plot_parametric_study


def show_parameter_analysis_page():
    """Display the parameter analysis page content."""
    st.title("Parameter Analysis")
    
    st.markdown(
        """
        This page allows you to explore the effects of different physical and model parameters 
        on the CVD simulation results. Select a parameter to vary and the output metrics to analyze.
        """
    )
    
    # Parameter selection
    parameter_categories = ["Physical Parameters", "Model Parameters", "Operating Conditions"]
    selected_category = st.selectbox("Select parameter category", options=parameter_categories)
    
    # Parameters for each category
    if selected_category == "Physical Parameters":
        parameters = [
            "Diffusion Coefficient (SiH4)",
            "Diffusion Coefficient (H2)",
            "Thermal Conductivity",
            "Reaction Rate Constant"
        ]
    elif selected_category == "Model Parameters":
        parameters = [
            "Number of Hidden Layers",
            "Neurons per Layer",
            "Learning Rate",
            "Ensemble Size"
        ]
    else:  # Operating Conditions
        parameters = [
            "Inlet Temperature",
            "Substrate Temperature",
            "Inlet SiH4 Concentration",
            "Reactor Pressure"
        ]
    
    selected_parameter = st.selectbox("Select parameter to analyze", options=parameters)
    
    # Metrics to analyze
    metric_options = [
        "Deposition Rate",
        "Film Thickness Uniformity",
        "Precursor Utilization",
        "Energy Efficiency"
    ]
    
    selected_metrics = st.multiselect(
        "Select metrics to analyze",
        options=metric_options,
        default=["Deposition Rate", "Film Thickness Uniformity"]
    )
    
    if not selected_metrics:
        st.warning("Please select at least one metric to analyze.")
        return
    
    # Generate analysis
    if st.button("Generate Analysis"):
        # Generate parameter range
        param_values, param_unit = _get_parameter_range(selected_parameter)
        
        # Generate results for each metric
        results = _generate_parametric_results(param_values, selected_metrics, selected_parameter)
        
        # Display as table
        st.subheader("Parametric Study Results")
        
        # Create DataFrame for display
        df = pd.DataFrame({"Parameter Value": param_values})
        
        for metric in selected_metrics:
            df[metric] = results[metric]
        
        if param_unit:
            df = df.rename(columns={"Parameter Value": f"Parameter Value ({param_unit})"})
            
        st.dataframe(df)
        
        # Plot parametric study
        st.subheader("Parametric Study Plot")
        fig = plot_parametric_study(param_values, results, selected_parameter, param_unit)
        st.pyplot(fig)
        plt.close(fig)
        
        # Display optimal values
        st.subheader("Optimal Values")
        
        for metric in selected_metrics:
            if metric in ["Deposition Rate", "Precursor Utilization", "Energy Efficiency"]:
                # For these metrics, higher is better
                best_idx = np.argmax(results[metric])
                optimal_type = "Maximum"
            else:
                # For uniformity, lower is better
                best_idx = np.argmin(results[metric])
                optimal_type = "Minimum"
            
            optimal_value = param_values[best_idx]
            
            st.markdown(
                f"**{optimal_type} {metric}:** {param_values[best_idx]:.4f} {param_unit} "
                f"(Value: {results[metric][best_idx]:.4f})"
            )
        
        # Show sensitivity analysis
        st.subheader("Sensitivity Analysis")
        
        # Calculate normalized sensitivities
        sensitivities = {}
        for metric in selected_metrics:
            # Calculate gradient
            grad = np.gradient(results[metric], param_values)
            
            # Normalize
            param_range = max(param_values) - min(param_values)
            metric_range = max(results[metric]) - min(results[metric])
            
            if metric_range > 0:
                norm_sensitivity = (param_range / metric_range) * grad
            else:
                norm_sensitivity = np.zeros_like(grad)
            
            sensitivities[metric] = norm_sensitivity
        
        # Plot sensitivity
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric, sensitivity in sensitivities.items():
            ax.plot(param_values, sensitivity, 'o-', linewidth=2, label=metric)
        
        ax.set_xlabel(f"{selected_parameter} {f'({param_unit})' if param_unit else ''}")
        ax.set_ylabel("Normalized Sensitivity")
        ax.set_title(f"Sensitivity Analysis for {selected_parameter}")
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def _get_parameter_range(parameter: str) -> Tuple[List[float], str]:
    """
    Get the range of values for a given parameter.
    
    Args:
        parameter: Parameter name
        
    Returns:
        Tuple of (parameter_values, parameter_unit)
    """
    # Define ranges and units for each parameter
    if parameter == "Diffusion Coefficient (SiH4)":
        values = np.linspace(0.5e-5, 2.0e-5, 10)
        unit = "m²/s"
    elif parameter == "Diffusion Coefficient (H2)":
        values = np.linspace(1.0e-5, 8.0e-5, 10)
        unit = "m²/s"
    elif parameter == "Thermal Conductivity":
        values = np.linspace(0.05, 0.5, 10)
        unit = "W/(m·K)"
    elif parameter == "Reaction Rate Constant":
        values = np.linspace(0.5e6, 5.0e6, 10)
        unit = "s⁻¹"
    elif parameter == "Number of Hidden Layers":
        values = np.array([2, 3, 4, 5, 6, 7, 8, 10, 12, 15])
        unit = ""
    elif parameter == "Neurons per Layer":
        values = np.array([16, 32, 48, 64, 96, 128, 192, 256, 384, 512])
        unit = ""
    elif parameter == "Learning Rate":
        values = np.logspace(-5, -2, 10)
        unit = ""
    elif parameter == "Ensemble Size":
        values = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 50])
        unit = ""
    elif parameter == "Inlet Temperature":
        values = np.linspace(300, 500, 10)
        unit = "K"
    elif parameter == "Substrate Temperature":
        values = np.linspace(600, 900, 10)
        unit = "K"
    elif parameter == "Inlet SiH4 Concentration":
        values = np.linspace(0.05, 0.5, 10)
        unit = "mol/m³"
    elif parameter == "Reactor Pressure":
        values = np.linspace(0.1, 1.0, 10)
        unit = "atm"
    else:
        values = np.linspace(0, 1, 10)
        unit = ""
    
    return values.tolist(), unit


def _generate_parametric_results(param_values: List[float], metrics: List[str], parameter: str) -> Dict[str, List[float]]:
    """
    Generate sample results for parametric study.
    
    Args:
        param_values: List of parameter values
        metrics: List of metrics to analyze
        parameter: Parameter name
        
    Returns:
        Dictionary with results for each metric
    """
    results = {}
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    for metric in metrics:
        # Different function shape based on metric and parameter
        if metric == "Deposition Rate":
            if "Temperature" in parameter:
                # Deposition rate increases with temperature (Arrhenius behavior)
                base = np.exp(0.01 * np.array(param_values)) - 1
            elif "Diffusion" in parameter or "Reaction" in parameter:
                # Deposition rate increases with diffusion and reaction rate
                base = np.sqrt(np.array(param_values))
            elif "Concentration" in parameter:
                # Linear with concentration
                base = np.array(param_values)
            else:
                # Default behavior - saturation curve
                base = 1 - np.exp(-0.2 * np.array(param_values))
        
        elif metric == "Film Thickness Uniformity":
            if "Diffusion" in parameter:
                # Uniformity improves with diffusion
                base = 0.2 / (0.1 + np.array(param_values))
            elif "Temperature" in parameter and "Substrate" in parameter:
                # Uniformity gets worse with higher substrate temperature
                base = 0.05 + 0.0001 * np.array(param_values)**2
            elif "Neurons" in parameter or "Layers" in parameter:
                # Model complexity helps uniformity up to a point
                x = np.array(param_values)
                base = 0.2 - 0.15 * (1 - np.exp(-0.01 * x))
            else:
                # Default behavior - parabola with minimum
                x = np.array(param_values)
                mid_point = (max(x) + min(x)) / 2
                base = 0.05 + 0.2 * ((x - mid_point) / (max(x) - min(x)))**2
        
        elif metric == "Precursor Utilization":
            if "Temperature" in parameter:
                # Utilization improves with temperature to a point then plateaus
                x = np.array(param_values)
                base = 0.7 * (1 - np.exp(-0.005 * x))
            elif "Pressure" in parameter:
                # Utilization decreases with pressure
                x = np.array(param_values)
                base = 0.8 - 0.4 * x / max(x)
            else:
                # Default - diminishing returns
                x = np.array(param_values)
                base = 0.3 + 0.5 * np.log(1 + x / min(x))
        
        elif metric == "Energy Efficiency":
            if "Temperature" in parameter:
                # Energy efficiency decreases with temperature
                x = np.array(param_values)
                base = 0.9 - 0.5 * (x - min(x)) / (max(x) - min(x))
            elif "Thermal Conductivity" in parameter:
                # Energy efficiency improves with thermal conductivity
                x = np.array(param_values)
                base = 0.4 + 0.5 * (x - min(x)) / (max(x) - min(x))
            else:
                # Default - slight curve
                x = np.array(param_values)
                base = 0.5 + 0.3 * np.sin(np.pi * (x - min(x)) / (max(x) - min(x)))
        
        else:
            # Default behavior - linear relationship
            x = np.array(param_values)
            base = 0.2 + 0.6 * (x - min(x)) / (max(x) - min(x))
        
        # Add some noise
        noise = 0.03 * np.random.randn(len(param_values))
        
        # Ensure values are reasonable (between 0 and 1 for most metrics)
        values = np.clip(base + noise, 0.01, 0.99)
        
        # Store results
        results[metric] = values.tolist()
    
    return results