"""
Main Streamlit application for CVD-PINN.

This module serves as the entry point for the Streamlit application.
"""
import streamlit as st
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to import the package
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the app components
from app.components.sidebar import create_sidebar
from app.pages.home import show_home_page
from app.pages.simulation import show_simulation_page
from app.pages.model_comparison import show_model_comparison_page
from app.pages.parameter_analysis import show_parameter_analysis_page
from app.pages.adaptive_sampling import show_adaptive_sampling_page


def main():
    """Main function for the Streamlit app."""
    st.set_page_config(
        page_title="CVD-PINN Simulator",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Create sidebar
    page = create_sidebar()

    # Display selected page
    if page == "Home":
        show_home_page()
    elif page == "Simulation":
        show_simulation_page()
    elif page == "Model Comparison":
        show_model_comparison_page()
    elif page == "Parameter Analysis":
        show_parameter_analysis_page()
    elif page == "Adaptive Sampling":
        show_adaptive_sampling_page()


if __name__ == "__main__":
    main()