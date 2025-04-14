"""
Sidebar component for the Streamlit application.
"""
import streamlit as st
import sys
from pathlib import Path

# Add the parent directory to sys.path to import the package
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import cvd_pinn
from src import __version__


def create_sidebar():
    """
    Create the sidebar for the Streamlit app.
    
    Returns:
        Selected page name
    """
    with st.sidebar:
        st.title("CVD-PINN Simulator")
        st.caption(f"Version {__version__}")
        
        st.markdown("---")
        
        # Navigation
        st.subheader("Navigation")
        page = st.radio(
            "Select a page:",
            options=["Home", "Simulation", "Model Comparison", "Parameter Analysis", "Adaptive Sampling"],
            index=0,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Information
        st.markdown(
            """
            ### About
            
            This application simulates Chemical Vapor Deposition (CVD) processes using 
            Physics-Informed Neural Networks (PINNs) with Entropy-Langevin dynamics.
            
            - **GitHub**: [CVD-PINN](https://github.com/dishantgupta2004/LPINN-CVD)
            """
        )
        
        # Footer
        st.markdown("---")
        st.caption("© 2025 | Dishant Gupta")
    
    return page