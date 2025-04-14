"""
Home page for the Streamlit application.
"""
import streamlit as st


def show_home_page():
    """Display the home page content."""
    st.title("Physics-Informed Neural Networks for CVD Modeling")
    
    st.markdown(
        """
        ## Welcome to CVD-PINN Simulator
        
        This application provides tools for modeling Chemical Vapor Deposition (CVD) processes 
        using Physics-Informed Neural Networks (PINNs) with an Entropy-Langevin optimization approach.
        
        ### Key Features
        
        - **Physics-Based Modeling**: Simulate CVD processes using fundamental physical laws
        - **Uncertainty Quantification**: Quantify uncertainty in predictions using ensemble methods
        - **Entropy-Langevin Optimization**: Improved convergence and robustness for stiff PDEs
        - **Adaptive Sampling**: Efficiently sample the domain to capture complex phenomena
        - **Parameter Analysis**: Explore the impact of different process parameters

        ### About CVD
        
        Chemical Vapor Deposition is a process used to produce high-quality, high-performance solid materials.
        The process involves chemical reactions of gas-phase precursors on a heated substrate to form a solid deposit.
        CVD is widely used in the semiconductor industry to produce thin films.
        
        ### About PINNs
        
        Physics-Informed Neural Networks (PINNs) are neural networks that are trained to solve supervised learning tasks
        while respecting physical laws described by general nonlinear partial differential equations (PDEs).
        PINNs incorporate physical laws as a regularization mechanism during training.
        
        ### Entropy-Langevin Approach
        
        The Entropy-Langevin approach enhances PINN training by:
        
        1. Using stochastic Langevin dynamics to explore complex loss landscapes
        2. Incorporating entropy-based regularization to balance exploration and exploitation
        3. Employing an ensemble of models to quantify prediction uncertainty
        
        ### Getting Started
        
        Use the sidebar to navigate to different sections of the application:
        
        - **Simulation**: Run CVD simulations with custom parameters
        - **Model Comparison**: Compare different PINN models and approaches
        - **Parameter Analysis**: Analyze the effect of process parameters
        - **Adaptive Sampling**: Explore adaptive sampling techniques
        """
    )
    
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Chemical_vapor_deposition.svg/1200px-Chemical_vapor_deposition.svg.png", 
             caption="Schematic representation of a CVD process",
             use_column_width=True)