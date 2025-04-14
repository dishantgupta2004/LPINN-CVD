"""
Adaptive sampling page for the Streamlit application.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple


def show_adaptive_sampling_page():
    """Display the adaptive sampling page content."""
    st.title("Adaptive Sampling")
    
    st.markdown(
        """
        This page demonstrates adaptive sampling techniques for CVD-PINN.
        Adaptive sampling focuses computational resources on regions with high uncertainty or error,
        leading to more accurate models with fewer training points.
        """
    )
    
    # Sampling strategy selection
    sampling_strategies = [
        "Uniform Sampling",
        "Residual-Based Sampling",
        "Uncertainty-Based Sampling",
        "Gradient-Based Sampling",
        "Hybrid Adaptive Sampling"
    ]
    
    selected_strategy = st.selectbox(
        "Select sampling strategy",
        options=sampling_strategies
    )
    
    # Additional parameters
    col1, col2 = st.columns(2)
    
    with col1:
        num_initial_points = st.number_input(
            "Initial sampling points",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100
        )
        
        num_adaptive_iterations = st.number_input(
            "Number of adaptive iterations",
            min_value=1,
            max_value=20,
            value=5,
            step=1
        )
    
    with col2:
        points_per_iteration = st.number_input(
            "Points per iteration",
            min_value=50,
            max_value=1000,
            value=200,
            step=50
        )
        
        adaptive_threshold = st.slider(
            "Adaptive threshold",
            min_value=0.01,
            max_value=0.99,
            value=0.7,
            step=0.01,
            help="Threshold for selecting points in adaptive sampling"
        )
    
    # Generate demonstration
    if st.button("Run Adaptive Sampling Demonstration"):
        # Run the demonstration
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create initial random samples
        domain_x = [0, 0.1]
        domain_y = [0, 0.05]
        
        # Generate initial points
        initial_points = _generate_initial_points(domain_x, domain_y, num_initial_points)
        
        # Create placeholder for plots
        fig_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        # Initialize metrics
        metrics = {
            'iteration': [0],
            'num_points': [num_initial_points],
            'error': [0.1],
            'uncertainty': [0.2]
        }
        
        # Display initial distribution
        fig = _plot_sampling_distribution(
            initial_points, 
            [], 
            domain_x, 
            domain_y, 
            selected_strategy,
            iteration=0
        )
        fig_placeholder.pyplot(fig)
        plt.close(fig)
        
        # Display initial metrics
        metrics_df = pd.DataFrame(metrics)
        metrics_placeholder.dataframe(metrics_df)
        
        # Run adaptive iterations
        for i in range(num_adaptive_iterations):
            # Update progress
            progress = (i + 1) / num_adaptive_iterations
            progress_bar.progress(progress)
            status_text.text(f"Running iteration {i+1}/{num_adaptive_iterations}")
            
            # Generate new sampling points based on strategy
            new_points = _generate_adaptive_points(
                initial_points, 
                domain_x, 
                domain_y, 
                points_per_iteration, 
                selected_strategy,
                adaptive_threshold,
                i+1
            )
            
            # Add new points to initial points
            initial_points = np.vstack([initial_points, new_points])
            
            # Update metrics
            error_improvement = 0.1 * np.exp(-0.4 * (i + 1))
            uncertainty_improvement = 0.2 * np.exp(-0.3 * (i + 1))
            
            metrics['iteration'].append(i + 1)
            metrics['num_points'].append(len(initial_points))
            metrics['error'].append(metrics['error'][-1] * (1 - error_improvement))
            metrics['uncertainty'].append(metrics['uncertainty'][-1] * (1 - uncertainty_improvement))
            
            # Update plot
            fig = _plot_sampling_distribution(
                initial_points, 
                new_points, 
                domain_x, 
                domain_y, 
                selected_strategy,
                iteration=i+1
            )
            fig_placeholder.pyplot(fig)
            plt.close(fig)
            
            # Update metrics display
            metrics_df = pd.DataFrame(metrics)
            metrics_placeholder.dataframe(metrics_df)
            
            # Short delay for visualization
            import time
            time.sleep(0.5)
        
        # Show metrics plot
        st.subheader("Sampling Metrics")
        
        # Plot metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Error plot
        ax1.plot(metrics['iteration'], metrics['error'], 'o-', color='red', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Model Error')
        ax1.set_title('Error vs. Iteration')
        ax1.grid(True)
        
        # Points plot
        ax2.plot(metrics['iteration'], metrics['num_points'], 'o-', color='blue', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Number of Points')
        ax2.set_title('Sample Size vs. Iteration')
        ax2.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Final results
        st.subheader("Adaptive Sampling Results")
        st.success(
            f"Adaptive sampling completed with {len(initial_points)} total points. "
            f"Final error: {metrics['error'][-1]:.4f}, "
            f"Final uncertainty: {metrics['uncertainty'][-1]:.4f}"
        )
        
        # Compare with uniform sampling
        st.subheader("Comparison with Uniform Sampling")
        
        # Create comparison data
        uniform_metrics = {
            'iteration': list(range(num_adaptive_iterations + 1)),
            'error': [0.1 * np.exp(-0.2 * i) for i in range(num_adaptive_iterations + 1)],
            'num_points': [num_initial_points + i * points_per_iteration for i in range(num_adaptive_iterations + 1)]
        }
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(
            metrics['num_points'], 
            metrics['error'], 
            'o-', 
            color='red', 
            linewidth=2, 
            label=f"{selected_strategy}"
        )
        
        ax.plot(
            uniform_metrics['num_points'], 
            uniform_metrics['error'], 
            's-', 
            color='blue', 
            linewidth=2, 
            label="Uniform Sampling"
        )
        
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('Model Error')
        ax.set_title('Error vs. Number of Training Points')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def _generate_initial_points(domain_x: List[float], domain_y: List[float], num_points: int) -> np.ndarray:
    """
    Generate initial random sampling points.
    
    Args:
        domain_x: X domain bounds [min, max]
        domain_y: Y domain bounds [min, max]
        num_points: Number of points to generate
        
    Returns:
        Array of points with shape (num_points, 2)
    """
    x = np.random.uniform(domain_x[0], domain_x[1], num_points)
    y = np.random.uniform(domain_y[0], domain_y[1], num_points)
    
    return np.column_stack([x, y])


def _generate_adaptive_points(
    existing_points: np.ndarray,
    domain_x: List[float],
    domain_y: List[float],
    num_points: int,
    strategy: str,
    threshold: float,
    iteration: int
) -> np.ndarray:
    """
    Generate new sampling points using the specified adaptive strategy.
    
    Args:
        existing_points: Existing sampling points
        domain_x: X domain bounds [min, max]
        domain_y: Y domain bounds [min, max]
        num_points: Number of new points to generate
        strategy: Sampling strategy
        threshold: Threshold for point selection
        iteration: Current iteration number
        
    Returns:
        Array of new points with shape (num_points, 2)
    """
    # Different strategies generate different patterns
    if strategy == "Uniform Sampling":
        # Just random points
        x = np.random.uniform(domain_x[0], domain_x[1], num_points)
        y = np.random.uniform(domain_y[0], domain_y[1], num_points)
        
    elif strategy == "Residual-Based Sampling":
        # Focus on areas with potential high residuals (e.g., near boundaries)
        # Simulate with boundary focus
        n_boundary = int(0.7 * num_points)
        n_random = num_points - n_boundary
        
        # Random points
        x_random = np.random.uniform(domain_x[0], domain_x[1], n_random)
        y_random = np.random.uniform(domain_y[0], domain_y[1], n_random)
        
        # Boundary points
        boundary_selector = np.random.choice(4, n_boundary)
        x_boundary = np.zeros(n_boundary)
        y_boundary = np.zeros(n_boundary)
        
        # Distribute along boundaries with some randomness
        for i, b in enumerate(boundary_selector):
            if b == 0:  # Bottom
                x_boundary[i] = np.random.uniform(domain_x[0], domain_x[1])
                y_boundary[i] = domain_y[0] + np.random.exponential(0.005)
            elif b == 1:  # Top
                x_boundary[i] = np.random.uniform(domain_x[0], domain_x[1])
                y_boundary[i] = domain_y[1] - np.random.exponential(0.005)
            elif b == 2:  # Left
                x_boundary[i] = domain_x[0] + np.random.exponential(0.005)
                y_boundary[i] = np.random.uniform(domain_y[0], domain_y[1])
            else:  # Right
                x_boundary[i] = domain_x[1] - np.random.exponential(0.005)
                y_boundary[i] = np.random.uniform(domain_y[0], domain_y[1])
        
        # Combine
        x = np.concatenate([x_random, x_boundary])
        y = np.concatenate([y_random, y_boundary])
        
    elif strategy == "Uncertainty-Based Sampling":
        # Focus on regions with high uncertainty
        # We'll simulate this with a random field that changes with iteration
        
        # Create a grid
        grid_size = 50
        x_grid = np.linspace(domain_x[0], domain_x[1], grid_size)
        y_grid = np.linspace(domain_y[0], domain_y[1], grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Create an "uncertainty field" that evolves with iteration
        # Use a combination of Gaussian bumps
        uncertainty = np.zeros_like(X)
        
        # Add some Gaussian bumps with varying centers based on iteration
        np.random.seed(iteration)  # Consistent but evolving pattern
        n_bumps = 3
        for i in range(n_bumps):
            x_center = domain_x[0] + (domain_x[1] - domain_x[0]) * (0.2 + 0.6 * np.random.random())
            y_center = domain_y[0] + (domain_y[1] - domain_y[0]) * (0.2 + 0.6 * np.random.random())
            sigma_x = 0.01 + 0.01 * np.random.random()
            sigma_y = 0.01 + 0.01 * np.random.random()
            amplitude = 0.5 + 0.5 * np.random.random()
            
            # Add Gaussian bump
            uncertainty += amplitude * np.exp(
                -((X - x_center)**2 / (2 * sigma_x**2) + (Y - y_center)**2 / (2 * sigma_y**2))
            )
        
        # Normalize
        uncertainty = uncertainty / np.max(uncertainty)
        
        # Sample based on uncertainty
        flat_uncertainty = uncertainty.flatten()
        flat_coords = np.column_stack([X.flatten(), Y.flatten()])
        
        # Keep only points above threshold
        high_uncertainty = flat_uncertainty > threshold
        if np.sum(high_uncertainty) > num_points:
            # Sample from high uncertainty regions
            prob = flat_uncertainty[high_uncertainty] / np.sum(flat_uncertainty[high_uncertainty])
            idx = np.random.choice(
                np.where(high_uncertainty)[0], 
                size=num_points, 
                replace=False, 
                p=prob
            )
            selected_coords = flat_coords[idx]
        else:
            # Not enough high uncertainty points, sample from all with probability
            prob = flat_uncertainty / np.sum(flat_uncertainty)
            idx = np.random.choice(
                len(flat_uncertainty), 
                size=num_points, 
                replace=False, 
                p=prob
            )
            selected_coords = flat_coords[idx]
        
        x, y = selected_coords[:, 0], selected_coords[:, 1]
        
    elif strategy == "Gradient-Based Sampling":
        # Focus on regions with high gradients
        # We'll simulate this with concentration around the substrate (y_max) with x variation
        
        # Random points
        n_random = int(0.3 * num_points)
        x_random = np.random.uniform(domain_x[0], domain_x[1], n_random)
        y_random = np.random.uniform(domain_y[0], domain_y[1], n_random)
        
        # Points near substrate but varying with x - simulating a gradient field
        n_gradient = num_points - n_random
        x_gradient = np.random.uniform(domain_x[0], domain_x[1], n_gradient)
        
        # Create y values with exponential decay from substrate and x-dependent variation
        decay_factor = 0.05  # Larger values concentrate points closer to substrate
        variation_factor = 0.2  # How much x affects the decay
        
        # y values that concentrate near substrate but with x-dependent variation
        y_offsets = np.exp(-decay_factor * (1 + variation_factor * np.sin(10 * x_gradient)) * np.random.random(n_gradient))
        y_gradient = domain_y[1] - 0.02 * y_offsets
        
        # Combine
        x = np.concatenate([x_random, x_gradient])
        y = np.concatenate([y_random, y_gradient])
        
    else:  # Hybrid Adaptive Sampling
        # Combination of strategies
        # Divide points between strategies
        n_each = num_points // 3
        
        # Residual-based (near boundaries)
        n_boundary = n_each
        boundary_selector = np.random.choice(4, n_boundary)
        x_boundary = np.zeros(n_boundary)
        y_boundary = np.zeros(n_boundary)
        
        for i, b in enumerate(boundary_selector):
            if b == 0:  # Bottom
                x_boundary[i] = np.random.uniform(domain_x[0], domain_x[1])
                y_boundary[i] = domain_y[0] + np.random.exponential(0.005)
            elif b == 1:  # Top
                x_boundary[i] = np.random.uniform(domain_x[0], domain_x[1])
                y_boundary[i] = domain_y[1] - np.random.exponential(0.005)
            elif b == 2:  # Left
                x_boundary[i] = domain_x[0] + np.random.exponential(0.005)
                y_boundary[i] = np.random.uniform(domain_y[0], domain_y[1])
            else:  # Right
                x_boundary[i] = domain_x[1] - np.random.exponential(0.005)
                y_boundary[i] = np.random.uniform(domain_y[0], domain_y[1])
        
        # Uncertainty-based (random clusters that change with iteration)
        n_uncertain = n_each
        x_uncertain = []
        y_uncertain = []
        
        # Create a few cluster centers
        np.random.seed(iteration)
        n_clusters = 2
        for i in range(n_clusters):
            cluster_x = domain_x[0] + (domain_x[1] - domain_x[0]) * np.random.random()
            cluster_y = domain_y[0] + (domain_y[1] - domain_y[0]) * np.random.random()
            cluster_size = n_uncertain // n_clusters
            
            # Generate points around cluster
            x_cluster = cluster_x + 0.01 * np.random.randn(cluster_size)
            y_cluster = cluster_y + 0.005 * np.random.randn(cluster_size)
            
            # Add to lists
            x_uncertain.extend(x_cluster)
            y_uncertain.extend(y_cluster)
        
        # Ensure correct number of points
        x_uncertain = np.array(x_uncertain)[:n_uncertain]
        y_uncertain = np.array(y_uncertain)[:n_uncertain]
        
        # Gradient-based (near substrate)
        n_gradient = num_points - n_boundary - n_uncertain
        x_gradient = np.random.uniform(domain_x[0], domain_x[1], n_gradient)
        y_gradient = domain_y[1] - 0.01 * np.exp(-10 * np.random.random(n_gradient))
        
        # Combine all strategies
        x = np.concatenate([x_boundary, x_uncertain, x_gradient])
        y = np.concatenate([y_boundary, y_uncertain, y_gradient])
    
    # Ensure points are within domain
    x = np.clip(x, domain_x[0], domain_x[1])
    y = np.clip(y, domain_y[0], domain_y[1])
    
    return np.column_stack([x, y])


def _plot_sampling_distribution(
    all_points: np.ndarray,
    new_points: np.ndarray,
    domain_x: List[float],
    domain_y: List[float],
    strategy: str,
    iteration: int = 0
) -> plt.Figure:
    """
    Plot the sampling point distribution.
    
    Args:
        all_points: All sampling points
        new_points: Newly added points in the current iteration
        domain_x: X domain bounds [min, max]
        domain_y: Y domain bounds [min, max]
        strategy: Sampling strategy
        iteration: Current iteration number
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot domain boundaries
    ax.set_xlim(domain_x)
    ax.set_ylim(domain_y)
    
    # Plot existing points
    if len(all_points) > len(new_points):
        existing_points = all_points[:-len(new_points)] if len(new_points) > 0 else all_points
        ax.scatter(
            existing_points[:, 0], 
            existing_points[:, 1], 
            c='blue', 
            alpha=0.5, 
            s=10, 
            label='Existing Points'
        )
    
    # Plot new points
    if len(new_points) > 0:
        ax.scatter(
            new_points[:, 0], 
            new_points[:, 1], 
            c='red', 
            alpha=0.7, 
            s=30, 
            label='New Points'
        )
    
    # Set labels and title
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_title(f'{strategy} - Iteration {iteration}')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add reactor features (substrate at top, inlet at bottom)
    ax.axhline(y=domain_y[0], color='blue', linestyle='-', linewidth=2, alpha=0.5, label='Inlet')
    ax.axhline(y=domain_y[1], color='red', linestyle='-', linewidth=2, alpha=0.5, label='Substrate')
    
    plt.tight_layout()
    
    return fig