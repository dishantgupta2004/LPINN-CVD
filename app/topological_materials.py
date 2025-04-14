"""
Topological Materials Module for CVD Simulation App

This module provides a Streamlit interface for exploring topological insulator materials
relevant to quantum computing applications and connects them to CVD simulation results.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional

# Database of topological insulator materials
TOPOLOGICAL_MATERIALS = {
    "Bi2Te3": {
        "full_name": "Bismuth Telluride",
        "formula": "Bi₂Te₃",
        "z2_indices": [1, 0, 0, 0],  # (ν₀; ν₁ν₂ν₃)
        "chern_number": 0,
        "band_gap": 0.15,  # eV
        "band_gap_temp_coef": -4.5e-4,  # eV/K
        "critical_thickness": 5.0,  # nm
        "optimal_thickness_range": (6.0, 15.0),  # nm
        "lattice_constant_a": 4.38,  # Å
        "lattice_constant_c": 30.50,  # Å
        "dirac_velocity": 4.0e5,  # m/s
        "qubit_suitability": "High",
        "debye_temperature": 155,  # K
        "notes": "Strong 3D TI with relatively large band gap. Surface states protected against backscattering.",
        "deposition_methods": ["MBE", "PVD", "CVD", "Sputtering"],
        "optimal_deposition_temp": 573,  # K
    },
    "Bi2Se3": {
        "full_name": "Bismuth Selenide",
        "formula": "Bi₂Se₃",
        "z2_indices": [1, 0, 0, 0],
        "chern_number": 0,
        "band_gap": 0.3,  # eV
        "band_gap_temp_coef": -3.8e-4,  # eV/K
        "critical_thickness": 3.0,  # nm
        "optimal_thickness_range": (4.0, 12.0),  # nm
        "lattice_constant_a": 4.14,  # Å
        "lattice_constant_c": 28.64,  # Å
        "dirac_velocity": 5.0e5,  # m/s
        "qubit_suitability": "High",
        "debye_temperature": 182,  # K
        "notes": "3D TI with larger band gap than Bi2Te3. Good candidate for room temperature applications.",
        "deposition_methods": ["MBE", "CVD", "Sputtering", "PLD"],
        "optimal_deposition_temp": 523,  # K
    },
    "Sb2Te3": {
        "full_name": "Antimony Telluride",
        "formula": "Sb₂Te₃",
        "z2_indices": [1, 0, 0, 0],
        "chern_number": 0,
        "band_gap": 0.2,  # eV
        "band_gap_temp_coef": -5.0e-4,  # eV/K
        "critical_thickness": 4.0,  # nm
        "optimal_thickness_range": (5.0, 14.0),  # nm
        "lattice_constant_a": 4.25,  # Å
        "lattice_constant_c": 30.35,  # Å
        "dirac_velocity": 3.8e5,  # m/s
        "qubit_suitability": "Medium",
        "debye_temperature": 160,  # K
        "notes": "Similar structure to Bi2Te3 but with weaker spin-orbit coupling.",
        "deposition_methods": ["MBE", "CVD", "Sputtering"],
        "optimal_deposition_temp": 553,  # K
    },
    "Bi0.9Sb0.1": {
        "full_name": "Bismuth Antimony Alloy",
        "formula": "Bi₀.₉Sb₀.₁",
        "z2_indices": [1, 1, 1, 1],
        "chern_number": 0,
        "band_gap": 0.03,  # eV
        "band_gap_temp_coef": -2.0e-4,  # eV/K
        "critical_thickness": 8.0,  # nm
        "optimal_thickness_range": (10.0, 25.0),  # nm
        "lattice_constant_a": 4.53,  # Å
        "lattice_constant_c": None,  # Not applicable
        "dirac_velocity": 6.0e5,  # m/s
        "qubit_suitability": "Medium",
        "debye_temperature": 140,  # K
        "notes": "First experimentally confirmed 3D TI. Has complex surface states.",
        "deposition_methods": ["MBE", "PVD"],
        "optimal_deposition_temp": 473,  # K
    },
    "HgTe": {
        "full_name": "Mercury Telluride",
        "formula": "HgTe",
        "z2_indices": [1, 0, 0, 0],
        "chern_number": 0,
        "band_gap": 0.0,  # eV (nominally zero gap)
        "band_gap_temp_coef": 0.0,  # eV/K
        "critical_thickness": 6.3,  # nm (critical QW thickness)
        "optimal_thickness_range": (7.0, 10.0),  # nm
        "lattice_constant_a": 6.46,  # Å
        "lattice_constant_c": None,  # Not applicable
        "dirac_velocity": 5.5e5,  # m/s
        "qubit_suitability": "Medium-High",
        "debye_temperature": 145,  # K
        "notes": "Quantum wells of HgTe/CdTe can form 2D TIs with quantum spin Hall effect.",
        "deposition_methods": ["MBE"],
        "optimal_deposition_temp": 453,  # K
    },
    "WTe2": {
        "full_name": "Tungsten Ditelluride",
        "formula": "WTe₂",
        "z2_indices": [0, 0, 0, 1],
        "chern_number": 0,
        "band_gap": 0.055,  # eV (indirect)
        "band_gap_temp_coef": -3.0e-4,  # eV/K
        "critical_thickness": 2.0,  # nm (monolayer)
        "optimal_thickness_range": (1.0, 5.0),  # nm
        "lattice_constant_a": 3.48,  # Å
        "lattice_constant_c": 14.07,  # Å
        "dirac_velocity": 3.0e5,  # m/s
        "qubit_suitability": "Medium",
        "debye_temperature": 250,  # K
        "notes": "2D TI in monolayer form. Also shows interesting Weyl semimetal behavior in bulk.",
        "deposition_methods": ["CVD", "MBE", "Exfoliation"],
        "optimal_deposition_temp": 823,  # K
    },
    "SmB6": {
        "full_name": "Samarium Hexaboride",
        "formula": "SmB₆",
        "z2_indices": [1, 0, 0, 0],
        "chern_number": 0,
        "band_gap": 0.02,  # eV
        "band_gap_temp_coef": -1.5e-4,  # eV/K
        "critical_thickness": 10.0,  # nm
        "optimal_thickness_range": (12.0, 30.0),  # nm
        "lattice_constant_a": 4.13,  # Å
        "lattice_constant_c": None,  # Not applicable (cubic)
        "dirac_velocity": 3.0e5,  # m/s
        "qubit_suitability": "Medium",
        "debye_temperature": 373,  # K
        "notes": "Topological Kondo insulator. Exhibits unusual surface conductivity at low temperatures.",
        "deposition_methods": ["MBE", "PLD"],
        "optimal_deposition_temp": 1073,  # K
    }
}


def get_material_info(material_name: str) -> Dict:
    """
    Get information about a topological insulator material.
    
    Args:
        material_name: Name of the material
        
    Returns:
        Dictionary with material properties
    """
    return TOPOLOGICAL_MATERIALS.get(material_name, {})


def is_topologically_nontrivial(material_name: str) -> bool:
    """
    Check if a material is topologically nontrivial.
    
    Args:
        material_name: Name of the material
        
    Returns:
        True if topologically nontrivial, False otherwise
    """
    material = get_material_info(material_name)
    z2_indices = material.get("z2_indices", [0, 0, 0, 0])
    chern_number = material.get("chern_number", 0)
    
    # Strong TI has ν₀ = 1, or non-zero Chern number
    return z2_indices[0] == 1 or chern_number != 0


def evaluate_thickness_suitability(material_name: str, thickness_nm: float) -> Tuple[str, float]:
    """
    Evaluate the suitability of a given film thickness for topological properties.
    
    Args:
        material_name: Name of the material
        thickness_nm: Film thickness in nanometers
        
    Returns:
        Tuple of (suitability_level, suitability_score)
        where suitability_level is one of "Poor", "Moderate", "Good", "Excellent"
        and suitability_score is between 0.0 and 1.0
    """
    material = get_material_info(material_name)
    
    critical_thickness = material.get("critical_thickness", 0.0)
    optimal_range = material.get("optimal_thickness_range", (0.0, 0.0))
    
    if thickness_nm < critical_thickness:
        return "Poor", 0.0
    
    # Calculate suitability score
    if thickness_nm < optimal_range[0]:
        # Linear interpolation between critical thickness and optimal minimum
        score = (thickness_nm - critical_thickness) / (optimal_range[0] - critical_thickness)
        level = "Moderate"
    elif thickness_nm <= optimal_range[1]:
        # Within optimal range
        score = 1.0
        level = "Excellent"
    else:
        # Beyond optimal range, decay exponentially
        decay_factor = 0.1  # Controls how quickly suitability decays
        excess = thickness_nm - optimal_range[1]
        score = np.exp(-decay_factor * excess)
        
        if score > 0.7:
            level = "Good"
        elif score > 0.4:
            level = "Moderate"
        else:
            level = "Poor"
    
    return level, min(max(score, 0.0), 1.0)


def get_temperature_effect(material_name: str, temperature_K: float) -> Dict:
    """
    Get the effects of temperature on material properties.
    
    Args:
        material_name: Name of the material
        temperature_K: Temperature in Kelvin
        
    Returns:
        Dictionary with temperature effects
    """
    material = get_material_info(material_name)
    
    # Adjust band gap based on temperature
    base_band_gap = material.get("band_gap", 0.0)
    temp_coef = material.get("band_gap_temp_coef", 0.0)
    adjusted_band_gap = base_band_gap + temp_coef * (temperature_K - 300)  # Adjust from room temp
    
    # Temperature factor for quantum applications (normalized to Debye temperature)
    debye_temp = material.get("debye_temperature", 200)
    temp_ratio = min(debye_temp / max(temperature_K, 1.0), 5.0)
    temp_factor = min(np.tanh(temp_ratio - 0.5) + 0.5, 1.0)
    
    temp_suitability = "Excellent" if temp_factor > 0.9 else \
                      "Good" if temp_factor > 0.7 else \
                      "Moderate" if temp_factor > 0.4 else "Poor"
    
    return {
        "adjusted_band_gap": max(adjusted_band_gap, 0.0),  # Ensure non-negative
        "temperature_factor": temp_factor,
        "temperature_suitability": temp_suitability,
        "debye_temperature": debye_temp
    }


def calculate_qubit_potential(material_name: str, thickness_nm: float, temp_K: float) -> Dict:
    """
    Calculate the potential for topological qubit applications.
    
    Args:
        material_name: Name of the material
        thickness_nm: Film thickness in nanometers
        temp_K: Operating temperature in Kelvin
        
    Returns:
        Dictionary with qubit potential metrics
    """
    material = get_material_info(material_name)
    
    # Base qubit suitability
    suitability_map = {
        "Low": 0.3,
        "Medium": 0.6,
        "Medium-High": 0.8,
        "High": 1.0
    }
    base_suitability = suitability_map.get(material.get("qubit_suitability", "Low"), 0.3)
    
    # Thickness suitability
    _, thickness_factor = evaluate_thickness_suitability(material_name, thickness_nm)
    
    # Temperature effects
    temp_effects = get_temperature_effect(material_name, temp_K)
    temp_factor = temp_effects["temperature_factor"]
    
    # Band gap effects on coherence
    band_gap = temp_effects["adjusted_band_gap"]
    band_gap_factor = min(band_gap / 0.1, 1.0) if band_gap > 0 else 0.1
    
    # Calculate overall qubit potential
    potential_score = base_suitability * thickness_factor * temp_factor * band_gap_factor
    
    # Interpret score
    if potential_score > 0.8:
        potential_level = "Excellent"
    elif potential_score > 0.6:
        potential_level = "Good"
    elif potential_score > 0.3:
        potential_level = "Moderate"
    else:
        potential_level = "Poor"
    
    # Coherence time estimate (very approximate, for illustration)
    # Based on band gap, temperature, and material suitability
    base_coherence_ns = 100.0  # base coherence in nanoseconds
    coherence_estimate = base_coherence_ns * potential_score * (band_gap / 0.1) * np.exp(-temp_K / material.get("debye_temperature", 200))
    
    return {
        "potential_score": potential_score,
        "potential_level": potential_level,
        "estimated_coherence_ns": min(coherence_estimate, 1000.0),  # Cap at 1 microsecond
        "factors": {
            "material_suitability": base_suitability,
            "thickness_factor": thickness_factor,
            "temperature_factor": temp_factor,
            "band_gap_factor": band_gap_factor
        }
    }


def plot_band_structure(material_name: str, ax=None) -> plt.Figure:
    """
    Plot a schematic band structure for the material.
    
    Args:
        material_name: Name of the material
        ax: Optional matplotlib axis to plot on
        
    Returns:
        Matplotlib figure
    """
    material = get_material_info(material_name)
    band_gap = material.get("band_gap", 0.0)
    z2_indices = material.get("z2_indices", [0, 0, 0, 0])
    
    is_strong_ti = z2_indices[0] == 1
    
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # k points
    k = np.linspace(-np.pi, np.pi, 1000)
    
    # Band structure
    if is_strong_ti:
        # For strong TI, show linear Dirac cone
        conduction = 0.5 * band_gap + np.sqrt((0.1 * k)**2 + (0.5 * band_gap)**2)
        valence = -0.5 * band_gap - np.sqrt((0.1 * k)**2 + (0.5 * band_gap)**2)
        surface = 0.8 * k  # Linear surface states
    else:
        # For trivial insulator or weak TI
        conduction = 0.5 * band_gap + 0.1 * k**2
        valence = -0.5 * band_gap - 0.1 * k**2
        surface = np.zeros_like(k)  # No surface states
    
    # Add gap at Γ point for weak TI
    if not is_strong_ti and any(z2_indices[1:]):
        idx = (k > -0.3) & (k < 0.3)
        surface[idx] = 0.3 * k[idx]
    
    # Plot bands
    ax.plot(k, conduction, 'b-', linewidth=2, label='Conduction Band')
    ax.plot(k, valence, 'b-', linewidth=2, label='Valence Band')
    
    # Plot surface states if topological
    if is_strong_ti or any(z2_indices[1:]):
        ax.plot(k, surface, 'r-', linewidth=2, label='Surface States')
    
    # Add zero lines
    ax.axhline(y=0, color='k', linestyle=':', alpha=0.7)
    ax.axvline(x=0, color='k', linestyle=':', alpha=0.7)
    
    # Labels
    high_symmetry_points = ['L', 'Γ', 'X']
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(high_symmetry_points)
    ax.set_xlabel('Wave Vector')
    ax.set_ylabel('Energy (eV)')
    ax.set_ylim(-0.8, 0.8)
    
    # Title
    if band_gap > 0:
        title = f"{material_name} - Band Gap: {band_gap:.2f} eV"
    else:
        title = f"{material_name} - Zero Gap"
    
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    return fig


def plot_thickness_dependence(material_name: str, current_thickness: Optional[float] = None) -> plt.Figure:
    """
    Plot the topological property dependence on film thickness.
    
    Args:
        material_name: Name of the material
        current_thickness: Current thickness to highlight
        
    Returns:
        Matplotlib figure
    """
    material = get_material_info(material_name)
    critical_thickness = material.get("critical_thickness", 0.0)
    optimal_range = material.get("optimal_thickness_range", (0.0, 0.0))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate thickness points
    thickness_range = np.linspace(0, 2 * optimal_range[1], 200)
    scores = []
    
    # Calculate suitability scores
    for t in thickness_range:
        _, score = evaluate_thickness_suitability(material_name, t)
        scores.append(score)
    
    # Plot suitability curve
    ax.plot(thickness_range, scores, 'b-', linewidth=2.5)
    
    # Add regions
    ax.axvspan(0, critical_thickness, alpha=0.2, color='red', label='Sub-Critical (Not Topological)')
    ax.axvspan(optimal_range[0], optimal_range[1], alpha=0.2, color='green', label='Optimal Range')
    
    # Add critical thickness line
    ax.axvline(x=critical_thickness, color='red', linestyle='--', 
               label=f'Critical Thickness: {critical_thickness} nm')
    
    # Add current thickness if provided
    if current_thickness is not None:
        ax.axvline(x=current_thickness, color='black', linestyle='-', linewidth=2,
                   label=f'Current Thickness: {current_thickness} nm')
    
    # Labels and title
    ax.set_xlabel('Film Thickness (nm)')
    ax.set_ylabel('Topological Suitability Score')
    ax.set_title(f'Thickness Dependence of Topological Properties - {material_name}')
    ax.set_xlim(0, thickness_range[-1])
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    return fig


def create_cvd_connection(mat_name: str, thickness: float, temperature: float) -> Dict:
    """
    Create connection between CVD simulation results and topological properties.
    
    Args:
        mat_name: Name of the material
        thickness: Film thickness from simulation (nm)
        temperature: Deposition temperature from simulation (K)
        
    Returns:
        Dictionary with recommendations and evaluations
    """
    material = get_material_info(mat_name)
    thickness_level, thickness_score = evaluate_thickness_suitability(mat_name, thickness)
    temp_effects = get_temperature_effect(mat_name, temperature)
    
    # Overall quality assessment
    overall_score = 0.7 * thickness_score + 0.3 * temp_effects["temperature_factor"]
    
    if overall_score > 0.8:
        quality_level = "Excellent"
    elif overall_score > 0.6:
        quality_level = "Good"
    elif overall_score > 0.4:
        quality_level = "Moderate"
    else:
        quality_level = "Poor"
    
    # Generate specific recommendations
    recommendations = []
    
    if thickness_score < 0.7:
        if thickness < material["critical_thickness"]:
            recommendations.append(f"Increase film thickness above critical value ({material['critical_thickness']} nm)")
        elif thickness < material["optimal_thickness_range"][0]:
            recommendations.append(f"Increase film thickness to optimal range ({material['optimal_thickness_range'][0]}-{material['optimal_thickness_range'][1]} nm)")
        elif thickness > material["optimal_thickness_range"][1]:
            recommendations.append(f"Decrease film thickness to optimal range ({material['optimal_thickness_range'][0]}-{material['optimal_thickness_range'][1]} nm)")
    
    temp_diff = abs(temperature - material["optimal_deposition_temp"])
    if temp_diff > 50:
        if temperature > material["optimal_deposition_temp"]:
            recommendations.append(f"Decrease deposition temperature to near {material['optimal_deposition_temp']} K")
        else:
            recommendations.append(f"Increase deposition temperature to near {material['optimal_deposition_temp']} K")
    
    return {
        "material": mat_name,
        "overall_quality": quality_level,
        "overall_score": overall_score,
        "thickness_quality": thickness_level,
        "thickness_score": thickness_score,
        "temperature_quality": temp_effects["temperature_suitability"],
        "temperature_score": temp_effects["temperature_factor"],
        "recommendations": recommendations,
        "is_topological": is_topologically_nontrivial(mat_name) and thickness >= material["critical_thickness"],
        "adjusted_band_gap": temp_effects["adjusted_band_gap"]
    }


def topological_materials_tab():
    """
    Main function to display the topological materials tab in the Streamlit app.
    """
    st.title("Topological Insulators for Quantum Computing")
    
    st.markdown("""
    Topological insulators (TIs) are a class of quantum materials with insulating bulk and 
    conducting surface states protected by time-reversal symmetry. These unique properties 
    make them promising candidates for topological quantum computing and other quantum applications.
    
    This module helps you understand the connection between CVD deposition parameters and 
    the topological properties relevant for quantum computing applications.
    """)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Material Properties", "CVD Integration", "Qubit Potential"])
    
    # Material selection (common across tabs)
    materials = list(TOPOLOGICAL_MATERIALS.keys())
    selected_material = st.sidebar.selectbox(
        "Select Topological Material",
        options=materials
    )
    
    material = get_material_info(selected_material)
    
    # Material Properties Tab
    with tab1:
        st.header("Topological Material Properties")
        
        # Material info section
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display material formula and image placeholder
            st.markdown(f"## {material['full_name']}")
            st.markdown(f"### {material['formula']}")
            
            # Placeholder for material structure image
            # In a real app, you would have actual crystal structure images
            st.image("https://via.placeholder.com/300x200?text=Crystal+Structure", 
                    caption=f"{material['formula']} crystal structure")
            
            # Topological status indicator
            is_topological = is_topologically_nontrivial(selected_material)
            if is_topological:
                st.success("✓ Strong Topological Insulator")
            else:
                st.error("✗ Trivial Insulator")
        
        with col2:
            # Material properties in table format
            st.subheader("Key Properties")
            
            # Create DataFrame for display
            props = {
                "Property": [
                    "Z₂ Indices (ν₀;ν₁ν₂ν₃)", 
                    "Chern Number", 
                    "Band Gap (eV)",
                    "Critical Thickness (nm)",
                    "Optimal Thickness Range (nm)",
                    "Lattice Constants (Å)",
                    "Dirac Velocity (m/s)",
                    "Qubit Suitability"
                ],
                "Value": [
                    str(material["z2_indices"]),
                    str(material["chern_number"]),
                    f"{material['band_gap']:.3f}",
                    f"{material['critical_thickness']:.1f}",
                    f"{material['optimal_thickness_range'][0]:.1f} - {material['optimal_thickness_range'][1]:.1f}",
                    f"a = {material['lattice_constant_a']:.2f}, c = {material.get('lattice_constant_c', 'N/A')}",
                    f"{material['dirac_velocity']:.1e}",
                    material["qubit_suitability"]
                ]
            }
            
            st.table(pd.DataFrame(props))
            
            # Material notes
            st.subheader("Notes")
            st.info(material["notes"])
        
        # Band structure plot
        st.subheader("Band Structure Visualization")
        band_fig = plot_band_structure(selected_material)
        st.pyplot(band_fig)
        
        with st.expander("About Topological Band Structure"):
            st.markdown("""
            ### Understanding Topological Band Structure
            
            The band structure shows the relationship between the energy of electron states and their momentum.
            In topological insulators:
            
            - **Bulk Bands**: The blue lines represent the bulk conduction and valence bands, separated by a band gap.
            - **Surface States**: The red line represents topologically protected surface states that cross the Fermi level (E=0).
            - **Dirac Point**: Where the surface states cross, forming a Dirac cone with linear dispersion.
            
            These surface states are protected by time-reversal symmetry and are robust against non-magnetic impurities,
            making them potentially useful for quantum computing applications.
            
            The **Z₂ indices** (ν₀;ν₁ν₂ν₃) classify the topological nature:
            - ν₀ = 1 indicates a **strong** topological insulator
            - ν₁, ν₂, ν₃ relate to weak topological indices
            
            The **Chern number** is another topological invariant, usually 0 for time-reversal invariant systems.
            """)
    
    # CVD Integration Tab
    with tab2:
        st.header("CVD Deposition Integration")
        
        st.markdown(f"""
        This section helps you optimize CVD deposition parameters for {selected_material} 
        to achieve the desired topological properties for quantum computing applications.
        """)
        
        # Get simulation parameters from session state or use defaults
        if 'cvd_sim_results' in st.session_state:
            sim_results = st.session_state.cvd_sim_results
            film_thickness = sim_results.get('thickness', material['optimal_thickness_range'][0])
            deposition_temp = sim_results.get('temperature', material['optimal_deposition_temp'])
        else:
            film_thickness = st.slider(
                "Film Thickness (nm)", 
                min_value=0.0, 
                max_value=30.0, 
                value=material['optimal_thickness_range'][0],
                step=0.1
            )
            
            deposition_temp = st.slider(
                "Deposition Temperature (K)",
                min_value=300,
                max_value=1200,
                value=material['optimal_deposition_temp'],
                step=10
            )
        
        # Analyze thickness suitability
        thickness_level, thickness_score = evaluate_thickness_suitability(selected_material, film_thickness)
        
        # Analyze temperature effects
        temp_effects = get_temperature_effect(selected_material, deposition_temp)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Thickness Analysis")
            
            # Create a styled progress bar
            st.markdown(f"**Suitability: {thickness_level}**")
            st.progress(thickness_score)
            
            # Additional thickness info
            st.markdown(f"""
            - **Critical Thickness:** {material['critical_thickness']} nm
            - **Optimal Range:** {material['optimal_thickness_range'][0]} - {material['optimal_thickness_range'][1]} nm
            - **Current Thickness:** {film_thickness} nm
            """)
            
            # Recommendations based on thickness
            if thickness_score < 0.3:
                st.error("Film is too thin or too thick for topological properties.")
                if film_thickness < material['critical_thickness']:
                    recommendation = f"Increase thickness above {material['critical_thickness']} nm"
                else:
                    recommendation = f"Reduce thickness to within {material['optimal_thickness_range'][0]} - {material['optimal_thickness_range'][1]} nm"
                st.markdown(f"**Recommendation:** {recommendation}")
            elif thickness_score > 0.9:
                st.success("Excellent thickness for topological properties!")
            else:
                st.warning(f"Consider adjusting thickness to {material['optimal_thickness_range'][0]} - {material['optimal_thickness_range'][1]} nm for optimal properties.")
        
        with col2:
            st.subheader("Temperature Analysis")
            
            # Temperature suitability bar
            st.markdown(f"**Temperature Suitability: {temp_effects['temperature_suitability']}**")
            st.progress(temp_effects['temperature_factor'])
            
            # Additional temperature info
            st.markdown(f"""
            - **Optimal Deposition Temp:** {material['optimal_deposition_temp']} K
            - **Current Temperature:** {deposition_temp} K
            - **Adjusted Band Gap:** {temp_effects['adjusted_band_gap']:.3f} eV
            """)
            
            # Temperature recommendations
            temp_diff = abs(deposition_temp - material['optimal_deposition_temp'])
            if temp_diff > 100:
                st.error(f"Temperature is significantly different from optimal ({temp_diff} K deviation).")
                st.markdown("**Recommendation:** Adjust temperature closer to optimal value.")
            elif temp_diff > 50:
                st.warning(f"Temperature deviation ({temp_diff} K) may affect film quality.")
                st.markdown("**Recommendation:** Consider adjusting temperature.")
            else:
                st.success("Temperature is near optimal for high-quality deposition.")
        
        # Thickness dependence plot
        st.subheader("Thickness Dependence of Topological Properties")
        thickness_fig = plot_thickness_dependence(selected_material, film_thickness)
        st.pyplot(thickness_fig)
        
        # CVD Parameter Recommendations
        st.subheader("CVD Process Parameter Recommendations")
        
        cvd_suitable = "CVD" in material["deposition_methods"]
        
        if cvd_suitable:
            st.success(f"{selected_material} is suitable for CVD deposition")
            
            # CVD parameter recommendations
            st.markdown(f"""
            ### Recommended CVD Parameters
            
            For optimal topological properties in {selected_material}, we recommend:
            
            - **Precursor Gas**: Metal-organic precursors for {material["formula"].split("₂")[0]}
            - **Carrier Gas**: High-purity Ar or H₂
            - **Substrate**: Si(111) or c-plane sapphire
            - **Chamber Pressure**: 0.1-1 Torr
            - **Gas Flow Rate**: 50-200 sccm
            - **Substrate Temperature**: {material["optimal_deposition_temp"]} K
            - **Deposition Time**: Adjust to achieve {material["optimal_thickness_range"][0]}-{material["optimal_thickness_range"][1]} nm thickness
            - **Post-deposition**: In-situ annealing at {int(material["optimal_deposition_temp"] * 0.8)} K under inert atmosphere
            """)
            
            # Connect to current simulation parameters
            if 'cvd_sim_results' in st.session_state:
                st.info("Your current simulation parameters can be optimized for this material. See recommendations above.")
        else:
            st.warning(f"{selected_material} is typically not deposited using CVD.")
            st.markdown(f"""
            ### Alternative Deposition Methods
            
            For {selected_material}, consider these methods instead:
            
            {", ".join(material["deposition_methods"])}
            """)
            
        # Display deposition integration tips
        with st.expander("Tips for High-Quality Topological Insulator Films"):
            st.markdown("""
            ### Critical Factors for Topological Properties
            
            1. **Crystalline Quality**: High crystallinity is essential for observing topological surface states.
               - Use low deposition rates
               - Consider epitaxial growth on lattice-matched substrates
               
            2. **Stoichiometry Control**: Precise control of composition is critical.
               - Monitor and adjust precursor flow rates carefully
               - Verify composition with post-deposition characterization (XPS, EDX)
               
            3. **Thickness Uniformity**: Non-uniform films may exhibit mixed topological behavior.
               - Ensure substrate rotation during deposition
               - Optimize gas flow dynamics in the chamber
               
            4. **Surface Protection**: Topological surface states are sensitive to oxidation and contamination.
               - Consider in-situ capping with inert materials
               - Minimize air exposure during transfer and handling
            """)