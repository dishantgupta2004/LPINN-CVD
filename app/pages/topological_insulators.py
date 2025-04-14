"""
Topological insulators page for the Streamlit application.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add the parent directory to sys.path to import the package
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the topological materials module
from src.analysis.topological_materials import (
    get_available_materials,
    get_material_info,
    evaluate_thickness_suitability,
    get_deposition_recommendations,
    calculate_qubit_potential,
    is_topologically_nontrivial
)


def show_topological_insulators_page():
    """Display the topological insulators page content."""
    st.title("Topological Insulators for Quantum Computing")
    
    st.markdown(
        """
        Topological insulators are a class of materials with insulating bulk and conducting surface states 
        protected by time-reversal symmetry. These unique properties make them promising candidates for 
        realizing fault-tolerant quantum computing through topological qubits.
        
        This module helps you explore the potential of different topological materials for quantum computing 
        applications and connect them with your CVD simulation results.
        """
    )
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Material Properties", "Deposition Analysis", "Quantum Applications"])
    
    # Material Properties tab
    with tab1:
        _show_material_properties()
    
    # Deposition Analysis tab
    with tab2:
        _show_deposition_analysis()
    
    # Quantum Applications tab
    with tab3:
        _show_quantum_applications()


def _show_material_properties():
    """Display topological material properties."""
    st.header("Topological Insulator Properties")
    
    # Get available materials
    materials = get_available_materials()
    
    # Material selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_material = st.selectbox(
            "Select Material",
            options=materials
        )
        
        # Get material info
        material_info = get_material_info(selected_material)
        
        # Material image placeholder
        st.image(
            f"https://via.placeholder.com/300x200?text={selected_material}",
            caption=f"{material_info['full_name']} ({material_info['formula']})"
        )
        
        # Topological status indicator
        is_topological = is_topologically_nontrivial(selected_material)
        topological_status = "✓ Strong Topological Insulator" if is_topological else "✗ Trivial Insulator"
        status_color = "green" if is_topological else "red"
        
        st.markdown(f"**Status:** <span style='color:{status_color}'>{topological_status}</span>", unsafe_allow_html=True)
    
    with col2:
        # Display material properties in a table
        st.subheader(f"{material_info['full_name']} Properties")
        
        # Create properties dataframe
        properties = {
            "Property": [
                "Chemical Formula", 
                "Z₂ Indices (ν₀; ν₁ν₂ν₃)", 
                "Chern Number",
                "Band Gap (eV)",
                "Critical Thickness (nm)",
                "Optimal Thickness Range (nm)",
                "Lattice Constant a (Å)",
                "Lattice Constant c (Å)",
                "Dirac Velocity (m/s)",
                "Qubit Suitability"
            ],
            "Value": [
                material_info["formula"],
                str(material_info["z2_indices"]),
                material_info["chern_number"],
                f"{material_info['band_gap']:.3f}",
                f"{material_info['critical_thickness']:.1f}",
                f"{material_info['optimal_thickness_range'][0]:.1f} - {material_info['optimal_thickness_range'][1]:.1f}",
                f"{material_info['lattice_constant_a']:.2f}",
                f"{material_info['lattice_constant_c']:.2f}" if material_info['lattice_constant_c'] else "N/A",
                f"{material_info['dirac_velocity']:.1e}",
                material_info["qubit_suitability"]
            ]
        }
        
        st.table(pd.DataFrame(properties))
        
        # Notes about the material
        st.markdown("**Notes:**")
        st.info(material_info["notes"])
        
        # References
        st.markdown("**References:**")
        for ref in material_info["references"]:
            st.markdown(f"- {ref}")
    
    # Visualize band structure (mock)
    st.subheader("Band Structure Visualization")
    
    # Mock band structure data
    k_points = np.linspace(-1, 1, 100)
    
    # Different band structures based on material
    if selected_material in ["Bi2Te3", "Bi2Se3"]:
        # Strong TI with linear Dirac cone
        band_gap = material_info["band_gap"]
        conduction = np.sqrt(k_points**2 + band_gap**2/4) + band_gap/2
        valence = -np.sqrt(k_points**2 + band_gap**2/4) + band_gap/2
        surface = k_points * 0.8  # Linear surface states
    elif selected_material in ["Bi0.9Sb0.1", "SmB6"]:
        # More complex surface states
        band_gap = material_info["band_gap"]
        conduction = np.sqrt(k_points**2 + band_gap**2/4) + band_gap/2
        valence = -np.sqrt(k_points**2 + band_gap**2/4) + band_gap/2
        surface = k_points * 0.6 * np.tanh(k_points * 3)  # Non-linear surface states
    else:
        # Generic band structure
        band_gap = material_info["band_gap"]
        conduction = k_points**2 + band_gap/2
        valence = -k_points**2 - band_gap/2
        surface = k_points * 0.5  # Surface states
    
    # Plot band structure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bulk bands
    ax.plot(k_points, conduction, 'b-', linewidth=2, label="Conduction Band")
    ax.plot(k_points, valence, 'b-', linewidth=2, label="Valence Band")
    
    # Surface states
    ax.plot(k_points, surface, 'r--', linewidth=2, label="Surface States")
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('k')
    ax.set_ylabel('Energy (eV)')
    ax.set_title(f'Schematic Band Structure of {selected_material}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close(fig)
    
    # Topological invariants explanation
    with st.expander("About Topological Invariants"):
        st.markdown("""
        ### Z₂ Indices

        The topological invariant Z₂ characterizes time-reversal invariant topological insulators:
        
        - For 3D topological insulators, there are four Z₂ indices: (ν₀; ν₁ν₂ν₃)
        - ν₀ = 1 indicates a **strong** topological insulator
        - (ν₁ν₂ν₃) ≠ (000) with ν₀ = 0 indicates a **weak** topological insulator
        - (0;000) indicates a trivial insulator
        
        ### Chern Number
        
        The Chern number (C) is an integer topological invariant related to the Berry curvature:
        
        - C = 0 for time-reversal invariant topological insulators
        - C ≠ 0 for Chern insulators and quantum anomalous Hall systems
        - |C| represents the number of chiral edge states
        """)


def _show_deposition_analysis():
    """Display deposition analysis for topological materials."""
    st.header("Deposition Analysis")
    
    st.markdown(
        """
        This section helps you analyze how CVD deposition parameters affect the topological properties
        of the materials and their suitability for quantum computing applications.
        """
    )
    
    # Get available materials
    materials = get_available_materials()
    
    # Material and parameters selection
    col1, col2 = st.columns(2)
    
    with col1:
        selected_material = st.selectbox(
            "Select Material",
            options=materials,
            key="deposition_material"
        )
        
        material_info = get_material_info(selected_material)
        
        st.markdown(f"**Selected Material:** {material_info['full_name']} ({material_info['formula']})")
        
        # Deposition method
        methods = material_info['deposition_methods']
        deposition_method = st.selectbox(
            "Deposition Method",
            options=methods,
            index=methods.index("CVD") if "CVD" in methods else 0
        )
    
    with col2:
        # Deposition parameters
        deposition_temp = st.slider(
            "Deposition Temperature (K)",
            min_value=300,
            max_value=1200,
            value=material_info.get("optimal_deposition_temp", 573),
            step=10
        )
        
        film_thickness = st.slider(
            "Film Thickness (nm)",
            min_value=1.0,
            max_value=50.0,
            value=material_info.get("optimal_thickness_range", (5.0, 15.0))[0],
            step=0.5
        )
    
    # Analyze deposition parameters
    st.subheader("Deposition Parameter Analysis")
    
    # Get deposition recommendations
    deposition_recommendations = get_deposition_recommendations(selected_material, deposition_temp)
    
    # Evaluate thickness suitability
    thickness_level, thickness_score = evaluate_thickness_suitability(selected_material, film_thickness)
    
    # Display results in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Temperature Analysis")
        
        temp_diff = abs(deposition_temp - material_info.get("optimal_deposition_temp", 573))
        
        # Temperature gauge
        temp_factor = deposition_recommendations.get("temperature_factor", 0.5)
        st.markdown(f"**Temperature Suitability:** {deposition_recommendations.get('temperature_suitability', 'N/A')}")
        st.progress(temp_factor)
        
        st.markdown(f"**Optimal Temperature:** {material_info.get('optimal_deposition_temp', 573)} K")
        st.markdown(f"**Temperature Difference:** {temp_diff} K")
        
        if temp_diff > 100:
            st.warning("Temperature is significantly different from optimal. Consider adjusting.")
        elif temp_diff < 30:
            st.success("Temperature is near optimal for this material.")
        
        # Band gap at temperature
        band_gap = deposition_recommendations.get("adjusted_band_gap", material_info.get("band_gap", 0.0))
        st.markdown(f"**Adjusted Band Gap at {deposition_temp} K:** {band_gap:.3f} eV")
    
    with col2:
        st.markdown("##### Thickness Analysis")
        
        # Thickness gauge
        st.markdown(f"**Thickness Suitability:** {thickness_level}")
        st.progress(thickness_score)
        
        st.markdown(f"**Critical Thickness:** {material_info.get('critical_thickness', 0.0)} nm")
        st.markdown(f"**Optimal Range:** {material_info.get('optimal_thickness_range', (0.0, 0.0))[0]} - {material_info.get('optimal_thickness_range', (0.0, 0.0))[1]} nm")
        
        if thickness_score < 0.3:
            st.error("Film thickness is not suitable for topological properties.")
        elif thickness_score > 0.9:
            st.success("Film thickness is ideal for topological properties.")
        else:
            st.info("Film thickness is acceptable but not optimal.")
    
    # Visualize thickness-dependent properties
    st.subheader("Thickness-Dependent Properties")
    
    # Generate data for plot
    thickness_range = np.linspace(1.0, 30.0, 100)
    suitability_scores = []
    
    for t in thickness_range:
        _, score = evaluate_thickness_suitability(selected_material, t)
        suitability_scores.append(score)
    
    # Plot thickness vs suitability
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thickness_range, suitability_scores, 'b-', linewidth=2)
    ax.axvline(x=material_info.get("critical_thickness", 0.0), color='r', linestyle='--', label='Critical Thickness')
    ax.axvspan(
        material_info.get("optimal_thickness_range", (0.0, 0.0))[0],
        material_info.get("optimal_thickness_range", (0.0, 0.0))[1],
        alpha=0.2, color='g', label='Optimal Range'
    )
    ax.axvline(x=film_thickness, color='k', linestyle='-', label='Current Selection')
    
    ax.set_xlabel('Film Thickness (nm)')
    ax.set_ylabel('Topological Suitability Score')
    ax.set_title('Thickness Dependence of Topological Properties')
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)
    plt.close(fig)
    
    # Recommendations
    st.subheader("Deposition Recommendations")
    
    # Display recommendations
    for recommendation in deposition_recommendations.get("recommendations", []):
        if recommendation:
            st.markdown(f"- {recommendation}")
    
    # Integration with CVD simulation
    st.subheader("Integration with CVD Simulation")
    
    st.markdown(
        """
        To optimize your CVD deposition for topological properties:
        
        1. Set the deposition temperature in your CVD simulation to match the optimal temperature shown above
        2. Adjust gas flow rates to achieve the target film thickness within the optimal range
        3. Focus on uniformity of deposition - topological properties are sensitive to film quality
        4. Consider post-deposition annealing to improve crystallinity and topological properties
        """
    )
    
    # CVD process parameter suggestions
    with st.expander("CVD Process Parameter Suggestions"):
        st.markdown(f"""
        #### Suggested CVD Parameters for {selected_material}
        
        - **Substrate Temperature:** {material_info.get('optimal_deposition_temp', 573)} K
        - **Pressure:** 0.1-1.0 Torr (lower pressures often yield more uniform films)
        - **Carrier Gas:** High-purity Ar or H₂
        - **Precursors:** Metal-organic compounds for {', '.join(material_info['formula'].split('₂')[0:1])}
        - **Growth Rate:** Target slow growth (1-10 nm/min) for better crystallinity
        - **Substrate:** Si(111) or sapphire with appropriate buffer layers
        - **Post-deposition:** Consider in-situ annealing at {int(material_info.get('optimal_deposition_temp', 573) * 0.8)} K under inert atmosphere
        """)


def _show_quantum_applications():
    """Display quantum computing applications for topological materials."""
    st.header("Quantum Computing Applications")
    
    st.markdown(
        """
        Topological insulators offer unique advantages for quantum computing due to their protected surface states.
        This section explores the potential of your selected material for topological qubits and related quantum applications.
        """
    )
    
    # Get available materials
    materials = get_available_materials()
    
    # Material and parameters selection
    col1, col2 = st.columns(2)
    
    with col1:
        selected_material = st.selectbox(
            "Select Material",
            options=materials,
            key="quantum_material"
        )
        
        material_info = get_material_info(selected_material)
        
        st.markdown(f"**Selected Material:** {material_info['full_name']} ({material_info['formula']})")
        
        # Film parameters
        film_thickness = st.slider(
            "Film Thickness (nm)",
            min_value=1.0,
            max_value=50.0,
            value=material_info.get("optimal_thickness_range", (5.0, 15.0))[0],
            step=0.5,
            key="quantum_thickness"
        )
    
    with col2:
        # Operating conditions
        operating_temp = st.slider(
            "Operating Temperature (K)",
            min_value=0.01,
            max_value=300.0,
            value=4.0,  # Default to typical dilution refrigerator temperature
            step=0.01 if operating_temp_scale == "Log" else 1.0
        )
        
        operating_temp_scale = st.radio(
            "Temperature Scale",
            options=["Linear", "Log"],
            horizontal=True
        )
        
        if operating_temp_scale == "Log":
            operating_temp = 10**operating_temp if operating_temp > 0 else 0.01
    
    # Calculate qubit potential
    qubit_potential = calculate_qubit_potential(selected_material, film_thickness, operating_temp)
    
    # Display qubit potential
    st.subheader("Topological Qubit Potential")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown(f"**Qubit Potential:** {qubit_potential['potential_level']}")
        st.progress(qubit_potential['potential_score'])
        
        st.markdown(f"**Estimated Coherence Time:** {qubit_potential['estimated_coherence_ns']:.1f} ns")
        
        # Display factors
        st.markdown("##### Contributing Factors")
        factors = qubit_potential['factors']
        
        # Material suitability
        st.markdown(f"Material Suitability: {factors['material_suitability']:.2f}")
        st.progress(factors['material_suitability'])
        
        # Thickness factor
        st.markdown(f"Thickness Factor: {factors['thickness_factor']:.2f}")
        st.progress(factors['thickness_factor'])
        
        # Temperature factor
        st.markdown(f"Temperature Factor: {factors['temperature_factor']:.2f}")
        st.progress(factors['temperature_factor'])
        
        # Band gap factor
        st.markdown(f"Band Gap Factor: {factors['band_gap_factor']:.2f}")
        st.progress(factors['band_gap_factor'])
    
    with col2:
        # Visualize temperature dependence
        temp_range = np.logspace(-2, 2, 100) if operating_temp_scale == "Log" else np.linspace(0.1, 300, 100)
        potential_scores = []
        coherence_times = []
        
        for temp in temp_range:
            potential = calculate_qubit_potential(selected_material, film_thickness, temp)
            potential_scores.append(potential['potential_score'])
            coherence_times.append(potential['estimated_coherence_ns'])
        
        # Create plot
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Qubit Potential Score', color=color)
        ax1.plot(temp_range, potential_scores, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.axvline(x=operating_temp, color='k', linestyle='--', label='Selected Temperature')
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Coherence Time (ns)', color=color)
        ax2.plot(temp_range, coherence_times, color=color, linestyle=':')
        ax2.tick_params(axis='y', labelcolor=color)
        
        if operating_temp_scale == "Log":
            ax1.set_xscale('log')
        
        plt.title(f'Temperature Dependence of Qubit Performance for {selected_material}')
        plt.grid(True, alpha=0.3)
        
        # Add a legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        st.pyplot(fig)
        plt.close(fig)
    
    # Limitations and recommendations
    st.subheader("Limitations & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Limitations")
        for limitation in qubit_potential['limitations']:
            if limitation:
                st.markdown(f"- {limitation}")
    
    with col2:
        st.markdown("##### Recommendations")
        for recommendation in qubit_potential['recommendations']:
            if recommendation:
                st.markdown(f"- {recommendation}")
    
    # Qubit implementations
    st.subheader("Potential Qubit Implementations")
    
    # Different implementation recommendations based on material properties
    if material_info['qubit_suitability'] in ["High", "Medium-High"]:
        st.markdown("""
        #### Recommended Implementations
        
        1. **Majorana Zero Modes**
           - Create heterostructures with s-wave superconductors
           - Potential for fault-tolerant topological quantum computing
           
        2. **Quantum Anomalous Hall Edge Qubits**
           - Dope with magnetic impurities to break time-reversal symmetry
           - Use chiral edge states for robust qubit states
           
        3. **Topological Josephson Junctions**
           - Form Josephson junctions with topological insulator barriers
           - Leverage 4π-periodic supercurrent for qubit operations
        """)
    else:
        st.markdown("""
        #### Potential Implementations
        
        1. **Enhanced Conventional Qubits**
           - Use as material platform for traditional superconducting or spin qubits
           - May offer improved coherence due to reduced backscattering
           
        2. **Hybrid Qubit Systems**
           - Combine with conventional materials for hybrid quantum systems
           - Explore spin-momentum locking for novel qubit operations
        """)
    
    # Topological qubit explainer
    with st.expander("What are Topological Qubits?"):
        st.markdown("""
        ### Topological Qubits
        
        Topological qubits are a proposed type of quantum bit that exploits the unique properties 
        of topological states of matter to achieve fault tolerance.
        
        #### Key advantages:
        
        - **Inherent error protection**: Information is encoded in non-local, topological degrees of freedom
        - **Reduced decoherence**: Less sensitive to local environmental perturbations
        - **Potential for fault-tolerant operations**: Certain operations can be performed without additional error correction
        
        #### Current challenges:
        
        - Difficult to realize experimentally
        - Requires ultra-low temperatures in most implementations
        - Complex materials engineering and precise control
        
        #### Common approaches:
        
        1. **Majorana Zero Modes**: Quasiparticles that appear at the ends of topological superconducting wires
        2. **Quantum Anomalous Hall Systems**: Uses chiral edge states for robust quantum information
        3. **Fractional Quantum Hall States**: Leverages anyonic excitations for topological protection
        
        While still largely theoretical, topological qubits represent one of the most promising paths 
        toward scalable, fault-tolerant quantum computing.
        """)
    
    # Recent research
    st.subheader("Recent Research")
    
    st.markdown("""
    #### Latest Developments in Topological Quantum Computing
    
    - **2023**: Improved coherence in hybrid superconductor-TI devices reaching microsecond timescales
    - **2022**: Demonstration of controlled braiding of Majorana-like excitations
    - **2021**: Fabrication of high-quality topological insulator thin films with CVD showing improved mobility
    - **2020**: First observation of quantized conductance in a candidate topological insulator edge
    
    > Note: This field is evolving rapidly. For the most up-to-date research, consult recent publications.
    """)