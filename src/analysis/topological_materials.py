"""
Topological insulator materials database and analysis module.

This module provides data and analysis functions for topological insulators
relevant to quantum computing applications.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# Database of topological insulator materials with their properties
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
        "references": ["Zhang et al., Nature Physics 5, 438 (2009)", "Chen et al., Science 325, 178 (2009)"]
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
        "references": ["Xia et al., Nature Physics 5, 398 (2009)", "Zhang et al., Nature Physics 5, 438 (2009)"]
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
        "references": ["Zhang et al., Nature Physics 5, 438 (2009)"]
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
        "references": ["Hsieh et al., Nature 452, 970 (2008)"]
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
        "references": ["König et al., Science 318, 766 (2007)"]
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
        "references": ["Qian et al., Science 346, 1344 (2014)"]
    },
    "Bi4I4": {
        "full_name": "Bismuth Tetraiodide",
        "formula": "Bi₄I₄",
        "z2_indices": [1, 0, 0, 0],
        "chern_number": 0,
        "band_gap": 0.2,  # eV
        "band_gap_temp_coef": -2.8e-4,  # eV/K
        "critical_thickness": 7.0,  # nm
        "optimal_thickness_range": (8.0, 20.0),  # nm
        "lattice_constant_a": 14.45,  # Å
        "lattice_constant_c": 12.67,  # Å
        "dirac_velocity": 4.5e5,  # m/s
        "qubit_suitability": "Medium-High",
        "debye_temperature": 160,  # K
        "notes": "Weak TI with quasi-1D structure. Interesting anisotropic properties.",
        "deposition_methods": ["CVD", "PVD"],
        "optimal_deposition_temp": 513,  # K
        "references": ["Noguchi et al., Nature 566, 518 (2019)"]
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
        "references": ["Dzero et al., Phys. Rev. Lett. 104, 106408 (2010)"]
    }
}


def get_material_info(material_name: str) -> Dict[str, Any]:
    """
    Get information about a topological insulator material.
    
    Args:
        material_name: Name of the material
        
    Returns:
        Dictionary with material properties
        
    Raises:
        KeyError: If material is not found in the database
    """
    if material_name not in TOPOLOGICAL_MATERIALS:
        raise KeyError(f"Material '{material_name}' not found in the database.")
    
    return TOPOLOGICAL_MATERIALS[material_name]


def get_available_materials() -> List[str]:
    """
    Get list of available topological insulator materials.
    
    Returns:
        List of material names
    """
    return list(TOPOLOGICAL_MATERIALS.keys())


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


def get_deposition_recommendations(material_name: str, cvd_temp_K: float) -> Dict[str, Any]:
    """
    Get recommendations for CVD deposition based on material and temperature.
    
    Args:
        material_name: Name of the material
        cvd_temp_K: CVD deposition temperature in Kelvin
        
    Returns:
        Dictionary with recommendations
    """
    material = get_material_info(material_name)
    
    optimal_temp = material.get("optimal_deposition_temp", 573)  # K
    deposition_methods = material.get("deposition_methods", [])
    
    # Check if CVD is a suitable deposition method
    cvd_suitable = "CVD" in deposition_methods
    
    # Calculate temperature suitability
    temp_diff = abs(cvd_temp_K - optimal_temp)
    if temp_diff < 30:
        temp_suitability = "Excellent"
        temp_factor = 1.0
    elif temp_diff < 70:
        temp_suitability = "Good"
        temp_factor = 0.8
    elif temp_diff < 150:
        temp_suitability = "Moderate"
        temp_factor = 0.5
    else:
        temp_suitability = "Poor"
        temp_factor = 0.2
    
    # Adjust band gap based on temperature
    base_band_gap = material.get("band_gap", 0.0)
    temp_coef = material.get("band_gap_temp_coef", 0.0)
    adjusted_band_gap = base_band_gap + temp_coef * (cvd_temp_K - 300)  # Adjust from room temp
    
    return {
        "cvd_suitable": cvd_suitable,
        "temperature_suitability": temp_suitability,
        "temperature_factor": temp_factor,
        "adjusted_band_gap": max(adjusted_band_gap, 0.0),  # Ensure non-negative
        "recommended_methods": deposition_methods,
        "recommendations": [
            f"Optimal deposition temperature: {optimal_temp} K",
            f"Temperature difference: {temp_diff} K from optimal",
            "Consider post-deposition annealing" if temp_diff > 100 else "",
            "Monitor thickness carefully to maintain topological properties",
            "Protect surface after deposition to preserve topological states"
        ],
        "expected_quality": temp_factor * (1.0 if cvd_suitable else 0.6)
    }


def calculate_qubit_potential(material_name: str, thickness_nm: float, temp_K: float) -> Dict[str, Any]:
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
    debye_temp = material.get("debye_temperature", 200)  # K
    temp_ratio = min(debye_temp / max(temp_K, 1.0), 5.0)  # Limit ratio to avoid extreme values
    temp_factor = min(np.tanh(temp_ratio - 0.5) + 0.5, 1.0)  # Scale to 0-1 range
    
    # Band gap effects on coherence
    band_gap = material.get("band_gap", 0.0)
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
    coherence_estimate = base_coherence_ns * potential_score * (band_gap / 0.1) * np.exp(-temp_K / debye_temp)
    
    return {
        "potential_score": potential_score,
        "potential_level": potential_level,
        "estimated_coherence_ns": min(coherence_estimate, 1000.0),  # Cap at 1 microsecond
        "factors": {
            "material_suitability": base_suitability,
            "thickness_factor": thickness_factor,
            "temperature_factor": temp_factor,
            "band_gap_factor": band_gap_factor
        },
        "limitations": [
            "Limited coherence time" if potential_score < 0.5 else "",
            "Temperature sensitivity" if temp_factor < 0.7 else "",
            "Thickness uniformity critical" if thickness_factor < 0.9 else "",
            "Surface protection required to maintain topological states"
        ],
        "recommendations": [
            "Operate at temperatures below 100K" if debye_temp < 200 else "Consider cryogenic operation",
            "Ensure precise thickness control during deposition",
            "Use epitaxial growth when possible for higher quality",
            "Consider heterostructures for enhanced topological properties"
        ]
    }


if __name__ == "__main__":
    # Simple test
    print("Available topological materials:")
    print(get_available_materials())
    
    material = "Bi2Te3"
    print(f"\nInformation for {material}:")
    info = get_material_info(material)
    print(f"Formula: {info['formula']}")
    print(f"Z2 indices: {info['z2_indices']}")
    print(f"Band gap: {info['band_gap']} eV")
    print(f"Qubit suitability: {info['qubit_suitability']}")