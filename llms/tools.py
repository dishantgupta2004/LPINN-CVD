"""
Tool definitions for LLM function calling capabilities.
"""
from typing import Dict, List, Optional, Union, Any, Callable
import json
import numpy as np
import tensorflow as tf
from dataclasses import dataclass


@dataclass
class ToolDefinition:
    """Definition of a tool for LLM function calling."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    

class CVDPINNTools:
    """Collection of tools for CVD-PINN integration with LLMs."""
    
    def __init__(self):
        """Initialize the tools collection."""
        self.tools = {}
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # Configuration tools
        self.register_tool(
            name="set_domain_parameters",
            description="Set the domain parameters for the CVD simulation",
            parameters={
                "type": "object",
                "properties": {
                    "x_min": {"type": "number", "description": "Minimum x-coordinate (m)"},
                    "x_max": {"type": "number", "description": "Maximum x-coordinate (m)"},
                    "y_min": {"type": "number", "description": "Minimum y-coordinate (m)"},
                    "y_max": {"type": "number", "description": "Maximum y-coordinate (m)"},
                    "t_min": {"type": "number", "description": "Start time (s)"},
                    "t_max": {"type": "number", "description": "End time (s)"},
                    "dimension": {"type": "integer", "description": "Domain dimension (2 or 3)"}
                },
                "required": ["x_min", "x_max", "y_min", "y_max", "t_min", "t_max"]
            },
            function=self.set_domain_parameters
        )
        
        self.register_tool(
            name="set_physical_parameters",
            description="Set the physical parameters for the CVD simulation",
            parameters={
                "type": "object",
                "properties": {
                    "D_SiH4": {"type": "number", "description": "Diffusion coefficient for SiH4 (m²/s)"},
                    "D_Si": {"type": "number", "description": "Diffusion coefficient for Si (m²/s)"},
                    "D_H2": {"type": "number", "description": "Diffusion coefficient for H2 (m²/s)"},
                    "D_SiH2": {"type": "number", "description": "Diffusion coefficient for SiH2 (m²/s)"},
                    "thermal_conductivity": {"type": "number", "description": "Thermal conductivity (W/(m·K))"},
                    "specific_heat": {"type": "number", "description": "Specific heat (J/(kg·K))"},
                    "density": {"type": "number", "description": "Density (kg/m³)"},
                    "A1": {"type": "number", "description": "Pre-exponential factor for reaction 1"},
                    "E1": {"type": "number", "description": "Activation energy for reaction 1 (J/mol)"},
                    "A2": {"type": "number", "description": "Pre-exponential factor for reaction 2"},
                    "E2": {"type": "number", "description": "Activation energy for reaction 2 (J/mol)"},
                    "A3": {"type": "number", "description": "Pre-exponential factor for reaction 3"},
                    "E3": {"type": "number", "description": "Activation energy for reaction 3 (J/mol)"}
                }
            },
            function=self.set_physical_parameters
        )
        
        self.register_tool(
            name="set_model_parameters",
            description="Set the neural network model parameters",
            parameters={
                "type": "object",
                "properties": {
                    "hidden_layers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of neurons in each hidden layer"
                    },
                    "activation": {"type": "string", "description": "Activation function name"},
                    "dropout_rate": {"type": "number", "description": "Dropout rate (0.0 to 0.5)"},
                    "learning_rate": {"type": "number", "description": "Learning rate for training"}
                },
                "required": ["hidden_layers"]
            },
            function=self.set_model_parameters
        )
        
        # Analysis tools
        self.register_tool(
            name="analyze_simulation_results",
            description="Analyze and interpret the simulation results",
            parameters={
                "type": "object",
                "properties": {
                    "analysis_type": {
                        "type": "string", 
                        "enum": ["deposition_rate", "uniformity", "efficiency", "parameter_sensitivity"],
                        "description": "Type of analysis to perform"
                    },
                    "time_point": {"type": "number", "description": "Time point for analysis (s)"},
                    "species": {"type": "string", "description": "Chemical species to analyze"}
                },
                "required": ["analysis_type"]
            },
            function=self.analyze_simulation_results
        )
        
        # Code generation tools
        self.register_tool(
            name="generate_reaction_code",
            description="Generate code for custom reaction models",
            parameters={
                "type": "object",
                "properties": {
                    "reaction_equations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of reaction equations (e.g., 'SiH4 -> Si + 2H2')"
                    },
                    "rate_constants": {
                        "type": "object",
                        "description": "Dictionary of rate constants and activation energies"
                    },
                    "temperature_dependent": {"type": "boolean", "description": "Whether rates are temperature-dependent"}
                },
                "required": ["reaction_equations"]
            },
            function=self.generate_reaction_code
        )
    
    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], function: Callable) -> None:
        """
        Register a new tool.
        
        Args:
            name: Tool name
            description: Tool description
            parameters: JSON Schema for tool parameters
            function: Function to call when tool is invoked
        """
        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            function=function
        )
    
    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Get tools formatted for OpenAI function calling.
        
        Returns:
            List of tool definitions in OpenAI format
        """
        openai_tools = []
        
        for tool_name, tool in self.tools.items():
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        
        return openai_tools
    
    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Call a tool with the provided arguments.
        
        Args:
            tool_name: Name of the tool to call
            **kwargs: Arguments for the tool
            
        Returns:
            Tool function result
            
        Raises:
            ValueError: If tool name is not found
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}")
        
        return self.tools[tool_name].function(**kwargs)
    
    # Tool implementations
    def set_domain_parameters(self, **kwargs) -> Dict[str, Any]:
        """Set domain parameters for the simulation."""
        # In a real implementation, this would update the simulation configuration
        # For now, we'll just return the validated parameters
        return {
            "status": "success",
            "message": "Domain parameters set successfully",
            "parameters": kwargs
        }
    
    def set_physical_parameters(self, **kwargs) -> Dict[str, Any]:
        """Set physical parameters for the simulation."""
        # In a real implementation, this would update the simulation configuration
        # For now, we'll just return the validated parameters
        return {
            "status": "success",
            "message": "Physical parameters set successfully",
            "parameters": kwargs
        }
    
    def set_model_parameters(self, **kwargs) -> Dict[str, Any]:
        """Set model parameters for the simulation."""
        # In a real implementation, this would update the model configuration
        # For now, we'll just return the validated parameters
        return {
            "status": "success",
            "message": "Model parameters set successfully",
            "parameters": kwargs
        }
    
    def analyze_simulation_results(self, analysis_type: str, **kwargs) -> Dict[str, Any]:
        """
        Analyze simulation results.
        
        Args:
            analysis_type: Type of analysis to perform
            **kwargs: Additional analysis parameters
            
        Returns:
            Analysis results
        """
        # This would actually analyze real simulation results
        # For now, we'll return mock analysis
        
        if analysis_type == "deposition_rate":
            return {
                "status": "success",
                "analysis_type": "deposition_rate",
                "average_rate": 0.025,  # nm/s
                "min_rate": 0.018,
                "max_rate": 0.032,
                "spatial_variation": "Higher rates observed near the center of substrate"
            }
        
        elif analysis_type == "uniformity":
            return {
                "status": "success",
                "analysis_type": "uniformity",
                "uniformity_metric": 0.87,  # higher is better
                "standard_deviation": 0.045,
                "recommendations": [
                    "Increase substrate temperature for better uniformity",
                    "Reduce inlet velocity to allow more time for diffusion"
                ]
            }
        
        elif analysis_type == "efficiency":
            return {
                "status": "success",
                "analysis_type": "efficiency",
                "precursor_utilization": 0.72,  # fraction of precursor converted to film
                "energy_efficiency": 0.58,
                "limiting_factors": "Reaction rate limited by substrate temperature"
            }
        
        elif analysis_type == "parameter_sensitivity":
            return {
                "status": "success",
                "analysis_type": "parameter_sensitivity",
                "most_sensitive_parameters": [
                    "Substrate temperature",
                    "SiH4 concentration",
                    "Reaction rate constant A1"
                ],
                "sensitivity_coefficients": {
                    "temperature": 0.82,
                    "sih4_concentration": 0.65,
                    "A1": 0.58
                }
            }
        
        else:
            return {
                "status": "error",
                "message": f"Unknown analysis type: {analysis_type}"
            }
    
    def generate_reaction_code(self, reaction_equations: List[str], **kwargs) -> Dict[str, Any]:
        """
        Generate code for custom reaction models.
        
        Args:
            reaction_equations: List of chemical reaction equations
            **kwargs: Additional code generation parameters
            
        Returns:
            Generated code and documentation
        """
        # This would generate actual implementation code
        # For now, return a template
        
        temperature_dependent = kwargs.get("temperature_dependent", True)
        
        code_template = """
def compute_reaction_rates(concentrations, temperature):
    \"\"\"
    Compute reaction rates for custom CVD reactions.
    
    Args:
        concentrations: Dictionary of species concentrations
        temperature: Temperature in Kelvin
        
    Returns:
        Dictionary of reaction rates
    \"\"\"
    # Constants
    R = 8.314  # Universal gas constant (J/mol·K)
    
    # Extract concentrations
    {concentration_extraction}
    
    # Compute reaction rates
    rates = {{}}
    
    {rate_calculations}
    
    return rates
"""
        
        # Build concentration extraction
        concentration_extractions = []
        species_set = set()
        
        for eq in reaction_equations:
            reactants = eq.split('->')[0].strip().split('+')
            products = eq.split('->')[1].strip().split('+')
            
            for reactant in reactants:
                species = reactant.strip().split(' ')[0]
                species_set.add(species)
            
            for product in products:
                species = product.strip().split(' ')[0]
                species_set.add(species)
        
        for species in species_set:
            concentration_extractions.append(f"{species} = concentrations.get('{species}', 0.0)")
        
        concentration_extraction_code = "\n    ".join(concentration_extractions)
        
        # Build rate calculations
        rate_calcs = []
        
        for i, eq in enumerate(reaction_equations, 1):
            reactants = eq.split('->')[0].strip().split('+')
            reactant_terms = []
            
            for reactant in reactants:
                parts = reactant.strip().split(' ')
                species = parts[-1]
                coef = 1.0
                if len(parts) > 1:
                    try:
                        coef = float(parts[0])
                    except ValueError:
                        coef = 1.0
                
                reactant_terms.append(f"{species}")
            
            if temperature_dependent:
                rate_calcs.append(f"# Reaction {i}: {eq}")
                rate_calcs.append(f"k{i} = A{i} * np.exp(-E{i}/(R*temperature))")
                rate_calcs.append(f"rates['R{i}'] = k{i} * " + " * ".join(reactant_terms))
            else:
                rate_calcs.append(f"# Reaction {i}: {eq}")
                rate_calcs.append(f"rates['R{i}'] = k{i} * " + " * ".join(reactant_terms))
        
        rate_calculations_code = "\n    ".join(rate_calcs)
        
        # Fill in the template
        generated_code = code_template.format(
            concentration_extraction=concentration_extraction_code,
            rate_calculations=rate_calculations_code
        )
        
        return {
            "status": "success",
            "message": "Reaction code generated successfully",
            "code": generated_code,
            "reactions_parsed": len(reaction_equations),
            "species_identified": list(species_set)
        }


def register_tools(tools_instance: Optional[CVDPINNTools] = None) -> CVDPINNTools:
    """
    Register tools with the CVD-PINN application.
    
    Args:
        tools_instance: Optional existing tools instance to extend
        
    Returns:
        CVDPINNTools instance with registered tools
    """
    if tools_instance is None:
        tools_instance = CVDPINNTools()
    
    # Register any additional custom tools here
    
    return tools_instance