"""
Prompt templates for CVD-PINN LLM integration.
"""
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Template for LLM prompts with placeholders."""
    template: str
    description: str
    
    def format(self, **kwargs: Any) -> str:
        """
        Format the template with the provided values.
        
        Args:
            **kwargs: Values to fill the template placeholders
            
        Returns:
            Formatted prompt string
        """
        return self.template.format(**kwargs)


# System prompt for CVD simulation assistant
CVD_SYSTEM_PROMPT = """
You are a specialized assistant for Chemical Vapor Deposition (CVD) simulation using Physics-Informed Neural Networks (PINNs).
Your expertise covers:

1. CVD processes, reactions, and physics
2. Neural network architecture and training
3. Physics-informed machine learning
4. Uncertainty quantification in simulations
5. Interpreting simulation results in scientific contexts

Help the user understand, configure, and interpret CVD-PINN models.
When asked about model parameters, chemical reactions, or physical configurations, provide scientifically accurate guidance.
When the user wants to change simulation parameters, help them determine reasonable values based on physical principles.

You can:
- Generate code for custom reaction models
- Suggest parameter configurations
- Interpret simulation results
- Explain physical phenomena in the context of CVD
- Guide the user through the workflow of setting up and running simulations
"""


# Dictionary of available prompt templates
PROMPT_TEMPLATES = {
    "configuration": PromptTemplate(
        template="""
Based on my simulation needs, help me configure the CVD-PINN model for:

Material: {material}
Reactor type: {reactor_type}
Desired outcome: {desired_outcome}

What parameters should I set for:
1. Temperature
2. Pressure
3. Precursor concentrations
4. Reaction rates
5. Neural network architecture
""",
        description="Template for model configuration assistance"
    ),
    
    "result_interpretation": PromptTemplate(
        template="""
I've run a CVD simulation with the following parameters:
- Temperature: {temperature}K
- Pressure: {pressure} atm
- SiH4 concentration: {sih4_concentration}
- H2 concentration: {h2_concentration}

Here are the key results:
{results}

Please help me interpret these results and understand their implications.
What insights can I draw from this simulation?
""",
        description="Template for interpreting simulation results"
    ),
    
    "reaction_modeling": PromptTemplate(
        template="""
I want to model a custom CVD reaction system with the following:

Main precursors: {precursors}
Substrate material: {substrate}
Reactions: {reactions}

Please help me understand how to model this in the CVD-PINN framework.
What equations should I use? What parameters are most important?
""",
        description="Template for custom reaction system modeling"
    ),
    
    "code_generation": PromptTemplate(
        template="""
I need to implement a custom component for my CVD-PINN simulation.

Component type: {component_type}
Functionality needed: {functionality}
Integration point: {integration_point}

Please generate the Python code that I can use to implement this.
""",
        description="Template for generating custom code components"
    ),
    
    "parameter_optimization": PromptTemplate(
        template="""
I'm trying to optimize these parameters for my CVD simulation:
{parameters}

Current results: {current_results}
Desired outcomes: {desired_outcomes}

What parameter adjustments would you recommend to improve my results?
""",
        description="Template for parameter optimization assistance"
    )
}


def get_prompt_template(template_name: str) -> PromptTemplate:
    """
    Get a prompt template by name.
    
    Args:
        template_name: Name of the template to retrieve
        
    Returns:
        Prompt template object
        
    Raises:
        ValueError: If template name is not found
    """
    if template_name not in PROMPT_TEMPLATES:
        raise ValueError(f"Template '{template_name}' not found. Available templates: {list(PROMPT_TEMPLATES.keys())}")
    
    return PROMPT_TEMPLATES[template_name]


def list_available_templates() -> List[Dict[str, str]]:
    """
    List all available prompt templates with descriptions.
    
    Returns:
        List of dictionaries containing template names and descriptions
    """
    return [
        {"name": name, "description": template.description}
        for name, template in PROMPT_TEMPLATES.items()
    ]