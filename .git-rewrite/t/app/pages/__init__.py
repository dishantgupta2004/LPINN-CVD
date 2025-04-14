"""
Page modules for the Streamlit application.
"""

from app.pages.home import show_home_page
from app.pages.simulation import show_simulation_page
from app.pages.model_comparison import show_model_comparison_page
from app.pages.parameter_analysis import show_parameter_analysis_page
from app.pages.adaptive_sampling import show_adaptive_sampling_page

__all__ = [
    'show_home_page',
    'show_simulation_page',
    'show_model_comparison_page',
    'show_parameter_analysis_page',
    'show_adaptive_sampling_page'
]