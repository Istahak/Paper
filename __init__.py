"""
Hawkes Process Granger Causality Detection

A Python package for learning causal relationships in multivariate Hawkes processes
using Maximum Likelihood Estimation with basis representation.
"""

__version__ = "1.0.0"
__author__ = "Converted from MATLAB research code"
__email__ = "your.email@example.com"

# Main imports for easy access
from .test_causality import main
from .simulation_branch_hp import simulation_branch_hp
from .simulation_thinning_hp import simulation_thinning_hp
from .learning_mle_basis import learning_mle_basis
from .initialization_basis import initialization_basis
from .impact_func import impact_func

__all__ = [
    'main',
    'simulation_branch_hp',
    'simulation_thinning_hp', 
    'learning_mle_basis',
    'initialization_basis',
    'impact_func'
]