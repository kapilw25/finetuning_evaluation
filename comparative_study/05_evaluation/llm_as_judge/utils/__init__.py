"""
LLM-as-Judge Evaluation Utilities

This package contains utility modules for dual-metric evaluation of alignment methods:
- Harmlessness evaluation (PKU-SafeRLHF test split)
- Helpfulness evaluation (AlpacaEval)
- Statistical analysis (Pareto plots, bootstrap CI, t-tests)

Modules:
    prompts: LLM-as-judge evaluation prompts (harmlessness, helpfulness, pairwise)
    fireworks_client: Fireworks API wrapper for Llama-3.3-70B judge
    statistical_analysis: Statistical tests and visualization
"""

# Import key classes and functions for easier access
from .prompts import (
    get_harmlessness_prompt,
    get_helpfulness_prompt,
    get_pairwise_prompt
)

from .fireworks_client import FireworksJudge

from .statistical_analysis import (
    bootstrap_ci,
    paired_t_test,
    create_pareto_plot,
    run_statistical_analysis
)

__all__ = [
    # Prompts
    'get_harmlessness_prompt',
    'get_helpfulness_prompt',
    'get_pairwise_prompt',

    # Fireworks client
    'FireworksJudge',

    # Statistical analysis
    'bootstrap_ci',
    'paired_t_test',
    'create_pareto_plot',
    'run_statistical_analysis',
]

__version__ = '1.0.0'
__author__ = 'CITA Research Team'
