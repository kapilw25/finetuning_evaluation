"""
Shared evaluation utilities for all evaluation scripts
"""

import sys
from pathlib import Path

# Add 0c_utils to path for logging_utils
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root / "comparative_study" / "0c_utils"))

from .model_loader import (
    MODELS,
    BASE_MODEL,
    MODEL_NAME,
    load_model_for_eval,
    unload_model,
    verify_hf_repos
)

from .checkpoint import (
    get_checkpoint_dir,
    get_checkpoint_path,
    save_checkpoint,
    load_checkpoint,
    delete_checkpoint
)

from .fireworks_client import FireworksJudge

from .generation import (
    batch_generate,
    cleanup_gpu,
    format_chat_messages
)

from .prompts import (
    # get_toxicity_prompt,  # LEGACY - Toxicity eval excluded from final comparison
    get_harmlessness_prompt,
    get_helpfulness_prompt,
    get_pairwise_prompt
)

from .statistical_analysis import (
    bootstrap_ci,
    paired_t_test,
    run_statistical_analysis
)

from .response_validator import (
    detect_gibberish,
    detect_repetition,
    validate_response,
    add_validation_columns,
    calculate_stratified_metrics,
    print_stratified_metrics,
    get_valid_mask,
    get_validation_summary
)

from .cli_menus import (
    show_cached_data_menu,
    show_mode_selection_menu,
    show_checkpoint_resume_menu,
    filter_model_keys
)

from .dataset_info import (
    get_isd_max_samples,
    get_truthfulqa_max_samples,
    get_conditional_safety_max_samples,
    get_length_control_max_samples,
    get_aqi_max_samples,
    get_all_max_samples
)

from .plotting import (
    get_model_color,
    get_model_colors,
    get_legend_elements,
    add_figure_legend,
    generate_comparison_plots,
    generate_boxviolin_chart
)

from .appendix_visualizations import (
    generate_isd_embedding_visualization,
    generate_aqi_3d_visualization,
    generate_instruction_fidelity_heatmap,
    generate_isd_fidelity_radar,
    generate_truthfulqa_category_heatmap,
    generate_length_control_distribution,
    generate_all_appendix_visualizations
)

from logging_utils import setup_training_logger, restore_logging

__all__ = [
    # Model loading
    'MODELS',
    'BASE_MODEL',
    'MODEL_NAME',
    'load_model_for_eval',
    'unload_model',
    'verify_hf_repos',
    # Checkpointing
    'get_checkpoint_dir',
    'get_checkpoint_path',
    'save_checkpoint',
    'load_checkpoint',
    'delete_checkpoint',
    # Generation
    'batch_generate',
    'cleanup_gpu',
    'format_chat_messages',
    # LLM-as-judge
    'FireworksJudge',
    # 'get_toxicity_prompt',  # LEGACY
    'get_harmlessness_prompt',
    'get_helpfulness_prompt',
    'get_pairwise_prompt',
    # Statistical analysis
    'bootstrap_ci',
    'paired_t_test',
    'run_statistical_analysis',
    # Response validation
    'detect_gibberish',
    'detect_repetition',
    'validate_response',
    'add_validation_columns',
    'calculate_stratified_metrics',
    'print_stratified_metrics',
    'get_valid_mask',
    'get_validation_summary',
    # CLI menus
    'show_cached_data_menu',
    'show_mode_selection_menu',
    'show_checkpoint_resume_menu',
    'filter_model_keys',
    # Dataset info
    'get_isd_max_samples',
    'get_truthfulqa_max_samples',
    'get_conditional_safety_max_samples',
    'get_length_control_max_samples',
    'get_aqi_max_samples',
    'get_all_max_samples',
    # Plotting
    'get_model_color',
    'get_model_colors',
    'get_legend_elements',
    'add_figure_legend',
    'generate_comparison_plots',
    'generate_boxviolin_chart',
    # Appendix Visualizations
    'generate_isd_embedding_visualization',
    'generate_aqi_3d_visualization',
    'generate_instruction_fidelity_heatmap',
    'generate_isd_fidelity_radar',
    'generate_truthfulqa_category_heatmap',
    'generate_length_control_distribution',
    'generate_all_appendix_visualizations',
    # Logging
    'setup_training_logger',
    'restore_logging',
]
