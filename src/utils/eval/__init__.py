"""Shared evaluation utilities for all evaluation scripts."""

from .model_loader import (
    MODELS,
    BASE_MODEL,
    MODEL_NAME,
    load_model_for_eval,
    unload_model,
    verify_hf_repos,
)

from .checkpoint import (
    get_checkpoint_dir,
    get_checkpoint_path,
    save_checkpoint,
    load_checkpoint,
    delete_checkpoint,
)

from .fireworks_client import FireworksJudge

from .generation import (
    batch_generate,
    cleanup_gpu,
    format_chat_messages,
)

from .prompts import (
    get_harmlessness_prompt,
    get_helpfulness_prompt,
    get_pairwise_prompt,
)

from ..statistical_analysis import (
    bootstrap_ci,
    paired_t_test,
    run_statistical_analysis,
)

from .response_validator import (
    detect_gibberish,
    detect_repetition,
    validate_response,
    add_validation_columns,
    calculate_stratified_metrics,
    print_stratified_metrics,
    get_valid_mask,
    get_validation_summary,
)

from .cli_menus import (
    show_cached_data_menu,
    show_mode_selection_menu,
    show_checkpoint_resume_menu,
    filter_model_keys,
)

from .dataset_info import (
    get_isd_max_samples,
    get_truthfulqa_max_samples,
    get_conditional_safety_max_samples,
    get_length_control_max_samples,
    get_aqi_max_samples,
    get_all_max_samples,
)

from .plotting import (
    get_model_color,
    get_model_colors,
    get_legend_elements,
    add_figure_legend,
    generate_comparison_plots,
    generate_boxviolin_chart,
)

from .appendix_visualizations import (
    generate_isd_embedding_visualization,
    generate_aqi_3d_visualization,
    generate_instruction_fidelity_heatmap,
    generate_isd_fidelity_radar,
    generate_truthfulqa_category_heatmap,
    generate_length_control_distribution,
    generate_all_appendix_visualizations,
)

from ..logging_utils import setup_training_logger, restore_logging

__all__ = [
    "MODELS",
    "BASE_MODEL",
    "MODEL_NAME",
    "load_model_for_eval",
    "unload_model",
    "verify_hf_repos",
    "get_checkpoint_dir",
    "get_checkpoint_path",
    "save_checkpoint",
    "load_checkpoint",
    "delete_checkpoint",
    "batch_generate",
    "cleanup_gpu",
    "format_chat_messages",
    "FireworksJudge",
    "get_harmlessness_prompt",
    "get_helpfulness_prompt",
    "get_pairwise_prompt",
    "bootstrap_ci",
    "paired_t_test",
    "run_statistical_analysis",
    "detect_gibberish",
    "detect_repetition",
    "validate_response",
    "add_validation_columns",
    "calculate_stratified_metrics",
    "print_stratified_metrics",
    "get_valid_mask",
    "get_validation_summary",
    "show_cached_data_menu",
    "show_mode_selection_menu",
    "show_checkpoint_resume_menu",
    "filter_model_keys",
    "get_isd_max_samples",
    "get_truthfulqa_max_samples",
    "get_conditional_safety_max_samples",
    "get_length_control_max_samples",
    "get_aqi_max_samples",
    "get_all_max_samples",
    "get_model_color",
    "get_model_colors",
    "get_legend_elements",
    "add_figure_legend",
    "generate_comparison_plots",
    "generate_boxviolin_chart",
    "generate_isd_embedding_visualization",
    "generate_aqi_3d_visualization",
    "generate_instruction_fidelity_heatmap",
    "generate_isd_fidelity_radar",
    "generate_truthfulqa_category_heatmap",
    "generate_length_control_distribution",
    "generate_all_appendix_visualizations",
    "setup_training_logger",
    "restore_logging",
]
