#!/usr/bin/env python3
"""
Comprehensive AQI Evaluation for All 7 Models
Evaluates: Baseline, SFT, SFT+GRIT, DPO, DPO+GRIT, CITA, CITA+GRIT
"""

import os
import sys
import gc

# IMPORTANT: Import unsloth FIRST (before transformers/peft) to apply optimizations
try:
    import unsloth
except ImportError:
    pass  # Optional - not required for evaluation

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

# Add AQI evaluation utilities
AQI_EVAL_SRC_PATH = "/lambda/nfs/DiskUsEast1/finetuning_evaluation/comparative_study/0a_AQI_EVAL_utils/src"
sys.path.insert(0, AQI_EVAL_SRC_PATH)
from aqi.aqi_dealign_xb_chi import *

# ============================================================================
# Configuration
# ============================================================================

# Paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent  # comparative_study directory
OUTPUT_DIR = SCRIPT_DIR / "AQI_Evaluation_Results"  # All results in 05_evaluation/AQI_Evaluation_Results
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
DATASET_NAME = "hasnat79/litmus"
SAMPLES_PER_CATEGORY = 100
GAMMA = 0.5
DIM_REDUCTION_METHOD = 'tsne'
RANDOM_SEED = 42

# Model definitions with their adapter paths
MODELS = {
    "Baseline": {
        "adapter_path": None,  # No adapter - just base model
        "display_name": "Baseline (Unaligned)",
        "output_subdir": "00_baseline_results"
    },
    "SFT_Baseline": {
        "adapter_path": BASE_DIR / "01a_SFT_Baseline" / "lora_model_SFT_Baseline",
        "display_name": "SFT Baseline",
        "output_subdir": "01a_sft_baseline_results"
    },
    "SFT_GRIT": {
        "adapter_path": BASE_DIR / "01b_SFT_GRIT" / "lora_model_SFT_GRIT",
        "display_name": "SFT + GRIT",
        "output_subdir": "01b_sft_grit_results"
    },
    "DPO_Baseline": {
        "adapter_path": BASE_DIR / "02a_DPO_Baseline" / "lora_model_DPO_Baseline",
        "display_name": "DPO Baseline",
        "output_subdir": "02a_dpo_baseline_results"
    },
    "DPO_GRIT": {
        "adapter_path": BASE_DIR / "02b_DPO_GRIT" / "lora_model_DPO_GRIT",
        "display_name": "DPO + GRIT",
        "output_subdir": "02b_dpo_grit_results"
    },
    "CITA_Baseline": {
        "adapter_path": BASE_DIR / "03a_CITA_Baseline" / "lora_model_CITA_Baseline",
        "display_name": "CITA Baseline",
        "output_subdir": "03a_cita_baseline_results"
    },
    "CITA_GRIT": {
        "adapter_path": BASE_DIR / "03b_CITA_GRIT" / "lora_model_CITA_GRIT",
        "display_name": "CITA + GRIT",
        "output_subdir": "03b_cita_grit_results"
    }
}

# ============================================================================
# Utility Functions
# ============================================================================

def check_model_exists(model_info):
    """Check if model adapter exists (or if it's baseline)"""
    if model_info["adapter_path"] is None:
        return True  # Baseline always exists
    return model_info["adapter_path"].exists()


def check_embeddings_exist(model_key):
    """Check if cached embeddings exist for this model"""
    results_dir = OUTPUT_DIR / MODELS[model_key]["output_subdir"]
    embeddings_file = results_dir / "embeddings.pkl"
    return embeddings_file.exists()


def check_results_exist(model_key):
    """Check if CSV results already exist for this model"""
    results_dir = OUTPUT_DIR / MODELS[model_key]["output_subdir"]
    csv_file = results_dir / f"{MODELS[model_key]['display_name'].replace(' ', '_')}_metrics_summary.csv"
    return csv_file.exists()


def generate_responses_batch(model, tokenizer, prompts, model_name, max_new_tokens=150, batch_size=16):
    """
    Generate responses for prompts using correct format based on model type.

    Format mapping:
    - SFT/DPO models (Baseline, GRIT) → Alpaca format
    - CITA models (Baseline, GRIT) → Llama-3 chat template

    This is the KEY modification for Response-AQI:
    Instead of embedding prompts, we generate responses and embed those.

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        prompts: List of prompt strings
        model_name: Model name to determine format
        max_new_tokens: Maximum tokens to generate per response
        batch_size: Number of prompts to process at once

    Returns:
        List of generated response strings
    """
    from tqdm import tqdm

    model.eval()
    responses = []

    # Determine format based on model name
    use_chat_template = "CITA" in model_name
    format_type = "Llama-3 Chat Template" if use_chat_template else "Alpaca"
    print(f"🔄 Generating {len(prompts)} responses using {format_type} format...")

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating Responses", unit="batch"):
        batch_prompts = prompts[i:i+batch_size]

        # Format prompts based on model type
        if use_chat_template:
            # CITA models: Use Llama-3 chat template
            formatted_prompts = []
            for prompt in batch_prompts:
                messages = [{"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                formatted_prompts.append(formatted)
        else:
            # SFT/DPO models: Use Alpaca format
            formatted_prompts = []
            for prompt in batch_prompts:
                full_prompt = f"""Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

### Instruction:
{prompt}

### Response:
"""
                formatted_prompts.append(full_prompt)

        # Tokenize batch
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        # Determine device from model
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = "cuda" if torch.cuda.is_available() else "cpu"

        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode only the generated portion (skip the prompt)
        for j, output in enumerate(outputs):
            prompt_length = inputs['input_ids'][j].shape[0]
            generated_ids = output[prompt_length:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            responses.append(response.strip())

    print(f"✅ Generated {len(responses)} responses using {format_type}")
    return responses


def generate_and_cache_responses(model, tokenizer, dataset_df, model_name, cache_file):
    """
    Generate responses, then call process_model_data with responses.

    This implements Response-AQI:
    1. Generate responses for each prompt
    2. Replace 'input' column with responses
    3. Embed responses (not prompts)
    4. Cluster by safety label

    Higher AQI = better separation of helpful vs refusal responses = better alignment

    Args:
        model: The language model
        tokenizer: Tokenizer
        dataset_df: DataFrame with 'input' column (prompts)
        model_name: Model name for logging
        cache_file: Path to cache file

    Returns:
        DataFrame with 'embedding' and 'original_prompt' columns
    """
    print(f"\n{'='*80}")
    print(f"Response-AQI Mode for {model_name}")
    print(f"{'='*80}")
    print("This will:")
    print("  1. Generate responses for each prompt")
    print("  2. Embed the RESPONSES (not prompts)")
    print("  3. Measure separation of helpful vs refusal responses")
    print(f"{'='*80}\n")

    # Step 1: Generate responses
    prompts = dataset_df['input'].tolist()
    responses = generate_responses_batch(model, tokenizer, prompts, model_name, max_new_tokens=150, batch_size=4)

    # Step 2: Create modified dataframe with responses as 'input'
    df_with_responses = dataset_df.copy()
    df_with_responses['original_prompt'] = df_with_responses['input']
    df_with_responses['input'] = responses  # Replace input with response

    print(f"\n✅ Replaced prompts with responses in 'input' column")
    print(f"   Sample response: {responses[0][:100]}...")

    # Step 3: Use existing process_model_data (it now embeds responses)
    print(f"\n🔄 Embedding responses (not prompts)...")
    processed_df = process_model_data(
        model, tokenizer, df_with_responses,
        model_name=model_name,
        cache_file=cache_file
    )

    return processed_df


def run_full_evaluation(model, tokenizer, model_display_name, output_sub_dir, balanced_df):
    """Run complete AQI evaluation for a model"""
    model_output_dir = OUTPUT_DIR / output_sub_dir
    model_output_dir.mkdir(parents=True, exist_ok=True)

    cache_file = model_output_dir / "embeddings.pkl"

    # If embeddings exist and model is None, load from cache directly
    if model is None and cache_file.exists():
        print(f"\n{'='*80}")
        print(f"Loading Cached Response Embeddings for {model_display_name}")
        print(f"{'='*80}")
        print(f"Loading cached response embeddings from {cache_file}")
        processed_df = pd.read_pickle(cache_file)
        print(f"✅ Loaded {len(processed_df)} samples from cache")
    else:
        # Response-AQI: Generate responses, then embed them (not prompts)
        print(f"\n{'='*80}")
        print(f"Response-AQI Evaluation for {model_display_name}")
        print(f"{'='*80}")
        processed_df = generate_and_cache_responses(
            model, tokenizer, balanced_df,
            model_name=model_display_name,
            cache_file=str(cache_file)
        )

    print(f"\n{'='*80}")
    print(f"Calculating AQI for {model_display_name}")
    print(f"{'='*80}")
    results, embeddings_3d, _, _ = analyze_by_axiom(
        processed_df,
        model_name=model_display_name,
        gamma=GAMMA,
        dim_reduction_method=DIM_REDUCTION_METHOD
    )

    create_metrics_summary(results, model_display_name, output_dir=str(model_output_dir))

    # Skip visualization for large datasets
    if 'overall' in embeddings_3d and embeddings_3d['overall'] is not None and len(processed_df) < 2000:
        try:
            visualize_clusters_3d(
                embeddings_3d['overall'],
                processed_df['safety_label_binary'].values,
                results['overall'],
                axiom='overall',
                title=f"{model_display_name} - Overall Clusters",
                output_dir=str(model_output_dir)
            )
        except (OverflowError, ValueError) as e:
            print(f"⚠️  Skipping visualization due to: {e}")

    overall_aqi = results.get('overall', {}).get('AQI', 'N/A')
    print(f"\n✅ Evaluation for {model_display_name} complete. Overall AQI: {overall_aqi}")
    return overall_aqi


def load_model(model_key):
    """Load model with or without adapter"""
    model_info = MODELS[model_key]

    print(f"\n{'='*80}")
    print(f"Loading {model_info['display_name']}")
    print(f"{'='*80}")

    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load adapter if specified
    if model_info["adapter_path"] is not None:
        print(f"Loading adapter from {model_info['adapter_path']}...")
        model = PeftModel.from_pretrained(base_model, str(model_info["adapter_path"]))
        print("Merging adapter weights...")
        model = model.merge_and_unload()
    else:
        print("Using base model (no adapter)")
        model = base_model

    model.eval()
    return model, tokenizer


# ============================================================================
# Comparative Analysis Functions
# ============================================================================

def create_comprehensive_comparison():
    """Create comprehensive comparison across all evaluated models"""
    print(f"\n{'='*80}")
    print("CREATING COMPREHENSIVE COMPARISON")
    print(f"{'='*80}")

    # Collect all CSV files
    all_results = {}
    for model_key, model_info in MODELS.items():
        csv_path = OUTPUT_DIR / model_info["output_subdir"] / f"{model_info['display_name'].replace(' ', '_')}_metrics_summary.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_results[model_key] = df
        else:
            print(f"⚠️  Skipping {model_key} (results not found)")

    if len(all_results) == 0:
        print("❌ No results found to compare")
        return

    # Create overall AQI comparison
    overall_scores = {}
    for model_key, df in all_results.items():
        overall_row = df[df['Category'] == 'overall']
        if not overall_row.empty:
            overall_scores[MODELS[model_key]['display_name']] = overall_row['AQI [0-100] (↑)'].values[0]

    # Print overall ranking
    print(f"\n{'='*80}")
    print("OVERALL AQI RANKING")
    print(f"{'='*80}")
    ranked = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (model_name, score) in enumerate(ranked, 1):
        print(f"  {i}. {model_name:<25}: {score:>8.4f}")

    # Create per-axiom comparison table
    print(f"\n{'='*80}")
    print("PER-AXIOM COMPARISON")
    print(f"{'='*80}")

    axioms = all_results[list(all_results.keys())[0]]['Category'].unique()
    axioms = [a for a in axioms if a != 'overall']

    # Build comparison DataFrame
    comparison_data = []
    for axiom in axioms:
        row = {'Axiom': axiom}
        for model_key, df in all_results.items():
            axiom_row = df[df['Category'] == axiom]
            if not axiom_row.empty:
                row[MODELS[model_key]['display_name']] = axiom_row['AQI [0-100] (↑)'].values[0]
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # Print table
    print(comparison_df.to_string(index=False))

    # Save comparison
    comparison_csv = OUTPUT_DIR / "All_Models_AQI_Comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\n✅ Saved comparison to: {comparison_csv}")

    # Create visualizations
    create_comparison_plots(comparison_df, overall_scores)


def create_comparison_plots(comparison_df, overall_scores):
    """Create comprehensive visualizations"""

    # Plot 1: Overall AQI Bar Chart (sorted descending)
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # Sort by scores in descending order
    sorted_items = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    models = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(models)))

    bars = ax1.bar(range(len(models)), scores, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Overall AQI Score [0-100]', fontsize=13, fontweight='bold')
    ax1.set_title('Overall AQI Comparison Across All Models', fontsize=15, fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plot1_path = OUTPUT_DIR / "Overall_AQI_Comparison.png"
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved overall comparison plot: {plot1_path}")
    plt.close()

    # Plot 2: Per-Axiom Heatmap
    fig2, ax2 = plt.subplots(figsize=(14, 8))

    # Prepare data for heatmap
    heatmap_data = comparison_df.set_index('Axiom').T

    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=100, cbar_kws={'label': 'AQI Score'},
                linewidths=0.5, ax=ax2)
    ax2.set_title('Per-Axiom AQI Heatmap Across All Models', fontsize=15, fontweight='bold', pad=20)
    ax2.set_xlabel('Ethical Axiom', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Model', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plot2_path = OUTPUT_DIR / "Per_Axiom_AQI_Heatmap.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved heatmap: {plot2_path}")
    plt.close()

    # Plot 3: Grouped Bar Chart (Per-Axiom with sorted bars within each axiom)
    fig3, ax3 = plt.subplots(figsize=(16, 8))

    axioms = comparison_df['Axiom'].values
    x = np.arange(len(axioms))
    model_names = [col for col in comparison_df.columns if col != 'Axiom']
    n_models = len(model_names)
    width = 0.8 / n_models

    # Create a consistent color mapping for each model across all axioms
    model_color_map = {name: plt.cm.tab10(i / n_models) for i, name in enumerate(model_names)}

    # For each axiom, sort models by their scores (descending)
    sorted_model_positions = {}  # axiom_idx -> list of (model_name, value, position)

    for axiom_idx, axiom in enumerate(axioms):
        # Get scores for this axiom across all models
        axiom_scores = [(model, comparison_df.loc[axiom_idx, model]) for model in model_names]
        # Sort by score descending
        sorted_scores = sorted(axiom_scores, key=lambda x: x[1], reverse=True)
        # Assign positions within the group
        sorted_model_positions[axiom_idx] = [
            (model, score, pos) for pos, (model, score) in enumerate(sorted_scores)
        ]

    # Plot bars with sorted positions within each axiom group
    plotted_models = set()
    for axiom_idx in range(len(axioms)):
        for model, score, position in sorted_model_positions[axiom_idx]:
            offset = (position - n_models/2 + 0.5) * width
            label = model if model not in plotted_models else None
            ax3.bar(axiom_idx + offset, score, width,
                   label=label, color=model_color_map[model], alpha=0.8)
            plotted_models.add(model)

    ax3.set_xlabel('Ethical Axiom', fontsize=13, fontweight='bold')
    ax3.set_ylabel('AQI Score [0-100]', fontsize=13, fontweight='bold')
    ax3.set_title('Per-Axiom AQI Comparison Across All Models', fontsize=15, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(axioms, rotation=45, ha='right')
    ax3.legend(fontsize=10, ncol=2, title='Models (sorted by score within each axiom)')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plot3_path = OUTPUT_DIR / "Per_Axiom_AQI_Grouped_Bars.png"
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved grouped bar chart: {plot3_path}")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print(f"\n{'='*80}")
    print("COMPREHENSIVE AQI EVALUATION - ALL 7 MODELS")
    print(f"{'='*80}")

    # Set random seed
    set_seed(RANDOM_SEED)

    # Load dataset once
    print(f"\n{'='*80}")
    print("Loading and Balancing Dataset")
    print(f"{'='*80}")
    balanced_eval_df = load_and_balance_dataset(
        dataset_name=DATASET_NAME,
        samples_per_category=SAMPLES_PER_CATEGORY,
        split='train'
    )

    # Add dummy axiom column if needed
    if 'axiom' not in balanced_eval_df.columns:
        balanced_eval_df['axiom'] = 'overall'
    if 'prompt' in balanced_eval_df.columns and 'input' not in balanced_eval_df.columns:
        balanced_eval_df = balanced_eval_df.rename(columns={'prompt': 'input'})

    print(f"\n✅ Dataset loaded: {len(balanced_eval_df)} samples")

    # Evaluate each model
    results_summary = {}

    for model_key in MODELS.keys():
        model_info = MODELS[model_key]

        # Check if model exists
        if not check_model_exists(model_info):
            print(f"\n⏭️  Skipping {model_info['display_name']} (adapter not found)")
            continue

        # Check if embeddings already cached - if so, we can skip model loading
        has_embeddings = check_embeddings_exist(model_key)

        # Load and evaluate model
        try:
            # Only load model if we need to extract embeddings
            if not has_embeddings:
                model, tokenizer = load_model(model_key)
            else:
                # If embeddings exist, we don't need to load the model
                # Just set tokenizer for process_model_data function signature
                print(f"\n{'='*80}")
                print(f"Processing {model_info['display_name']} (using cached embeddings)")
                print(f"{'='*80}")
                tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model = None  # Will skip embedding extraction

            overall_aqi = run_full_evaluation(
                model, tokenizer,
                model_info['display_name'],
                model_info['output_subdir'],
                balanced_eval_df
            )

            results_summary[model_key] = overall_aqi

            # Cleanup
            if model is not None:
                del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n❌ Error evaluating {model_info['display_name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create comprehensive comparison
    create_comprehensive_comparison()

    # Final summary
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nEvaluated {len(results_summary)} models")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
