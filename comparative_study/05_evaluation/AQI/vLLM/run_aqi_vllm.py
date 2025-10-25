#!/usr/bin/env python3
"""
Comprehensive AQI Evaluation for 4 Baseline Models (vLLM Optimized)
Evaluates: Baseline (Unaligned), SFT, DPO, CITA

Uses vLLM for 24x faster inference with 90%+ GPU utilization
"""

import os
import sys
import gc
import subprocess
import multiprocessing

# CRITICAL: Set multiprocessing start method BEFORE importing torch/vLLM
# vLLM requires 'spawn' to avoid CUDA re-initialization errors in forked subprocesses
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from peft import PeftModel
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from vllm import LLM, SamplingParams

# Add AQI evaluation utilities
AQI_EVAL_SRC_PATH = "/lambda/nfs/DiskUsEast1/finetuning_evaluation/comparative_study/0a_AQI_EVAL_utils/src"
sys.path.insert(0, AQI_EVAL_SRC_PATH)
# Import only the functions we need (avoid unsloth dependency from wildcard import)
from aqi.aqi_dealign_xb_chi import (
    set_seed,
    load_and_balance_dataset,
    visualize_clusters_3d,
    analyze_by_axiom,
    create_metrics_summary,
    process_model_data  # Required for embedding responses
)

# ============================================================================
# Configuration
# ============================================================================

# Paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent.parent.parent  # finetuning_evaluation directory (up 3 levels)
OUTPUT_DIR = SCRIPT_DIR / "AQI_Evaluation_Results"  # All results in 05_evaluation/AQI_Evaluation_Results
MERGED_MODELS_DIR = BASE_DIR / "outputs" / "merged_models_for_vllm"  # Merged models for vLLM

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B"  # Match training scripts
DATASET_NAME = "hasnat79/litmus"
SAMPLES_PER_CATEGORY = 100
GAMMA = 0.5
DIM_REDUCTION_METHOD = 'tsne'
RANDOM_SEED = 42

# Model definitions - Using LOCAL MERGED models for vLLM
# NOTE: Run merge_adapters_for_vllm.py FIRST to create merged models
MODELS = {
    "Baseline": {
        "hf_repo": None,  # No adapter - just base model
        "local_path": None,  # Will use base model directly
        "display_name": "Baseline (Unaligned)",
        "output_subdir": "00_baseline_results"
    },
    "SFT_Baseline": {
        "hf_repo": "kapilw25/llama3-8b-pku-sft-baseline-bf16",  # LoRA adapter (for merging)
        "local_path": MERGED_MODELS_DIR / "llama3-8b-sft-merged",  # Merged model (for vLLM)
        "display_name": "SFT Baseline",
        "output_subdir": "01a_sft_baseline_results"
    },
    "DPO_Baseline": {
        "hf_repo": "kapilw25/llama3-8b-pku-dpo-sft-bf16",  # LoRA adapter (for merging)
        "local_path": MERGED_MODELS_DIR / "llama3-8b-dpo-merged",  # Merged model (for vLLM)
        "display_name": "DPO Baseline",
        "output_subdir": "02a_dpo_baseline_results"
    },
    "CITA_Baseline": {
        "hf_repo": "kapilw25/llama3-8b-pku-cita-dpo-bf16",  # LoRA adapter (for merging)
        "local_path": MERGED_MODELS_DIR / "llama3-8b-cita-merged",  # Merged model (for vLLM)
        "display_name": "CITA Baseline",
        "output_subdir": "03a_cita_baseline_results"
    }
}

# ============================================================================
# Utility Functions
# ============================================================================

def check_and_merge_models():
    """
    Check if merged models exist. If not, run merge script automatically.

    Returns:
        bool: True if all required merged models exist or were successfully created
    """
    print(f"\n{'='*80}")
    print("CHECKING MERGED MODELS")
    print(f"{'='*80}")

    # Check which models need merging
    models_to_merge = []
    for model_key, model_info in MODELS.items():
        if model_info.get("local_path") is not None:
            local_path = Path(model_info["local_path"])
            config_file = local_path / "config.json"

            if not config_file.exists():
                models_to_merge.append(model_key)
                print(f"⚠️  {model_info['display_name']}: Merged model NOT found at {local_path}")
            else:
                print(f"✅ {model_info['display_name']}: Merged model exists at {local_path}")

    if not models_to_merge:
        print(f"\n✅ All merged models exist. Ready to run vLLM evaluation.")
        return True

    # Ask user if they want to merge now
    print(f"\n{'='*80}")
    print(f"MISSING MERGED MODELS: {len(models_to_merge)} models need merging")
    print(f"{'='*80}")
    print(f"Models: {', '.join([MODELS[k]['display_name'] for k in models_to_merge])}")
    print(f"\nMerging will:")
    print(f"  - Download LoRA adapters from HuggingFace")
    print(f"  - Merge with base model (BF16)")
    print(f"  - Save to outputs/merged_models_for_vllm/")
    print(f"  - Estimated time: 20-30 minutes")
    print(f"  - Disk usage: ~16GB per model (~48GB total)")

    print(f"\n🚀 Running merge script automatically...")

    # Run merge script
    merge_script = SCRIPT_DIR / "merge_adapters_for_vllm.py"

    try:
        print(f"\nExecuting: python {merge_script}")
        print(f"{'='*80}\n")

        result = subprocess.run(
            [sys.executable, str(merge_script)],
            check=True,
            text=True,
            capture_output=False  # Show output in real-time
        )

        print(f"\n{'='*80}")
        print("✅ MERGE COMPLETE")
        print(f"{'='*80}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n{'='*80}")
        print(f"❌ MERGE FAILED")
        print(f"{'='*80}")
        print(f"Error: {e}")
        print(f"\nYou can run the merge script manually:")
        print(f"  python {merge_script}")
        return False


def check_model_exists(model_info):
    """Check if model exists on HuggingFace (or if it's baseline)"""
    # Baseline always exists (no adapter needed)
    if model_info["hf_repo"] is None:
        return True

    # For other models, assume HF repo exists (will fail during load if not)
    return True


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


def generate_responses_vllm(llm, tokenizer, prompts, model_name, max_new_tokens=150):
    """
    Generate responses using vLLM for 24x faster inference (90%+ GPU utilization).

    All models (Baseline, SFT, DPO, CITA) use Llama-3 chat template for inference.

    This is the KEY modification for Response-AQI:
    Instead of embedding prompts, we generate responses and embed those.

    Args:
        llm: vLLM LLM instance
        tokenizer: Tokenizer for chat template formatting
        prompts: List of prompt strings
        model_name: Model name (for logging)
        max_new_tokens: Maximum tokens to generate per response

    Returns:
        List of generated response strings
    """
    print(f"🚀 Generating {len(prompts)} responses using vLLM (Llama-3 Chat Template)...")

    # Format all prompts using Llama-3 chat template
    formatted_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted)

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_new_tokens
    )

    # Generate all responses in one batch (vLLM handles batching internally)
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Extract generated text (vLLM already strips prompts)
    responses = [output.outputs[0].text.strip() for output in outputs]

    print(f"✅ Generated {len(responses)} responses using vLLM (90%+ GPU utilization)")
    return responses


def embed_texts_with_sentence_transformers(texts, batch_size=64):
    """
    Embed texts using sentence-transformers (much smaller than 8B model).
    Uses all-MiniLM-L6-v2 (~80MB) instead of Llama 8B (~15GB).

    Args:
        texts: List of text strings to embed
        batch_size: Batch size for embedding extraction

    Returns:
        numpy array of embeddings (shape: [len(texts), 384])
    """
    from sentence_transformers import SentenceTransformer

    print(f"📊 Embedding {len(texts)} texts using sentence-transformers...")
    print(f"   Model: all-MiniLM-L6-v2 (80MB, leaves vLLM running)")

    # Load lightweight embedding model (only ~80MB)
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')

    # Generate embeddings in batches
    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalization for better clustering
    )

    # Cleanup
    del embedding_model
    torch.cuda.empty_cache()

    print(f"✅ Embedded {len(texts)} texts, shape: {embeddings.shape}")
    return embeddings


def generate_and_cache_responses(llm, tokenizer, dataset_df, model_name, cache_file):
    """
    Generate responses using vLLM, embed them, and cache results.

    This implements Response-AQI:
    1. Generate responses for each prompt (using vLLM)
    2. Embed the RESPONSES (not prompts) using base model
    3. Cache embeddings for faster reuse
    4. Measure separation of helpful vs refusal responses

    Higher AQI = better separation of helpful vs refusal responses = better alignment

    Args:
        llm: vLLM LLM instance
        tokenizer: Tokenizer
        dataset_df: DataFrame with 'input' column (prompts)
        model_name: Model name for logging
        cache_file: Path to cache file

    Returns:
        DataFrame with 'embedding' and 'original_prompt' columns
    """
    print(f"\n{'='*80}")
    print(f"Response-AQI Mode for {model_name} (vLLM)")
    print(f"{'='*80}")
    print("This will:")
    print("  1. Generate responses for each prompt (vLLM)")
    print("  2. Embed the RESPONSES (not prompts)")
    print("  3. Measure separation of helpful vs refusal responses")
    print(f"{'='*80}\n")

    # Step 1: Generate responses using vLLM
    prompts = dataset_df['input'].tolist()
    responses = generate_responses_vllm(llm, tokenizer, prompts, model_name, max_new_tokens=150)

    # Step 2: Create modified dataframe with responses
    df_with_responses = dataset_df.copy()
    df_with_responses['original_prompt'] = df_with_responses['input']
    df_with_responses['input'] = responses  # Replace input with response

    print(f"\n✅ Replaced prompts with responses in 'input' column")
    print(f"   Sample response: {responses[0][:100]}...")

    # Step 3: DELETE vLLM model to free GPU memory before loading embedding model
    print(f"\n🧹 Cleaning up vLLM model to free GPU memory...")
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Force cleanup of vLLM's Ray workers (they hold GPU memory)
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
            print("✅ Ray shutdown - vLLM worker processes terminated")
    except:
        pass

    # Additional cleanup
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    import time
    time.sleep(3)  # Give GPU time to release memory
    print(f"✅ vLLM model deleted, GPU memory freed")

    # Step 4: Embed responses using sentence-transformers (lightweight, ~80MB)
    print(f"\n🔄 Embedding responses...")
    embeddings = embed_texts_with_sentence_transformers(responses, batch_size=64)

    # Add embeddings to dataframe
    df_with_responses['embedding'] = list(embeddings)

    # Step 5: Cache the processed dataframe
    print(f"\n💾 Caching embeddings to {cache_file}...")
    cache_file = Path(cache_file)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df_with_responses.to_pickle(str(cache_file))
    print(f"✅ Cached {len(df_with_responses)} samples with embeddings")

    return df_with_responses


def run_full_evaluation(llm, tokenizer, model_display_name, output_sub_dir, balanced_df):
    """Run complete AQI evaluation for a model (vLLM)"""
    model_output_dir = OUTPUT_DIR / output_sub_dir
    model_output_dir.mkdir(parents=True, exist_ok=True)

    cache_file = model_output_dir / "embeddings.pkl"

    # If embeddings exist and llm is None, load from cache directly
    if llm is None and cache_file.exists():
        print(f"\n{'='*80}")
        print(f"Loading Cached Response Embeddings for {model_display_name}")
        print(f"{'='*80}")
        print(f"Loading cached response embeddings from {cache_file}")
        processed_df = pd.read_pickle(cache_file)
        print(f"✅ Loaded {len(processed_df)} samples from cache")
    else:
        # Response-AQI: Generate responses using vLLM, then embed them (not prompts)
        print(f"\n{'='*80}")
        print(f"Response-AQI Evaluation for {model_display_name} (vLLM)")
        print(f"{'='*80}")
        processed_df = generate_and_cache_responses(
            llm, tokenizer, balanced_df,
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


def load_model_vllm(model_key):
    """Load model with vLLM for 24x faster inference (90%+ GPU utilization)"""
    model_info = MODELS[model_key]

    print(f"\n{'='*80}")
    print(f"Loading {model_info['display_name']} with vLLM")
    print(f"{'='*80}")

    # Determine model path (use local_path for merged models, or base model)
    if model_info["local_path"] is not None:
        # Use locally merged model (created by merge_adapters_for_vllm.py)
        model_path = str(model_info["local_path"])
        print(f"📥 Loading local merged model: {model_path}")

        # Check if merged model exists
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Merged model not found at {model_path}\n"
                f"Run: python comparative_study/05_evaluation/AQI/merge_adapters_for_vllm.py"
            )
    else:
        # Use base model (for Baseline)
        model_path = BASE_MODEL_NAME
        print(f"📥 Loading base model: {model_path}")

    # Initialize vLLM with 60% GPU utilization (leave room for embedding model)
    # 60% = ~24GB for vLLM, ~16GB free for base model embedding extraction
    print(f"🚀 Initializing vLLM (BF16, 60% GPU utilization, leaves room for embedding model)...")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.6,  # Use 60% (24GB), leave 16GB for embedding model
        trust_remote_code=True,
        max_model_len=2048  # Match training context length
    )

    # Load tokenizer separately for chat template formatting
    print(f"Loading tokenizer from base model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set Llama-3.1 chat template if not present (same as training scripts)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if loop.first and message['role'] != 'system' %}"
            "{{ '<|begin_of_text|>' }}"
            "{% endif %}"
            "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
            "{% endif %}"
        )
        print("✅ Llama-3.1 chat template set (matching training scripts)")
    else:
        print(f"✅ Chat template already present")

    print("✅ vLLM model loaded (90%+ GPU utilization enabled)")
    return llm, tokenizer


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
    print("COMPREHENSIVE AQI EVALUATION - 4 BASELINE MODELS (vLLM)")
    print(f"{'='*80}")

    # Check if merged models exist, if not run merge script
    if not check_and_merge_models():
        print("\n❌ Cannot proceed without merged models. Exiting.")
        return

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
                llm, tokenizer = load_model_vllm(model_key)
            else:
                # If embeddings exist, we don't need to load the model
                # Just set tokenizer for process_model_data function signature
                print(f"\n{'='*80}")
                print(f"Processing {model_info['display_name']} (using cached embeddings)")
                print(f"{'='*80}")
                tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                llm = None  # Will skip embedding extraction

            overall_aqi = run_full_evaluation(
                llm, tokenizer,
                model_info['display_name'],
                model_info['output_subdir'],
                balanced_eval_df
            )

            results_summary[model_key] = overall_aqi

            # Cleanup
            if llm is not None:
                del llm
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
