"""
Dual-Metric Evaluation for 4 Baseline Models
Evaluates: Baseline (Unaligned), SFT, DPO, CITA
Metrics: Harmlessness (PKU test) + Helpfulness (AlpacaEval)
LLM-as-judge: Llama-3-70B via Fireworks AI

IMPORTANT: Loads models from HuggingFace (not local paths)
Since training instances auto-shutdown after pushing to HF, evaluation runs on fresh instances.

Usage:
    # Sanity check: 50+50 samples (~5-10 min, ~$0.10)
    python3 comparative_study/05_evaluation/llm_as_judge/dual_metric.py --mode sanity

    # Full evaluation: 1000+805 samples (~40-60 min, ~$1.80)
    python3 comparative_study/05_evaluation/llm_as_judge/dual_metric.py --mode full

    # Evaluate specific models
    python3 comparative_study/05_evaluation/llm_as_judge/dual_metric.py --mode sanity --models SFT_Baseline CITA_Baseline

    # Custom sample counts (overrides --mode)
    python3 comparative_study/05_evaluation/llm_as_judge/dual_metric.py --harmlessness-samples 100 --helpfulness-samples 100

Note: Uses BF16 precision only (required for Tier-1 publication quality)
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

# Add utils to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))
sys.path.insert(0, str(Path(__file__).parent / "utils"))

from model_utils import load_hf_token, get_model_repo_name
from prompts import get_harmlessness_prompt, get_helpfulness_prompt
from fireworks_client import FireworksJudge
from statistical_analysis import run_statistical_analysis
from logging_utils import setup_training_logger, restore_logging

# ===================================================================
# Configuration
# ===================================================================

# Model configurations - Using HuggingFace repos only (no local paths)
# All models use Llama-3 chat template (matching training scripts)
MODELS = {
    "Baseline": {
        "hf_repo": None,  # No adapter - just base model
        "display_name": "Baseline (Unaligned)",
    },
    "SFT_Baseline": {
        "hf_repo": get_model_repo_name("SFT_Baseline", precision="bf16"),
        "display_name": "SFT Baseline",
    },
    "DPO_Baseline": {
        "hf_repo": get_model_repo_name("DPO_Baseline", precision="bf16"),
        "display_name": "DPO Baseline",
    },
    "CITA_Baseline": {
        "hf_repo": get_model_repo_name("CITA_Baseline", precision="bf16"),
        "display_name": "CITA Baseline",
    },
}

BASE_MODEL = "meta-llama/Llama-3.1-8B"
EVAL_OUTPUT_DIR = Path(__file__).parent / "DualMetric_Evaluation_Results"
EVAL_OUTPUT_DIR.mkdir(exist_ok=True)

# Load HF_TOKEN for gated model access
HF_TOKEN = load_hf_token(project_root)

# ===================================================================
# Dataset Loaders
# ===================================================================

def load_harmlessness_test_set(max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load PKU-SafeRLHF test split for harmlessness evaluation

    Returns:
        DataFrame with columns: ['prompt', 'harm_categories', 'severity']
    """
    print("\n" + "="*80)
    print("Loading PKU-SafeRLHF Test Split (Harmlessness)")
    print("="*80)

    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="test")

    # Filter for harmful prompts only (at least one unsafe response)
    harmful_dataset = dataset.filter(
        lambda x: not x['is_response_0_safe'] or not x['is_response_1_safe']
    )

    if max_samples:
        harmful_dataset = harmful_dataset.select(range(min(max_samples, len(harmful_dataset))))

    # Extract prompts and harm categories
    df_rows = []
    for example in harmful_dataset:
        # Get harm categories from unsafe response
        if not example['is_response_0_safe']:
            harm_cat = example['response_0_harm_category']
        else:
            harm_cat = example['response_1_harm_category']

        categories = [k for k, v in harm_cat.items() if v == True]

        df_rows.append({
            "prompt": example['prompt'],
            "harm_categories": categories,
            "severity": "unknown"  # PKU-SafeRLHF doesn't have severity in test split
        })

    df = pd.DataFrame(df_rows)
    print(f"✅ Loaded {len(df)} harmful prompts from PKU test split")
    print(f"   Harm category distribution:")
    all_cats = [cat for cats in df['harm_categories'] for cat in cats]
    from collections import Counter
    for cat, count in Counter(all_cats).most_common():
        print(f"      {cat}: {count}")

    return df


def load_helpfulness_test_set(max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load AlpacaEval (805 prompts) for helpfulness evaluation

    Returns:
        DataFrame with columns: ['prompt', 'reference_output']
    """
    print("\n" + "="*80)
    print("Loading AlpacaEval Test Set (Helpfulness)")
    print("="*80)

    # Download JSON directly from HuggingFace (bypass dataset script)
    import requests
    url = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval_gpt4_baseline.json"

    print(f"Downloading from: {url}")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data)

    if max_samples:
        df = df.head(max_samples)

    print(f"✅ Loaded {len(df)} prompts from AlpacaEval")
    return df[['instruction', 'output']].rename(columns={'instruction': 'prompt', 'output': 'reference_output'})


def load_air_bench_test_set(max_samples: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Load AIR-Bench 2024 (5,694 prompts, 314 risk categories) - OPTIONAL

    Returns:
        DataFrame with columns: ['prompt', 'risk_category', 'risk_tier'] or None
    """
    print("\n" + "="*80)
    print("Loading AIR-Bench 2024 Test Set (Optional)")
    print("="*80)

    try:
        dataset = load_dataset("stanford-crfm/air-bench-2024", split="test")

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        df = pd.DataFrame({
            "prompt": dataset['prompt'],
            "risk_category": dataset['risk_category'],
            "risk_tier": dataset['tier']
        })

        print(f"✅ Loaded {len(df)} prompts from AIR-Bench 2024")
        return df
    except Exception as e:
        print(f"⚠️  AIR-Bench 2024 not available: {e}")
        return None


# ===================================================================
# Model Loader (from HuggingFace)
# ===================================================================

def load_model_for_eval(model_key: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model from HuggingFace in BF16 (with or without adapter)

    Args:
        model_key: One of ["Baseline", "SFT_Baseline", "DPO_Baseline", "CITA_Baseline"]

    Returns:
        Tuple of (model, tokenizer)

    Note:
        - All models use BF16 precision (Tier-1 publication quality)
        - All models use Llama-3 chat template (matching training)
        - Models loaded from HuggingFace (pushed by push_automation.py)
    """
    print(f"\n{'='*80}")
    print(f"Loading {MODELS[model_key]['display_name']} for evaluation")
    print(f"{'='*80}")

    model_info = MODELS[model_key]
    hf_repo = model_info["hf_repo"]

    if hf_repo:
        print(f"Loading from HuggingFace: {hf_repo}")
    else:
        print(f"Loading base model (no adapter)")

    # Load tokenizer from base model
    print(f"Loading tokenizer from base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    # Set Llama-3.1 chat template (same as training scripts)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if loop.first and message['role'] != 'system' %}"
            "{{ '<|begin_of_text|>' }}"
            "{% endif %}"
            "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )
        print("✅ Llama-3.1 chat template set (matching training scripts)")
    else:
        print("✅ Chat template already present")

    # Load base model in BF16 (required for Tier-1 publication quality)
    # Try Flash Attention 2 first, fallback to eager if unavailable
    print(f"Loading base model in BF16: {BASE_MODEL}")

    attn_impl = "flash_attention_2"
    try:
        print(f"Attempting to load with Flash Attention 2...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_impl,
            token=HF_TOKEN
        )
        print(f"✅ Using Flash Attention 2 (faster inference)")
    except Exception as e:
        print(f"⚠️  Flash Attention 2 unavailable: {type(e).__name__}")
        print(f"   Falling back to eager mode (20-30% slower)...")
        attn_impl = "eager"
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_impl,
            token=HF_TOKEN
        )
        print(f"✅ Using eager attention")

    # Load LoRA adapter from HuggingFace if specified
    if hf_repo is not None:
        print(f"📥 Downloading adapter from HuggingFace: {hf_repo}...")
        try:
            model = PeftModel.from_pretrained(model, hf_repo, token=HF_TOKEN)
            print("🔄 Merging adapter weights...")
            model = model.merge_and_unload()
            print("✅ Adapter merged")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load adapter from HuggingFace: {hf_repo}\n"
                f"Error: {e}\n\n"
                f"Possible causes:\n"
                f"  1. Model not yet trained and pushed to HuggingFace\n"
                f"  2. Training performance did not improve (push_automation skipped push)\n"
                f"  3. HuggingFace authentication issue (check HF_TOKEN in .env)\n"
            )
    else:
        print("Using base model (no adapter)")

    model.eval()
    print(f"✅ {MODELS[model_key]['display_name']} loaded successfully")

    return model, tokenizer


# ===================================================================
# Checkpointing Functions
# ===================================================================

def get_checkpoint_path(model_key: str, dataset_type: str) -> Path:
    """Get checkpoint file path for a model and dataset type"""
    checkpoint_dir = project_root / "comparative_study" / "05_evaluation" / "llm_as_judge" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{model_key}_{dataset_type}_checkpoint.json"


def save_checkpoint(
    model_key: str,
    dataset_type: str,
    responses: List[str],
    prompts: List[str],
    completed: bool = False
):
    """Save checkpoint with responses generated so far"""
    checkpoint_path = get_checkpoint_path(model_key, dataset_type)

    checkpoint_data = {
        "model_key": model_key,
        "dataset_type": dataset_type,
        "n_prompts_total": len(prompts),
        "n_responses_completed": len(responses),
        "completed": completed,
        "responses": responses,
        "timestamp": pd.Timestamp.now().isoformat()
    }

    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

    status = "✅ COMPLETED" if completed else f"💾 Checkpoint saved ({len(responses)}/{len(prompts)} prompts)"
    print(f"{status}: {checkpoint_path.name}")


def load_checkpoint(model_key: str, dataset_type: str) -> Optional[Dict]:
    """Load checkpoint if exists"""
    checkpoint_path = get_checkpoint_path(model_key, dataset_type)

    if not checkpoint_path.exists():
        return None

    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)

    print(f"📂 Found checkpoint: {checkpoint['n_responses_completed']}/{checkpoint['n_prompts_total']} prompts")
    return checkpoint


# ===================================================================
# Response Generation
# ===================================================================

def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    model_key: str,
    dataset_type: str,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    checkpoint_interval: int = 100
    ) -> List[str]:
    """
    Generate responses using Llama-3 chat template (matches training)

    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer (with Llama-3 chat template)
        prompts: List of user prompts
        model_key: Model identifier (for checkpointing)
        dataset_type: "harmlessness" or "helpfulness" (for checkpointing)
        max_new_tokens: Max tokens to generate
        batch_size: Batch size for generation
        checkpoint_interval: Save checkpoint every N prompts

    Returns:
        List of generated responses
    """
    from tqdm import tqdm

    # Check for existing checkpoint
    checkpoint = load_checkpoint(model_key, dataset_type)

    if checkpoint and checkpoint['completed']:
        print(f"✅ Inference already completed for {model_key} ({dataset_type})")
        return checkpoint['responses']

    # Resume from checkpoint if exists
    if checkpoint and not checkpoint['completed']:
        responses = checkpoint['responses']
        start_idx = len(responses)
        print(f"🔄 Resuming from prompt {start_idx + 1}/{len(prompts)}")
    else:
        responses = []
        start_idx = 0

    for i in tqdm(range(start_idx, len(prompts), batch_size), desc="Generating (Llama-3)"):
        batch_prompts = prompts[i:i+batch_size]

        # Format all prompts using Llama-3 chat template (matches training)
        formatted = []
        for p in batch_prompts:
            messages = [{"role": "user", "content": p}]
            formatted.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))

        # Tokenize
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode (skip prompt)
        for j, output in enumerate(outputs):
            prompt_length = inputs['input_ids'][j].shape[0]
            generated_ids = output[prompt_length:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            responses.append(response.strip())

        # Checkpoint every N prompts
        if len(responses) % checkpoint_interval == 0:
            save_checkpoint(model_key, dataset_type, responses, prompts, completed=False)

    # Final checkpoint after all prompts completed
    save_checkpoint(model_key, dataset_type, responses, prompts, completed=True)

    return responses


# ===================================================================
# Dual-Metric Evaluation
# ===================================================================

def evaluate_harmlessness(
    prompts: List[str],
    responses: List[str],
    harm_categories: List[List[str]],
    judge: FireworksJudge
    ) -> pd.DataFrame:
    """
    Evaluate harmlessness using LLM-as-judge

    Returns:
        DataFrame with columns: ['prompt', 'response', 'refusal_score', 'harm_categories', 'reasoning']
    """
    print("\n" + "="*80)
    print("Evaluating Harmlessness (LLM-as-judge)")
    print("="*80)

    # Create evaluation prompts
    eval_prompts = [
        get_harmlessness_prompt(p, r) for p, r in zip(prompts, responses)
    ]

    # Judge all responses
    results = judge.judge_batch(eval_prompts, batch_size=10)

    # Parse results
    df_rows = []
    for i, (p, r, cats, result) in enumerate(zip(prompts, responses, harm_categories, results)):
        if "error" in result:
            print(f"⚠️  Error in evaluation {i}: {result['error']}")
            refusal_score = None
            reasoning = f"Error: {result.get('error')}"
        else:
            refusal_score = result.get('refusal_score', None)
            reasoning = result.get('reasoning', '')

        df_rows.append({
            "prompt": p,
            "response": r,
            "refusal_score": refusal_score,
            "harm_categories": cats,
            "reasoning": reasoning
        })

    df = pd.DataFrame(df_rows)

    # Summary statistics
    valid_scores = df[df['refusal_score'].notna()]['refusal_score']
    if len(valid_scores) > 0:
        print(f"✅ Harmlessness Evaluation Complete")
        print(f"   Mean refusal score: {valid_scores.mean():.2f} / 10")
        print(f"   Median: {valid_scores.median():.2f}")
        print(f"   Std: {valid_scores.std():.2f}")
        print(f"   Valid evaluations: {len(valid_scores)} / {len(df)}")

    return df


def evaluate_helpfulness(
    prompts: List[str],
    responses: List[str],
    judge: FireworksJudge
    ) -> pd.DataFrame:
    """
    Evaluate helpfulness using LLM-as-judge

    Returns:
        DataFrame with columns: ['prompt', 'response', 'helpfulness_score', 'reasoning']
    """
    print("\n" + "="*80)
    print("Evaluating Helpfulness (LLM-as-judge)")
    print("="*80)

    # Create evaluation prompts
    eval_prompts = [
        get_helpfulness_prompt(p, r) for p, r in zip(prompts, responses)
    ]

    # Judge all responses
    results = judge.judge_batch(eval_prompts, batch_size=10)

    # Parse results
    df_rows = []
    for i, (p, r, result) in enumerate(zip(prompts, responses, results)):
        if "error" in result:
            print(f"⚠️  Error in evaluation {i}: {result['error']}")
            helpfulness_score = None
            reasoning = f"Error: {result.get('error')}"
        else:
            helpfulness_score = result.get('helpfulness_score', None)
            reasoning = result.get('reasoning', '')

        df_rows.append({
            "prompt": p,
            "response": r,
            "helpfulness_score": helpfulness_score,
            "reasoning": reasoning
        })

    df = pd.DataFrame(df_rows)

    # Summary statistics
    valid_scores = df[df['helpfulness_score'].notna()]['helpfulness_score']
    if len(valid_scores) > 0:
        print(f"✅ Helpfulness Evaluation Complete")
        print(f"   Mean helpfulness score: {valid_scores.mean():.2f} / 10")
        print(f"   Median: {valid_scores.median():.2f}")
        print(f"   Std: {valid_scores.std():.2f}")
        print(f"   Valid evaluations: {len(valid_scores)} / {len(df)}")

    return df


# ===================================================================
# Main Evaluation Pipeline
# ===================================================================

def run_dual_metric_evaluation(
    model_key: str,
    harmlessness_test: pd.DataFrame,
    helpfulness_test: pd.DataFrame,
    judge: FireworksJudge
    ) -> Dict:
    """
    Run full dual-metric evaluation for a single model (BF16 only)

    Returns:
        Dict with harmlessness_df, helpfulness_df, summary_stats
    """
    print(f"\n{'='*80}")
    print(f"DUAL-METRIC EVALUATION: {model_key}")
    print(f"{'='*80}")

    # Check if both datasets already completed
    harm_checkpoint = load_checkpoint(model_key, "harmlessness")
    help_checkpoint = load_checkpoint(model_key, "helpfulness")

    both_completed = (
        harm_checkpoint and harm_checkpoint['completed'] and
        help_checkpoint and help_checkpoint['completed']
    )

    if both_completed:
        print(f"\n{'='*80}")
        print(f"✅ INFERENCE ALREADY COMPLETED for {model_key}")
        print(f"{'='*80}")
        print(f"  Harmlessness: {harm_checkpoint['n_responses_completed']} responses")
        print(f"  Helpfulness:  {help_checkpoint['n_responses_completed']} responses")
        print(f"{'='*80}")
        print("\nChoose action:")
        print("  1) Evaluate using cached responses (skip inference)")
        print("  2) Re-run inference (overwrite checkpoints)")
        print("="*80)

        while True:
            choice = input("\nEnter choice (1 or 2): ").strip()
            if choice == "1":
                # Use cached responses
                harm_responses = harm_checkpoint['responses']
                help_responses = help_checkpoint['responses']
                # Skip model loading (already returned early)
                model = None
                tokenizer = None
                break
            elif choice == "2":
                # Re-run inference - DELETE checkpoints first
                print("\n🔄 Re-running inference from scratch...")
                harm_checkpoint_path = get_checkpoint_path(model_key, "harmlessness")
                help_checkpoint_path = get_checkpoint_path(model_key, "helpfulness")

                if harm_checkpoint_path.exists():
                    harm_checkpoint_path.unlink()
                    print(f"   🗑️  Deleted: {harm_checkpoint_path.name}")

                if help_checkpoint_path.exists():
                    help_checkpoint_path.unlink()
                    print(f"   🗑️  Deleted: {help_checkpoint_path.name}")

                break
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")

    # Load model only if needed
    if not both_completed or (both_completed and choice == "2"):
        model, tokenizer = load_model_for_eval(model_key)

        # Generate responses for harmlessness test
        print(f"\n--- Harmlessness Test ({len(harmlessness_test)} prompts) ---")
        harm_responses = generate_responses(
            model, tokenizer,
            harmlessness_test['prompt'].tolist(),
            model_key=model_key,
            dataset_type="harmlessness",
            max_new_tokens=256
        )

        # Generate responses for helpfulness test
        print(f"\n--- Helpfulness Test ({len(helpfulness_test)} prompts) ---")
        help_responses = generate_responses(
            model, tokenizer,
            helpfulness_test['prompt'].tolist(),
            model_key=model_key,
            dataset_type="helpfulness",
            max_new_tokens=256
        )

    # Cleanup model (if loaded)
    if model is not None:
        del model, tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Evaluate harmlessness
    harmlessness_df = evaluate_harmlessness(
        harmlessness_test['prompt'].tolist(),
        harm_responses,
        harmlessness_test['harm_categories'].tolist(),
        judge
    )

    # Evaluate helpfulness
    helpfulness_df = evaluate_helpfulness(
        helpfulness_test['prompt'].tolist(),
        help_responses,
        judge
    )

    # Calculate summary statistics
    harm_mean = harmlessness_df['refusal_score'].mean()
    help_mean = helpfulness_df['helpfulness_score'].mean()

    summary = {
        "model": model_key,
        "harmlessness_mean": harm_mean,
        "helpfulness_mean": help_mean,
        "harmlessness_n": len(harmlessness_df),
        "helpfulness_n": len(helpfulness_df)
    }

    print(f"\n{'='*80}")
    print(f"SUMMARY: {model_key}")
    print(f"{'='*80}")
    print(f"  Harmlessness: {harm_mean:.2f} / 10 (n={len(harmlessness_df)})")
    print(f"  Helpfulness:  {help_mean:.2f} / 10 (n={len(helpfulness_df)})")
    print(f"{'='*80}")

    return {
        "harmlessness_df": harmlessness_df,
        "helpfulness_df": helpfulness_df,
        "summary": summary
    }


# ===================================================================
# Main Script
# ===================================================================

def main():
    import argparse

    # Setup logging to capture ALL terminal output
    log_file, log_filename, original_stdout, original_stderr = setup_training_logger(
        run_name="dual_metric_evaluation",
        project_root=project_root
    )

    try:
        main_inner()
    finally:
        # Restore logging on exit
        restore_logging(log_file, original_stdout, original_stderr)
        print(f"\n✅ Log saved to: {log_filename}")


def main_inner():
    import argparse
    import shutil

    parser = argparse.ArgumentParser(
        description="Dual-Metric Evaluation (Harmlessness + Helpfulness)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Evaluate all 3 baselines (loads from HuggingFace)
        python dual_metric_eval.py

        # Evaluate specific models
        python dual_metric_eval.py --models SFT_Baseline CITA_Baseline

        # Quick test with small samples
        python dual_metric_eval.py --harmlessness-samples 100 --helpfulness-samples 100

        Note: All models loaded in BF16 (Tier-1 publication quality)
        Note: Models are loaded from HuggingFace (pushed by training scripts via push_automation.py)
        """
    )
    parser.add_argument("--mode", choices=["sanity", "full"], default="full",
                       help="Evaluation mode: sanity (50+50 samples) or full (1000+805 samples)")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                       help="Models to evaluate (default: all 4)")
    parser.add_argument("--harmlessness-samples", type=int, default=None,
                       help="Max samples for harmlessness test (overrides --mode)")
    parser.add_argument("--helpfulness-samples", type=int, default=None,
                       help="Max samples for helpfulness test (overrides --mode)")

    args = parser.parse_args()

    # ===================================================================
    # Cleanup Helper: Handle cached data from previous runs
    # ===================================================================
    checkpoint_dir = project_root / "comparative_study" / "05_evaluation" / "llm_as_judge" / "checkpoints"
    results_dir = EVAL_OUTPUT_DIR

    checkpoints_exist = checkpoint_dir.exists() and any(checkpoint_dir.iterdir())
    results_exist = results_dir.exists() and any(results_dir.iterdir())

    if checkpoints_exist or results_exist:
        print("\n" + "="*80)
        print("🗑️  CLEANUP: Cached Data Detected")
        print("="*80)
        print("\nWhat changed since last evaluation?")
        print("\n  1) Nothing - Use cached data (fastest)")
        print("  2) New LLM-as-Judge - Keep model responses, delete old evaluations")
        print("     (e.g., switched from Llama-3.3-70B to GPT-4)")
        print("  3) Retrained Models - Delete all cached data")
        print("     (e.g., new LoRA adapters pushed to HuggingFace)")
        print("="*80)

        while True:
            cleanup_choice = input("\nEnter choice (1, 2, or 3): ").strip()

            if cleanup_choice == "1":
                print("\n✅ Using cached data (no cleanup)")
                break

            elif cleanup_choice == "2":
                print("\n🗑️  Scenario: New LLM-as-Judge")
                print("   Keeping: checkpoints/ (model responses still valid)")
                print("   Deleting: DualMetric_Evaluation_Results/ (old judge scores)")

                if results_exist:
                    shutil.rmtree(results_dir)
                    print(f"   ✅ Deleted: {results_dir.name}/")

                results_dir.mkdir(exist_ok=True)
                break

            elif cleanup_choice == "3":
                print("\n🗑️  Scenario: Retrained Models")
                print("   Deleting: checkpoints/ (old model responses)")
                print("   Deleting: DualMetric_Evaluation_Results/ (old evaluations)")

                if checkpoints_exist:
                    shutil.rmtree(checkpoint_dir)
                    print(f"   ✅ Deleted: {checkpoint_dir.name}/")

                if results_exist:
                    shutil.rmtree(results_dir)
                    print(f"   ✅ Deleted: {results_dir.name}/")

                checkpoint_dir.mkdir(exist_ok=True)
                results_dir.mkdir(exist_ok=True)
                break

            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")

        print("="*80)

    # Interactive mode selection if --mode not provided
    print("\n" + "="*80)
    print("🎯 DUAL-METRIC EVALUATION: Mode Selection")
    print("="*80)
    print("\nChoose evaluation mode:")
    print("  1) Sanity Check  - 50 harmful + 50 helpful prompts (~5-10 min, ~$0.10)")
    print("  2) Full Evaluation - 1000 harmful + 805 helpful prompts (~40-60 min, ~$1.80)")
    print("="*80)

    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice == "1":
            args.mode = "sanity"
            break
        elif choice == "2":
            args.mode = "full"
            break
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")

    # Set sample counts based on mode (if not explicitly provided)
    if args.mode == "sanity":
        args.harmlessness_samples = args.harmlessness_samples or 50
        args.helpfulness_samples = args.helpfulness_samples or 50
        print(f"\n✅ Sanity mode selected: {args.harmlessness_samples} + {args.helpfulness_samples} samples")
    else:  # full mode
        args.harmlessness_samples = args.harmlessness_samples or 1000
        args.helpfulness_samples = args.helpfulness_samples or 805
        print(f"\n✅ Full mode selected: {args.harmlessness_samples} + {args.helpfulness_samples} samples")

    # Initialize LLM judge
    judge = FireworksJudge()

    # Load test sets
    harmlessness_test = load_harmlessness_test_set(max_samples=args.harmlessness_samples)
    helpfulness_test = load_helpfulness_test_set(max_samples=args.helpfulness_samples)

    # Evaluate all models
    all_results = {}
    for model_key in args.models:
        if model_key not in MODELS:
            print(f"⚠️  Unknown model: {model_key}, skipping")
            continue

        try:
            results = run_dual_metric_evaluation(
                model_key,
                harmlessness_test,
                helpfulness_test,
                judge
            )

            all_results[model_key] = results

            # Save individual results
            model_dir = EVAL_OUTPUT_DIR / model_key
            model_dir.mkdir(exist_ok=True)

            results['harmlessness_df'].to_csv(model_dir / "harmlessness_results.csv", index=False)
            results['helpfulness_df'].to_csv(model_dir / "helpfulness_results.csv", index=False)

            with open(model_dir / "summary.json", 'w') as f:
                json.dump(results['summary'], f, indent=2)

        except RuntimeError as e:
            print(f"\n❌ Failed to evaluate {model_key}: {e}")
            print(f"   Skipping this model...")
            continue

    # Statistical analysis and Pareto plot (only if we have results)
    if len(all_results) >= 2:
        run_statistical_analysis(all_results, output_dir=EVAL_OUTPUT_DIR)
    else:
        print(f"\n⚠️  Need at least 2 models for statistical analysis (got {len(all_results)})")

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {EVAL_OUTPUT_DIR}")
    print(f"Evaluated models: {list(all_results.keys())}")


if __name__ == "__main__":
    main()
