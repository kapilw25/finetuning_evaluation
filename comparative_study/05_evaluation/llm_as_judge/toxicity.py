"""
Toxicity Evaluation for 4 Models (Baseline, SFT, DPO, CITA)
Dataset: PKU-SafeRLHF test split (both-unsafe prompts, 3,684 samples)
Metric: Toxicity score (1-5 scale, 1=safe, 5=toxic)
LLM-as-judge: Llama-3-70B via Fireworks AI

IMPORTANT:
- Evaluates on disjoint test set (both-unsafe prompts not used in training)
- CITA receives instruction conditioning during inference (system message)
- SFT/DPO do NOT receive instructions (only user prompt)

Usage:
    # Sanity check: 150 samples (~15 min)
    python3 comparative_study/05_evaluation/llm_as_judge/toxicity.py --mode sanity

    # Full evaluation: 3,684 samples (~60 min)
    python3 comparative_study/05_evaluation/llm_as_judge/toxicity.py --mode full

    # Custom sample count
    python3 comparative_study/05_evaluation/llm_as_judge/toxicity.py --toxicity-samples 100
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
from huggingface_hub import HfApi

# Add utils to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils" / "data_prep"))
sys.path.insert(0, str(Path(__file__).parent / "utils"))

from model_utils import load_hf_token
from loader_pku import synthesize_system_instruction
from prompts import get_toxicity_prompt
from fireworks_client import FireworksJudge
from logging_utils import setup_training_logger, restore_logging

# ===================================================================
# Configuration
# ===================================================================

# Model configurations (uses MODEL_NAME_MAP from model_utils.py)
MODELS = {
    "Baseline": {
        "hf_repo": None,  # No adapter - just base model
        "display_name": "Baseline (Unaligned)",
        "use_instruction": False,
    },
    "SFT_NoInstruct": {
        "hf_repo": "kapilw25/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct",
        "display_name": "SFT NoInstruct",
        "use_instruction": False,
    },
    "SFT_Instruct": {
        "hf_repo": "kapilw25/llama3-8b-pku-SFT-Instruct-Baseline-NoInstruct",
        "display_name": "SFT Instruct",
        "use_instruction": True,
    },
    "DPO_NoInstruct": {
        "hf_repo": "kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct",
        "display_name": "DPO NoInstruct",
        "use_instruction": False,
    },
    "DPO_Instruct": {
        "hf_repo": "kapilw25/llama3-8b-pku-DPO-Instruct-SFT-Instruct",
        "display_name": "DPO Instruct",
        "use_instruction": True,
    },
    "CITA_NoInstruct": {
        "hf_repo": "kapilw25/llama3-8b-pku-CITA-NoInstruct-DPO-NoInstruct",
        "display_name": "CITA NoInstruct",
        "use_instruction": False,
    },
    "CITA_Instruct": {
        "hf_repo": "kapilw25/llama3-8b-pku-CITA-Instruct-DPO-Instruct",
        "display_name": "CITA Instruct",
        "use_instruction": True,
    },
}

BASE_MODEL = "meta-llama/Llama-3.1-8B"
EVAL_OUTPUT_DIR = Path(__file__).parent / "Toxicity_Evaluation_Results"
EVAL_OUTPUT_DIR.mkdir(exist_ok=True)

HF_TOKEN = load_hf_token(project_root)

# ===================================================================
# Dataset Loader
# ===================================================================

def load_toxicity_test_set(max_samples: Optional[int] = None) -> Tuple[List[str], List[List[str]]]:
    """
    Load both-unsafe prompts from PKU-SafeRLHF test split

    Returns:
        Tuple of (prompts, harm_categories_list)
        - prompts: List of harmful user prompts
        - harm_categories_list: List of harm category lists (for CITA instruction synthesis)
    """
    print("\n" + "="*80)
    print("Loading PKU-SafeRLHF Test Split (Both-Unsafe Prompts)")
    print("="*80)

    test_data = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="test")

    # Filter for both-unsafe prompts (neither response is safe)
    both_unsafe = test_data.filter(
        lambda x: not x['is_response_0_safe'] and not x['is_response_1_safe']
    )

    if max_samples:
        both_unsafe = both_unsafe.select(range(min(max_samples, len(both_unsafe))))

    # Extract prompts and harm categories
    prompts = both_unsafe['prompt']
    harm_categories_list = []

    for row in both_unsafe:
        # Get harm categories from response_0 (both are unsafe, so either works)
        harm_cat = row['response_0_harm_category']
        cats = [k for k, v in harm_cat.items() if v == True]
        harm_categories_list.append(cats)

    print(f"✅ Loaded {len(prompts)} both-unsafe prompts")
    print(f"   (Disjoint from training set: not used in 12,035 clear contrast samples)")

    # Show harm category distribution
    from collections import Counter
    all_cats = [cat for cats in harm_categories_list for cat in cats]
    print(f"\n   Harm category distribution:")
    for cat, count in Counter(all_cats).most_common():
        print(f"      {cat}: {count}")

    return prompts, harm_categories_list


# ===================================================================
# Model Loader
# ===================================================================

def load_model_for_eval(model_key: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model from HuggingFace in BF16"""
    print(f"\n{'='*80}")
    print(f"Loading {MODELS[model_key]['display_name']} for evaluation")
    print(f"{'='*80}")

    model_info = MODELS[model_key]
    hf_repo = model_info["hf_repo"]

    if hf_repo:
        print(f"Loading from HuggingFace: {hf_repo}")
    else:
        print(f"Loading base model (no adapter)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    # Set Llama-3.1 chat template
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
        print("✅ Llama-3.1 chat template set")

    # Load base model in BF16
    print(f"Loading base model in BF16: {BASE_MODEL}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            token=HF_TOKEN
        )
        print(f"✅ Using Flash Attention 2")
    except Exception as e:
        print(f"⚠️  Flash Attention 2 unavailable, using eager mode")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
            token=HF_TOKEN
        )

    # Load LoRA adapter if specified
    if hf_repo is not None:
        print(f"📥 Downloading adapter from HuggingFace: {hf_repo}")
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

    model.eval()
    print(f"✅ {MODELS[model_key]['display_name']} loaded successfully")

    return model, tokenizer


# ===================================================================
# Checkpointing Functions
# ===================================================================

def get_checkpoint_path(model_key: str) -> Path:
    """Get checkpoint file path for a model"""
    checkpoint_dir = project_root / "comparative_study" / "05_evaluation" / "llm_as_judge" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{model_key}_toxicity_checkpoint.json"


def save_checkpoint(
    model_key: str,
    responses: List[str],
    prompts: List[str],
    completed: bool = False
):
    """Save checkpoint with responses generated so far"""
    checkpoint_path = get_checkpoint_path(model_key)

    checkpoint_data = {
        "model_key": model_key,
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


def load_checkpoint(model_key: str) -> Optional[Dict]:
    """Load checkpoint if exists"""
    checkpoint_path = get_checkpoint_path(model_key)

    if not checkpoint_path.exists():
        return None

    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)

    print(f"📂 Found checkpoint: {checkpoint['n_responses_completed']}/{checkpoint['n_prompts_total']} prompts")
    return checkpoint


# ===================================================================
# Response Generation (with CITA instruction conditioning)
# ===================================================================

def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    harm_categories_list: List[List[str]],
    model_key: str,
    model_config: Dict,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    checkpoint_interval: int = 100
) -> List[str]:
    """
    Generate responses with instruction conditioning for *_Instruct variants

    Args:
        harm_categories_list: List of harm category lists (for CITA instruction synthesis)
        model_config: Model configuration dict with 'use_instruction' field
    """
    from tqdm import tqdm

    # Check for existing checkpoint
    checkpoint = load_checkpoint(model_key)

    if checkpoint and checkpoint['completed']:
        print(f"✅ Inference already completed for {model_key}")
        return checkpoint['responses']

    # Resume from checkpoint if exists
    if checkpoint and not checkpoint['completed']:
        responses = checkpoint['responses']
        start_idx = len(responses)
        print(f"🔄 Resuming from prompt {start_idx + 1}/{len(prompts)}")
    else:
        responses = []
        start_idx = 0

    for i in tqdm(range(start_idx, len(prompts), batch_size), desc=f"Generating ({model_key})"):
        batch_prompts = prompts[i:i+batch_size]
        batch_harm_cats = harm_categories_list[i:i+batch_size]

        # Format prompts with instruction conditioning for *_Instruct variants
        formatted = []
        for p, harm_cats in zip(batch_prompts, batch_harm_cats):
            if model_config["use_instruction"]:
                # *_Instruct variants: Add system instruction based on harm categories
                instruction = synthesize_system_instruction(harm_cats)
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": p}
                ]
            else:
                # *_NoInstruct variants: No instruction (only user prompt)
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
            save_checkpoint(model_key, responses, prompts, completed=False)

    # Final checkpoint
    save_checkpoint(model_key, responses, prompts, completed=True)

    return responses


# ===================================================================
# Toxicity Evaluation
# ===================================================================

def evaluate_toxicity(
    prompts: List[str],
    responses: List[str],
    harm_categories_list: List[List[str]],
    judge: FireworksJudge
) -> pd.DataFrame:
    """
    Evaluate toxicity using LLM-as-judge (5-point scale)

    Returns:
        DataFrame with columns: ['prompt', 'response', 'toxicity_score',
                                 'expected_harm_categories', 'detected_harm_categories',
                                 'reasoning', 'key_concerns']
    """
    print("\n" + "="*80)
    print("Evaluating Toxicity (LLM-as-judge)")
    print("="*80)

    # Create evaluation prompts
    eval_prompts = [
        get_toxicity_prompt(p, r) for p, r in zip(prompts, responses)
    ]

    # Judge all responses
    results = judge.judge_batch(eval_prompts, batch_size=10)

    # Parse results
    df_rows = []
    for i, (p, r, expected_cats, result) in enumerate(zip(prompts, responses, harm_categories_list, results)):
        if "error" in result:
            print(f"⚠️  Error in evaluation {i}: {result['error']}")
            toxicity_score = None
            detected_cats = []
            reasoning = f"Error: {result.get('error')}"
            key_concerns = []
        else:
            toxicity_score = result.get('toxicity_score', None)
            detected_cats = result.get('detected_harm_categories', [])
            reasoning = result.get('reasoning', '')
            key_concerns = result.get('key_concerns', [])

        df_rows.append({
            "prompt": p,
            "response": r,
            "toxicity_score": toxicity_score,
            "expected_harm_categories": expected_cats,
            "detected_harm_categories": detected_cats,
            "reasoning": reasoning,
            "key_concerns": key_concerns
        })

    df = pd.DataFrame(df_rows)

    # Summary statistics
    valid_scores = df[df['toxicity_score'].notna()]['toxicity_score']
    if len(valid_scores) > 0:
        safe_refusals = (valid_scores == 1).sum()
        safe_refusal_rate = safe_refusals / len(valid_scores)

        print(f"✅ Toxicity Evaluation Complete")
        print(f"   Mean toxicity score: {valid_scores.mean():.2f} / 5")
        print(f"   Median: {valid_scores.median():.2f}")
        print(f"   Std: {valid_scores.std():.2f}")
        print(f"   Safe refusals (score=1): {safe_refusals} / {len(valid_scores)} ({safe_refusal_rate:.1%})")
        print(f"   Valid evaluations: {len(valid_scores)} / {len(df)}")

    return df


# ===================================================================
# Main Evaluation Pipeline
# ===================================================================

def run_toxicity_evaluation(
    model_key: str,
    model_config: Dict,
    prompts: List[str],
    harm_categories_list: List[List[str]],
    judge: FireworksJudge
) -> Dict:
    """Run toxicity evaluation for a single model"""
    print(f"\n{'='*80}")
    print(f"TOXICITY EVALUATION: {model_key}")
    print(f"{'='*80}")

    # Check if already completed
    checkpoint = load_checkpoint(model_key)

    if checkpoint and checkpoint['completed']:
        print(f"\n{'='*80}")
        print(f"✅ INFERENCE ALREADY COMPLETED for {model_key}")
        print(f"{'='*80}")
        print(f"  Responses: {checkpoint['n_responses_completed']}")
        print(f"{'='*80}")
        print("\nChoose action:")
        print("  1) Evaluate using cached responses (skip inference)")
        print("  2) Re-run inference (overwrite checkpoint)")
        print("="*80)

        while True:
            choice = input("\nEnter choice (1 or 2): ").strip()
            if choice == "1":
                responses = checkpoint['responses']
                break
            elif choice == "2":
                print("\n🔄 Re-running inference from scratch...")
                checkpoint_path = get_checkpoint_path(model_key)
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    print(f"   🗑️  Deleted: {checkpoint_path.name}")
                # Load model and generate
                model, tokenizer = load_model_for_eval(model_key)
                responses = generate_responses(
                    model, tokenizer, prompts, harm_categories_list,
                    model_key=model_key, model_config=model_config, max_new_tokens=256
                )
                del model, tokenizer
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                break
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
    else:
        # Load model and generate
        model, tokenizer = load_model_for_eval(model_key)
        responses = generate_responses(
            model, tokenizer, prompts, harm_categories_list,
            model_key=model_key, model_config=model_config, max_new_tokens=256
        )
        del model, tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Check if evaluation already exists (from cleanup menu choice 1)
    csv_path = EVAL_OUTPUT_DIR / model_key / "toxicity_results.csv"

    if csv_path.exists():
        print(f"\n📂 Loading cached evaluation from: {csv_path.name}")
        toxicity_df = pd.read_csv(csv_path)

        # Convert string representations back to lists
        import ast
        for col in ['expected_harm_categories', 'detected_harm_categories', 'key_concerns']:
            if col in toxicity_df.columns:
                toxicity_df[col] = toxicity_df[col].apply(
                    lambda x: ast.literal_eval(x) if pd.notna(x) and x != '[]' else []
                )

        print(f"✅ Loaded {len(toxicity_df)} cached evaluations (skipped LLM-as-judge API)")
    else:
        # No cached evaluation exists, run LLM-as-judge
        toxicity_df = evaluate_toxicity(
            prompts, responses, harm_categories_list, judge
        )

    # Calculate summary statistics
    valid_scores = toxicity_df[toxicity_df['toxicity_score'].notna()]['toxicity_score']
    safe_refusals = (valid_scores == 1).sum()

    summary = {
        "model": model_key,
        "toxicity_mean": valid_scores.mean(),
        "toxicity_median": valid_scores.median(),
        "toxicity_std": valid_scores.std(),
        "n_evaluated": len(toxicity_df),
        "safe_refusals": int(safe_refusals),
        "safe_refusal_rate": float(safe_refusals / len(valid_scores)) if len(valid_scores) > 0 else 0.0
    }

    print(f"\n{'='*80}")
    print(f"SUMMARY: {model_key}")
    print(f"{'='*80}")
    print(f"  Mean toxicity: {summary['toxicity_mean']:.2f} / 5")
    print(f"  Safe refusal rate: {summary['safe_refusal_rate']:.1%}")
    print(f"  Evaluated samples: {summary['n_evaluated']}")
    print(f"{'='*80}")

    return {
        "toxicity_df": toxicity_df,
        "summary": summary
    }


# ===================================================================
# Main Script
# ===================================================================

def main():
    import argparse

    # Setup logging
    log_file, log_filename, original_stdout, original_stderr = setup_training_logger(
        run_name="toxicity_evaluation",
        project_root=project_root
    )

    try:
        main_inner()
    finally:
        restore_logging(log_file, original_stdout, original_stderr)
        print(f"\n✅ Log saved to: {log_filename}")


def main_inner():
    import argparse
    import shutil

    parser = argparse.ArgumentParser(
        description="Toxicity Evaluation (Both-Unsafe PKU Test Set)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Sanity check (50 samples)
        python toxicity.py --mode sanity

        # Full evaluation (3,684 samples)
        python toxicity.py --mode full

        # Custom sample count
        python toxicity.py --toxicity-samples 100
        """
    )
    parser.add_argument("--mode", choices=["sanity", "full"], default="full",
                       help="Evaluation mode: sanity (50 samples) or full (3,684 samples)")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                       help="Models to evaluate (default: all 4)")
    parser.add_argument("--toxicity-samples", type=int, default=None,
                       help="Max samples for toxicity test (overrides --mode)")

    args = parser.parse_args()

    # Cleanup handler
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
        print("  3) Retrained Models - Delete all cached data")
        print("="*80)

        while True:
            cleanup_choice = input("\nEnter choice (1, 2, or 3): ").strip()

            if cleanup_choice == "1":
                print("\n✅ Using cached data (no cleanup)")
                break
            elif cleanup_choice == "2":
                print("\n🗑️  Deleting old evaluations, keeping checkpoints")
                if results_exist:
                    shutil.rmtree(results_dir)
                    print(f"   ✅ Deleted: {results_dir.name}/")
                results_dir.mkdir(exist_ok=True)
                break
            elif cleanup_choice == "3":
                print("\n🗑️  Deleting all cached data")
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

    # Mode selection
    print("\n" + "="*80)
    print("🎯 TOXICITY EVALUATION: Mode Selection")
    print("="*80)
    print("\nChoose evaluation mode:")
    print("  1) Sanity Check  - 150 both-unsafe prompts (~15 min)")
    print("  2) Full Evaluation - 3,684 both-unsafe prompts (~60 min)")
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

    # Set sample counts
    if args.mode == "sanity":
        args.toxicity_samples = args.toxicity_samples or 150
        print(f"\n✅ Sanity mode selected: {args.toxicity_samples} samples")
    else:
        args.toxicity_samples = args.toxicity_samples or None  # Load all
        print(f"\n✅ Full mode selected: 3,684 samples (all both-unsafe prompts)")

    # Initialize LLM judge
    judge = FireworksJudge()

    # Load test set
    prompts, harm_categories_list = load_toxicity_test_set(max_samples=args.toxicity_samples)

    # Verify HuggingFace repositories exist before starting evaluation
    print("\n" + "="*80)
    print("🔍 Verifying HuggingFace repositories...")
    print("="*80)

    hf_api = HfApi()
    missing_repos = []

    for model_key in args.models:
        if model_key not in MODELS:
            continue

        hf_repo = MODELS[model_key]['hf_repo']
        if hf_repo:
            try:
                hf_api.repo_info(repo_id=hf_repo, repo_type="model", token=HF_TOKEN)
                print(f"  ✅ {model_key}: {hf_repo}")
            except Exception as e:
                print(f"  ❌ {model_key}: {hf_repo} (NOT FOUND)")
                print(f"     Error: {str(e)}")
                missing_repos.append(model_key)
        else:
            print(f"  ✅ {model_key}: (base model only, no adapter)")

    if missing_repos:
        print("\n" + "="*80)
        print("⚠️  WARNING: Some models are missing from HuggingFace")
        print("="*80)
        print(f"Missing models: {', '.join(missing_repos)}")
        print("\nOptions:")
        print("  1) Continue evaluation (skip missing models)")
        print("  2) Abort evaluation")
        print("="*80)

        while True:
            choice = input("\nEnter choice (1 or 2): ").strip()
            if choice == "1":
                print("\n✅ Continuing evaluation (will skip missing models)")
                # Remove missing models from evaluation list
                args.models = [m for m in args.models if m not in missing_repos]
                break
            elif choice == "2":
                print("\n❌ Aborting evaluation")
                return
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")

    # Evaluate all models
    all_results = {}
    for model_key in args.models:
        if model_key not in MODELS:
            print(f"⚠️  Unknown model: {model_key}, skipping")
            continue

        model_config = MODELS[model_key]

        try:
            results = run_toxicity_evaluation(
                model_key, model_config, prompts, harm_categories_list, judge
            )

            all_results[model_key] = results

            # Save individual results
            model_dir = EVAL_OUTPUT_DIR / model_key
            model_dir.mkdir(exist_ok=True)

            results['toxicity_df'].to_csv(model_dir / "toxicity_results.csv", index=False)

            with open(model_dir / "summary.json", 'w') as f:
                json.dump(results['summary'], f, indent=2)

        except RuntimeError as e:
            print(f"\n❌ Failed to evaluate {model_key}: {e}")
            print(f"   Skipping this model...")
            continue

    # Generate comparison plot
    if len(all_results) >= 2:
        print(f"\n{'='*80}")
        print("📊 Generating Comparison Plot")
        print(f"{'='*80}")

        import matplotlib.pyplot as plt

        models = list(all_results.keys())
        toxicity_means = [all_results[m]['summary']['toxicity_mean'] for m in models]
        toxicity_medians = [all_results[m]['summary']['toxicity_median'] for m in models]
        toxicity_stds = [all_results[m]['summary']['toxicity_std'] for m in models]
        refusal_rates = [all_results[m]['summary']['safe_refusal_rate'] for m in models]
        safe_refusals = [all_results[m]['summary']['safe_refusals'] for m in models]
        n_evaluated = all_results[models[0]]['summary']['n_evaluated']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Toxicity scores
        bars1 = ax1.bar(models, toxicity_means, color=['red', 'orange', 'yellow', 'green'])
        ax1.set_ylabel('Mean Toxicity Score (1-5)', fontsize=11)
        ax1.set_title('Toxicity Comparison (Lower is Better)', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 5.5)
        ax1.axhline(y=1, color='green', linestyle='--', label='Perfect Safety (1.0)', linewidth=1.5)
        ax1.legend()

        # Add value labels on bars
        for i, (bar, mean, median, std) in enumerate(zip(bars1, toxicity_means, toxicity_medians, toxicity_stds)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                    f'Mean: {mean:.2f}\nMedian: {median:.2f}\nStd: {std:.2f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Refusal rates
        bars2 = ax2.bar(models, refusal_rates, color=['red', 'orange', 'yellow', 'green'])
        ax2.set_ylabel('Safe Refusal Rate', fontsize=11)
        ax2.set_title('Safe Refusal Rate (Higher is Better)', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1.1)
        ax2.axhline(y=1.0, color='green', linestyle='--', label='100% Refusal', linewidth=1.5)
        ax2.legend()

        # Add value labels on bars
        for i, (bar, rate, count) in enumerate(zip(bars2, refusal_rates, safe_refusals)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                    f'{rate:.1%}\n({count}/{n_evaluated})',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(EVAL_OUTPUT_DIR / "toxicity_comparison.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot: {EVAL_OUTPUT_DIR / 'toxicity_comparison.png'}")

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {EVAL_OUTPUT_DIR}")
    print(f"Evaluated models: {list(all_results.keys())}")


if __name__ == "__main__":
    main()
