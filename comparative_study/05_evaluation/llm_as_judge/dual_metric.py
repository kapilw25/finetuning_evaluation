"""
Dual-Metric Evaluation for SFT, DPO, and CITA Baselines
Evaluates: Harmlessness (PKU test) + Helpfulness (AlpacaEval)
LLM-as-judge: Llama-3-70B via Fireworks AI

IMPORTANT: Loads models from HuggingFace (not local paths)
Since training instances auto-shutdown after pushing to HF, evaluation runs on fresh instances.

Usage:
    # Sanity check: 50+50 samples (~5-10 min, ~$0.10)
    python comparative_study/05_evaluation/llm_as_judge/dual_metric.py --mode sanity

    # Full evaluation: 1000+805 samples (~40-60 min, ~$1.80)
    python comparative_study/05_evaluation/llm_as_judge/dual_metric.py --mode full

    # Evaluate specific models
    python comparative_study/05_evaluation/llm_as_judge/dual_metric.py --mode sanity --models SFT_Baseline CITA_Baseline

    # Custom sample counts (overrides --mode)
    python comparative_study/05_evaluation/llm_as_judge/dual_metric.py --harmlessness-samples 100 --helpfulness-samples 100

    # Use INT4 quantization for faster inference
    python comparative_study/05_evaluation/llm_as_judge/dual_metric.py --mode sanity --quantization int4
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

# ===================================================================
# Configuration
# ===================================================================

# Model configurations (loads from HuggingFace, not local)
# After training, models are pushed to HF by push_automation.py
MODELS = {
    "SFT_Baseline": {
        "hf_repo": get_model_repo_name("SFT_Baseline", precision="bf16"),
        "chat_template": "alpaca",  # SFT uses Alpaca format
    },
    "DPO_Baseline": {
        "hf_repo": get_model_repo_name("DPO_Baseline", precision="bf16"),
        "chat_template": "alpaca",  # DPO uses Alpaca format
    },
    "CITA_Baseline": {
        "hf_repo": get_model_repo_name("CITA_Baseline", precision="bf16"),
        "chat_template": "llama3",  # CITA uses Llama-3 chat template
    },
}

BASE_MODEL = "meta-llama/Llama-3.1-8B"
EVAL_OUTPUT_DIR = Path(__file__).parent / "DualMetric_Evaluation_Results"
EVAL_OUTPUT_DIR.mkdir(exist_ok=True)

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

def load_model_for_eval(
    model_key: str,
    quantization: str = "bf16"  # or "int4" for faster inference
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model with LoRA adapter from HuggingFace for evaluation

    Args:
        model_key: One of ["SFT_Baseline", "DPO_Baseline", "CITA_Baseline"]
        quantization: "bf16" (accurate) or "int4" (fast)

    Returns:
        Tuple of (model, tokenizer)

    Note:
        Models are loaded from HuggingFace (pushed by push_automation.py after training)
        This allows evaluation on different GPU instances than training
    """
    print(f"\n{'='*80}")
    print(f"Loading {model_key} for evaluation")
    print(f"{'='*80}")

    model_info = MODELS[model_key]
    hf_repo = model_info["hf_repo"]

    print(f"Loading from HuggingFace: {hf_repo}")
    print(f"(Models pushed automatically by push_automation.py after training)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    if quantization == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    elif quantization == "int4":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map="auto"
        )
    else:
        raise ValueError(f"Unknown quantization: {quantization}")

    # Load LoRA adapter from HuggingFace
    try:
        print(f"Loading LoRA adapter from HuggingFace: {hf_repo}...")
        model = PeftModel.from_pretrained(model, hf_repo)
        model = model.merge_and_unload()
        model.eval()
        print(f"✅ {model_key} loaded successfully from HuggingFace")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load adapter from HuggingFace: {hf_repo}\n"
            f"Error: {e}\n\n"
            f"Possible causes:\n"
            f"  1. Model not yet trained and pushed to HuggingFace\n"
            f"  2. Training performance did not improve (push_automation skipped push)\n"
            f"  3. HuggingFace authentication issue (check HF_TOKEN in .env)\n\n"
            f"To train this model:\n"
            f"  python comparative_study/0{list(MODELS.keys()).index(model_key)+1}a_{model_key}/Llama3_BF16.py --mode full\n"
        )

    return model, tokenizer


# ===================================================================
# Response Generation
# ===================================================================

def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    chat_template: str,
    max_new_tokens: int = 256,
    batch_size: int = 8
    ) -> List[str]:
    """
    Generate responses using correct chat template per model

    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        prompts: List of user prompts
        chat_template: "alpaca" (SFT/DPO) or "llama3" (CITA)
        max_new_tokens: Max tokens to generate
        batch_size: Batch size for generation

    Returns:
        List of generated responses
    """
    from tqdm import tqdm

    responses = []

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating ({chat_template})"):
        batch_prompts = prompts[i:i+batch_size]

        # Format prompts based on chat template
        if chat_template == "alpaca":
            formatted = [
                f"Below are some instructions that describe some tasks. Write responses that appropriately complete each request.\n\n### Instruction:\n{p}\n\n### Response:\n"
                for p in batch_prompts
            ]
        elif chat_template == "llama3":
            formatted = []
            for p in batch_prompts:
                messages = [{"role": "user", "content": p}]
                formatted.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ))
        else:
            raise ValueError(f"Unknown chat_template: {chat_template}")

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
    judge: FireworksJudge,
    quantization: str = "bf16"
    ) -> Dict:
    """
    Run full dual-metric evaluation for a single model

    Returns:
        Dict with harmlessness_df, helpfulness_df, summary_stats
    """
    print(f"\n{'='*80}")
    print(f"DUAL-METRIC EVALUATION: {model_key}")
    print(f"{'='*80}")

    # Load model from HuggingFace
    model, tokenizer = load_model_for_eval(model_key, quantization=quantization)
    chat_template = MODELS[model_key]["chat_template"]

    # Generate responses for harmlessness test
    print(f"\n--- Harmlessness Test ({len(harmlessness_test)} prompts) ---")
    harm_responses = generate_responses(
        model, tokenizer,
        harmlessness_test['prompt'].tolist(),
        chat_template=chat_template,
        max_new_tokens=256
    )

    # Generate responses for helpfulness test
    print(f"\n--- Helpfulness Test ({len(helpfulness_test)} prompts) ---")
    help_responses = generate_responses(
        model, tokenizer,
        helpfulness_test['prompt'].tolist(),
        chat_template=chat_template,
        max_new_tokens=256
    )

    # Cleanup model
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

        # Use INT4 for faster inference
        python dual_metric_eval.py --quantization int4

        Note: Models are loaded from HuggingFace (pushed by training scripts via push_automation.py)
        """
    )
    parser.add_argument("--mode", choices=["sanity", "full"], default="full",
                       help="Evaluation mode: sanity (50+50 samples) or full (1000+805 samples)")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                       help="Models to evaluate (default: all 3)")
    parser.add_argument("--harmlessness-samples", type=int, default=None,
                       help="Max samples for harmlessness test (overrides --mode)")
    parser.add_argument("--helpfulness-samples", type=int, default=None,
                       help="Max samples for helpfulness test (overrides --mode)")
    parser.add_argument("--quantization", choices=["bf16", "int4"], default="bf16",
                       help="Model quantization (default: bf16)")

    args = parser.parse_args()

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

    # Load HF token
    load_hf_token(project_root)

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
                judge,
                quantization=args.quantization
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
