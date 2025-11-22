"""
ISD Evaluation Runner

Evaluates models on Instruction Switch Dataset to measure:
1. Fidelity: Does response match expected characteristics for the instruction?
2. Semantic Shift: How much does response change across different instructions?
3. Consistency: Is behavior consistent across instruction variants?

Tests the core claim: "Instruction-Aware: DPO=No, CITA=Yes"

Usage:
    # Run specific models
    python isd_evaluation.py --models CITA_Instruct DPO_Instruct

    # Run all models
    python isd_evaluation.py --num_prompts 50

    # Available models: Baseline, SFT_NoInstruct, SFT_Instruct,
    #                   DPO_NoInstruct, DPO_Instruct, CITA_NoInstruct, CITA_Instruct

Uses shared eval_utils for model loading (same pattern as toxicity.py).
"""

import sys
import json
import torch
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import pandas as pd
from tqdm import tqdm

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))
sys.path.insert(0, str(project_root / "comparative_study" / "05_evaluation"))

from instruction_switch_dataset import (
    InstructionSwitchDataset,
    ISDTestCase
)

from eval_utils import MODELS, load_model_for_eval, unload_model


@dataclass
class ISDResponse:
    """Response data for a single ISD test case"""
    prompt_id: int
    instruction_type: str
    instruction: str
    prompt: str
    response: str
    response_length: int
    generation_time: float
    expected_characteristics: List[str]


@dataclass
class ISDModelResult:
    """Complete evaluation result for a model"""
    model_name: str
    responses: List[ISDResponse]
    total_prompts: int
    total_test_cases: int
    evaluation_time: float
    timestamp: str


class ISDEvaluator:
    """
    Evaluator for Instruction Switch Dataset

    Runs inference and collects responses for metrics calculation.
    """

    def __init__(
        self,
        model_key: str,
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        batch_size: int = 1
    ):
        self.model_key = model_key
        self.model_name = MODELS[model_key]["display_name"]
        self.use_instruction = MODELS[model_key]["use_instruction"]
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model and tokenizer using shared utilities"""
        self.model, self.tokenizer = load_model_for_eval(self.model_key)
        print(f"Model loaded on {self.device}")

    def generate_response(self, test_case: ISDTestCase) -> ISDResponse:
        """Generate response for a single test case"""
        start_time = time.time()

        # Format input based on whether we use instruction
        if self.use_instruction:
            messages = [
                {"role": "system", "content": test_case.instruction},
                {"role": "user", "content": test_case.prompt}
            ]
        else:
            # For models without instruction awareness (like DPO)
            messages = [
                {"role": "user", "content": test_case.prompt}
            ]

        # Tokenize
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode response only
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        generation_time = time.time() - start_time

        return ISDResponse(
            prompt_id=test_case.prompt_id,
            instruction_type=test_case.instruction_type,
            instruction=test_case.instruction,
            prompt=test_case.prompt,
            response=response,
            response_length=len(response),
            generation_time=generation_time,
            expected_characteristics=test_case.expected_characteristics
        )

    def evaluate(
        self,
        test_cases: List[ISDTestCase],
        progress_bar: bool = True
    ) -> ISDModelResult:
        """
        Run evaluation on all test cases

        Args:
            test_cases: List of ISDTestCase objects
            progress_bar: Show progress bar

        Returns:
            ISDModelResult with all responses
        """
        if self.model is None:
            self.load_model()

        start_time = time.time()
        responses = []

        iterator = tqdm(test_cases, desc=f"Evaluating {self.model_name}") if progress_bar else test_cases

        for test_case in iterator:
            response = self.generate_response(test_case)
            responses.append(response)

        evaluation_time = time.time() - start_time

        result = ISDModelResult(
            model_name=self.model_name,
            responses=responses,
            total_prompts=len(set(tc.prompt_id for tc in test_cases)),
            total_test_cases=len(test_cases),
            evaluation_time=evaluation_time,
            timestamp=datetime.now().isoformat()
        )

        return result

    def unload_model(self):
        """Free GPU memory using shared utilities"""
        if self.model is not None:
            unload_model(self.model)
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            print(f"Model {self.model_name} unloaded")


def save_results(
    result: ISDModelResult,
    output_dir: Path
) -> Path:
    """Save evaluation results to JSON and CSV"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results as JSON
    json_path = output_dir / f"{result.model_name}_isd_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            "metadata": {
                "model_name": result.model_name,
                "total_prompts": result.total_prompts,
                "total_test_cases": result.total_test_cases,
                "evaluation_time": result.evaluation_time,
                "timestamp": result.timestamp
            },
            "responses": [asdict(r) for r in result.responses]
        }, f, indent=2)

    # Save responses as CSV for easier analysis
    csv_path = output_dir / f"{result.model_name}_isd_responses.csv"
    df = pd.DataFrame([asdict(r) for r in result.responses])
    df.to_csv(csv_path, index=False)

    print(f"Results saved to {output_dir}")
    return output_dir


def run_isd_evaluation(
    model_keys: List[str],
    num_prompts: int = 500,
    output_dir: Optional[Path] = None,
    seed: int = 42
) -> Dict[str, ISDModelResult]:
    """
    Run ISD evaluation on multiple models

    Args:
        model_keys: List of model keys from MODELS dict (e.g., ["CITA_Instruct", "DPO_Instruct"])
        num_prompts: Number of prompts (x10 instructions = test cases)
        output_dir: Where to save results
        seed: Random seed for dataset generation

    Returns:
        Dict mapping model_key -> ISDModelResult
    """
    # Generate dataset
    isd = InstructionSwitchDataset(seed=seed)
    test_cases = isd.generate_dataset(num_prompts)

    print("=" * 80)
    print("ISD Evaluation")
    print("=" * 80)
    print(f"Prompts: {num_prompts}")
    print(f"Instructions: {len(isd.get_instruction_types())}")
    print(f"Total test cases: {len(test_cases)}")
    print(f"Models: {len(model_keys)}")
    print("=" * 80)

    if output_dir is None:
        output_dir = project_root / "outputs" / "ISD_Evaluation"

    results = {}

    for model_key in model_keys:
        model_info = MODELS[model_key]

        print(f"\n{'=' * 80}")
        print(f"Evaluating: {model_info['display_name']}")
        print(f"Instruction-Aware: {model_info['use_instruction']}")
        print(f"{'=' * 80}")

        evaluator = ISDEvaluator(model_key=model_key)

        result = evaluator.evaluate(test_cases)
        results[model_key] = result

        # Save results
        model_output_dir = output_dir / model_key
        save_results(result, model_output_dir)

        # Unload to free memory
        evaluator.unload_model()

        # Print summary
        avg_length = sum(r.response_length for r in result.responses) / len(result.responses)
        avg_time = sum(r.generation_time for r in result.responses) / len(result.responses)

        print(f"\nSummary for {model_key}:")
        print(f"  Test cases: {result.total_test_cases}")
        print(f"  Avg response length: {avg_length:.0f} chars")
        print(f"  Avg generation time: {avg_time:.2f}s")
        print(f"  Total time: {result.evaluation_time:.1f}s")

    return results


# =============================================================================
# DEFAULT MODEL KEYS (uses shared MODELS from eval_utils)
# =============================================================================

# All available models from eval_utils.MODELS
MODEL_KEYS = list(MODELS.keys())
# ["Baseline", "SFT_NoInstruct", "SFT_Instruct", "DPO_NoInstruct",
#  "DPO_Instruct", "CITA_NoInstruct", "CITA_Instruct"]


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ISD Evaluation")
    parser.add_argument("--num_prompts", type=int, default=50, help="Number of prompts (x10 = test cases)")
    parser.add_argument("--models", nargs="+", default=None, help="Specific models to evaluate (e.g., CITA_Instruct DPO_Instruct)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Filter models if specified
    if args.models:
        # Validate model keys
        invalid = [m for m in args.models if m not in MODELS]
        if invalid:
            print(f"Invalid model keys: {invalid}")
            print(f"Available models: {list(MODELS.keys())}")
            sys.exit(1)
        model_keys = args.models
    else:
        model_keys = MODEL_KEYS

    if not model_keys:
        print("No models selected. Available models:")
        for key in MODELS.keys():
            print(f"  - {key}")
        sys.exit(1)

    # Run evaluation
    results = run_isd_evaluation(
        model_keys=model_keys,
        num_prompts=args.num_prompts,
        seed=args.seed
    )

    print("\n" + "=" * 80)
    print("ISD Evaluation Complete")
    print("=" * 80)
    print("\nNext step: Run isd_metrics.py to calculate fidelity, semantic shift, consistency")
