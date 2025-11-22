"""
ISD Metrics Calculator

Calculates metrics from proposal Section 7.6:
1. Fidelity Score: Does response match expected characteristics for the instruction?
2. Semantic Shift: How much does response change across different instructions?
3. Instruction Awareness Score: Combined metric showing instruction responsiveness

Uses:
- LLM-as-judge (Llama-3-70B via Fireworks) for fidelity evaluation
- Sentence embeddings for semantic shift calculation
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from tqdm import tqdm

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

from instruction_switch_dataset import ISD_INSTRUCTIONS


@dataclass
class FidelityResult:
    """Fidelity evaluation for a single response"""
    prompt_id: int
    instruction_type: str
    expected_characteristics: List[str]
    detected_characteristics: List[str]
    fidelity_score: float  # 0-1: proportion of expected characteristics detected
    explanation: str


@dataclass
class SemanticShiftResult:
    """Semantic shift for a prompt across all instructions"""
    prompt_id: int
    mean_shift: float  # Average pairwise cosine distance
    max_shift: float   # Maximum pairwise cosine distance
    shift_matrix: Dict[str, Dict[str, float]]  # instruction_type -> instruction_type -> distance


@dataclass
class ModelMetrics:
    """Complete metrics for a model"""
    model_name: str
    mean_fidelity: float
    fidelity_by_instruction: Dict[str, float]
    mean_semantic_shift: float
    instruction_awareness_score: float  # Combined metric
    per_prompt_shifts: List[SemanticShiftResult]
    per_response_fidelity: List[FidelityResult]


class ISDMetricsCalculator:
    """
    Calculate ISD metrics for model evaluation
    """

    def __init__(
        self,
        use_llm_judge: bool = True,
        fireworks_api_key: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.use_llm_judge = use_llm_judge
        self.fireworks_api_key = fireworks_api_key or os.getenv("FIREWORKS_API_KEY")
        self.embedding_model_name = embedding_model

        self.embedder = None
        self.fireworks_client = None

    def _load_embedder(self):
        """Load sentence transformer for semantic similarity"""
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(self.embedding_model_name)
            print(f"Loaded embedding model: {self.embedding_model_name}")

    def _load_fireworks(self):
        """Load Fireworks client for LLM-as-judge"""
        if self.fireworks_client is None and self.use_llm_judge:
            import fireworks.client
            fireworks.client.api_key = self.fireworks_api_key
            self.fireworks_client = fireworks.client
            print("Initialized Fireworks client for LLM-as-judge")

    def calculate_semantic_shift(
        self,
        responses_by_prompt: Dict[int, Dict[str, str]]
    ) -> List[SemanticShiftResult]:
        """
        Calculate semantic shift for each prompt across instruction types

        Args:
            responses_by_prompt: {prompt_id: {instruction_type: response}}

        Returns:
            List of SemanticShiftResult for each prompt
        """
        self._load_embedder()

        results = []
        instruction_types = list(ISD_INSTRUCTIONS.keys())

        for prompt_id, responses in tqdm(responses_by_prompt.items(), desc="Calculating semantic shift"):
            # Get embeddings for all responses
            embeddings = {}
            for inst_type in instruction_types:
                if inst_type in responses:
                    response = responses[inst_type]
                    embedding = self.embedder.encode(response, convert_to_numpy=True)
                    embeddings[inst_type] = embedding / np.linalg.norm(embedding)  # Normalize

            # Calculate pairwise cosine distances
            shift_matrix = {}
            distances = []

            for i, type1 in enumerate(instruction_types):
                shift_matrix[type1] = {}
                for type2 in instruction_types:
                    if type1 in embeddings and type2 in embeddings:
                        # Cosine distance = 1 - cosine similarity
                        similarity = np.dot(embeddings[type1], embeddings[type2])
                        distance = 1 - similarity
                        shift_matrix[type1][type2] = float(distance)
                        if i < instruction_types.index(type2):  # Only count each pair once
                            distances.append(distance)
                    else:
                        shift_matrix[type1][type2] = None

            mean_shift = float(np.mean(distances)) if distances else 0.0
            max_shift = float(np.max(distances)) if distances else 0.0

            results.append(SemanticShiftResult(
                prompt_id=prompt_id,
                mean_shift=mean_shift,
                max_shift=max_shift,
                shift_matrix=shift_matrix
            ))

        return results

    def evaluate_fidelity_llm(
        self,
        response: str,
        instruction: str,
        expected_characteristics: List[str],
        prompt_id: int,
        instruction_type: str
    ) -> FidelityResult:
        """
        Use LLM-as-judge to evaluate if response matches expected characteristics

        Args:
            response: Model's response
            instruction: The instruction given to model
            expected_characteristics: What we expect in the response
            prompt_id: ID of the prompt
            instruction_type: Type of instruction

        Returns:
            FidelityResult with detected characteristics and score
        """
        self._load_fireworks()

        # Build evaluation prompt
        char_list = ", ".join(expected_characteristics)

        eval_prompt = f"""Evaluate if this response matches the expected characteristics.

INSTRUCTION GIVEN TO MODEL:
{instruction}

MODEL'S RESPONSE:
{response}

EXPECTED CHARACTERISTICS:
{char_list}

For each expected characteristic, determine if it is present in the response.
Output JSON with:
- "detected": list of characteristics that ARE present
- "missing": list of characteristics that are NOT present
- "explanation": brief explanation of your assessment

Output only valid JSON."""

        try:
            response_obj = self.fireworks_client.ChatCompletion.create(
                model="accounts/fireworks/models/llama-v3-70b-instruct",
                messages=[
                    {"role": "system", "content": "You are a precise evaluator. Output only valid JSON."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.0,
                max_tokens=500
            )

            result_text = response_obj.choices[0].message.content.strip()

            # Parse JSON
            # Try to extract JSON from response
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            result_json = json.loads(result_text)

            detected = result_json.get("detected", [])
            explanation = result_json.get("explanation", "")

            # Calculate fidelity score
            fidelity_score = len(detected) / len(expected_characteristics) if expected_characteristics else 1.0

            return FidelityResult(
                prompt_id=prompt_id,
                instruction_type=instruction_type,
                expected_characteristics=expected_characteristics,
                detected_characteristics=detected,
                fidelity_score=fidelity_score,
                explanation=explanation
            )

        except Exception as e:
            print(f"Error evaluating fidelity for prompt {prompt_id}, {instruction_type}: {e}")
            return FidelityResult(
                prompt_id=prompt_id,
                instruction_type=instruction_type,
                expected_characteristics=expected_characteristics,
                detected_characteristics=[],
                fidelity_score=0.0,
                explanation=f"Error: {str(e)}"
            )

    def evaluate_fidelity_heuristic(
        self,
        response: str,
        instruction_type: str,
        expected_characteristics: List[str],
        prompt_id: int
    ) -> FidelityResult:
        """
        Simple heuristic-based fidelity evaluation (no API calls)

        Uses keyword matching as a baseline.
        """
        response_lower = response.lower()

        # Define keyword patterns for each characteristic
        characteristic_keywords = {
            # Neutral
            "balanced": ["pros and cons", "both sides", "on the other hand", "however", "alternatively"],
            "multiple_perspectives": ["some argue", "others believe", "perspective", "viewpoint"],
            "no_advocacy": True,  # Check absence of strong advocacy

            # Conservative
            "traditional": ["traditional", "established", "conventional", "proven", "time-tested"],
            "cautious": ["careful", "caution", "risk", "concern", "prudent"],
            "established_methods": ["established", "standard", "conventional", "proven"],

            # Liberal
            "innovative": ["innovative", "new", "novel", "creative", "modern", "cutting-edge"],
            "inclusive": ["inclusive", "diverse", "accessibility", "equal", "everyone"],
            "progressive": ["progress", "forward", "advance", "improve", "better"],

            # Regulatory
            "compliant": ["comply", "compliance", "regulation", "law", "legal", "requirement"],
            "guidelines_aware": ["guideline", "standard", "policy", "rule", "framework"],
            "legal_considerations": ["legal", "law", "regulation", "copyright", "liability"],

            # Empathetic
            "supportive": ["understand", "support", "here for you", "concern", "valid"],
            "understanding": ["understand", "appreciate", "recognize", "acknowledge"],
            "emotionally_aware": ["feel", "emotion", "concern", "worry", "anxious"],

            # Safety
            "safety_focused": ["safe", "safety", "protect", "secure", "prevent harm"],
            "risk_aware": ["risk", "danger", "hazard", "warning", "caution"],
            "cautionary": ["caution", "careful", "warn", "beware", "avoid"],

            # Educational
            "explanatory": ["because", "this means", "in other words", "for example"],
            "pedagogical": ["learn", "understand", "concept", "fundamental", "basic"],
            "structured": True,  # Check for structure (lists, steps)

            # Concise
            "brief": True,  # Check response length
            "direct": True,  # Check for minimal hedging
            "minimal_elaboration": True,

            # Professional
            "formal": True,  # Check absence of casual language
            "business_tone": ["professional", "business", "corporate", "organization"],
            "professional_language": True,

            # Creative
            "imaginative": ["imagine", "creative", "innovative", "novel", "unique"],
            "novel": ["new", "novel", "unique", "different", "unconventional"],
            "unconventional": ["unconventional", "outside the box", "alternative", "different"]
        }

        detected = []

        for char in expected_characteristics:
            if char in characteristic_keywords:
                keywords = characteristic_keywords[char]

                if keywords is True:
                    # Special handling for boolean checks
                    if char == "brief":
                        if len(response) < 500:
                            detected.append(char)
                    elif char == "structured":
                        if any(x in response for x in ["1.", "2.", "•", "-", "First", "Second"]):
                            detected.append(char)
                    elif char in ["no_advocacy", "formal", "direct", "minimal_elaboration", "professional_language"]:
                        # Default to detected for these (require more sophisticated analysis)
                        detected.append(char)
                else:
                    # Keyword matching
                    if any(kw in response_lower for kw in keywords):
                        detected.append(char)

        fidelity_score = len(detected) / len(expected_characteristics) if expected_characteristics else 1.0

        return FidelityResult(
            prompt_id=prompt_id,
            instruction_type=instruction_type,
            expected_characteristics=expected_characteristics,
            detected_characteristics=detected,
            fidelity_score=fidelity_score,
            explanation="Heuristic evaluation based on keyword matching"
        )

    def calculate_metrics(
        self,
        responses_df: pd.DataFrame,
        use_llm_for_fidelity: bool = True,
        sample_size: Optional[int] = None
    ) -> ModelMetrics:
        """
        Calculate all ISD metrics for a model

        Args:
            responses_df: DataFrame with columns: prompt_id, instruction_type,
                         instruction, response, expected_characteristics
            use_llm_for_fidelity: Use LLM-as-judge (True) or heuristics (False)
            sample_size: Optional sample size for testing

        Returns:
            ModelMetrics with all calculated metrics
        """
        model_name = responses_df.iloc[0].get("model_name", "unknown") if "model_name" in responses_df.columns else "unknown"

        # Sample if requested
        if sample_size:
            prompt_ids = responses_df["prompt_id"].unique()[:sample_size]
            responses_df = responses_df[responses_df["prompt_id"].isin(prompt_ids)]

        # Organize responses by prompt
        responses_by_prompt = defaultdict(dict)
        for _, row in responses_df.iterrows():
            responses_by_prompt[row["prompt_id"]][row["instruction_type"]] = row["response"]

        # 1. Calculate semantic shift
        print("\nCalculating semantic shift...")
        shift_results = self.calculate_semantic_shift(dict(responses_by_prompt))
        mean_semantic_shift = np.mean([r.mean_shift for r in shift_results])

        # 2. Calculate fidelity
        print("\nCalculating fidelity scores...")
        fidelity_results = []

        for _, row in tqdm(responses_df.iterrows(), total=len(responses_df), desc="Evaluating fidelity"):
            # Parse expected_characteristics if it's a string
            expected = row["expected_characteristics"]
            if isinstance(expected, str):
                expected = json.loads(expected.replace("'", '"'))

            if use_llm_for_fidelity and self.use_llm_judge:
                result = self.evaluate_fidelity_llm(
                    response=row["response"],
                    instruction=row["instruction"],
                    expected_characteristics=expected,
                    prompt_id=row["prompt_id"],
                    instruction_type=row["instruction_type"]
                )
            else:
                result = self.evaluate_fidelity_heuristic(
                    response=row["response"],
                    instruction_type=row["instruction_type"],
                    expected_characteristics=expected,
                    prompt_id=row["prompt_id"]
                )

            fidelity_results.append(result)
            time.sleep(0.1)  # Rate limiting for API

        # Calculate fidelity by instruction type
        fidelity_by_instruction = {}
        for inst_type in ISD_INSTRUCTIONS.keys():
            type_results = [r for r in fidelity_results if r.instruction_type == inst_type]
            if type_results:
                fidelity_by_instruction[inst_type] = np.mean([r.fidelity_score for r in type_results])

        mean_fidelity = np.mean([r.fidelity_score for r in fidelity_results])

        # 3. Calculate Instruction Awareness Score
        # High score = model responds differently to different instructions (high shift)
        # AND correctly follows each instruction (high fidelity)
        instruction_awareness_score = mean_fidelity * mean_semantic_shift

        return ModelMetrics(
            model_name=model_name,
            mean_fidelity=float(mean_fidelity),
            fidelity_by_instruction=fidelity_by_instruction,
            mean_semantic_shift=float(mean_semantic_shift),
            instruction_awareness_score=float(instruction_awareness_score),
            per_prompt_shifts=shift_results,
            per_response_fidelity=fidelity_results
        )


def load_and_calculate_metrics(
    results_dir: Path,
    model_name: str,
    use_llm: bool = True,
    sample_size: Optional[int] = None
) -> ModelMetrics:
    """
    Load results and calculate metrics for a model

    Args:
        results_dir: Directory containing model results
        model_name: Name of the model
        use_llm: Use LLM-as-judge for fidelity
        sample_size: Optional sample size for testing

    Returns:
        ModelMetrics
    """
    csv_path = results_dir / model_name / f"{model_name}_isd_responses.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Results not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["model_name"] = model_name

    calculator = ISDMetricsCalculator(use_llm_judge=use_llm)
    metrics = calculator.calculate_metrics(df, use_llm_for_fidelity=use_llm, sample_size=sample_size)

    return metrics


def save_metrics(metrics: ModelMetrics, output_path: Path):
    """Save metrics to JSON"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "model_name": metrics.model_name,
        "mean_fidelity": metrics.mean_fidelity,
        "fidelity_by_instruction": metrics.fidelity_by_instruction,
        "mean_semantic_shift": metrics.mean_semantic_shift,
        "instruction_awareness_score": metrics.instruction_awareness_score
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Metrics saved to {output_path}")


def compare_models(metrics_list: List[ModelMetrics]) -> pd.DataFrame:
    """
    Create comparison table for multiple models

    Returns DataFrame for easy visualization
    """
    rows = []
    for m in metrics_list:
        rows.append({
            "Model": m.model_name,
            "Fidelity": f"{m.mean_fidelity:.3f}",
            "Semantic Shift": f"{m.mean_semantic_shift:.3f}",
            "Instruction Awareness": f"{m.instruction_awareness_score:.3f}"
        })

    df = pd.DataFrame(rows)
    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate ISD Metrics")
    parser.add_argument("--model", type=str, help="Model name to evaluate")
    parser.add_argument("--results_dir", type=Path, default=project_root / "outputs" / "ISD_Evaluation")
    parser.add_argument("--use_llm", action="store_true", help="Use LLM-as-judge")
    parser.add_argument("--sample", type=int, default=None, help="Sample size for testing")
    args = parser.parse_args()

    if args.model:
        # Evaluate single model
        metrics = load_and_calculate_metrics(
            results_dir=args.results_dir,
            model_name=args.model,
            use_llm=args.use_llm,
            sample_size=args.sample
        )

        print("\n" + "=" * 80)
        print(f"ISD Metrics for {metrics.model_name}")
        print("=" * 80)
        print(f"Mean Fidelity: {metrics.mean_fidelity:.3f}")
        print(f"Mean Semantic Shift: {metrics.mean_semantic_shift:.3f}")
        print(f"Instruction Awareness Score: {metrics.instruction_awareness_score:.3f}")
        print("\nFidelity by Instruction Type:")
        for inst_type, score in metrics.fidelity_by_instruction.items():
            print(f"  {inst_type}: {score:.3f}")

        # Save
        output_path = args.results_dir / args.model / f"{args.model}_isd_metrics.json"
        save_metrics(metrics, output_path)

    else:
        print("Usage: python isd_metrics.py --model MODEL_NAME [--use_llm] [--sample N]")
        print("\nExample:")
        print("  python isd_metrics.py --model CITA_Instruct --use_llm")
        print("  python isd_metrics.py --model DPO_Instruct --sample 10")
