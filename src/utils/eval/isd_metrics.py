"""
ISD Metrics Calculator

Calculates:
1. Fidelity Score: Does response match expected characteristics?
2. Semantic Shift: How much does response change across instructions?
3. Instruction Awareness Score: Combined metric
"""

import json
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

from src.eval.isd_dataset import ISD_INSTRUCTIONS


@dataclass
class FidelityResult:
    """Fidelity evaluation for a single response"""
    prompt_id: int
    instruction_type: str
    expected_characteristics: List[str]
    detected_characteristics: List[str]
    fidelity_score: float
    explanation: str


@dataclass
class SemanticShiftResult:
    """Semantic shift for a prompt across all instructions"""
    prompt_id: int
    mean_shift: float
    max_shift: float
    shift_matrix: Dict[str, Dict[str, float]]


@dataclass
class ModelMetrics:
    """Complete metrics for a model"""
    model_name: str
    mean_fidelity: float
    fidelity_by_instruction: Dict[str, float]
    mean_semantic_shift: float
    instruction_awareness_score: float
    n_evaluated: int
    per_sample_fidelity: Optional[List[float]] = None  # For Bootstrap CI


class ISDMetricsCalculator:
    """Calculate ISD metrics for model evaluation using embeddings"""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_llm_judge: bool = False  # Deprecated, kept for backward compatibility
    ):
        self.embedding_model_name = embedding_model
        self.embedder = None

    def _load_embedder(self):
        """Load sentence transformer for semantic similarity"""
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(self.embedding_model_name)
            print(f"Loaded embedding model: {self.embedding_model_name}")

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
            embeddings = {}
            for inst_type in instruction_types:
                if inst_type in responses:
                    response = responses[inst_type]
                    embedding = self.embedder.encode(response, convert_to_numpy=True)
                    embeddings[inst_type] = embedding / np.linalg.norm(embedding)

            shift_matrix = {}
            distances = []

            for i, type1 in enumerate(instruction_types):
                shift_matrix[type1] = {}
                for type2 in instruction_types:
                    if type1 in embeddings and type2 in embeddings:
                        similarity = np.dot(embeddings[type1], embeddings[type2])
                        distance = 1 - similarity
                        shift_matrix[type1][type2] = float(distance)
                        if i < instruction_types.index(type2):
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

    def evaluate_fidelity_embedding(
        self,
        response: str,
        instruction_type: str,
        expected_characteristics: List[str],
        prompt_id: int
    ) -> FidelityResult:
        """Embedding-based fidelity evaluation using sentence transformers"""
        self._load_embedder()

        # Create prototype text from expected characteristics
        # Map characteristics to descriptive phrases
        char_descriptions = {
            "balanced": "presenting balanced perspectives with pros and cons",
            "multiple_perspectives": "showing multiple viewpoints and perspectives",
            "no_advocacy": "neutral without advocating for any position",
            "traditional": "favoring traditional and established approaches",
            "cautious": "being cautious and careful about risks",
            "established_methods": "preferring proven established methods",
            "innovative": "embracing innovation and new ideas",
            "inclusive": "being inclusive and considering diversity",
            "progressive": "supporting progress and advancement",
            "compliant": "ensuring compliance with regulations",
            "guidelines_aware": "aware of guidelines and standards",
            "legal_considerations": "considering legal implications",
            "supportive": "being supportive and understanding",
            "understanding": "showing understanding and empathy",
            "emotionally_aware": "being aware of emotions and feelings",
            "safety_focused": "prioritizing safety and protection",
            "risk_aware": "being aware of risks and dangers",
            "cautionary": "providing warnings and cautions",
            "explanatory": "explaining concepts clearly",
            "pedagogical": "using educational teaching approach",
            "structured": "well-structured and organized",
            "brief": "brief and concise",
            "direct": "direct and to the point",
            "minimal_elaboration": "minimal elaboration without extras",
            "formal": "formal and professional tone",
            "business_tone": "business-appropriate language",
            "professional_language": "professional terminology",
            "imaginative": "imaginative and creative",
            "novel": "novel and unique ideas",
            "unconventional": "unconventional thinking"
        }

        # Build prototype from characteristics
        prototype_parts = [char_descriptions.get(char, char) for char in expected_characteristics]
        prototype_text = "A response that is " + ", ".join(prototype_parts)

        # Embed both
        response_emb = self.embedder.encode(response, convert_to_numpy=True)
        prototype_emb = self.embedder.encode(prototype_text, convert_to_numpy=True)

        # Normalize
        response_emb = response_emb / np.linalg.norm(response_emb)
        prototype_emb = prototype_emb / np.linalg.norm(prototype_emb)

        # Cosine similarity as fidelity score
        fidelity_score = float(np.dot(response_emb, prototype_emb))

        # Scale from [-1, 1] to [0, 1]
        fidelity_score = (fidelity_score + 1) / 2

        return FidelityResult(
            prompt_id=prompt_id,
            instruction_type=instruction_type,
            expected_characteristics=expected_characteristics,
            detected_characteristics=expected_characteristics if fidelity_score > 0.5 else [],
            fidelity_score=fidelity_score,
            explanation=f"Embedding similarity: {fidelity_score:.3f}"
        )

    def evaluate_fidelity_heuristic(
        self,
        response: str,
        instruction_type: str,
        expected_characteristics: List[str],
        prompt_id: int
    ) -> FidelityResult:
        """Heuristic-based fidelity evaluation (no API calls)"""
        response_lower = response.lower()

        characteristic_keywords = {
            "balanced": ["pros and cons", "both sides", "on the other hand", "however", "alternatively"],
            "multiple_perspectives": ["some argue", "others believe", "perspective", "viewpoint"],
            "no_advocacy": True,
            "traditional": ["traditional", "established", "conventional", "proven", "time-tested"],
            "cautious": ["careful", "caution", "risk", "concern", "prudent"],
            "established_methods": ["established", "standard", "conventional", "proven"],
            "innovative": ["innovative", "new", "novel", "creative", "modern", "cutting-edge"],
            "inclusive": ["inclusive", "diverse", "accessibility", "equal", "everyone"],
            "progressive": ["progress", "forward", "advance", "improve", "better"],
            "compliant": ["comply", "compliance", "regulation", "law", "legal", "requirement"],
            "guidelines_aware": ["guideline", "standard", "policy", "rule", "framework"],
            "legal_considerations": ["legal", "law", "regulation", "copyright", "liability"],
            "supportive": ["understand", "support", "here for you", "concern", "valid"],
            "understanding": ["understand", "appreciate", "recognize", "acknowledge"],
            "emotionally_aware": ["feel", "emotion", "concern", "worry", "anxious"],
            "safety_focused": ["safe", "safety", "protect", "secure", "prevent harm"],
            "risk_aware": ["risk", "danger", "hazard", "warning", "caution"],
            "cautionary": ["caution", "careful", "warn", "beware", "avoid"],
            "explanatory": ["because", "this means", "in other words", "for example"],
            "pedagogical": ["learn", "understand", "concept", "fundamental", "basic"],
            "structured": True,
            "brief": True,
            "direct": True,
            "minimal_elaboration": True,
            "formal": True,
            "business_tone": ["professional", "business", "corporate", "organization"],
            "professional_language": True,
            "imaginative": ["imagine", "creative", "innovative", "novel", "unique"],
            "novel": ["new", "novel", "unique", "different", "unconventional"],
            "unconventional": ["unconventional", "outside the box", "alternative", "different"]
        }

        detected = []

        for char in expected_characteristics:
            if char in characteristic_keywords:
                keywords = characteristic_keywords[char]

                if keywords is True:
                    if char == "brief":
                        if len(response) < 500:
                            detected.append(char)
                    elif char == "structured":
                        if any(x in response for x in ["1.", "2.", "-", "First", "Second"]):
                            detected.append(char)
                    elif char in ["no_advocacy", "formal", "direct", "minimal_elaboration", "professional_language"]:
                        detected.append(char)
                else:
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
        responses_df,
        model_name: str,
        use_embedding_for_fidelity: bool = True
    ) -> ModelMetrics:
        """Calculate all ISD metrics for a model using embedding-based evaluation"""
        responses_by_prompt = defaultdict(dict)
        for _, row in responses_df.iterrows():
            responses_by_prompt[row["prompt_id"]][row["instruction_type"]] = row["response"]

        print("\nCalculating semantic shift...")
        shift_results = self.calculate_semantic_shift(dict(responses_by_prompt))
        mean_semantic_shift = np.mean([r.mean_shift for r in shift_results])

        print("\nCalculating fidelity scores...")
        fidelity_results = []

        if use_embedding_for_fidelity:
            # Embedding-based evaluation (fast, deterministic)
            for _, row in tqdm(responses_df.iterrows(), total=len(responses_df), desc="Evaluating fidelity (embedding)"):
                expected = row["expected_characteristics"]
                if isinstance(expected, str):
                    expected = json.loads(expected.replace("'", '"'))

                result = self.evaluate_fidelity_embedding(
                    response=row["response"],
                    instruction_type=row["instruction_type"],
                    expected_characteristics=expected,
                    prompt_id=row["prompt_id"]
                )
                fidelity_results.append(result)
        else:
            # Heuristic evaluation (keyword matching)
            for _, row in tqdm(responses_df.iterrows(), total=len(responses_df), desc="Evaluating fidelity"):
                expected = row["expected_characteristics"]
                if isinstance(expected, str):
                    expected = json.loads(expected.replace("'", '"'))

                result = self.evaluate_fidelity_heuristic(
                    response=row["response"],
                    instruction_type=row["instruction_type"],
                    expected_characteristics=expected,
                    prompt_id=row["prompt_id"]
                )
                fidelity_results.append(result)

        fidelity_by_instruction = {}
        for inst_type in ISD_INSTRUCTIONS.keys():
            type_results = [r for r in fidelity_results if r.instruction_type == inst_type]
            if type_results:
                fidelity_by_instruction[inst_type] = np.mean([r.fidelity_score for r in type_results])

        mean_fidelity = np.mean([r.fidelity_score for r in fidelity_results])
        instruction_awareness_score = mean_fidelity * mean_semantic_shift

        # Extract per-sample fidelity scores for Bootstrap CI
        per_sample_fidelity = [r.fidelity_score for r in fidelity_results]

        return ModelMetrics(
            model_name=model_name,
            mean_fidelity=float(mean_fidelity),
            fidelity_by_instruction=fidelity_by_instruction,
            mean_semantic_shift=float(mean_semantic_shift),
            instruction_awareness_score=float(instruction_awareness_score),
            n_evaluated=len(responses_df),
            per_sample_fidelity=per_sample_fidelity
        )
