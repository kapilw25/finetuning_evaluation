"""ECLIPTICA scoring helpers shared by Exp 1, 1.5, 2, 4 (CHAR_DESCRIPTIONS + bootstrap)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import bootstrap as scipy_bootstrap

from src.eval.isd_dataset import ISD_INSTRUCTIONS

# ============================================================================
# Static lookup: characteristic name → natural-language description.
# Used to build prototype texts that response embeddings are scored against.
# Mirror of the dict originally defined inline at
#   src/utils/eval/isd_metrics.py::ISDMetricsCalculator.evaluate_fidelity_embedding
# Extracted to module level so Exp 1 / 1.5 / 2 / 4 can share without duplication.
# ============================================================================
CHAR_DESCRIPTIONS: dict[str, str] = {
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
    "unconventional": "unconventional thinking",
}


def build_prototype_embeddings(embedder, instructions: dict | None = None) -> dict[str, np.ndarray]:
    """Pre-compute one normalized prototype embedding per instruction type.

    Args:
        embedder: SentenceTransformer (already loaded)
        instructions: dict like ISD_INSTRUCTIONS. Defaults to ISD_INSTRUCTIONS.

    Returns:
        dict {instruction_type: normalized_embedding}
    """
    if instructions is None:
        instructions = ISD_INSTRUCTIONS
    prototypes = {}
    for inst_type, info in instructions.items():
        chars = info["expected_characteristics"]
        parts = [CHAR_DESCRIPTIONS.get(c, c) for c in chars]
        prototype_text = "A response that is " + ", ".join(parts)
        emb = embedder.encode(prototype_text, convert_to_numpy=True)
        norm = np.linalg.norm(emb)
        prototypes[inst_type] = emb / norm if norm > 0 else emb
    return prototypes


def score_fidelities(responses: Sequence[str], instruction_types: Sequence[str],
                     embedder, prototypes: dict[str, np.ndarray]) -> np.ndarray:
    """Per-row fidelity = cosine(response_emb, prototype_emb), scaled to [0, 1]."""
    embs = embedder.encode([r if isinstance(r, str) else "" for r in responses],
                           batch_size=64, show_progress_bar=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms

    scores = np.empty(len(responses), dtype=np.float32)
    for i, inst_type in enumerate(instruction_types):
        if inst_type not in prototypes:
            scores[i] = np.nan
            continue
        sim = float(np.dot(embs[i], prototypes[inst_type]))
        scores[i] = (sim + 1.0) / 2.0
    return scores


def bca_ci(scores, n_bootstrap: int = 10000, confidence: float = 0.95,
           seed: int = 42) -> dict:
    """95% BCa bootstrap CI via scipy (CLAUDE.md §7.4 spec).

    Returns dict with keys: mean, ci_lo, ci_hi, ci_half.
    """
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[~np.isnan(scores)]
    if len(scores) < 2:
        return {"mean": float(np.nan), "ci_lo": float(np.nan),
                "ci_hi": float(np.nan), "ci_half": float(np.nan)}
    rng = np.random.default_rng(seed)
    res = scipy_bootstrap(
        (scores,), np.mean,
        n_resamples=n_bootstrap,
        confidence_level=confidence,
        method="BCa",
        random_state=rng,
    )
    mean_val = float(np.mean(scores))
    lo = float(res.confidence_interval.low)
    hi = float(res.confidence_interval.high)
    return {"mean": mean_val, "ci_lo": lo, "ci_hi": hi, "ci_half": (hi - lo) / 2.0}


__all__ = ["CHAR_DESCRIPTIONS", "build_prototype_embeddings", "score_fidelities", "bca_ci"]
