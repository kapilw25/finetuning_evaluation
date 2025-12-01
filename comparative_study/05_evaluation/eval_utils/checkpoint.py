"""
Shared Checkpoint System for All Evaluations

Saves/loads inference results to avoid re-running expensive model generation.
Supports: ISD, TruthfulQA, Conditional Safety, Length Control
"""

import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime


def get_checkpoint_dir(eval_type: str) -> Path:
    """Get checkpoint directory for a specific evaluation type"""
    eval_dirs = {
        "isd": Path(__file__).parent.parent / "isd" / "checkpoints",
        "truthfulqa": Path(__file__).parent.parent / "truthfulqa" / "checkpoints",
        "conditional_safety": Path(__file__).parent.parent / "conditional_safety" / "checkpoints",
        "length_control": Path(__file__).parent.parent / "length_control" / "checkpoints",
        "aqi": Path(__file__).parent.parent / "AQI" / "checkpoints",
    }
    checkpoint_dir = eval_dirs.get(eval_type, Path(__file__).parent.parent / "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_checkpoint_path(model_key: str, eval_type: str = "isd", variant: str = None) -> Path:
    """Get checkpoint file path for a model

    Args:
        model_key: Model identifier (e.g., "CITA_Instruct")
        eval_type: Evaluation type ("isd", "truthfulqa", etc.)
        variant: Optional variant (e.g., "HONEST", "CONFIDENT" for TruthfulQA)
    """
    checkpoint_dir = get_checkpoint_dir(eval_type)
    if variant:
        return checkpoint_dir / f"{model_key}_{variant}_checkpoint.json"
    return checkpoint_dir / f"{model_key}_{eval_type}_checkpoint.json"


def save_checkpoint(
    model_key: str,
    responses: List[Dict],
    total_test_cases: int,
    eval_type: str = "isd",
    variant: str = None,
    completed: bool = False
):
    """
    Save checkpoint with responses generated so far

    Args:
        model_key: Model identifier
        responses: List of response dicts
        total_test_cases: Total number of test cases
        eval_type: Evaluation type ("isd", "truthfulqa", etc.)
        variant: Optional variant (e.g., "HONEST" for TruthfulQA)
        completed: Whether inference is complete
    """
    checkpoint_path = get_checkpoint_path(model_key, eval_type, variant)

    checkpoint_data = {
        "model_key": model_key,
        "eval_type": eval_type,
        "variant": variant,
        "n_total": total_test_cases,
        "n_completed": len(responses),
        "completed": completed,
        "responses": responses,
        "timestamp": datetime.now().isoformat()
    }

    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

    label = f"{model_key}/{variant}" if variant else model_key
    if completed:
        print(f"✅ Checkpoint COMPLETED: {label} ({len(responses)}/{total_test_cases})")
    else:
        print(f"💾 Checkpoint saved: {label} ({len(responses)}/{total_test_cases})")


def load_checkpoint(model_key: str, eval_type: str = "isd", variant: str = None) -> Optional[Dict]:
    """
    Load checkpoint if exists

    Args:
        model_key: Model identifier
        eval_type: Evaluation type ("isd", "truthfulqa", etc.)
        variant: Optional variant (e.g., "HONEST" for TruthfulQA)

    Returns:
        Dict with keys: model_key, n_total, n_completed, completed, responses, timestamp
        or None if no checkpoint exists
    """
    checkpoint_path = get_checkpoint_path(model_key, eval_type, variant)

    if not checkpoint_path.exists():
        return None

    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)

    label = f"{model_key}/{variant}" if variant else model_key
    status = "COMPLETED" if checkpoint['completed'] else f"{checkpoint['n_completed']}/{checkpoint['n_total']}"
    print(f"📂 Found checkpoint: {label} ({status})")

    return checkpoint


def delete_checkpoint(model_key: str, eval_type: str = "isd", variant: str = None) -> bool:
    """Delete checkpoint for a model"""
    checkpoint_path = get_checkpoint_path(model_key, eval_type, variant)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        label = f"{model_key}/{variant}" if variant else model_key
        print(f"🗑️ Deleted checkpoint: {label}")
        return True
    return False
