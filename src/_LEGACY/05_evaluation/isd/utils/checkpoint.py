"""
ISD Checkpoint System

Saves/loads inference results to avoid re-running expensive model generation.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime


def get_checkpoint_dir() -> Path:
    """Get checkpoint directory"""
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_checkpoint_path(model_key: str) -> Path:
    """Get checkpoint file path for a model"""
    return get_checkpoint_dir() / f"{model_key}_isd_checkpoint.json"


def save_checkpoint(
    model_key: str,
    responses: List[Dict],
    total_test_cases: int,
    completed: bool = False
):
    """
    Save checkpoint with responses generated so far

    Args:
        model_key: Model identifier
        responses: List of response dicts (from ISDResponse dataclass)
        total_test_cases: Total number of test cases
        completed: Whether inference is complete
    """
    checkpoint_path = get_checkpoint_path(model_key)

    checkpoint_data = {
        "model_key": model_key,
        "n_total": total_test_cases,
        "n_completed": len(responses),
        "completed": completed,
        "responses": responses,
        "timestamp": datetime.now().isoformat()
    }

    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

    if completed:
        print(f"✅ Checkpoint COMPLETED: {model_key} ({len(responses)}/{total_test_cases})")
    else:
        print(f"💾 Checkpoint saved: {model_key} ({len(responses)}/{total_test_cases})")


def load_checkpoint(model_key: str) -> Optional[Dict]:
    """
    Load checkpoint if exists

    Returns:
        Dict with keys: model_key, n_total, n_completed, completed, responses, timestamp
        or None if no checkpoint exists
    """
    checkpoint_path = get_checkpoint_path(model_key)

    if not checkpoint_path.exists():
        return None

    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)

    status = "COMPLETED" if checkpoint['completed'] else f"{checkpoint['n_completed']}/{checkpoint['n_total']}"
    print(f"📂 Found checkpoint: {model_key} ({status})")

    return checkpoint


def delete_checkpoint(model_key: str) -> bool:
    """Delete checkpoint for a model"""
    checkpoint_path = get_checkpoint_path(model_key)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"🗑️ Deleted checkpoint: {model_key}")
        return True
    return False
