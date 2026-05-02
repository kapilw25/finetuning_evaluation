"""YAML config loader (CLAUDE.md §15: no hardcoded magic numbers).

Usage:
    from src.utils.config import load_pipeline_config, load_train_config, load_eval_config

    cfg = load_pipeline_config()  # configs/pipeline.yaml
    sanity_n = cfg["eval"]["sanity_n_prompts"]

    train_cfg = load_train_config("cita")  # configs/train/cita.yaml
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

# Repo root: src/utils/config.py → up 2 = repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


@lru_cache(maxsize=8)
def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_pipeline_config() -> Dict[str, Any]:
    """Load configs/pipeline.yaml (cross-cutting params: sample limits, seeds, modes)."""
    return _load_yaml(CONFIG_DIR / "pipeline.yaml")


def load_train_config(method: str) -> Dict[str, Any]:
    """Load configs/train/<method>.yaml. method ∈ {sft, dpo, ppo, grpo, cita, cita_optuna}."""
    return _load_yaml(CONFIG_DIR / "train" / f"{method}.yaml")


def load_eval_config(benchmark: str) -> Dict[str, Any]:
    """Load configs/eval/<benchmark>.yaml. benchmark ∈ {aqi, conditional_safety, isd, length_control, truthfulqa}."""
    return _load_yaml(CONFIG_DIR / "eval" / f"{benchmark}.yaml")


def load_data_config(name: str) -> Dict[str, Any]:
    """Load configs/data/<name>.yaml. e.g. 'pku_safe_rlhf'."""
    return _load_yaml(CONFIG_DIR / "data" / f"{name}.yaml")
