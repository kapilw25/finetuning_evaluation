"""Shared wandb integration (CLAUDE.md §9: --no-wandb flag, no-op when disabled).

Usage:
    from src.utils.wandb_utils import (
        add_wandb_args, init_wandb, log_metrics, log_image, log_artifact, finish_wandb
    )
    parser = argparse.ArgumentParser()
    add_wandb_args(parser)
    args = parser.parse_args()
    run = init_wandb(module="sft", mode=args.mode, config=vars(args), enabled=not args.no_wandb)
    log_metrics(run, {"loss": 0.5}, step=10)
    finish_wandb(run)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    """Add --no-wandb / --wandb-project / --wandb-entity flags to a parser."""
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", default="cita_ecliptica", help="wandb project name")
    parser.add_argument("--wandb-entity", default=None, help="wandb entity (username or team)")


def init_wandb(
    module: str,
    mode: str,
    config: Dict[str, Any],
    enabled: bool = True,
) -> Optional[Any]:
    """Initialize wandb run, or return None if disabled / unavailable.

    `module`: e.g. "sft", "dpo", "cita" — used as run-name prefix.
    `mode`: e.g. "sanity", "full" — appended to run name.
    `config`: hyperparameters / args dict to log.
    """
    if not enabled or wandb is None:
        return None
    return wandb.init(
        project=config.get("wandb_project", "cita_ecliptica"),
        entity=config.get("wandb_entity"),
        name=f"{module}_{mode}",
        config=config,
    )


def log_metrics(run: Optional[Any], metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """No-op if run is None."""
    if run is None:
        return
    run.log(metrics, step=step)


def log_image(run: Optional[Any], key: str, path: Path) -> None:
    """No-op if run is None or wandb missing."""
    if run is None or wandb is None:
        return
    run.log({key: wandb.Image(str(path))})


def log_artifact(run: Optional[Any], name: str, path: Path, artifact_type: str = "result") -> None:
    """No-op if run is None or wandb missing."""
    if run is None or wandb is None:
        return
    art = wandb.Artifact(name, type=artifact_type)
    if path.is_dir():
        art.add_dir(str(path))
    else:
        art.add_file(str(path))
    run.log_artifact(art)


def finish_wandb(run: Optional[Any]) -> None:
    """No-op if run is None."""
    if run is None:
        return
    run.finish()
