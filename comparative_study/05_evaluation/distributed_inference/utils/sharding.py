"""
Data Sharding Utilities for Distributed Evaluation

Enables running evaluations across multiple nodes by splitting datasets.

Usage:
    from sharding import add_sharding_args, shard_dataset, get_shard_suffix

    # In argparse setup:
    add_sharding_args(parser)

    # After loading dataset:
    dataset = shard_dataset(dataset, args.shard, args.total_shards)

    # For output filenames:
    suffix = get_shard_suffix(args.shard, args.total_shards)
    output_file = f"results{suffix}.json"
"""

import argparse
from typing import Any, List, Optional, Union


def add_sharding_args(parser: argparse.ArgumentParser) -> None:
    """Add sharding arguments to an argument parser."""
    parser.add_argument(
        "--shard",
        type=int,
        default=None,
        help="Shard index (0 to total_shards-1) for distributed evaluation"
    )
    parser.add_argument(
        "--total-shards",
        type=int,
        default=1,
        help="Total number of shards for distributed evaluation"
    )


def shard_dataset(dataset, shard_idx: Optional[int], total_shards: int = 1):
    """
    Split dataset into shards for distributed processing.

    Args:
        dataset: HuggingFace Dataset or list to shard
        shard_idx: Which shard to return (0-indexed). If None, return full dataset.
        total_shards: Total number of shards

    Returns:
        The specified shard of the dataset
    """
    if shard_idx is None or total_shards <= 1:
        return dataset

    if shard_idx < 0 or shard_idx >= total_shards:
        raise ValueError(f"shard_idx ({shard_idx}) must be in [0, {total_shards})")

    total_samples = len(dataset)
    samples_per_shard = total_samples // total_shards
    remainder = total_samples % total_shards

    start_idx = shard_idx * samples_per_shard + min(shard_idx, remainder)
    end_idx = start_idx + samples_per_shard + (1 if shard_idx < remainder else 0)

    print(f"📊 Sharding: shard {shard_idx+1}/{total_shards}, samples {start_idx}-{end_idx} of {total_samples}")

    if isinstance(dataset, list):
        return dataset[start_idx:end_idx]
    else:
        return dataset.select(range(start_idx, end_idx))


def get_shard_suffix(shard_idx: Optional[int], total_shards: int = 1) -> str:
    """Get filename suffix for sharded outputs."""
    if shard_idx is None or total_shards <= 1:
        return ""
    return f"_shard_{shard_idx}_of_{total_shards}"


def is_sharded(args: argparse.Namespace) -> bool:
    """Check if running in sharded mode."""
    return hasattr(args, 'shard') and args.shard is not None and args.total_shards > 1
