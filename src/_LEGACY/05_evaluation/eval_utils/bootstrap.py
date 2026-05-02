"""
Bootstrap Confidence Interval Utilities

Computes 95% confidence intervals using bootstrap resampling.
Used for statistical significance in evaluation plots.

Usage:
    from eval_utils.bootstrap import compute_bootstrap_ci, BootstrapResult

    result = compute_bootstrap_ci(per_sample_scores, n_bootstrap=1000)
    print(f"Mean: {result.mean:.3f} +/- {result.ci_half_width:.3f}")
    print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class BootstrapResult:
    """Result of bootstrap confidence interval computation."""
    mean: float           # Point estimate (sample mean)
    ci_lower: float       # Lower bound of 95% CI
    ci_upper: float       # Upper bound of 95% CI
    ci_half_width: float  # Half-width for +/- notation
    std_error: float      # Bootstrap standard error
    n_samples: int        # Number of original samples
    n_bootstrap: int      # Number of bootstrap iterations

    def __str__(self) -> str:
        return f"{self.mean:.3f} +/- {self.ci_half_width:.3f} [{self.ci_lower:.3f}, {self.ci_upper:.3f}]"

    def format_short(self, decimals: int = 3) -> str:
        """Format as 'mean +/- ci' for plot annotations."""
        return f"{self.mean:.{decimals}f} +/- {self.ci_half_width:.{decimals}f}"

    def format_with_ci(self, decimals: int = 3) -> str:
        """Format as 'mean [lower, upper]' for tables."""
        return f"{self.mean:.{decimals}f} [{self.ci_lower:.{decimals}f}, {self.ci_upper:.{decimals}f}]"


def compute_bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = 42
) -> BootstrapResult:
    """
    Compute bootstrap confidence interval for a list of scores.

    Algorithm:
        1. For i in 1..n_bootstrap:
            a. Resample N scores with replacement
            b. Compute mean of resampled scores
        2. CI_lower = percentile(bootstrap_means, (1-confidence)/2 * 100)
        3. CI_upper = percentile(bootstrap_means, (1+confidence)/2 * 100)

    Args:
        scores: List of per-sample scores
        n_bootstrap: Number of bootstrap iterations (default 1000)
        confidence_level: Confidence level (default 0.95 for 95% CI)
        random_seed: Random seed for reproducibility (default 42)

    Returns:
        BootstrapResult with mean, CI bounds, and metadata
    """
    scores = np.array(scores)
    n_samples = len(scores)

    if n_samples == 0:
        return BootstrapResult(
            mean=0.0, ci_lower=0.0, ci_upper=0.0, ci_half_width=0.0,
            std_error=0.0, n_samples=0, n_bootstrap=n_bootstrap
        )

    if n_samples == 1:
        val = float(scores[0])
        return BootstrapResult(
            mean=val, ci_lower=val, ci_upper=val, ci_half_width=0.0,
            std_error=0.0, n_samples=1, n_bootstrap=n_bootstrap
        )

    # Set random seed for reproducibility
    rng = np.random.RandomState(random_seed)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled = rng.choice(scores, size=n_samples, replace=True)
        bootstrap_means.append(np.mean(resampled))

    bootstrap_means = np.array(bootstrap_means)

    # Compute percentiles for CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    # Point estimate and standard error
    mean = np.mean(scores)
    std_error = np.std(bootstrap_means)
    ci_half_width = (ci_upper - ci_lower) / 2

    return BootstrapResult(
        mean=float(mean),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        ci_half_width=float(ci_half_width),
        std_error=float(std_error),
        n_samples=n_samples,
        n_bootstrap=n_bootstrap
    )


def compute_delta_bootstrap_ci(
    scores_instruct: List[float],
    scores_noinstruct: List[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = 42
) -> BootstrapResult:
    """
    Compute bootstrap CI for the difference (Instruct - NoInstruct).

    Uses paired bootstrap: resample the same indices from both lists.

    Args:
        scores_instruct: Per-sample scores for Instruct variant
        scores_noinstruct: Per-sample scores for NoInstruct variant
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for CI
        random_seed: Random seed for reproducibility

    Returns:
        BootstrapResult for the delta (Instruct - NoInstruct)
    """
    scores_inst = np.array(scores_instruct)
    scores_no = np.array(scores_noinstruct)

    # Must have same length for paired bootstrap
    n_samples = min(len(scores_inst), len(scores_no))
    if n_samples == 0:
        return BootstrapResult(
            mean=0.0, ci_lower=0.0, ci_upper=0.0, ci_half_width=0.0,
            std_error=0.0, n_samples=0, n_bootstrap=n_bootstrap
        )

    scores_inst = scores_inst[:n_samples]
    scores_no = scores_no[:n_samples]

    # Set random seed
    rng = np.random.RandomState(random_seed)

    # Paired bootstrap
    bootstrap_deltas = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        resampled_inst = scores_inst[indices]
        resampled_no = scores_no[indices]
        delta = np.mean(resampled_inst) - np.mean(resampled_no)
        bootstrap_deltas.append(delta)

    bootstrap_deltas = np.array(bootstrap_deltas)

    # Compute CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_deltas, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_deltas, (1 - alpha / 2) * 100)

    mean_delta = np.mean(scores_inst) - np.mean(scores_no)
    std_error = np.std(bootstrap_deltas)
    ci_half_width = (ci_upper - ci_lower) / 2

    return BootstrapResult(
        mean=float(mean_delta),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        ci_half_width=float(ci_half_width),
        std_error=float(std_error),
        n_samples=n_samples,
        n_bootstrap=n_bootstrap
    )


def compute_batch_bootstrap_ci(
    scores_by_model: Dict[str, List[float]],
    n_bootstrap: int = 1000,
    random_seed: Optional[int] = 42
) -> Dict[str, BootstrapResult]:
    """
    Compute bootstrap CI for multiple models at once.

    Args:
        scores_by_model: Dict mapping model name to per-sample scores
        n_bootstrap: Number of bootstrap iterations
        random_seed: Random seed for reproducibility

    Returns:
        Dict mapping model name to BootstrapResult
    """
    results = {}
    for model_name, scores in scores_by_model.items():
        results[model_name] = compute_bootstrap_ci(
            scores, n_bootstrap=n_bootstrap, random_seed=random_seed
        )
    return results


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    test_scores = np.random.normal(0.5, 0.1, 100).tolist()

    result = compute_bootstrap_ci(test_scores)
    print(f"Test result: {result}")
    print(f"Short format: {result.format_short()}")
    print(f"CI format: {result.format_with_ci()}")
