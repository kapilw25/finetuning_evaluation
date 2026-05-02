"""
Dataset Info Utilities - Dynamically fetch max available samples from datasets

This module provides functions to query datasets and return max available
sample counts for each evaluation benchmark.

Each function returns a tuple with 'source' field indicating:
- 'fetched': Data was loaded live from HuggingFace/URL
- 'fallback': Used hardcoded fallback (network error, dataset unavailable)
"""

from typing import Dict, Tuple
import requests


def get_isd_max_samples() -> Tuple[int, int, str]:
    """
    Get max available prompts from ISD dataset on HuggingFace

    Returns:
        Tuple of (num_prompts, total_test_cases, source)
        where total_test_cases = num_prompts * 10 instruction types
        source = 'fetched' or 'fallback'
    """
    from datasets import load_dataset

    try:
        dataset = load_dataset("kapilw25/ISD-Instruction-Switch-Dataset", split="train")
        unique_prompts = len(set(dataset['prompt_id']))
        total_cases = unique_prompts * 10  # 10 instruction types
        return unique_prompts, total_cases, "fetched"
    except Exception as e:
        print(f"⚠️  ISD: Using fallback (reason: {e})")
        return 300, 3000, "fallback"


def get_truthfulqa_max_samples() -> Tuple[int, int, str]:
    """
    Get max available questions from TruthfulQA validation set

    Returns:
        Tuple of (num_questions, total_test_cases, source)
        where total_test_cases = num_questions * 2 variants
    """
    from datasets import load_dataset

    try:
        dataset = load_dataset("truthful_qa", "generation", split="validation")
        num_questions = len(dataset)
        total_cases = num_questions * 2  # HONEST, CONFIDENT variants
        return num_questions, total_cases, "fetched"
    except Exception as e:
        print(f"⚠️  TruthfulQA: Using fallback (reason: {e})")
        return 817, 1634, "fallback"


def get_conditional_safety_max_samples() -> Tuple[int, int, str]:
    """
    Get max available borderline prompts from PKU-SafeRLHF test split

    Borderline = one response safe, one unsafe (is_response_0_safe != is_response_1_safe)

    Returns:
        Tuple of (num_prompts, total_test_cases, source)
        where total_test_cases = num_prompts * 2 variants
    """
    from datasets import load_dataset

    try:
        test_data = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="test")
        borderline = test_data.filter(
            lambda x: (x['is_response_0_safe'] != x['is_response_1_safe'])
        )
        num_prompts = len(borderline)
        total_cases = num_prompts * 2  # STRICT, PERMISSIVE variants
        return num_prompts, total_cases, "fetched"
    except Exception as e:
        print(f"⚠️  Conditional Safety: Using fallback (reason: {e})")
        return 1222, 2444, "fallback"


def get_length_control_max_samples() -> Tuple[int, int, str]:
    """
    Get max available prompts from AlpacaEval dataset

    Returns:
        Tuple of (num_prompts, total_test_cases, source)
        where total_test_cases = num_prompts * 2 variants
    """
    try:
        url = 'https://huggingface.co/datasets/tatsu-lab/alpaca_eval/raw/main/alpaca_eval.json'
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        num_prompts = len(data)
        total_cases = num_prompts * 2  # CONCISE, DETAILED variants
        return num_prompts, total_cases, "fetched"
    except Exception as e:
        print(f"⚠️  Length Control: Using fallback (reason: {e})")
        return 805, 1610, "fallback"


def get_aqi_max_samples() -> Tuple[int, int, Dict[str, int], str]:
    """
    Get max available samples from litmus dataset for AQI evaluation

    Returns:
        Tuple of (total_samples, samples_per_axiom_min, axiom_counts_dict, source)
    """
    from datasets import load_dataset
    from collections import Counter

    try:
        dataset = load_dataset("hasnat79/litmus", split="train")
        total_samples = len(dataset)

        if 'axiom' in dataset.column_names:
            axiom_counts = Counter(dataset['axiom'])
            min_per_axiom = min(axiom_counts.values())
            return total_samples, min_per_axiom, dict(axiom_counts), "fetched"
        else:
            return total_samples, total_samples, {"overall": total_samples}, "fetched"
    except Exception as e:
        print(f"⚠️  AQI: Using fallback (reason: {e})")
        return 20439, 2919, {}, "fallback"


def get_all_max_samples() -> Dict[str, Dict]:
    """
    Get max available samples for all evaluation benchmarks

    Returns:
        Dict with benchmark name -> {prompts, total_cases, source, description}
    """
    results = {}

    # ISD
    prompts, cases, source = get_isd_max_samples()
    results['isd'] = {
        'prompts': prompts,
        'total_cases': cases,
        'multiplier': 10,
        'source': source,
        'description': f"{prompts} prompts x 10 = {cases:,} test cases"
    }

    # TruthfulQA
    questions, cases, source = get_truthfulqa_max_samples()
    results['truthfulqa'] = {
        'prompts': questions,
        'total_cases': cases,
        'multiplier': 2,
        'source': source,
        'description': f"{questions} Qs x 2 = {cases:,} test cases"
    }

    # Conditional Safety
    prompts, cases, source = get_conditional_safety_max_samples()
    results['conditional_safety'] = {
        'prompts': prompts,
        'total_cases': cases,
        'multiplier': 2,
        'source': source,
        'description': f"{prompts:,} prompts x 2 = {cases:,} test cases"
    }

    # Length Control
    prompts, cases, source = get_length_control_max_samples()
    results['length_control'] = {
        'prompts': prompts,
        'total_cases': cases,
        'multiplier': 2,
        'source': source,
        'description': f"{prompts} prompts x 2 = {cases:,} test cases"
    }

    # AQI
    total, min_per_axiom, axiom_counts, source = get_aqi_max_samples()
    results['aqi'] = {
        'prompts': total,
        'total_cases': total,
        'samples_per_axiom': min_per_axiom,
        'axiom_counts': axiom_counts,
        'source': source,
        'description': f"{total:,} total samples ({len(axiom_counts)} axioms)"
    }

    return results


def print_max_samples_summary():
    """Print summary of max available samples for all benchmarks"""
    print("\n" + "=" * 80)
    print("MAX AVAILABLE SAMPLES")
    print("=" * 80)

    results = get_all_max_samples()

    print(f"\n{'Benchmark':<20} {'Prompts':<10} {'Total':<12} {'Source':<12} {'Description'}")
    print("-" * 90)

    for name, info in results.items():
        source_tag = f"[{info['source']}]"
        print(f"{name:<20} {info['prompts']:<10} {info['total_cases']:<12} {source_tag:<12} {info['description']}")

    print("=" * 90)

    # Summary
    fetched_count = sum(1 for info in results.values() if info['source'] == 'fetched')
    fallback_count = sum(1 for info in results.values() if info['source'] == 'fallback')
    print(f"\nSummary: {fetched_count} fetched, {fallback_count} fallback")

    return results


if __name__ == "__main__":
    print_max_samples_summary()
