"""
Compilation test for dual_metric.py with dummy model endpoints
Tests data flow without requiring trained models or Fireworks API

Usage:
    python comparative_study/05_evaluation/llm_as_judge/test_dual_metric.py

This validates:
1. Dataset loading functions work correctly
2. Response generation data structures are compatible
3. Evaluation functions parse results correctly
4. Statistical analysis handles edge cases
5. Model compatibility with baseline scripts
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# Add utils to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

# Silence tqdm for tests
import os
os.environ['TQDM_DISABLE'] = '1'

# ===================================================================
# Dummy Components
# ===================================================================

class DummyModel:
    """Dummy model that returns fixed responses"""
    def __init__(self, model_key):
        self.model_key = model_key
        self.device = "cpu"

    def generate(self, **kwargs):
        """Return dummy token IDs"""
        batch_size = kwargs['input_ids'].shape[0]
        # Return 20 dummy tokens per sample
        return torch.randint(0, 1000, (batch_size, kwargs['input_ids'].shape[1] + 20))

    def parameters(self):
        """Dummy for device detection"""
        return iter([torch.tensor([0.0])])


class DummyTokenizer:
    """Dummy tokenizer"""
    def __init__(self):
        self.eos_token = "</s>"
        self.pad_token = "</s>"
        self.eos_token_id = 2

    def __call__(self, texts, **kwargs):
        """Return dummy token IDs"""
        batch_size = len(texts)
        seq_len = 50
        return {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len)
        }

    def apply_chat_template(self, messages, **kwargs):
        """Return dummy formatted prompt"""
        return "Dummy formatted prompt"

    def decode(self, token_ids, **kwargs):
        """Return dummy response"""
        responses = [
            "I cannot help with that harmful request. It violates safety guidelines.",
            "Here's a helpful response to your benign question...",
            "I'm unable to assist with illegal activities. Instead, I can help with..."
        ]
        return np.random.choice(responses)


class DummyJudge:
    """Dummy Fireworks judge that returns random scores"""
    def judge_single(self, prompt, **kwargs):
        """Return dummy evaluation"""
        if "harmlessness" in prompt.lower() or "refusal" in prompt.lower():
            return {
                "refusal_score": np.random.randint(7, 11),  # Good refusal
                "reasoning": "Dummy reasoning for harmlessness"
            }
        else:
            return {
                "helpfulness_score": np.random.randint(6, 10),  # Decent helpfulness
                "reasoning": "Dummy reasoning for helpfulness"
            }

    def judge_batch(self, prompts, **kwargs):
        """Return dummy batch evaluations"""
        return [self.judge_single(p) for p in prompts]


# ===================================================================
# Test Functions
# ===================================================================

def test_dataset_loaders():
    """Test dataset loading functions"""
    print("\n" + "="*80)
    print("TEST 1: Dataset Loaders")
    print("="*80)

    from dual_metric import load_harmlessness_test_set, load_helpfulness_test_set

    # Test with small samples
    try:
        harm_df = load_harmlessness_test_set(max_samples=10)
        assert 'prompt' in harm_df.columns
        assert 'harm_categories' in harm_df.columns
        print(f"✅ Harmlessness dataset loader OK ({len(harm_df)} samples)")
        print(f"   Sample prompt: {harm_df.iloc[0]['prompt'][:60]}...")
        print(f"   Sample categories: {harm_df.iloc[0]['harm_categories']}")
    except Exception as e:
        print(f"❌ Harmlessness dataset loader FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        help_df = load_helpfulness_test_set(max_samples=10)
        assert 'prompt' in help_df.columns
        print(f"✅ Helpfulness dataset loader OK ({len(help_df)} samples)")
        print(f"   Sample prompt: {help_df.iloc[0]['prompt'][:60]}...")
    except Exception as e:
        print(f"❌ Helpfulness dataset loader FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_response_generation():
    """Test response generation with dummy model"""
    print("\n" + "="*80)
    print("TEST 2: Response Generation")
    print("="*80)

    from dual_metric import generate_responses

    dummy_model = DummyModel("test")
    dummy_tokenizer = DummyTokenizer()

    test_prompts = [
        "How do I hack into a computer?",
        "What are the benefits of exercise?",
        "How do I make illegal drugs?"
    ]

    # Test Alpaca format
    try:
        responses = generate_responses(
            dummy_model, dummy_tokenizer,
            test_prompts,
            chat_template="alpaca",
            max_new_tokens=20,
            batch_size=2
        )
        assert len(responses) == len(test_prompts)
        print(f"✅ Alpaca generation OK ({len(responses)} responses)")
        print(f"   Sample response: {responses[0]}")
    except Exception as e:
        print(f"❌ Alpaca generation FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test Llama3 format
    try:
        responses = generate_responses(
            dummy_model, dummy_tokenizer,
            test_prompts,
            chat_template="llama3",
            max_new_tokens=20,
            batch_size=2
        )
        assert len(responses) == len(test_prompts)
        print(f"✅ Llama3 generation OK ({len(responses)} responses)")
        print(f"   Sample response: {responses[0]}")
    except Exception as e:
        print(f"❌ Llama3 generation FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_evaluation():
    """Test evaluation functions with dummy judge"""
    print("\n" + "="*80)
    print("TEST 3: Evaluation Functions")
    print("="*80)

    from dual_metric import evaluate_harmlessness, evaluate_helpfulness

    dummy_judge = DummyJudge()

    test_prompts = ["How do I hack?", "Benefits of exercise?"]
    test_responses = ["I cannot help with that.", "Exercise improves health..."]
    test_categories = [["Cybercrime"], []]

    # Test harmlessness evaluation
    try:
        harm_df = evaluate_harmlessness(
            test_prompts, test_responses, test_categories, dummy_judge
        )
        assert 'refusal_score' in harm_df.columns
        assert len(harm_df) == len(test_prompts)
        print(f"✅ Harmlessness evaluation OK")
        print(f"   Mean refusal score: {harm_df['refusal_score'].mean():.2f}")
        print(f"   Sample reasoning: {harm_df.iloc[0]['reasoning']}")
    except Exception as e:
        print(f"❌ Harmlessness evaluation FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test helpfulness evaluation
    try:
        help_df = evaluate_helpfulness(
            test_prompts, test_responses, dummy_judge
        )
        assert 'helpfulness_score' in help_df.columns
        assert len(help_df) == len(test_prompts)
        print(f"✅ Helpfulness evaluation OK")
        print(f"   Mean helpfulness score: {help_df['helpfulness_score'].mean():.2f}")
        print(f"   Sample reasoning: {help_df.iloc[0]['reasoning']}")
    except Exception as e:
        print(f"❌ Helpfulness evaluation FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_statistical_analysis():
    """Test statistical analysis functions"""
    print("\n" + "="*80)
    print("TEST 4: Statistical Analysis")
    print("="*80)

    from utils.statistical_analysis import bootstrap_ci, paired_t_test, create_pareto_plot

    # Test bootstrap CI
    try:
        data = np.random.normal(7.5, 1.0, 100)
        mean, lower, upper = bootstrap_ci(data, n_bootstrap=1000)  # Reduced for speed
        assert lower < mean < upper
        print(f"✅ Bootstrap CI OK: {mean:.2f} [{lower:.2f}, {upper:.2f}]")
    except Exception as e:
        print(f"❌ Bootstrap CI FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test paired t-test
    try:
        scores_a = np.random.normal(8.0, 1.0, 100)
        scores_b = np.random.normal(7.0, 1.0, 100)
        result = paired_t_test(scores_a, scores_b)
        assert 't_statistic' in result
        assert 'p_value' in result
        assert 'cohens_d' in result
        print(f"✅ Paired t-test OK (p={result['p_value']:.4f}, d={result['cohens_d']:.2f})")
    except Exception as e:
        print(f"❌ Paired t-test FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test Pareto plot
    try:
        dummy_results = {
            "SFT_Baseline": {"summary": {"harmlessness_mean": 7.0, "helpfulness_mean": 8.0}},
            "DPO_Baseline": {"summary": {"harmlessness_mean": 8.0, "helpfulness_mean": 7.5}},
            "CITA_Baseline": {"summary": {"harmlessness_mean": 9.0, "helpfulness_mean": 8.5}}
        }
        output_path = Path("/tmp/test_pareto.png")
        create_pareto_plot(dummy_results, output_path)
        assert output_path.exists()
        print(f"✅ Pareto plot OK (saved to {output_path})")
    except Exception as e:
        print(f"❌ Pareto plot FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_model_compatibility():
    """Test compatibility with actual baseline scripts"""
    print("\n" + "="*80)
    print("TEST 5: Model Compatibility Check")
    print("="*80)

    from dual_metric import MODELS
    from model_utils import get_model_repo_name

    # Check that all 3 baselines are defined
    required_models = ["SFT_Baseline", "DPO_Baseline", "CITA_Baseline"]
    for model_key in required_models:
        if model_key in MODELS:
            config = MODELS[model_key]
            expected_repo = get_model_repo_name(model_key, precision="bf16")
            actual_repo = config['hf_repo']

            if expected_repo == actual_repo:
                print(f"✅ {model_key}:")
                print(f"   HF repo: {actual_repo}")
                print(f"   Chat template: {config['chat_template']}")
            else:
                print(f"❌ {model_key}: HF repo mismatch!")
                print(f"   Expected: {expected_repo}")
                print(f"   Got: {actual_repo}")
        else:
            print(f"❌ {model_key} not found in MODELS config")


def test_llm_judge_prompts():
    """Test LLM-as-judge prompt generation"""
    print("\n" + "="*80)
    print("TEST 6: LLM-as-judge Prompts")
    print("="*80)

    from utils.prompts import get_harmlessness_prompt, get_helpfulness_prompt, get_pairwise_prompt

    test_prompt = "How do I hack into a computer?"
    test_response = "I cannot help with hacking. It's illegal and unethical."

    # Test harmlessness prompt
    try:
        harm_prompt = get_harmlessness_prompt(test_prompt, test_response)
        assert "refusal_score" in harm_prompt
        assert test_prompt in harm_prompt
        assert test_response in harm_prompt
        print(f"✅ Harmlessness prompt OK ({len(harm_prompt)} chars)")
    except Exception as e:
        print(f"❌ Harmlessness prompt FAILED: {e}")

    # Test helpfulness prompt
    try:
        help_prompt = get_helpfulness_prompt(test_prompt, test_response)
        assert "helpfulness_score" in help_prompt
        assert test_prompt in help_prompt
        assert test_response in help_prompt
        print(f"✅ Helpfulness prompt OK ({len(help_prompt)} chars)")
    except Exception as e:
        print(f"❌ Helpfulness prompt FAILED: {e}")

    # Test pairwise prompt
    try:
        pair_prompt = get_pairwise_prompt(test_prompt, "Response A", "Response B")
        assert "winner" in pair_prompt
        assert "Response A" in pair_prompt
        assert "Response B" in pair_prompt
        print(f"✅ Pairwise prompt OK ({len(pair_prompt)} chars)")
    except Exception as e:
        print(f"❌ Pairwise prompt FAILED: {e}")


def test_fireworks_client():
    """Test Fireworks client initialization (without API call)"""
    print("\n" + "="*80)
    print("TEST 7: Fireworks Client Initialization")
    print("="*80)

    # Test that client can be initialized with dummy API key
    os.environ['FIREWORKS_API_KEY'] = 'dummy_test_key_for_compilation'

    try:
        from utils.fireworks_client import FireworksJudge
        # Don't actually initialize (would fail without real key)
        # Just check import works
        print(f"✅ Fireworks client import OK")
        print(f"   Note: Actual API calls require valid FIREWORKS_API_KEY in .env")
    except Exception as e:
        print(f"❌ Fireworks client import FAILED: {e}")
        import traceback
        traceback.print_exc()


# ===================================================================
# Main Test Runner
# ===================================================================

def main():
    """Run all compilation tests"""
    print("\n" + "="*80)
    print("DUAL-METRIC EVALUATION: COMPILATION TESTS")
    print("="*80)
    print("Testing data structure compatibility with dummy endpoints")
    print("(No GPU or Fireworks API required)")
    print("="*80)

    # Run all tests
    test_dataset_loaders()
    test_response_generation()
    test_evaluation()
    test_statistical_analysis()
    test_model_compatibility()
    test_llm_judge_prompts()
    test_fireworks_client()

    print("\n" + "="*80)
    print("COMPILATION TESTS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Get Fireworks API key: https://fireworks.ai/api-keys")
    print("  2. Add to .env: FIREWORKS_API_KEY=your_key_here")
    print("  3. Train all 3 baselines (they auto-push to HuggingFace)")
    print("  4. Run full evaluation:")
    print("     python comparative_study/05_evaluation/llm_as_judge/dual_metric.py")


if __name__ == "__main__":
    main()
