"""
Quick compilation test (no dataset downloads)
Tests all components without requiring large dataset downloads
"""

import sys
from pathlib import Path

# Add utils to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

print('='*80)
print('QUICK COMPILATION TEST (No dataset downloads)')
print('='*80)

# Test 1: Imports
print('\nTEST 1: Module Imports')
from utils import FireworksJudge, bootstrap_ci, get_harmlessness_prompt
from dual_metric import MODELS
from test_dual_metric import DummyModel, DummyTokenizer, DummyJudge
print('✅ All imports OK')

# Test 2: Model config
print('\nTEST 2: Model Configuration')
from model_utils import get_model_repo_name
for model_key in MODELS:
    expected = get_model_repo_name(model_key, precision='bf16')
    actual = MODELS[model_key]['hf_repo']
    assert expected == actual, f'{model_key} mismatch!'
    print(f'✅ {model_key}: {actual}')

# Test 3: Dummy components
print('\nTEST 3: Dummy Test Components')
import torch
model = DummyModel('test')
tokenizer = DummyTokenizer()
judge = DummyJudge()
print('✅ Dummy model/tokenizer/judge OK')

# Test 4: Prompts
print('\nTEST 4: LLM-as-judge Prompts')
from utils import get_helpfulness_prompt, get_pairwise_prompt
harm_prompt = get_harmlessness_prompt('Test prompt', 'Test response')
help_prompt = get_helpfulness_prompt('Test prompt', 'Test response')
pair_prompt = get_pairwise_prompt('Test prompt', 'Response A', 'Response B')
assert 'refusal_score' in harm_prompt
assert 'helpfulness_score' in help_prompt
assert 'winner' in pair_prompt
print('✅ All prompt templates OK')

# Test 5: Statistical functions
print('\nTEST 5: Statistical Analysis')
import numpy as np
from utils import paired_t_test, create_pareto_plot
data_a = np.random.normal(8.0, 1.0, 100)
data_b = np.random.normal(7.0, 1.0, 100)
result = paired_t_test(data_a, data_b)
assert 't_statistic' in result
assert 'p_value' in result
assert 'cohens_d' in result
print(f'✅ T-test OK (p={result["p_value"]:.4f})')

dummy_results = {
    'SFT_Baseline': {'summary': {'harmlessness_mean': 7.0, 'helpfulness_mean': 8.0}},
    'DPO_Baseline': {'summary': {'harmlessness_mean': 8.0, 'helpfulness_mean': 7.5}},
    'CITA_Baseline': {'summary': {'harmlessness_mean': 9.0, 'helpfulness_mean': 8.5}}
}
output_path = Path('/tmp/test_pareto.png')
create_pareto_plot(dummy_results, output_path)
assert output_path.exists()
print(f'✅ Pareto plot OK (saved to {output_path})')

print('\n' + '='*80)
print('✅ ALL QUICK TESTS PASSED!')
print('='*80)
print('\nFull test (with datasets) will download:')
print('  - PKU-SafeRLHF test split (~500MB)')
print('  - AlpacaEval dataset (~10MB)')
print('\nRun when ready:')
print('  python comparative_study/05_evaluation/llm_as_judge/test_dual_metric.py')
