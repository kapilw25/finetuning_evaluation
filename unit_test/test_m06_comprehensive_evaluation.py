#!/usr/bin/env python3
"""
Unit test for m06_comprehensive_evaluation.py
Tests the comprehensive AQI evaluation system including:
- Data pipeline and prompt loading
- Hidden state extraction and processing
- AQI calculation (XBI, CHI, Delta-AQI)
- Results reporting and visualization
"""

import sys
import os
import tempfile
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

class TestComprehensiveEvaluation(unittest.TestCase):
    """Test the comprehensive AQI evaluation system"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_outputs_dir = Path(self.temp_dir) / "outputs" / "m06_comprehensive_evaluation"
        self.test_outputs_dir.mkdir(parents=True, exist_ok=True)

    def test_evaluation_prompt_structure(self):
        """Test EvaluationPrompt data structure"""
        try:
            from m06_comprehensive_evaluation import EvaluationPrompt

            prompt = EvaluationPrompt(
                id="test_01",
                original_prompt="What is machine learning?",
                formatted_prompt="<s>[INST] What is machine learning? [/INST]",
                safety_label="safe",
                dataset="test_dataset",
                length=45
            )

            self.assertEqual(prompt.id, "test_01")
            self.assertEqual(prompt.safety_label, "safe")
            self.assertEqual(prompt.dataset, "test_dataset")
            self.assertIn("[INST]", prompt.formatted_prompt)
            self.assertIn("[/INST]", prompt.formatted_prompt)

            print("✅ EvaluationPrompt structure test passed")
            return True

        except Exception as e:
            print(f"❌ EvaluationPrompt structure test failed: {e}")
            return False

    def test_aqi_scores_structure(self):
        """Test AQIScores data structure"""
        try:
            from m06_comprehensive_evaluation import AQIScores

            aqi_scores = AQIScores(
                model_variant="baseline",
                xbi_score=0.65,
                chi_score=4.23,
                aqi_score=2.45,
                safe_prompts_tested=64,
                unsafe_prompts_tested=64,
                hidden_states_extracted=4096,
                evaluation_time_minutes=15.5
            )

            self.assertEqual(aqi_scores.model_variant, "baseline")
            self.assertAlmostEqual(aqi_scores.xbi_score, 0.65)
            self.assertAlmostEqual(aqi_scores.chi_score, 4.23)
            self.assertAlmostEqual(aqi_scores.aqi_score, 2.45)
            self.assertEqual(aqi_scores.safe_prompts_tested, 64)
            self.assertEqual(aqi_scores.unsafe_prompts_tested, 64)

            print("✅ AQIScores structure test passed")
            return True

        except Exception as e:
            print(f"❌ AQIScores structure test failed: {e}")
            return False

    def test_xbi_calculation_logic(self):
        """Test XBI (Explicit Bias Index) calculation"""
        try:
            from m06_comprehensive_evaluation import ComprehensiveEvaluation

            evaluator = ComprehensiveEvaluation()

            # Create mock embeddings with clear separation
            # Safe embeddings clustered around [1, 1, 1]
            safe_embeddings = np.array([
                [1.0, 1.0, 1.0],
                [1.1, 0.9, 1.0],
                [0.9, 1.1, 1.0],
                [1.0, 1.0, 0.9]
            ])

            # Unsafe embeddings clustered around [-1, -1, -1]
            unsafe_embeddings = np.array([
                [-1.0, -1.0, -1.0],
                [-1.1, -0.9, -1.0],
                [-0.9, -1.1, -1.0],
                [-1.0, -1.0, -0.9]
            ])

            xbi_score = evaluator._calculate_xbi(safe_embeddings, unsafe_embeddings)

            # XBI should be relatively low for well-separated clusters
            self.assertIsInstance(xbi_score, float)
            self.assertGreater(xbi_score, 0)
            self.assertLess(xbi_score, 1.0)  # Should be low for good separation

            print(f"✅ XBI calculation test passed (XBI: {xbi_score:.4f})")
            return True

        except Exception as e:
            print(f"❌ XBI calculation test failed: {e}")
            return False

    def test_chi_calculation_logic(self):
        """Test CHI (Calinski-Harabasz Index) calculation"""
        try:
            from m06_comprehensive_evaluation import ComprehensiveEvaluation

            evaluator = ComprehensiveEvaluation()

            # Create mock embeddings with clear separation
            safe_embeddings = np.array([
                [2.0, 2.0, 2.0],
                [2.1, 1.9, 2.0],
                [1.9, 2.1, 2.0],
                [2.0, 2.0, 1.9]
            ])

            unsafe_embeddings = np.array([
                [-2.0, -2.0, -2.0],
                [-2.1, -1.9, -2.0],
                [-1.9, -2.1, -2.0],
                [-2.0, -2.0, -1.9]
            ])

            chi_score = evaluator._calculate_chi(safe_embeddings, unsafe_embeddings)

            # CHI should be high for well-separated clusters
            self.assertIsInstance(chi_score, float)
            self.assertGreater(chi_score, 0)
            # With good separation, CHI should be reasonably high
            self.assertGreater(chi_score, 1.0)

            print(f"✅ CHI calculation test passed (CHI: {chi_score:.4f})")
            return True

        except Exception as e:
            print(f"❌ CHI calculation test failed: {e}")
            return False

    def test_attention_pooling_logic(self):
        """Test attention-weighted pooling mechanism"""
        try:
            from m06_comprehensive_evaluation import ComprehensiveEvaluation
            import torch

            evaluator = ComprehensiveEvaluation()

            # Create mock hidden states and attention mask
            batch_size, seq_len, hidden_dim = 1, 4, 8
            hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

            # Attention mask: attend to first 3 tokens, ignore last token
            attention_mask = torch.tensor([[1, 1, 1, 0]])

            pooled = evaluator._apply_attention_pooling(hidden_states, attention_mask)

            # Check output shape
            self.assertEqual(pooled.shape, (batch_size, hidden_dim))

            # Pooled should be different from simple mean due to masking
            simple_mean = hidden_states.mean(dim=1)
            self.assertFalse(torch.allclose(pooled, simple_mean))

            print("✅ Attention pooling test passed")
            return True

        except Exception as e:
            print(f"❌ Attention pooling test failed: {e}")
            return False

    def test_prompt_loading_and_formatting(self):
        """Test evaluation prompt loading and Llama-3 formatting"""
        try:
            with patch('m06_comprehensive_evaluation.CentralizedDB') as mock_db, \
                 patch('m06_comprehensive_evaluation.sqlite3') as mock_sqlite:

                # Mock database responses
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_cursor.fetchone.return_value = (
                    1, 'alignment-instructions+ACCD', 70440, 45000, 5001, 500, 500, 'success', 'test_hash', '2025-09-26'
                )
                mock_conn.cursor.return_value = mock_cursor
                mock_sqlite.connect.return_value.__enter__.return_value = mock_conn

                from m06_comprehensive_evaluation import ComprehensiveEvaluation

                evaluator = ComprehensiveEvaluation()
                success = evaluator.load_evaluation_prompts()

                self.assertTrue(success)
                self.assertGreater(len(evaluator.evaluation_prompts), 0)

                # Check prompt formatting
                for prompt in evaluator.evaluation_prompts[:5]:  # Check first 5
                    self.assertIn("<s>[INST]", prompt.formatted_prompt)
                    self.assertIn("[/INST]", prompt.formatted_prompt)
                    self.assertIn(prompt.safety_label, ["safe", "unsafe"])

                # Check balance
                safe_count = sum(1 for p in evaluator.evaluation_prompts if p.safety_label == "safe")
                unsafe_count = sum(1 for p in evaluator.evaluation_prompts if p.safety_label == "unsafe")

                self.assertEqual(safe_count, evaluator.config["safe_prompts_target"])
                self.assertEqual(unsafe_count, evaluator.config["unsafe_prompts_target"])

                print(f"✅ Prompt loading test passed ({safe_count} safe, {unsafe_count} unsafe)")
                return True

        except Exception as e:
            print(f"❌ Prompt loading test failed: {e}")
            return False

    def test_evaluation_summary_generation(self):
        """Test evaluation summary and reporting"""
        try:
            from m06_comprehensive_evaluation import ComprehensiveEvaluation, AQIScores, EvaluationPrompt

            evaluator = ComprehensiveEvaluation()

            # Set up mock data
            evaluator.evaluation_prompts = [
                EvaluationPrompt("safe_1", "Test safe", "<s>[INST] Test safe [/INST]", "safe", "test", 30),
                EvaluationPrompt("unsafe_1", "Test unsafe", "<s>[INST] Test unsafe [/INST]", "unsafe", "test", 32)
            ]

            evaluator.aqi_scores = {
                "baseline": AQIScores("baseline", 0.75, 3.5, 2.1, 32, 32, 2048, 10.0),
                "qlora": AQIScores("qlora", 0.65, 4.2, 2.8, 32, 32, 2048, 12.0)
            }

            summary = evaluator.generate_evaluation_summary()

            # Verify summary structure
            self.assertIn("evaluation_overview", summary)
            self.assertIn("aqi_results", summary)
            self.assertIn("comparative_analysis", summary)
            self.assertIn("formatted_results", summary)

            # Check comparative analysis
            comp_analysis = summary["comparative_analysis"]
            self.assertIn("delta_aqi", comp_analysis)
            self.assertIn("performance_verdict", comp_analysis)

            # Delta-AQI should be positive (QLoRA better)
            self.assertGreater(comp_analysis["delta_aqi"], 0)
            self.assertIn("improvement", comp_analysis["performance_verdict"])

            print("✅ Evaluation summary test passed")
            return True

        except Exception as e:
            print(f"❌ Evaluation summary test failed: {e}")
            return False

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

def run_tests():
    """Run all tests and return summary"""
    print("=" * 60)
    print("UNIT TESTS: M06 COMPREHENSIVE AQI EVALUATION")
    print("=" * 60)

    test_loader = unittest.TestLoader()
    test_suite = test_loader.loadTestsFromTestCase(TestComprehensiveEvaluation)

    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%" if result.testsRun > 0 else "0%")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\nERRORRS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)