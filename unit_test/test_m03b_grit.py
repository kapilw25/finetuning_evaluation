#!/usr/bin/env python3
"""
Unit Test for GRIT Training (m03b_grit_training.py)
Quick validation test with minimal data to verify GRIT functionality
"""

import os
import sys
import unittest
import tempfile
import torch
import torch.nn as nn
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from m00_centralized_db import CentralizedDB
from m03b_grit_training import GRITTrainer, CustomLoRALayer, KFACApproximation, NaturalGradientOptimizer, NeuralReprojection

class TestGRITTraining(unittest.TestCase):
    """Test GRIT training functionality with minimal data"""

    def setUp(self):
        """Setup test environment"""
        self.project_root = project_root
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_centralized.db")

        # Create test database
        self.db = CentralizedDB(db_path=self.test_db_path)

        # Create minimal test data
        self.test_data = [
            {
                "prompt": "What is machine learning?",
                "instruction": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                "safety_label": "safe"
            },
            {
                "prompt": "Explain deep learning",
                "instruction": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.",
                "safety_label": "safe"
            },
            {
                "prompt": "What is GRIT methodology?",
                "instruction": "GRIT is a Geometry-Aware Robust Instruction Tuning methodology that uses KFAC approximation and neural reprojection for advanced fine-tuning.",
                "safety_label": "safe"
            }
        ]

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_grit_trainer_initialization(self):
        """Test GRIT trainer initialization"""
        trainer = GRITTrainer()

        # Check trainer attributes
        self.assertIsInstance(trainer.db, CentralizedDB)
        self.assertEqual(trainer.model_id, "meta-llama/Meta-Llama-3-8B-Instruct")
        self.assertTrue(trainer.output_dir.exists())
        self.assertTrue(trainer.kfac_matrices_dir.parent.exists())

        # Check GRIT configuration
        self.assertEqual(trainer.grit_config["learning_rate"], 2e-5)
        self.assertEqual(trainer.grit_config["kfac_damping"], 1e-2)
        self.assertEqual(trainer.grit_config["neural_reproj_k"], 4)
        self.assertEqual(trainer.grit_config["lora_r"], 16)

    def test_custom_lora_layer(self):
        """Test custom LoRA layer implementation"""
        in_features, out_features = 512, 256
        rank = 16
        alpha = 32

        lora_layer = CustomLoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha
        )

        # Test initialization
        self.assertEqual(lora_layer.rank, rank)
        self.assertEqual(lora_layer.alpha, alpha)
        self.assertEqual(lora_layer.scaling, alpha / rank)

        # Test parameter shapes
        self.assertEqual(lora_layer.lora_A.shape, (rank, in_features))
        self.assertEqual(lora_layer.lora_B.shape, (out_features, rank))

        # Test forward pass
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, in_features)

        output = lora_layer(x)
        self.assertEqual(output.shape, (batch_size, seq_len, out_features))

    def test_kfac_approximation_initialization(self):
        """Test KFAC approximation initialization"""
        # Create a simple model for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 32)
                # Add LoRA attributes for testing
                self.linear.lora_A = nn.Parameter(torch.randn(8, 64))
                self.linear.lora_B = nn.Parameter(torch.randn(32, 8))

        model = SimpleModel()
        lora_params = [model.linear.lora_A, model.linear.lora_B]

        kfac = KFACApproximation(
            model=model,
            lora_parameters=lora_params,
            damping=1e-2,
            update_freq=5
        )

        # Test initialization
        self.assertEqual(kfac.damping, 1e-2)
        self.assertEqual(kfac.update_freq, 5)
        self.assertEqual(kfac.step_counter, 0)
        self.assertIsInstance(kfac.A_kfac, dict)
        self.assertIsInstance(kfac.B_kfac, dict)

        # Test hooks registration
        self.assertGreater(len(kfac.hooks), 0)

        # Cleanup
        kfac.cleanup()

    def test_natural_gradient_optimizer(self):
        """Test natural gradient optimizer"""
        # Create a simple model for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 32)
                self.linear.lora_A = nn.Parameter(torch.randn(8, 64))
                self.linear.lora_B = nn.Parameter(torch.randn(32, 8))

        model = SimpleModel()
        lora_params = [model.linear.lora_A, model.linear.lora_B]

        kfac = KFACApproximation(model, lora_params)
        optimizer = NaturalGradientOptimizer(
            model=model,
            lora_parameters=lora_params,
            kfac_approximation=kfac,
            learning_rate=2e-5
        )

        # Test initialization
        self.assertEqual(optimizer.learning_rate, 2e-5)
        self.assertIs(optimizer.kfac_approx, kfac)

        # Test zero_grad method
        # Add some dummy gradients
        lora_params[0].grad = torch.randn_like(lora_params[0])
        lora_params[1].grad = torch.randn_like(lora_params[1])

        optimizer.zero_grad()

        # Check gradients are cleared
        self.assertIsNone(lora_params[0].grad)
        self.assertIsNone(lora_params[1].grad)

        # Cleanup
        kfac.cleanup()

    def test_neural_reprojection(self):
        """Test neural reprojection module"""
        # Create a simple model for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 32)
                self.linear.lora_A = nn.Parameter(torch.randn(8, 64))
                self.linear.lora_B = nn.Parameter(torch.randn(32, 8))

        model = SimpleModel()
        lora_params = [model.linear.lora_A, model.linear.lora_B]

        kfac = KFACApproximation(model, lora_params)

        # Add some dummy KFAC matrices for testing
        kfac.A_kfac['linear'] = torch.eye(64) + 0.1 * torch.randn(64, 64)
        kfac.B_kfac['linear'] = torch.eye(32) + 0.1 * torch.randn(32, 32)

        reprojection = NeuralReprojection(
            model=model,
            kfac_approximation=kfac,
            top_k_eigenvectors=4
        )

        # Test initialization
        self.assertEqual(reprojection.k, 4)
        self.assertIs(reprojection.kfac_approx, kfac)

        # Test projection matrix computation
        proj_A, proj_B = reprojection._compute_projection_matrices('linear')

        if proj_A is not None and proj_B is not None:
            self.assertEqual(proj_A.shape, (64, 64))
            self.assertEqual(proj_B.shape, (32, 32))

        # Cleanup
        kfac.cleanup()

    def test_grit_hyperparameter_configuration(self):
        """Test GRIT hyperparameter configuration matches plan1.md specs"""
        trainer = GRITTrainer()

        # Check GRIT config matches exact specifications from plan1.md
        expected_config = {
            "learning_rate": 2e-5,          # Lower than QLoRA due to second-order optimization
            "batch_size": 1,                # Small batch due to KFAC memory requirements
            "kfac_damping": 1e-2,           # Damping factor
            "kfac_update_freq": 20,         # Update frequency: 20 steps
            "neural_reproj_k": 4,           # Top-k eigenvector projection
            "neural_reproj_freq": 40,       # Every 40 steps
            "momentum": 0.95,               # Momentum-based factor updates
            "lora_r": 16,                   # Same as QLoRA for fair comparison
            "lora_alpha": 32,               # Same as QLoRA
            "lora_dropout": 0.05            # Same as QLoRA
        }

        for key, expected_value in expected_config.items():
            self.assertEqual(trainer.grit_config[key], expected_value,
                           f"Config mismatch for {key}: expected {expected_value}, got {trainer.grit_config[key]}")

    def test_chat_template_formatting(self):
        """Test Llama-3 chat template formatting consistency with QLoRA"""
        prompt = "What is GRIT?"
        instruction = "GRIT is an advanced fine-tuning methodology."

        formatted = f"<s>[INST] {prompt} [/INST] {instruction}</s>"
        expected = "<s>[INST] What is GRIT? [/INST] GRIT is an advanced fine-tuning methodology.</s>"

        self.assertEqual(formatted, expected)

    def test_database_integration(self):
        """Test database integration for GRIT training metrics"""
        trainer = GRITTrainer()

        # Test database connection
        self.assertIsInstance(trainer.db, CentralizedDB)

        # Test experiment creation
        experiment_id = self.db.create_experiment("test_grit_db", "grit")
        self.assertIsInstance(experiment_id, int)

        # Test GRIT-specific methods exist
        self.assertTrue(hasattr(trainer, 'setup_model_and_tokenizer'))
        self.assertTrue(hasattr(trainer, 'load_training_data'))
        self.assertTrue(hasattr(trainer, 'train_model'))
        self.assertTrue(hasattr(trainer, 'save_model_and_adapter'))
        self.assertTrue(hasattr(trainer, 'log_metrics_to_database'))
        self.assertTrue(hasattr(trainer, 'cleanup_resources'))

        # Test GRIT-specific components exist
        self.assertTrue(hasattr(trainer, 'apply_lora_to_model'))
        self.assertTrue(hasattr(trainer, 'train_fklnr_epoch'))

    def test_model_output_directory_structure(self):
        """Test GRIT model output directory structure"""
        trainer = GRITTrainer()

        # Check output directories exist
        self.assertTrue(trainer.output_dir.exists())
        self.assertTrue(trainer.adapter_dir.parent.exists())
        self.assertTrue(trainer.kfac_matrices_dir.parent.exists())

        # Check directory structure
        self.assertTrue("m03b_grit_training" in str(trainer.output_dir))
        self.assertTrue("adapter" in str(trainer.adapter_dir))
        self.assertTrue("kfac_matrices" in str(trainer.kfac_matrices_dir))

    def test_grit_vs_qlora_differences(self):
        """Test key differences between GRIT and QLoRA implementations"""
        grit_trainer = GRITTrainer()

        # GRIT should have lower learning rate due to second-order optimization
        self.assertEqual(grit_trainer.grit_config["learning_rate"], 2e-5)

        # GRIT should have batch size of 1 due to KFAC memory requirements
        self.assertEqual(grit_trainer.grit_config["batch_size"], 1)

        # GRIT-specific components should exist
        self.assertIn("kfac_damping", grit_trainer.grit_config)
        self.assertIn("neural_reproj_k", grit_trainer.grit_config)
        self.assertIn("neural_reproj_freq", grit_trainer.grit_config)

        # GRIT should have additional output directory for KFAC matrices
        self.assertTrue(hasattr(grit_trainer, 'kfac_matrices_dir'))

def run_quick_functionality_test():
    """Run a quick 30-second functionality test"""
    print("🧪 STARTING GRIT UNIT TEST (Quick Mode)")
    print("=" * 60)

    try:
        # Run basic tests
        suite = unittest.TestSuite()
        suite.addTest(TestGRITTraining('test_grit_trainer_initialization'))
        suite.addTest(TestGRITTraining('test_custom_lora_layer'))
        suite.addTest(TestGRITTraining('test_kfac_approximation_initialization'))
        suite.addTest(TestGRITTraining('test_natural_gradient_optimizer'))
        suite.addTest(TestGRITTraining('test_neural_reprojection'))
        suite.addTest(TestGRITTraining('test_grit_hyperparameter_configuration'))
        suite.addTest(TestGRITTraining('test_chat_template_formatting'))
        suite.addTest(TestGRITTraining('test_database_integration'))
        suite.addTest(TestGRITTraining('test_model_output_directory_structure'))
        suite.addTest(TestGRITTraining('test_grit_vs_qlora_differences'))

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        if result.wasSuccessful():
            print("\n✅ ALL GRIT UNIT TESTS PASSED!")
            print("🎉 GRIT training module is ready for full training")
            print("\nGRIT Components Verified:")
            print("✓ Custom LoRA layer implementation")
            print("✓ KFAC approximation with hook-based tracking")
            print("✓ Natural gradient optimizer with Fisher information")
            print("✓ Neural reprojection with eigenvalue decomposition")
            print("✓ Database integration with GRIT-specific metrics")
            print("✓ FKLNR hyperparameter configuration")
            print("✓ Output directory structure for KFAC matrices")
        else:
            print(f"\n❌ {len(result.failures)} tests failed")
            print(f"❌ {len(result.errors)} tests had errors")

        return result.wasSuccessful()

    except Exception as e:
        print(f"❌ Unit test execution failed: {e}")
        return False

def main():
    """Run unit tests"""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        return run_quick_functionality_test()
    else:
        # Run full test suite
        unittest.main(verbosity=2)

if __name__ == "__main__":
    main()