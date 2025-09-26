#!/usr/bin/env python3
"""
Unit Tests for Module 05: MVP Inference Backends
Purpose: Test MVP inference backend functionality

Tests:
- Class instantiation and configuration
- Model loading simulation (without actual ML models)
- API interface validation
- Database integration
- Performance tracking
- Memory management
"""

import unittest
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestMVPInferenceBackend(unittest.TestCase):
    """Test cases for MVPInferenceBackend class"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())

        # Mock the imports that require ML packages
        self.transformers_mock = MagicMock()
        self.peft_mock = MagicMock()

        # Create mock modules
        sys.modules['transformers'] = self.transformers_mock
        sys.modules['peft'] = self.peft_mock

        # Set up mock classes
        self.transformers_mock.AutoModelForCausalLM = MagicMock()
        self.transformers_mock.AutoTokenizer = MagicMock()
        self.transformers_mock.BitsAndBytesConfig = MagicMock()
        self.peft_mock.PeftModel = MagicMock()
        self.peft_mock.PeftConfig = MagicMock()

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
        # Remove mock modules
        if 'transformers' in sys.modules:
            del sys.modules['transformers']
        if 'peft' in sys.modules:
            del sys.modules['peft']

    @patch('sys.path')
    @patch('src.m00_centralized_db.CentralizedDB')
    def test_class_instantiation(self, mock_db, mock_path):
        """Test MVPInferenceBackend class instantiation"""
        # Import after mocking
        from m05_inference_backends import MVPInferenceBackend

        # Mock database
        mock_db_instance = MagicMock()
        mock_db.return_value = mock_db_instance

        # Create instance
        backend = MVPInferenceBackend()

        # Verify basic attributes
        self.assertIsNotNone(backend.db)
        self.assertEqual(backend.base_model_name, "meta-llama/Meta-Llama-3-8B-Instruct")
        self.assertIn("baseline", backend.models)
        self.assertIn("qlora", backend.models)
        self.assertIsNone(backend.current_model)

    @patch('src.m00_centralized_db.CentralizedDB')
    def test_quantization_config(self, mock_db):
        """Test 4-bit quantization configuration"""
        from m05_inference_backends import MVPInferenceBackend

        backend = MVPInferenceBackend()

        # Mock BitsAndBytesConfig
        mock_config = MagicMock()
        self.transformers_mock.BitsAndBytesConfig.return_value = mock_config

        config = backend.setup_quantization_config()

        # Verify BitsAndBytesConfig was called with correct parameters
        self.transformers_mock.BitsAndBytesConfig.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=unittest.mock.ANY
        )

    @patch('src.m00_centralized_db.CentralizedDB')
    def test_model_switching(self, mock_db):
        """Test model switching functionality"""
        from m05_inference_backends import MVPInferenceBackend

        backend = MVPInferenceBackend()

        # Test invalid model type
        self.assertFalse(backend.switch_model("invalid_model"))

        # Test switching to baseline (but model not loaded)
        self.assertFalse(backend.switch_model("baseline"))

        # Mock loaded baseline model
        backend.models["baseline"] = MagicMock()
        self.assertTrue(backend.switch_model("baseline"))
        self.assertEqual(backend.current_model, "baseline")

        # Mock loaded QLoRA model
        backend.models["qlora"] = MagicMock()
        self.assertTrue(backend.switch_model("qlora"))
        self.assertEqual(backend.current_model, "qlora")

    @patch('src.m00_centralized_db.CentralizedDB')
    def test_generate_response_no_model(self, mock_db):
        """Test generate_response with no active model"""
        from m05_inference_backends import MVPInferenceBackend

        backend = MVPInferenceBackend()

        result = backend.generate_response("Test prompt")

        self.assertIn("error", result)
        self.assertEqual(result["error"], "No active model selected")

    @patch('torch.no_grad')
    @patch('src.m00_centralized_db.CentralizedDB')
    def test_generate_response_success(self, mock_db, mock_no_grad):
        """Test successful response generation"""
        from m05_inference_backends import MVPInferenceBackend

        backend = MVPInferenceBackend()

        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_model.device = "cuda:0"
        mock_model.generate.return_value = [torch.tensor([1, 2, 3, 4, 5])]

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Test response"

        backend.models["baseline"] = mock_model
        backend.tokenizer = mock_tokenizer
        backend.current_model = "baseline"

        # Mock torch tensor
        with patch('torch.tensor') as mock_tensor:
            mock_tensor.return_value = MagicMock()
            mock_tensor.return_value.to.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

            result = backend.generate_response("Test prompt")

        self.assertTrue(result.get("success", False))
        self.assertEqual(result.get("model"), "baseline")
        self.assertIn("response", result)

    @patch('src.m00_centralized_db.CentralizedDB')
    def test_batch_generate(self, mock_db):
        """Test batch generation functionality"""
        from m05_inference_backends import MVPInferenceBackend

        backend = MVPInferenceBackend()

        # Test with no model loaded
        prompts = ["Prompt 1", "Prompt 2"]
        results = backend.batch_generate(prompts, "baseline")

        self.assertEqual(len(results), 2)
        self.assertIn("error", results[0])

    @patch('src.m00_centralized_db.CentralizedDB')
    def test_memory_usage_calculation(self, mock_db):
        """Test memory usage calculation"""
        from m05_inference_backends import MVPInferenceBackend

        backend = MVPInferenceBackend()

        # Mock model with get_memory_footprint
        mock_model = MagicMock()
        mock_model.get_memory_footprint.return_value = 1024 * 1024 * 100  # 100 MB

        memory_mb = backend._get_model_memory_usage(mock_model)
        self.assertEqual(memory_mb, 100.0)

        # Mock model without get_memory_footprint
        mock_model_no_footprint = MagicMock()
        del mock_model_no_footprint.get_memory_footprint

        # Mock parameters method
        mock_param = MagicMock()
        mock_param.numel.return_value = 1000000  # 1M parameters
        mock_model_no_footprint.parameters.return_value = [mock_param]

        memory_mb = backend._get_model_memory_usage(mock_model_no_footprint)
        self.assertGreater(memory_mb, 0)

    @patch('src.m00_centralized_db.CentralizedDB')
    def test_cleanup_models(self, mock_db):
        """Test model cleanup functionality"""
        from m05_inference_backends import MVPInferenceBackend

        backend = MVPInferenceBackend()

        # Mock loaded models
        backend.models["baseline"] = MagicMock()
        backend.models["qlora"] = MagicMock()
        backend.tokenizer = MagicMock()

        with patch('gc.collect') as mock_gc, \
             patch('torch.cuda.empty_cache') as mock_empty_cache:

            backend.cleanup_models()

            # Verify cleanup
            self.assertIsNone(backend.models["baseline"])
            self.assertIsNone(backend.models["qlora"])
            self.assertIsNone(backend.tokenizer)
            mock_gc.assert_called_once()
            mock_empty_cache.assert_called_once()

    @patch('builtins.open', create=True)
    @patch('json.dump')
    @patch('src.m00_centralized_db.CentralizedDB')
    def test_run_mvp_setup(self, mock_db, mock_json_dump, mock_open):
        """Test MVP setup workflow"""
        from m05_inference_backends import MVPInferenceBackend

        backend = MVPInferenceBackend()

        # Mock successful loading
        with patch.object(backend, 'load_baseline_model', return_value=True), \
             patch.object(backend, 'load_qlora_model', return_value=True), \
             patch.object(backend, 'log_to_database'):

            results = backend.run_mvp_setup()

            self.assertTrue(results["setup_success"])
            self.assertTrue(results["baseline_loaded"])
            self.assertTrue(results["qlora_loaded"])
            self.assertIn("next_steps", results)

    @patch('src.m00_centralized_db.CentralizedDB')
    def test_database_logging(self, mock_db):
        """Test database logging functionality"""
        from m05_inference_backends import MVPInferenceBackend

        backend = MVPInferenceBackend()

        # Mock database instance
        mock_db_instance = MagicMock()
        backend.db = mock_db_instance

        # Set up performance metrics
        backend.performance_metrics = {
            "baseline": {
                "test_successful": True,
                "model_size_mb": 8000,
                "generation_time_ms": 150
            },
            "qlora": {
                "test_successful": True,
                "model_size_mb": 8200,
                "generation_time_ms": 160
            }
        }

        backend.log_to_database()

        # Verify database logging was called twice (baseline + qlora)
        self.assertEqual(mock_db_instance.log_module_data.call_count, 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for m05_inference_backends"""

    def test_file_structure(self):
        """Test that required files exist"""
        src_dir = Path(__file__).parent.parent / "src"

        # Check main module exists
        module_path = src_dir / "m05_inference_backends.py"
        self.assertTrue(module_path.exists())

        # Check database module exists
        db_path = src_dir / "m00_centralized_db.py"
        self.assertTrue(db_path.exists())

    def test_qlora_adapter_structure(self):
        """Test QLoRA adapter directory structure"""
        adapter_path = Path("outputs/m03a_qlora_training/adapter")

        if adapter_path.exists():
            # Check required files
            required_files = [
                "adapter_config.json",
                "adapter_model.safetensors"
            ]

            for req_file in required_files:
                file_path = adapter_path / req_file
                self.assertTrue(file_path.exists(), f"Required file missing: {req_file}")

    def test_database_table_exists(self):
        """Test that database table exists"""
        import sqlite3

        db_path = Path("outputs/centralized.db")
        if db_path.exists():
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='m04_inference_backends'
                """)
                result = cursor.fetchone()
                self.assertIsNotNone(result, "m04_inference_backends table not found")


def run_tests():
    """Run all tests and return results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestMVPInferenceBackend))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("UNIT TESTS: m05_inference_backends.py")
    print("=" * 60)

    # Import torch mock for tests
    import torch
    if 'torch' not in sys.modules:
        sys.modules['torch'] = MagicMock()
        sys.modules['torch'].tensor = MagicMock()
        sys.modules['torch'].no_grad = MagicMock()
        sys.modules['torch'].cuda = MagicMock()
        sys.modules['torch'].cuda.empty_cache = MagicMock()
        sys.modules['torch'].float16 = MagicMock()

    result = run_tests()

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print("\nFAILURES:")
        for test, trace in result.failures:
            print(f"- {test}: {trace}")

    if result.errors:
        print("\nERRORS:")
        for test, trace in result.errors:
            print(f"- {test}: {trace}")

    print("\n✅ Unit tests completed!")