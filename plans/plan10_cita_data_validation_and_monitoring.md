# Plan 10: CITA Data Validation & Training Monitoring

**Goal**: Verify CITA data formatting is correct and add monitoring to detect gibberish generation during training

**Problem**: CITA_Baseline_BF16 generates gibberish ("however###...") instead of meaningful text, indicating training failure

**Root Causes to Investigate**:
1. ❓ Data formatting issue (chat template mismatch)
2. ❓ Training monitoring missing (no early detection of mode collapse)
3. ❓ Hyperparameters (KL penalty too high)

---

## 📋 Execution Plan

### Phase 1: Data Validation Script (High Priority)
**File**: `comparative_study/0c_DATA_PREP_utils/src/validate_cita_data.py`

**Purpose**: Verify CITA formatted data matches expected Llama-3 chat template

**Tasks**:
1. ✅ Load PKU-SafeRLHF filtered dataset
2. ✅ Format with CITA method
3. ✅ Apply Llama-3 chat template
4. ✅ Validate structure (system/user/assistant turns)
5. ✅ Check for template artifacts
6. ✅ Generate sample previews

**Expected Output**:
```
✅ CITA Data Validation Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset: PKU-SafeRLHF (filtered)
Samples: 10,813
Format: CITA (chosen/rejected pairs)

Structure Validation:
  ✅ All samples have 'chosen' and 'rejected' keys
  ✅ All trajectories start with 'system' message
  ✅ All trajectories have 3 turns (system/user/assistant)
  ✅ System message contains safety instructions

Template Validation:
  ✅ Chat template applied successfully
  ✅ No template artifacts detected
  ✅ Tokenization successful
  ⚠️  Max length: 2048 tokens (within limit)

Sample Preview (Chosen):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant...<|eot_id|><|start_header_id|>user<|end_header_id|>

My roommate constantly...<|eot_id|><|start_header_id|>assistant<|end_header_id|>

You can express your anger...<|eot_id|>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### Phase 2: Training Monitoring Callback (Critical)
**File**: `comparative_study/03a_CITA_Baseline/monitoring_callback.py`

**Purpose**: Detect gibberish generation during training and stop early

**Tasks**:
1. ✅ Create custom TrainerCallback
2. ✅ Run inference every N steps (e.g., 50 steps)
3. ✅ Detect gibberish patterns (repetition, nonsense tokens)
4. ✅ Log sample outputs to TensorBoard
5. ✅ Auto-stop training if gibberish detected
6. ✅ Save checkpoint before stopping

**Key Features**:
- **Gibberish Detection**:
  - Repetition score > 0.5 (severe repetition)
  - Token diversity < 10 unique tokens
  - Contains patterns like "however###", "####", etc.
- **Early Stopping**: Stops training at first sign of mode collapse
- **Checkpoint Recovery**: Saves last good checkpoint

**Expected Output** (during training):
```
Step 100: ✅ Sample generation OK
  Prompt: "Explain healthy breakfast"
  Response: "1. Choose whole grain cereal. 2. Add fresh fruit..."
  Repetition: 0.12 | Diversity: 45 tokens ✅

Step 150: ⚠️ Sample generation DEGRADING
  Prompt: "Explain healthy breakfast"
  Response: "however## however### however###..."
  Repetition: 0.87 | Diversity: 3 tokens ⚠️

Step 200: ❌ GIBBERISH DETECTED - STOPPING TRAINING
  Last good checkpoint: checkpoint-150
  Training stopped to prevent mode collapse
```

---

### Phase 3: Unit Test for Data Pipeline
**File**: `comparative_study/0c_DATA_PREP_utils/unit_test/test_cita_formatting.py`

**Purpose**: Automated tests to catch formatting bugs

**Tasks**:
1. ✅ Test `load_pku_filtered()` output structure
2. ✅ Test `format_dataset(method="cita")` output
3. ✅ Test chat template application
4. ✅ Test tokenization
5. ✅ Test edge cases (empty messages, long texts)

**Expected Output**:
```
test_cita_formatting.py::test_load_pku_filtered ✅ PASSED
test_cita_formatting.py::test_format_dataset_cita ✅ PASSED
test_cita_formatting.py::test_chat_template_application ✅ PASSED
test_cita_formatting.py::test_tokenization_within_limits ✅ PASSED
test_cita_formatting.py::test_edge_cases ✅ PASSED

5 passed in 12.34s
```

---

### Phase 4: Updated Training Notebook
**File**: `comparative_study/03a_CITA_Baseline/Llama3_BF16_v2.ipynb`

**Purpose**: Re-train with monitoring and validation

**Changes**:
1. ✅ Import validation script at start
2. ✅ Run data validation before training
3. ✅ Add monitoring callback to trainer
4. ✅ Reduce KL penalty (`lambda_kl = 0.001` instead of `0.01`)
5. ✅ Add intermediate inference tests
6. ✅ Save checkpoints more frequently (every 50 steps)

**New Cell Structure**:
```python
# Cell 1: Validate data BEFORE training
from validate_cita_data import validate_cita_dataset

print("🔍 Validating CITA data formatting...")
validation_report = validate_cita_dataset(dataset)
if not validation_report["all_passed"]:
    raise ValueError("❌ Data validation failed! Check report above.")
print("✅ Data validation passed\n")

# Cell 2: Add monitoring callback
from monitoring_callback import GibberishDetectionCallback

gibberish_monitor = GibberishDetectionCallback(
    test_prompts=[
        "Explain how to make a healthy breakfast in 3 steps.",
        "What are the benefits of regular exercise?"
    ],
    check_every_n_steps=50,
    stop_on_gibberish=True
)

# Cell 3: Train with monitoring
cita_trainer = CITATrainer(
    model=model,
    ref_model=None,
    args=DPOConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=1000,
        learning_rate=2e-5,
        lambda_kl=0.001,  # ← REDUCED from 0.01
        save_steps=50,  # ← More frequent checkpoints
        ...
    ),
    callbacks=[gibberish_monitor],  # ← Add monitoring
    ...
)

cita_trainer.train()
```

---

## 🔧 Implementation Details

### 1. `validate_cita_data.py` (Core Logic)

```python
#!/usr/bin/env python3
"""
CITA Data Validation Script
Validates PKU-SafeRLHF data formatting for CITA training
"""

import sys
from pathlib import Path
from collections import Counter
import re

# Add data_prep to path
sys.path.insert(0, str(Path(__file__).parent))
from data_prep import load_pku_filtered, format_dataset

def validate_cita_dataset(dataset=None, verbose=True):
    """
    Validate CITA formatted dataset

    Returns:
        dict: Validation report with pass/fail for each check
    """
    report = {
        "all_passed": True,
        "checks": {}
    }

    # Load dataset if not provided
    if dataset is None:
        dataset = load_pku_filtered(split="train", max_samples=None)
        dataset = format_dataset(dataset, method="cita")

    # Check 1: Structure validation
    report["checks"]["structure"] = validate_structure(dataset)

    # Check 2: Chat template validation
    report["checks"]["chat_template"] = validate_chat_template(dataset[:100])

    # Check 3: Tokenization validation
    report["checks"]["tokenization"] = validate_tokenization(dataset[:10])

    # Check 4: Content quality
    report["checks"]["content"] = validate_content(dataset[:100])

    # Overall result
    report["all_passed"] = all(
        check["passed"] for check in report["checks"].values()
    )

    if verbose:
        print_validation_report(report, dataset)

    return report


def validate_structure(dataset):
    """Validate dataset structure (chosen/rejected keys, message format)"""
    errors = []

    for i, sample in enumerate(dataset):
        # Check keys
        if "chosen" not in sample or "rejected" not in sample:
            errors.append(f"Sample {i}: Missing 'chosen' or 'rejected' key")
            continue

        # Check message structure
        for traj_type in ["chosen", "rejected"]:
            traj = sample[traj_type]

            if not isinstance(traj, list):
                errors.append(f"Sample {i}/{traj_type}: Not a list")
                continue

            if len(traj) != 3:
                errors.append(f"Sample {i}/{traj_type}: Expected 3 turns, got {len(traj)}")

            if traj[0]["role"] != "system":
                errors.append(f"Sample {i}/{traj_type}: First role should be 'system', got '{traj[0]['role']}'")

            if traj[1]["role"] != "user":
                errors.append(f"Sample {i}/{traj_type}: Second role should be 'user', got '{traj[1]['role']}'")

            if traj[2]["role"] != "assistant":
                errors.append(f"Sample {i}/{traj_type}: Third role should be 'assistant', got '{traj[2]['role']}'")

    return {
        "passed": len(errors) == 0,
        "errors": errors[:10],  # Show first 10 errors
        "total_errors": len(errors)
    }


def validate_chat_template(dataset_sample):
    """Validate chat template can be applied"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    tokenizer.chat_template = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"""

    errors = []

    for i, sample in enumerate(dataset_sample):
        try:
            # Try applying chat template to chosen trajectory
            text = tokenizer.apply_chat_template(
                sample["chosen"],
                tokenize=False,
                add_generation_prompt=False
            )

            # Check for template artifacts
            if "<|start_header_id|>" not in text:
                errors.append(f"Sample {i}: Missing Llama-3 template markers")

            if "<|eot_id|>" not in text:
                errors.append(f"Sample {i}: Missing end-of-turn markers")

        except Exception as e:
            errors.append(f"Sample {i}: Template application failed - {str(e)}")

    return {
        "passed": len(errors) == 0,
        "errors": errors[:10],
        "total_errors": len(errors)
    }


def validate_tokenization(dataset_sample):
    """Validate tokenization works and lengths are reasonable"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    tokenizer.chat_template = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"""

    errors = []
    max_length = 0

    for i, sample in enumerate(dataset_sample):
        try:
            text = tokenizer.apply_chat_template(
                sample["chosen"],
                tokenize=False,
                add_generation_prompt=False
            )

            tokens = tokenizer.encode(text)
            length = len(tokens)
            max_length = max(max_length, length)

            if length > 2048:
                errors.append(f"Sample {i}: Length {length} exceeds max (2048)")

        except Exception as e:
            errors.append(f"Sample {i}: Tokenization failed - {str(e)}")

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "max_length": max_length,
        "total_errors": len(errors)
    }


def validate_content(dataset_sample):
    """Validate content quality (no empty messages, reasonable length)"""
    errors = []

    for i, sample in enumerate(dataset_sample):
        for traj_type in ["chosen", "rejected"]:
            traj = sample[traj_type]

            for j, msg in enumerate(traj):
                content = msg.get("content", "")

                if not content or len(content.strip()) == 0:
                    errors.append(f"Sample {i}/{traj_type}/turn{j}: Empty content")

                if len(content) < 10:
                    errors.append(f"Sample {i}/{traj_type}/turn{j}: Content too short ({len(content)} chars)")

    return {
        "passed": len(errors) == 0,
        "errors": errors[:10],
        "total_errors": len(errors)
    }


def print_validation_report(report, dataset):
    """Pretty print validation report"""
    print("\n" + "="*80)
    print("✅ CITA Data Validation Report" if report["all_passed"] else "❌ CITA Data Validation Report")
    print("="*80)
    print(f"Dataset: PKU-SafeRLHF (filtered)")
    print(f"Samples: {len(dataset):,}")
    print(f"Format: CITA (chosen/rejected pairs)\n")

    for check_name, result in report["checks"].items():
        status = "✅" if result["passed"] else "❌"
        print(f"{status} {check_name.title()} Check:")

        if result["passed"]:
            print(f"   All checks passed")
        else:
            print(f"   {result['total_errors']} errors found:")
            for error in result.get("errors", [])[:3]:
                print(f"   - {error}")
            if result['total_errors'] > 3:
                print(f"   ... and {result['total_errors'] - 3} more")
        print()

    # Show sample
    if report["all_passed"]:
        print("Sample Preview (Chosen):")
        print("━" * 80)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        tokenizer.chat_template = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"""

        sample_text = tokenizer.apply_chat_template(
            dataset[0]["chosen"],
            tokenize=False,
            add_generation_prompt=False
        )
        print(sample_text[:500] + "..." if len(sample_text) > 500 else sample_text)
        print("━" * 80)

    print("="*80)


if __name__ == "__main__":
    # Run validation
    dataset = load_pku_filtered(split="train", max_samples=None)
    dataset = format_dataset(dataset, method="cita")

    report = validate_cita_dataset(dataset, verbose=True)

    # Exit with error code if validation failed
    sys.exit(0 if report["all_passed"] else 1)
```

---

### 2. `monitoring_callback.py` (Gibberish Detection)

```python
#!/usr/bin/env python3
"""
Gibberish Detection Callback for CITA Training
Monitors training outputs and stops if mode collapse detected
"""

import torch
import re
from collections import Counter
from transformers import TrainerCallback


class GibberishDetectionCallback(TrainerCallback):
    """
    Callback to detect gibberish generation during training

    Features:
    - Runs inference every N steps
    - Detects repetition patterns
    - Detects low token diversity
    - Auto-stops training on gibberish detection
    """

    def __init__(
        self,
        test_prompts,
        check_every_n_steps=50,
        repetition_threshold=0.5,
        diversity_threshold=15,
        stop_on_gibberish=True
    ):
        self.test_prompts = test_prompts
        self.check_every_n_steps = check_every_n_steps
        self.repetition_threshold = repetition_threshold
        self.diversity_threshold = diversity_threshold
        self.stop_on_gibberish = stop_on_gibberish

        self.last_good_step = 0

    def on_step_end(self, args, state, control, model, tokenizer, **kwargs):
        """Called at the end of each training step"""

        if state.global_step % self.check_every_n_steps != 0:
            return control

        print(f"\n{'='*80}")
        print(f"🔍 Gibberish Check - Step {state.global_step}")
        print(f"{'='*80}")

        # Run inference on test prompts
        gibberish_detected = False

        for prompt in self.test_prompts:
            response = self._generate_sample(model, tokenizer, prompt)

            # Analyze response
            repetition_score = self._detect_repetition(response)
            diversity_score = self._calculate_diversity(response)
            is_gibberish = self._is_gibberish(response, repetition_score, diversity_score)

            # Log results
            status = "❌ GIBBERISH" if is_gibberish else "✅ OK"
            print(f"\n{status} | Prompt: {prompt[:50]}...")
            print(f"  Response: {response[:100]}...")
            print(f"  Repetition: {repetition_score:.2f} | Diversity: {diversity_score} tokens")

            if is_gibberish:
                gibberish_detected = True

        print(f"{'='*80}\n")

        # Handle gibberish detection
        if gibberish_detected:
            print(f"\n{'!'*80}")
            print(f"⚠️  GIBBERISH DETECTED AT STEP {state.global_step}")
            print(f"{'!'*80}")
            print(f"Last good checkpoint: checkpoint-{self.last_good_step}")
            print(f"Training will {'STOP' if self.stop_on_gibberish else 'CONTINUE'}")
            print(f"{'!'*80}\n")

            if self.stop_on_gibberish:
                control.should_training_stop = True
        else:
            self.last_good_step = state.global_step

        return control

    def _generate_sample(self, model, tokenizer, prompt, max_new_tokens=100):
        """Generate a sample response"""
        # Format prompt with Llama-3 chat template
        messages = [{"role": "user", "content": prompt}]
        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()

    def _detect_repetition(self, text):
        """Detect n-gram repetition (0.0 = no repetition, 1.0 = severe)"""
        words = text.split()
        if len(words) < 10:
            return 0.0

        repetition_score = 0.0
        for n in [3, 4, 5]:
            ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
            if len(ngrams) == 0:
                continue
            unique_ngrams = len(set(ngrams))
            total_ngrams = len(ngrams)
            repetition_rate = 1.0 - (unique_ngrams / total_ngrams)
            repetition_score = max(repetition_score, repetition_rate)

        return repetition_score

    def _calculate_diversity(self, text):
        """Calculate unique token count"""
        tokens = text.split()
        return len(set(tokens))

    def _is_gibberish(self, text, repetition_score, diversity_score):
        """Determine if text is gibberish"""
        # Check 1: High repetition
        if repetition_score > self.repetition_threshold:
            return True

        # Check 2: Low diversity
        if diversity_score < self.diversity_threshold:
            return True

        # Check 3: Known gibberish patterns
        gibberish_patterns = [
            r'however#{3,}',  # "however###..."
            r'#{10,}',  # "##########..."
            r'(\w+#{2,}){3,}',  # "word## word### word##"
        ]

        for pattern in gibberish_patterns:
            if re.search(pattern, text):
                return True

        return False
```

---

## ✅ Success Criteria

After implementing this plan:

1. **Data Validation** ✅:
   - All 10,813 samples pass structure validation
   - Chat template applies without errors
   - No tokenization issues

2. **Training Monitoring** ✅:
   - Inference samples logged every 50 steps
   - Gibberish detected before step 200 (if it occurs)
   - Training auto-stops with saved checkpoint

3. **Model Quality** ✅:
   - CITA_Baseline_BF16 generates coherent text
   - Properly refuses harmful requests
   - No repetition or mode collapse

---

## 🚀 Execution Order

1. **First**: Run `validate_cita_data.py` standalone
2. **Second**: Add unit tests and verify
3. **Third**: Create `monitoring_callback.py`
4. **Fourth**: Update training notebook with validation + monitoring
5. **Fifth**: Re-train CITA_Baseline_BF16 with monitoring

---

## 📊 Expected Timeline

| Phase | Time | Priority |
|-------|------|----------|
| Data Validation Script | 30 min | ⚡ HIGH |
| Monitoring Callback | 45 min | ⚡ HIGH |
| Unit Tests | 30 min | 🔵 MEDIUM |
| Updated Training Notebook | 15 min | ⚡ HIGH |
| **Re-training** | 2-3 hours | ⚡ HIGH |

**Total**: ~1.5 hours coding + 2-3 hours training

---

## 🎯 Next Steps

After this plan is executed:
1. ✅ CITA_Baseline_BF16 will be properly trained
2. ✅ Same validation can be applied to CITA_GRIT_BF16
3. ✅ Monitoring callback can be reused for SFT/DPO training
4. ✅ Data validation becomes part of standard workflow
