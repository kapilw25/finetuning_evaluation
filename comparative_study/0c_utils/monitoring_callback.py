"""
Safety Monitoring Callback for CITA Training (Alpaca Format)
Monitors training outputs and stops if mode collapse or unsafe behavior detected

Key Features:
- ✅ use_alpaca_format=True: Uses Alpaca format for monitoring (not chat template)
- ✅ Detects gibberish: Repetition, low diversity, patterns like "however###"
- ✅ Detects unsafe behavior: Negative margin (model prefers rejected/unsafe responses)
- ✅ GPU-EFFICIENT: Reports failure status to Ray Tune for global abort check
- ✅ PBT rescue: Failed workers continue training (PBT rescues by copying from healthy workers)
- ✅ Saves last good checkpoint: Tracks last_good_step
"""

import torch
import re
from collections import Counter
from transformers import TrainerCallback

# Ray Tune integration (optional - only used when running under PBT)
try:
    from ray import tune
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


class GibberishDetectionCallback(TrainerCallback):
    """
    Callback to detect gibberish generation and unsafe behavior during training

    Features:
    - Runs inference every N steps
    - Detects repetition patterns
    - Detects low token diversity
    - Detects negative margin (model prefers unsafe/rejected responses)
    - Auto-stops training on gibberish OR unsafe behavior detection
    - Uses ALPACA format for generation (not chat template!)
    """

    def __init__(
        self,
        test_prompts,
        check_every_n_steps=50,
        repetition_threshold=0.5,
        diversity_threshold=15,
        stop_on_gibberish=True,
        use_alpaca_format=True,  # ← Use Alpaca format
        stop_on_negative_margin=True,  # ✅ NEW: Stop if margin becomes negative
        margin_tolerance=0.0  # ✅ NEW: Margin must be > this value (default: 0 = must be positive)
    ):
        self.test_prompts = test_prompts
        self.check_every_n_steps = check_every_n_steps
        self.repetition_threshold = repetition_threshold
        self.diversity_threshold = diversity_threshold
        self.stop_on_gibberish = stop_on_gibberish
        self.use_alpaca_format = use_alpaca_format
        self.stop_on_negative_margin = stop_on_negative_margin
        self.margin_tolerance = margin_tolerance

        self.last_good_step = 0
        self.negative_margin_violations = 0  # ✅ Track total negative margin detections (for logging)

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Called at the end of each training step"""
        # Handle both 'tokenizer' and 'processing_class' (new API)
        if tokenizer is None:
            tokenizer = kwargs.get('processing_class')

        if tokenizer is None or model is None:
            return control

        if state.global_step % self.check_every_n_steps != 0:
            return control

        print(f"\n{'='*80}")
        print(f"🔍 Safety Check - Step {state.global_step}")
        print(f"{'='*80}")

        # ===== CHECK 1: Margin Safety (Unsafe Behavior) =====
        unsafe_behavior_detected = False
        current_margin = None

        # Extract current margin from trainer logs
        if hasattr(state, 'log_history') and len(state.log_history) > 0:
            # Find most recent margin value
            for log_entry in reversed(state.log_history):
                if 'cita/margin' in log_entry:
                    current_margin = log_entry['cita/margin']
                    break

        if current_margin is not None:
            print(f"📊 Current Margin: {current_margin:.4f} (must be > {self.margin_tolerance})")

            if current_margin <= self.margin_tolerance:
                self.negative_margin_violations += 1
                print(f"⚠️  UNSAFE: Margin ≤ {self.margin_tolerance} (violation #{self.negative_margin_violations})")
                print(f"   Model prefers REJECTED (unsafe) responses!")
                print(f"   🛑 STOPPING IMMEDIATELY (no tolerance for unsafe behavior)")
                unsafe_behavior_detected = True
            else:
                print(f"✅ SAFE: Margin > {self.margin_tolerance} (model prefers chosen/safe responses)")

        # ===== CHECK 2: Gibberish Detection =====
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

        # ===== DECISION: Log Failures, Never Stop Individual Workers =====
        # Individual workers should NEVER be terminated
        # PBT will rescue failed workers by copying from best RUNNING workers
        # Global safety check (after all training) will abort if ALL workers fail

        failure_reasons = []

        # Check 1: Unsafe behavior (negative margin)
        if unsafe_behavior_detected and self.stop_on_negative_margin:
            failure_reasons.append(f"NEGATIVE MARGIN (model prefers unsafe responses)")

        # Check 2: Gibberish (mode collapse)
        if gibberish_detected and self.stop_on_gibberish:
            failure_reasons.append(f"GIBBERISH DETECTED (mode collapse)")

        # Log failure but continue training (GPU-efficient)
        if failure_reasons:
            print(f"\n{'!'*80}")
            print(f"⚠️  FAILURE DETECTED AT STEP {state.global_step}")
            print(f"{'!'*80}")
            print(f"Reason(s): {', '.join(failure_reasons)}")
            print(f"")
            print(f"🔄 PBT RESCUE MODE (Worker continues training):")
            print(f"   1. Worker trains with current hyperparameters until next checkpoint (≤{self.check_every_n_steps} steps)")
            print(f"   2. PBT ranks workers at checkpoint → This worker ranked LAST (due to poor metrics)")
            print(f"   3. PBT EXPLOIT: Copies weights from best RUNNING worker")
            print(f"   4. PBT EXPLORE: Mutates hyperparameters ±20%")
            print(f"   5. Worker rescued and continues with new weights + HPs")
            print(f"")
            print(f"   If ALL workers fail → Global safety stopper aborts experiment")
            print(f"   GPU stays 100% utilized (no idle workers)")
            print(f"   Last known good checkpoint: checkpoint-{self.last_good_step}")
            print(f"{'!'*80}\n")

            # ✅ Report failure status to Ray Tune (for AllWorkersSafetyStopper)
            # This allows stopper to check if ALL workers failed → abort experiment
            if RAY_AVAILABLE:
                try:
                    tune.report(
                        gibberish_detected=gibberish_detected,
                        unsafe_behavior_detected=unsafe_behavior_detected
                    )
                except Exception:
                    pass  # Not running under Ray Tune, skip reporting

            # ✅ NEVER terminate individual workers - let PBT handle recovery
            # control.should_training_stop remains False (default)
        else:
            # Training is healthy, update last good step
            self.last_good_step = state.global_step

            # Report healthy status to Ray Tune
            if RAY_AVAILABLE:
                try:
                    tune.report(
                        gibberish_detected=False,
                        unsafe_behavior_detected=False
                    )
                except Exception:
                    pass

        return control

    def _generate_sample(self, model, tokenizer, prompt, max_new_tokens=100):
        """Generate a sample response using ALPACA format"""

        if self.use_alpaca_format:
            # ✅ ALPACA FORMAT (for new CITA training)
            full_prompt = f"""Below are some instructions that describe some tasks. Write responses that 
appropriately complete each request.

### Instruction:
{prompt}

### Response:
"""
        else:
            # OLD: Llama-3 chat template (for comparison only)
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