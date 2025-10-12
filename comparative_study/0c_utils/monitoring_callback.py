"""
Gibberish Detection Callback for CITA Training (Alpaca Format)
Monitors training outputs and stops if mode collapse detected
# Key Features:
# - ✅ use_alpaca_format=True: Uses Alpaca format for monitoring (not chat template)
# - ✅ Detects gibberish: Repetition, low diversity, patterns like "however###"
# - ✅ Auto-stops: Stops training when gibberish detected
# - ✅ Saves last good checkpoint: Tracks last_good_step
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
    - Uses ALPACA format for generation (not chat template!)
    """

    def __init__(
        self,
        test_prompts,
        check_every_n_steps=50,
        repetition_threshold=0.5,
        diversity_threshold=15,
        stop_on_gibberish=True,
        use_alpaca_format=True  # ← NEW: Use Alpaca format
    ):
        self.test_prompts = test_prompts
        self.check_every_n_steps = check_every_n_steps
        self.repetition_threshold = repetition_threshold
        self.diversity_threshold = diversity_threshold
        self.stop_on_gibberish = stop_on_gibberish
        self.use_alpaca_format = use_alpaca_format

        self.last_good_step = 0

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