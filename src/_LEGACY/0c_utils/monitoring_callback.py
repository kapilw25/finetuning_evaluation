"""
Safety Monitoring Callback for CITA Training (Alpaca Format)
Monitors training outputs and stops if mode collapse or unsafe behavior detected

Key Features:
- ✅ use_alpaca_format=True: Uses Alpaca format for monitoring (not chat template)
- ✅ Detects gibberish: Repetition, low diversity, patterns like "however###"
- ✅ Detects unsafe behavior: Negative margin (model prefers rejected/unsafe responses)
- ✅ Optuna integration: Reports metrics for pruning
- ✅ Saves last good checkpoint: Tracks last_good_step
"""

import torch
import re
from collections import Counter
from transformers import TrainerCallback

# Optuna integration (optional - only used for pruning)
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None



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
        margin_tolerance=0.0,  # ✅ NEW: Margin must be > this value (default: 0 = must be positive)
        stop_on_high_kl=True,  # ✅ NEW: Stop if KL divergence too high (drift from reference)
        kl_threshold=0.5,  # ✅ NEW: KL must be < this value (default: 0.5)
        trial=None  # ✅ Optuna trial object for pruning
    ):
        self.test_prompts = test_prompts
        self.check_every_n_steps = check_every_n_steps
        self.repetition_threshold = repetition_threshold
        self.diversity_threshold = diversity_threshold
        self.stop_on_gibberish = stop_on_gibberish
        self.use_alpaca_format = use_alpaca_format
        self.stop_on_negative_margin = stop_on_negative_margin
        self.margin_tolerance = margin_tolerance
        self.stop_on_high_kl = stop_on_high_kl
        self.kl_threshold = kl_threshold
        self.trial = trial  # Store Optuna trial for pruning

        self.last_good_step = 0
        self.negative_margin_violations = 0  # ✅ Track total negative margin detections (for logging)
        self.kl_violations = 0  # ✅ Track KL divergence violations

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

        # Extract current margin from trainer logs (use EVAL margin, not train margin)
        if hasattr(state, 'log_history') and len(state.log_history) > 0:
            # Find most recent EVAL margin value (more reliable than training margin)
            for log_entry in reversed(state.log_history):
                if 'eval_rewards/margins' in log_entry:
                    current_margin = log_entry['eval_rewards/margins']
                    break

            # Fallback: If no eval margin yet, use training margin (less reliable)
            if current_margin is None:
                for log_entry in reversed(state.log_history):
                    if 'cita/margin' in log_entry:
                        current_margin = log_entry['cita/margin']
                        print(f"⚠️  Note: Using TRAIN margin (eval margin not available yet)")
                        break

        if current_margin is not None:
            print(f"📊 Current Eval Margin: {current_margin:.4f} (must be > {self.margin_tolerance})")

            if current_margin <= self.margin_tolerance:
                self.negative_margin_violations += 1
                print(f"⚠️  UNSAFE: Margin ≤ {self.margin_tolerance} (violation #{self.negative_margin_violations})")
                print(f"   Model prefers REJECTED (unsafe) responses!")
                print(f"   🛑 STOPPING IMMEDIATELY (no tolerance for unsafe behavior)")
                unsafe_behavior_detected = True
            else:
                print(f"✅ SAFE: Margin > {self.margin_tolerance} (model prefers chosen/safe responses)")

        # ===== CHECK 2: KL Divergence (Drift Detection) =====
        kl_drift_detected = False
        current_kl = None

        # Extract current KL divergence from trainer logs
        if hasattr(state, 'log_history') and len(state.log_history) > 0:
            for log_entry in reversed(state.log_history):
                if 'cita/loss_kl' in log_entry:
                    current_kl = log_entry['cita/loss_kl']
                    break

        if current_kl is not None:
            print(f"📊 Current KL: {current_kl:.4f} (must be < {self.kl_threshold})")

            if current_kl >= self.kl_threshold:
                self.kl_violations += 1
                print(f"⚠️  DRIFT: KL ≥ {self.kl_threshold} (violation #{self.kl_violations})")
                print(f"   Model drifting from reference (mode collapse risk)!")
                print(f"   🛑 STOPPING IMMEDIATELY (preventive)")
                kl_drift_detected = True
            else:
                print(f"✅ STABLE: KL < {self.kl_threshold} (model aligned with reference)")

        # ===== CHECK 3: Gibberish Detection =====
        gibberish_detected = False

        for prompt in self.test_prompts:
            response = self._generate_sample(model, tokenizer, prompt)

            # Analyze response
            repetition_score = self._detect_repetition(response)
            diversity_score = self._calculate_diversity(response)
            is_gibberish = self._is_gibberish(response, repetition_score, diversity_score)

            # Log results
            status = "❌ GIBBERISH" if is_gibberish else "✅ OK"

            # Show more context for gibberish cases (full response), less for OK cases
            if is_gibberish:
                print(f"\n{status} | Prompt: {prompt}")
                print(f"  Full Response ({len(response)} chars):")
                print(f"  → {response}")
                print(f"  Metrics → Repetition: {repetition_score:.2f} | Diversity: {diversity_score} tokens")
            else:
                print(f"\n{status} | Prompt: {prompt[:70]}...")
                print(f"  Response: {response[:150]}...")
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

        # Check 2: KL drift (preventive)
        if kl_drift_detected and self.stop_on_high_kl:
            failure_reasons.append(f"KL DIVERGENCE TOO HIGH (drift from reference)")

        # Check 3: Gibberish (mode collapse)
        if gibberish_detected and self.stop_on_gibberish:
            failure_reasons.append(f"GIBBERISH DETECTED (mode collapse)")

        # Handle failures based on stop flags
        if failure_reasons:
            print(f"\n{'!'*80}")
            print(f"⚠️  FAILURE DETECTED AT STEP {state.global_step}")
            print(f"{'!'*80}")
            print(f"Reason(s): {', '.join(failure_reasons)}")

            # Check if any stop flag is True
            should_stop = (
                (unsafe_behavior_detected and self.stop_on_negative_margin) or
                (kl_drift_detected and self.stop_on_high_kl) or
                (gibberish_detected and self.stop_on_gibberish)
            )

            if should_stop:
                print(f"\n🛑 STOPPING TRAINING")
                print(f"   stop_on_gibberish={self.stop_on_gibberish}, "
                      f"stop_on_negative_margin={self.stop_on_negative_margin}, "
                      f"stop_on_high_kl={self.stop_on_high_kl}")
                print(f"{'!'*80}\n")
                control.should_training_stop = True
            else:
                print(f"\n▶️  CONTINUING (all stop flags = False)")
                print(f"   stop_on_gibberish={self.stop_on_gibberish}, "
                      f"stop_on_negative_margin={self.stop_on_negative_margin}, "
                      f"stop_on_high_kl={self.stop_on_high_kl}")
                print(f"{'!'*80}\n")
        else:
            # Training is healthy, update last good step
            self.last_good_step = state.global_step

        # ===== OPTUNA PRUNING =====
        if self.trial is not None and hasattr(state, 'log_history') and len(state.log_history) > 0:
            try:
                # Get latest margin from log history
                latest_log = state.log_history[-1]
                if 'eval_rewards/margins' in latest_log:
                    margin = latest_log['eval_rewards/margins']
                    step = state.global_step

                    # Report intermediate value to Optuna
                    self.trial.report(margin, step)

                    # Check if trial should be pruned
                    if self.trial.should_prune():
                        print(f"\n🔪 OPTUNA PRUNING: Trial pruned at step {step} (margin={margin:.4f})")
                        raise optuna.TrialPruned()
            except Exception as e:
                # Don't fail training if Optuna pruning has issues
                if "TrialPruned" in str(type(e).__name__):
                    raise  # Re-raise TrialPruned
                pass  # Ignore other errors

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

        # ✅ FIX: Flash Attention requires BF16 during inference
        # Solution: Same as SFT/DPO inference - temporarily set model to BF16 + disable gradient checkpointing
        # Reference: comparative_study/01a_SFT_Baseline/Llama3_BF16.py:332-338

        # Save original training state
        was_training = model.training

        # Switch to eval mode + BF16 + disable gradient checkpointing
        model.eval()
        model = model.to(torch.bfloat16)
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()

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

        # Restore training state
        if was_training:
            model.train()
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()

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


# ===================================================================
# Training Summary Callback (ZERO overhead - just formats existing metrics)
# ===================================================================

class TrainingSummaryCallback(TrainerCallback):
    """
    Print training summary at checkpoints (NO inference, ZERO overhead)
    Just parses state.log_history and prints key metrics in clear format

    Works for:
    - SFT: loss trajectory, eval_loss
    - DPO: margin, accuracy, rewards
    - CITA: margin, accuracy, KL divergence, all loss components

    Usage:
        from monitoring_callback import TrainingSummaryCallback

        callback = TrainingSummaryCallback(
            check_every_n_steps=50,
            training_method="cita"  # or "sft" or "dpo"
        )
    """

    def __init__(
        self,
        check_every_n_steps=50,
        training_method="cita",  # "sft", "dpo", or "cita"
        window_size=50  # Number of recent batches to analyze
    ):
        self.check_every_n_steps = check_every_n_steps
        self.training_method = training_method.lower()
        self.window_size = window_size

    def on_step_end(self, args, state, control, **kwargs):
        """Print summary at checkpoint steps"""
        if state.global_step % self.check_every_n_steps != 0:
            return control

        print(f"\n{'='*80}")
        print(f"📊 TRAINING SUMMARY - Step {state.global_step}")
        print(f"{'='*80}")

        # Get recent logs
        recent_logs = state.log_history[-self.window_size:] if len(state.log_history) > self.window_size else state.log_history

        if self.training_method == "sft":
            self._print_sft_summary(recent_logs, state)
        elif self.training_method == "dpo":
            self._print_dpo_summary(recent_logs, state)
        elif self.training_method == "cita":
            self._print_cita_summary(recent_logs, state)
        elif self.training_method == "grpo":
            self._print_grpo_summary(recent_logs, state)
        else:
            print(f"⚠️  Unknown training_method: {self.training_method}")

        print(f"{'='*80}\n")

        return control

    def _print_sft_summary(self, recent_logs, state):
        """Print SFT-specific summary"""
        # Extract loss values
        losses = [log['loss'] for log in recent_logs if 'loss' in log]

        if losses:
            print(f"LOSS trajectory (last {len(losses)} batches):")
            print(f"  Current: {losses[-1]:.4f}")
            print(f"  Average: {sum(losses)/len(losses):.4f}")
            print(f"  Min: {min(losses):.4f}")
            print(f"  Max: {max(losses):.4f}")

            # Trend analysis
            if len(losses) > 10:
                first_half = sum(losses[:len(losses)//2]) / (len(losses)//2)
                second_half = sum(losses[len(losses)//2:]) / (len(losses) - len(losses)//2)
                trend = "↓ DECREASING" if second_half < first_half else "↑ INCREASING"
                print(f"  Trend: {trend} (first half: {first_half:.4f}, second half: {second_half:.4f})")

        # Extract eval_loss
        eval_losses = [log['eval_loss'] for log in recent_logs if 'eval_loss' in log]
        if eval_losses:
            print(f"\nVALIDATION:")
            print(f"  eval_loss: {eval_losses[-1]:.4f}")

    def _print_dpo_summary(self, recent_logs, state):
        """Print DPO-specific summary"""
        # Extract DPO metrics
        margins = [log.get('rewards/margins', log.get('rewards/margin')) for log in recent_logs if 'rewards/margins' in log or 'rewards/margin' in log]
        accuracies = [log['rewards/accuracies'] for log in recent_logs if 'rewards/accuracies' in log]
        losses = [log['loss'] for log in recent_logs if 'loss' in log]

        if losses:
            print(f"LOSS trajectory:")
            print(f"  Current: {losses[-1]:.4f}")
            print(f"  Average: {sum(losses)/len(losses):.4f}")

        if margins:
            avg_margin = sum(margins) / len(margins)
            negative_count = sum(1 for m in margins if m < 0)

            print(f"\nMARGIN (chosen - rejected logps):")
            print(f"  Current: {margins[-1]:.4f}")
            print(f"  Average: {avg_margin:.4f}")
            print(f"  Min: {min(margins):.4f}")
            print(f"  Negative samples: {negative_count}/{len(margins)} ({negative_count/len(margins)*100:.0f}%)")

            if negative_count > len(margins) * 0.1:
                print(f"  ⚠️  WARNING: >10% negative margins = model prefers UNSAFE responses!")

        if accuracies:
            avg_acc = sum(accuracies) / len(accuracies) * 100
            print(f"\nACCURACY (chosen > rejected):")
            print(f"  Current: {accuracies[-1]*100:.0f}%")
            print(f"  Average: {avg_acc:.0f}%")

            if avg_acc < 80:
                print(f"  ⚠️  WARNING: Accuracy < 80% = model learning WRONG preferences!")

        # Eval metrics
        eval_margins = [log.get('eval_rewards/margins', log.get('eval_rewards/margin')) for log in recent_logs if 'eval_rewards/margins' in log or 'eval_rewards/margin' in log]
        eval_accs = [log['eval_rewards/accuracies'] for log in recent_logs if 'eval_rewards/accuracies' in log]

        if eval_margins or eval_accs:
            print(f"\nVALIDATION:")
            if eval_margins:
                print(f"  eval_margin: {eval_margins[-1]:.4f}")
            if eval_accs:
                print(f"  eval_accuracy: {eval_accs[-1]*100:.0f}%")

    def _print_cita_summary(self, recent_logs, state):
        """Print CITA-specific summary"""
        # Extract CITA metrics
        total_losses = [log['cita/loss_total'] for log in recent_logs if 'cita/loss_total' in log]
        loss_sft = [log['cita/loss_sft'] for log in recent_logs if 'cita/loss_sft' in log]
        loss_dpo = [log['cita/loss_dpo'] for log in recent_logs if 'cita/loss_dpo' in log]
        loss_kl = [log['cita/loss_kl'] for log in recent_logs if 'cita/loss_kl' in log]
        margins = [log['cita/margin'] for log in recent_logs if 'cita/margin' in log]
        accuracies = [log['rewards/accuracies'] for log in recent_logs if 'rewards/accuracies' in log]
        reward_margins = [log.get('rewards/margins', log.get('rewards/margin')) for log in recent_logs if 'rewards/margins' in log or 'rewards/margin' in log]

        # Loss breakdown
        if total_losses:
            print(f"LOSS COMPONENTS:")
            print(f"  Total loss: {total_losses[-1]:.4f} (avg: {sum(total_losses)/len(total_losses):.4f})")
            if loss_sft:
                print(f"  L_SFT: {loss_sft[-1]:.4f} (avg: {sum(loss_sft)/len(loss_sft):.4f})")
            if loss_dpo:
                print(f"  L_DPO: {loss_dpo[-1]:.4f} (avg: {sum(loss_dpo)/len(loss_dpo):.4f})")
            if loss_kl:
                avg_kl = sum(loss_kl) / len(loss_kl)
                print(f"  L_KL: {loss_kl[-1]:.4f} (avg: {avg_kl:.4f})")
                if avg_kl < -2.0:
                    print(f"  ⚠️  WARNING: High negative KL = model drifting from reference!")

        # Margin analysis
        if margins:
            avg_margin = sum(margins) / len(margins)
            negative_count = sum(1 for m in margins if m < 0)

            print(f"\nMARGIN (chosen - rejected logps):")
            print(f"  Current: {margins[-1]:.1f}")
            print(f"  Average: {avg_margin:.1f}")
            print(f"  Min: {min(margins):.1f} | Max: {max(margins):.1f}")
            print(f"  Negative samples: {negative_count}/{len(margins)} ({negative_count/len(margins)*100:.0f}%)")

            if negative_count > len(margins) * 0.1:
                print(f"  🚨 CRITICAL: {negative_count/len(margins)*100:.0f}% negative margins!")
                print(f"     Model systematically prefers UNSAFE responses!")

        # Accuracy analysis
        if accuracies:
            avg_acc = sum(accuracies) / len(accuracies) * 100
            print(f"\nACCURACY (chosen > rejected):")
            print(f"  Current: {accuracies[-1]*100:.0f}%")
            print(f"  Average: {avg_acc:.0f}%")

            if avg_acc < 60:
                print(f"  🚨 CRITICAL: Accuracy < 60% = worse than random guessing!")
            elif avg_acc < 80:
                print(f"  ⚠️  WARNING: Accuracy < 80% = model learning WRONG preferences!")

        # Reward margins
        if reward_margins:
            avg_reward_margin = sum(reward_margins) / len(reward_margins)
            print(f"\nREWARD MARGIN:")
            print(f"  Current: {reward_margins[-1]:.4f}")
            print(f"  Average: {avg_reward_margin:.4f}")

        # Validation metrics
        eval_margins = [log.get('eval_rewards/margins', log.get('eval_rewards/margin')) for log in recent_logs if 'eval_rewards/margins' in log or 'eval_rewards/margin' in log]
        eval_accs = [log['eval_rewards/accuracies'] for log in recent_logs if 'eval_rewards/accuracies' in log]

        if eval_margins or eval_accs:
            print(f"\nVALIDATION:")
            if eval_margins:
                print(f"  eval_margin: {eval_margins[-1]:.4f}")
            if eval_accs:
                val_acc = eval_accs[-1]*100
                print(f"  eval_accuracy: {val_acc:.0f}%")
                if val_acc < 70:
                    print(f"  🚨 CRITICAL: Validation accuracy < 70%!")

    def _print_grpo_summary(self, recent_logs, state):
        """Print GRPO-specific summary"""
        # Extract GRPO metrics (TRL GRPOTrainer logs these)
        losses = [log['loss'] for log in recent_logs if 'loss' in log]
        rewards_mean = [log.get('reward', log.get('rewards/mean')) for log in recent_logs if 'reward' in log or 'rewards/mean' in log]
        rewards_std = [log['rewards/std'] for log in recent_logs if 'rewards/std' in log]
        kl_values = [log['kl'] for log in recent_logs if 'kl' in log]
        completion_lengths = [log.get('completion_length', log.get('completion_length/mean')) for log in recent_logs if 'completion_length' in log or 'completion_length/mean' in log]

        # Loss trajectory
        if losses:
            print(f"LOSS trajectory:")
            print(f"  Current: {losses[-1]:.4f}")
            print(f"  Average: {sum(losses)/len(losses):.4f}")
            print(f"  Min: {min(losses):.4f} | Max: {max(losses):.4f}")

            # Trend analysis
            if len(losses) > 10:
                first_half = sum(losses[:len(losses)//2]) / (len(losses)//2)
                second_half = sum(losses[len(losses)//2:]) / (len(losses) - len(losses)//2)
                trend = "↓ DECREASING" if second_half < first_half else "↑ INCREASING"
                print(f"  Trend: {trend}")

        # Reward trajectory (key metric for GRPO)
        if rewards_mean:
            avg_reward = sum(rewards_mean) / len(rewards_mean)
            positive_count = sum(1 for r in rewards_mean if r > 0)

            print(f"\nREWARD (from reward function):")
            print(f"  Current: {rewards_mean[-1]:.4f}")
            print(f"  Average: {avg_reward:.4f}")
            print(f"  Min: {min(rewards_mean):.4f} | Max: {max(rewards_mean):.4f}")
            print(f"  Positive samples: {positive_count}/{len(rewards_mean)} ({positive_count/len(rewards_mean)*100:.0f}%)")

            if rewards_std:
                print(f"  Std dev: {rewards_std[-1]:.4f}")

            # Trend analysis
            if len(rewards_mean) > 10:
                first_half = sum(rewards_mean[:len(rewards_mean)//2]) / (len(rewards_mean)//2)
                second_half = sum(rewards_mean[len(rewards_mean)//2:]) / (len(rewards_mean) - len(rewards_mean)//2)
                if second_half > first_half:
                    print(f"  Trend: ↑ IMPROVING (first half: {first_half:.4f}, second half: {second_half:.4f})")
                else:
                    print(f"  Trend: ↓ DECLINING (first half: {first_half:.4f}, second half: {second_half:.4f})")
                    print(f"  ⚠️  WARNING: Reward declining - model may be learning wrong behavior!")

        # KL divergence (drift from reference)
        if kl_values:
            avg_kl = sum(kl_values) / len(kl_values)
            print(f"\nKL DIVERGENCE (drift from reference):")
            print(f"  Current: {kl_values[-1]:.4f}")
            print(f"  Average: {avg_kl:.4f}")
            print(f"  Min: {min(kl_values):.4f} | Max: {max(kl_values):.4f}")

            if avg_kl > 0.5:
                print(f"  ⚠️  WARNING: KL > 0.5 = significant drift from reference model!")
            elif avg_kl > 1.0:
                print(f"  🚨 CRITICAL: KL > 1.0 = severe drift, risk of mode collapse!")

        # Completion length (useful for detecting degenerate behavior)
        if completion_lengths:
            avg_len = sum(completion_lengths) / len(completion_lengths)
            print(f"\nCOMPLETION LENGTH:")
            print(f"  Current: {completion_lengths[-1]:.0f} tokens")
            print(f"  Average: {avg_len:.0f} tokens")

            if avg_len < 10:
                print(f"  ⚠️  WARNING: Very short completions - possible reward hacking!")
            elif avg_len > 500:
                print(f"  ⚠️  WARNING: Very long completions - possible verbosity issue!")

        # Summary status
        if rewards_mean and kl_values:
            print(f"\nSTATUS:")
            if rewards_mean[-1] > 0 and kl_values[-1] < 0.5:
                print(f"  ✅ HEALTHY: Positive reward with controlled KL")
            elif rewards_mean[-1] <= 0:
                print(f"  ⚠️  CONCERNING: Negative/zero reward")
            elif kl_values[-1] >= 0.5:
                print(f"  ⚠️  CONCERNING: High KL divergence")


# ===================================================================
# Standalone PPO Summary Function (PPOTrainer doesn't support callbacks)
# ===================================================================

def print_ppo_summary(
    step: int,
    reward_history: list,
    kl_history: list,
    policy_loss_history: list = None,
    value_loss_history: list = None,
    window_size: int = 50
):
    """
    Print PPO training summary (standalone function for PPOTrainer compatibility)

    PPOTrainer uses a custom training loop and doesn't support HuggingFace callbacks.
    Call this function directly from the PPO training loop.

    Args:
        step: Current training step
        reward_history: List of mean rewards
        kl_history: List of KL divergence values
        policy_loss_history: List of policy losses (optional)
        value_loss_history: List of value losses (optional)
        window_size: Number of recent batches to analyze
    """
    print(f"\n{'='*80}")
    print(f"📊 PPO TRAINING SUMMARY - Step {step}")
    print(f"{'='*80}")

    # Recent window
    window = min(window_size, len(reward_history))
    recent_rewards = reward_history[-window:]
    recent_kl = kl_history[-window:] if kl_history else []

    # Reward trajectory
    if recent_rewards:
        print(f"\nREWARD trajectory (last {window} batches):")
        print(f"  Current: {recent_rewards[-1]:.4f}")
        print(f"  Average: {sum(recent_rewards)/len(recent_rewards):.4f}")
        print(f"  Min: {min(recent_rewards):.4f}")
        print(f"  Max: {max(recent_rewards):.4f}")

        positive_count = sum(1 for r in recent_rewards if r > 0)
        print(f"  Positive samples: {positive_count}/{len(recent_rewards)} ({positive_count/len(recent_rewards)*100:.0f}%)")

        # Trend analysis
        if len(recent_rewards) > 10:
            first_half = sum(recent_rewards[:len(recent_rewards)//2]) / (len(recent_rewards)//2)
            second_half = sum(recent_rewards[len(recent_rewards)//2:]) / (len(recent_rewards) - len(recent_rewards)//2)
            if second_half > first_half:
                print(f"  Trend: ↑ IMPROVING (first half: {first_half:.4f}, second half: {second_half:.4f})")
            else:
                print(f"  Trend: ↓ DECLINING (first half: {first_half:.4f}, second half: {second_half:.4f})")
                print(f"  ⚠️  WARNING: Reward declining - model may be learning wrong behavior!")

    # KL divergence
    if recent_kl:
        avg_kl = sum(recent_kl) / len(recent_kl)
        print(f"\nKL DIVERGENCE (policy vs reference):")
        print(f"  Current: {recent_kl[-1]:.4f}")
        print(f"  Average: {avg_kl:.4f}")

        if avg_kl > 0.5:
            print(f"  ⚠️  WARNING: KL > 0.5 = significant drift from reference!")
        elif avg_kl > 1.0:
            print(f"  🚨 CRITICAL: KL > 1.0 = severe drift, risk of mode collapse!")

    # Policy loss
    if policy_loss_history:
        recent_policy_loss = policy_loss_history[-window:]
        print(f"\nPOLICY LOSS:")
        print(f"  Current: {recent_policy_loss[-1]:.4f}")
        print(f"  Average: {sum(recent_policy_loss)/len(recent_policy_loss):.4f}")

    # Value loss
    if value_loss_history:
        recent_value_loss = value_loss_history[-window:]
        print(f"\nVALUE LOSS:")
        print(f"  Current: {recent_value_loss[-1]:.4f}")
        print(f"  Average: {sum(recent_value_loss)/len(recent_value_loss):.4f}")

    # Summary status
    if recent_rewards and recent_kl:
        print(f"\nSTATUS:")
        if recent_rewards[-1] > 0 and recent_kl[-1] < 0.5:
            print(f"  ✅ HEALTHY: Positive reward with controlled KL")
        elif recent_rewards[-1] <= 0:
            print(f"  ⚠️  CONCERNING: Negative/zero reward")
        elif recent_kl[-1] >= 0.5:
            print(f"  ⚠️  CONCERNING: High KL divergence")

    print(f"{'='*80}\n")