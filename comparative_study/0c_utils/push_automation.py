"""
Push Automation Module for CITA PBT Training
Handles automatic pushing to HuggingFace and GitHub after training completes

Features:
- Push weights to HuggingFace (overwrites previous regardless of performance)
- Push codebase to GitHub ALWAYS (especially logs/)
- Handle large log files (>100MB): Split into chunks or use Git LFS
- GitHub credentials: email=kapilw25@gmail.com, username=kapilw25
- Integrates with auto-shutdown (runs before shutdown)

Usage:
    from push_automation import PushAutomation

    pusher = PushAutomation(
        hf_token="hf_xxx",
        github_email="kapilw25@gmail.com",
        github_username="kapilw25"
    )

    # After training completes
    pusher.push_all(
        best_trial=best_trial,
        best_checkpoint=best_checkpoint,
        hf_repo="kapilw25/llama3-8b-pku-cita-baseline-bf16",
        config_path="./outputs/training/CITA_Baseline/CITA_Baseline_config.json"
    )
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Load .env file to get GITHUB_TOKEN, HF_TOKEN, etc.
load_dotenv()


class PushAutomation:
    """
    Automated push to HuggingFace and GitHub after training

    Features:
    - HuggingFace push (overwrites previous regardless of performance)
    - Always push to GitHub (especially logs/)
    - Large log file handling (>100MB)
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        github_email: str = "kapilw25@gmail.com",
        github_username: str = "kapilw25",
        project_root: Optional[Path] = None
    ):
        """
        Initialize push automation

        Args:
            hf_token: HuggingFace API token (optional)
            github_email: GitHub email for git config
            github_username: GitHub username for git config
            project_root: Project root directory (auto-detected if None)
        """
        self.hf_token = hf_token
        self.github_email = github_email
        self.github_username = github_username

        # Auto-detect project root (3 levels up from this file)
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)

        self.logs_dir = self.project_root / "logs"
        self.outputs_dir = self.project_root / "outputs" / "training"

        # Configure git (idempotent - safe to run multiple times)
        self._configure_git()

    def _configure_git(self):
        """Configure git user credentials (if not already set)"""
        try:
            # Check if git user is configured
            result = subprocess.run(
                ["git", "config", "user.email"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )

            if result.returncode != 0 or not result.stdout.strip():
                # Not configured, set it
                subprocess.run(
                    ["git", "config", "user.email", self.github_email],
                    cwd=self.project_root,
                    check=True
                )
                subprocess.run(
                    ["git", "config", "user.name", self.github_username],
                    cwd=self.project_root,
                    check=True
                )
                print(f"✅ Configured git user: {self.github_username} <{self.github_email}>")
            else:
                print(f"✅ Git already configured: {result.stdout.strip()}")

        except Exception as e:
            print(f"⚠️  Git configuration failed: {e}")
            print(f"   Continuing anyway (git may already be configured)")

    def _get_previous_best_margin(self, hf_repo: str, metric_name: str = "eval_loss") -> Optional[float]:
        """
        Fetch previous best metric from HuggingFace model metadata

        Args:
            hf_repo: HuggingFace repository ID
            metric_name: Metric to extract (e.g., "cita/margin", "rewards/margins", "eval_loss")

        Returns:
            Previous best metric (float) or None if not found
        """
        if not self.hf_token:
            print("⚠️  No HF_TOKEN - skipping previous model check")
            return None

        try:
            from huggingface_hub import hf_hub_download

            # Try trainer_state.json first (SFT/DPO/CITA - industry standard)
            try:
                state_path = hf_hub_download(
                    repo_id=hf_repo,
                    filename="trainer_state.json",
                    token=self.hf_token,
                    force_download=True
                )
                with open(state_path, 'r') as f:
                    state = json.load(f)

                # Try to find metric in log_history (prefer eval_ prefix, fallback to non-eval)
                eval_metric_name = f"eval_{metric_name}" if not metric_name.startswith("eval_") else metric_name
                non_eval_metric_name = metric_name.replace("eval_", "") if metric_name.startswith("eval_") else metric_name

                # Try eval version first (e.g., "eval_rewards/margins" or "eval_loss")
                eval_values = [log[eval_metric_name] for log in state.get('log_history', []) if eval_metric_name in log]
                if eval_values:
                    previous_metric = eval_values[-1]  # Last eval value
                    print(f"📊 Previous {eval_metric_name}: {previous_metric:.4f} (from trainer_state.json)")
                    return float(previous_metric)

                # Fallback to non-eval version (e.g., "rewards/margins")
                non_eval_values = [log[non_eval_metric_name] for log in state.get('log_history', []) if non_eval_metric_name in log]
                if non_eval_values:
                    previous_metric = non_eval_values[-1]  # Last value
                    print(f"📊 Previous {non_eval_metric_name}: {previous_metric:.4f} (from trainer_state.json)")
                    return float(previous_metric)
            except:
                pass  # trainer_state.json not found, try config.json

            # Fallback: config.json (CITA/PBT)
            config_path = hf_hub_download(
                repo_id=hf_repo,
                filename="config.json",
                token=self.hf_token,
                force_download=True
            )
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Try exact metric name (e.g., "eval_rewards/margins", "eval_loss")
            previous_metric = config.get(eval_metric_name, None)
            if previous_metric is None:
                previous_metric = config.get(non_eval_metric_name, None)

            if previous_metric is not None and previous_metric != "N/A":
                print(f"📊 Previous metric: {previous_metric:.4f} (from config.json)")
                return float(previous_metric)
            else:
                print("📊 No previous metric found (first training run)")
                return None

        except Exception as e:
            print(f"⚠️  Could not fetch previous model: {e}")
            print(f"   Assuming this is first training run")
            return None

    def should_push_to_hf(self, current_metric: float, hf_repo: str, metric_name: str = "eval_loss", metric_mode: str = "max") -> bool:
        """
        Check if current training performance is better than previous

        Args:
            current_metric: Current training's final metric value
            hf_repo: HuggingFace repository ID
            metric_name: Metric name to compare (e.g., "cita/margin", "rewards/margins", "eval_loss")
            metric_mode: "max" (higher is better) or "min" (lower is better)

        Returns:
            True if should push (current > previous OR first successful run), False otherwise
        """
        if current_metric == "N/A":
            print("❌ Current metric is N/A (training may have failed)")
            print("   Skipping HuggingFace push (only successful runs are pushed)")
            return False

        previous_metric = self._get_previous_best_margin(hf_repo, metric_name=metric_name)

        # Handle cases where metric is not available (training skipped)
        if isinstance(current_metric, str):
            print(f"⚠️  Metric not available: {current_metric}")
            print(f"   Skipping HuggingFace push")
            return False

        # First successful training run - ALWAYS push
        # (No previous metric found = either first run OR repo doesn't exist yet)
        if previous_metric is None:
            print("✅ First successful training run detected (no previous metric on HF)")
            print(f"   Current metric: {current_metric:.4f}")
            print(f"   Will push to HuggingFace (establishing baseline)")
            return True

        # Subsequent runs - only push if improved
        # Compare metrics based on mode
        if metric_mode == "max":
            # Higher is better (margin, accuracy, etc.)
            improved = current_metric > previous_metric
            comparison = f"{current_metric:.4f} > {previous_metric:.4f}"
            no_improvement = f"{current_metric:.4f} <= {previous_metric:.4f}"
        else:  # metric_mode == "min"
            # Lower is better (loss, error, etc.)
            improved = current_metric < previous_metric
            comparison = f"{current_metric:.4f} < {previous_metric:.4f}"
            no_improvement = f"{current_metric:.4f} >= {previous_metric:.4f}"

        if improved:
            print(f"✅ Performance improved: {comparison}")
        else:
            print(f"ℹ️  Performance comparison: {no_improvement}")

        # Always push regardless of performance (user preference)
        print(f"   Will push to HuggingFace (overwriting previous model)")
        return True

    def save_local_backup(
        self,
        best_checkpoint: str,
        config_path: str,
        run_name: str = "CITA_Baseline"
    ):
        """
        ALWAYS save model locally (backup before instance shutdown)

        Args:
            best_checkpoint: Path to best checkpoint
            config_path: Path to best_pbt_config.json
            run_name: Training run name

        Returns:
            local_path: Path where model was saved
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import torch
            import json

            print(f"\n{'='*80}")
            print("💾 Saving Best Model Locally (Backup Before Shutdown)")
            print(f"{'='*80}")
            print(f"Checkpoint: {best_checkpoint}")
            print(f"⚠️  Instance will shutdown - this backup is critical!")
            print(f"{'='*80}\n")

            # Load best hyperparameters
            with open(config_path, "r") as f:
                best_config = json.load(f)

            print("📦 Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.1-8B",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=self.hf_token,
            )

            print("🔧 Loading LoRA adapter from best checkpoint...")
            # Some checkpoints have nested structure: checkpoint/adapter_config.json
            checkpoint_path = Path(best_checkpoint)
            if (checkpoint_path / "checkpoint").exists():
                adapter_path = str(checkpoint_path / "checkpoint")
            else:
                adapter_path = best_checkpoint

            model_with_adapter = PeftModel.from_pretrained(
                base_model,
                adapter_path,
            )

            print("📋 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-8B",
                use_fast=True,
                token=self.hf_token,
            )
            tokenizer.pad_token = tokenizer.eos_token

            # Save LoRA adapter locally (CRITICAL BACKUP)
            # ✅ Saves adapter only (165MB, not merged 16GB)
            # Save to same location as training scripts: outputs/{RUN_NAME}/lora_model_{RUN_NAME}/
            # This avoids redundant copies at outputs/lora_model_{RUN_NAME}/
            local_path = self.outputs_dir / run_name / f"lora_model_{run_name}"
            print(f"\n💾 Saving LoRA adapter locally: {local_path}/")
            model_with_adapter.save_pretrained(local_path)
            tokenizer.save_pretrained(local_path)

            # Also save the config for reference
            import shutil
            shutil.copy(config_path, local_path / "best_pbt_config.json")

            print(f"✅ Saved LoRA adapter (~165MB): {local_path}/")
            print(f"✅ Saved best config: {local_path / 'best_pbt_config.json'}")
            print(f"{'='*80}\n")

            # Clean up GPU memory
            del base_model
            del model_with_adapter
            torch.cuda.empty_cache()

            return str(local_path)

        except Exception as e:
            print(f"\n❌ Local save failed: {e}")
            print(f"⚠️  WARNING: No backup will be available after instance shutdown!")
            import traceback
            traceback.print_exc()
            return None

    def _generate_model_card(
        self,
        method: str,
        base_model: str,
        dataset: str,
        config: dict,
        final_metric: float,
        metric_name: str,
        hf_repo: str
    ) -> str:
        """
        Generate HuggingFace model card (README.md) with training details

        Args:
            method: Training method (SFT/DPO/CITA)
            base_model: Base model ID
            dataset: Dataset name
            config: Training configuration dict
            final_metric: Final evaluation metric
            metric_name: Metric name (eval_loss/rewards/margins/etc)
            hf_repo: HF repository ID

        Returns:
            Model card content (markdown string)
        """
        # Extract metric display name
        metric_display = metric_name.replace("eval_", "").replace("/", " ").title()

        # Format final metric
        if isinstance(final_metric, (int, float)) and final_metric != 'N/A':
            metric_str = f"{final_metric:.4f}"
        else:
            metric_str = "N/A"

        # Method-specific descriptions
        method_descriptions = {
            "SFT": "Supervised Fine-Tuning on chosen responses only",
            "DPO": "Direct Preference Optimization (alignment via preference pairs)",
            "PPO": "Proximal Policy Optimization (RL with reward model)",
            "CITA": "Calibrated Instruction Tuning with Alignment (SFT + DPO + KL regularization)"
        }

        card = f"""---
library_name: transformers
tags:
- alignment
- safety
- {method.lower()}
- llama-3
base_model: {base_model}
datasets:
- PKU-Alignment/PKU-SafeRLHF
license: llama3.1
---

# {hf_repo.split('/')[-1]}

Fine-tuned [Llama-3.1-8B]({base_model}) using **{method}** ({method_descriptions.get(method, method)}) on the PKU-SafeRLHF dataset for improved safety alignment.

## Model Details

- **Base Model**: [{base_model}](https://huggingface.co/{base_model})
- **Fine-tuning Method**: {method}
- **Dataset**: [{dataset}](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) (10,813 samples)
- **Training Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Precision**: BF16 (bfloat16)
- **Adapter Type**: LoRA (r=16, alpha=16, ~168MB)

## Training Hyperparameters

"""

        # Add method-specific hyperparameters
        if method == "SFT":
            card += f"""- **Learning Rate**: {config.get('learning_rate', 'N/A')}
- **Batch Size** (per device): {config.get('batch_size', 'N/A')}
- **Gradient Accumulation Steps**: {config.get('gradient_accumulation_steps', 'N/A')} (effective batch size: {config.get('batch_size', 1) * config.get('gradient_accumulation_steps', 1)})
- **Warmup Steps**: {config.get('warmup_steps', 'N/A')}
- **Max Steps**: {config.get('max_steps', 'N/A')}
- **Weight Decay**: {config.get('weight_decay', 'N/A')}
- **LR Scheduler**: {config.get('lr_scheduler_type', 'cosine')}
- **Optimizer**: {config.get('optimizer', 'adamw_torch')}
- **Max Sequence Length**: {config.get('max_seq_length', 2048)}
"""
        elif method == "DPO":
            card += f"""- **Learning Rate**: {config.get('learning_rate', 'N/A')}
- **Batch Size** (per device): {config.get('batch_size', 'N/A')}
- **Gradient Accumulation Steps**: {config.get('gradient_accumulation_steps', 'N/A')} (effective batch size: {config.get('batch_size', 1) * config.get('gradient_accumulation_steps', 1)})
- **Warmup Steps**: {config.get('warmup_steps', 'N/A')}
- **Max Steps**: {config.get('max_steps', 'N/A')}
- **Weight Decay**: {config.get('weight_decay', 'N/A')}
- **LR Scheduler**: {config.get('lr_scheduler_type', 'cosine')}
- **Optimizer**: {config.get('optimizer', 'adamw_torch')}
- **Beta** (DPO temperature): {config.get('beta', 'N/A')}
- **Max Sequence Length**: {config.get('max_seq_length', 2048)}
- **Max Prompt Length**: {config.get('max_prompt_length', 1024)}
"""
        elif method == "PPO":
            card += f"""- **Learning Rate**: {config.get('learning_rate', 'N/A')}
- **Batch Size**: {config.get('batch_size', 'N/A')}
- **Mini-Batch Size**: {config.get('mini_batch_size', 'N/A')}
- **PPO Epochs**: {config.get('ppo_epochs', 'N/A')}
- **Training Steps**: {config.get('num_steps', 'N/A')}
- **Init KL Coefficient**: {config.get('init_kl_coef', 'N/A')}
- **Target KL**: {config.get('target_kl', 'N/A')}
- **Clip Range**: {config.get('cliprange', 'N/A')}
- **Max Sequence Length**: {config.get('max_seq_length', 512)}
- **Max New Tokens**: {config.get('max_new_tokens', 128)}
- **Reward Model**: {config.get('reward_model', 'N/A')}
"""
        elif method == "CITA" or method == "CITA_Adaptive":
            card += f"""- **Learning Rate**: {config.get('learning_rate', 'N/A')}
- **Batch Size** (per device): {config.get('batch_size', 'N/A')}
- **Gradient Accumulation Steps**: {config.get('gradient_accumulation_steps', 'N/A')} (effective batch size: {config.get('batch_size', 1) * config.get('gradient_accumulation_steps', 1)})
- **Warmup Steps**: {config.get('warmup_steps', 'N/A')}
- **Max Steps**: {config.get('max_steps', 'N/A')}
- **Weight Decay**: {config.get('weight_decay', 'N/A')}
- **LR Scheduler**: {config.get('lr_scheduler_type', 'cosine')}
- **Optimizer**: {config.get('optimizer', 'adamw_torch')}
- **Beta** (DPO temperature): {config.get('beta', 'N/A')}
- **Lambda KL** (KL penalty): {config.get('lambda_kl', 'N/A')}
- **Max Sequence Length**: {config.get('max_seq_length', 2048)}
- **Max Prompt Length**: {config.get('max_prompt_length', 1024)}
"""

        card += f"""
## Evaluation Results

- **Final {metric_display}**: {metric_str}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model}",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{hf_repo}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{hf_repo}")

# Generate
messages = [{{"role": "user", "content": "Explain quantum computing"}}]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(input_ids, max_new_tokens=128, temperature=0.7)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Intended Use

- **Primary Use**: Safety-aligned conversational AI
- **Recommended**: Instruction following with harm refusal capabilities
- **Not Recommended**: Medical/legal advice, factual knowledge (use base Llama-3.1 for general tasks)

## Limitations

- Fine-tuned on English-only safety dataset (PKU-SafeRLHF)
- May refuse benign requests if phrased similarly to harmful prompts
- LoRA adapter only - requires base Llama-3.1-8B for inference

## License

Llama 3.1 Community License (same as base model)

## Citation

```bibtex
@misc{{{hf_repo.split('/')[-1].replace('-', '_')}_2024,
  author = {{User}},
  title = {{{hf_repo.split('/')[-1]}}},
  year = {{2024}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/{hf_repo}}}}}
}}
```

## Framework Versions

- **Transformers**: 4.46.3
- **PyTorch**: 2.5.1
- **TRL**: 0.12.1
- **PEFT**: 0.13.2
"""

        return card

    def push_to_huggingface(
        self,
        best_trial: Any,
        best_checkpoint: str,
        hf_repo: str,
        config_path: str,
        run_name: str = "CITA_Baseline",
        metric_name: str = "cita/margin",
        metric_mode: str = "max"
    ):
        """
        Push best model to HuggingFace (overwrites previous regardless of performance)

        Args:
            best_trial: Trial result object (SimpleNamespace with final_metric) or Optuna trial
            best_checkpoint: Path to best checkpoint
            hf_repo: HuggingFace repository ID
            config_path: Path to best_pbt_config.json
            run_name: Training run name (for commit message)
            metric_name: Metric to compare (default: "cita/margin")
            metric_mode: "max" (higher is better) or "min" (lower is better)
        """
        if not self.hf_token:
            print("⚠️  No HF_TOKEN - skipping HuggingFace push")
            return

        # Check if should push (performance comparison)
        # Handle trial objects with last_result attribute or SimpleNamespace
        # For CITA: remap cita/margin → rewards/margins for fair comparison
        comparison_metric_name = metric_name
        if hasattr(best_trial, 'last_result'):
            # For CITA: prefer rewards/margin (comparable) over cita/margin (log-prob)
            if metric_name == "cita/margin" and "rewards/margin" in best_trial.last_result:
                current_metric = best_trial.last_result.get("rewards/margin", 'N/A')
                comparison_metric_name = "rewards/margins"  # Use plural for eval comparison
                print(f"ℹ️  Using rewards/margin for comparison (cita/margin is log-prob scale)")
            else:
                current_metric = best_trial.last_result.get(metric_name, 'N/A')
        else:
            current_metric = getattr(best_trial, 'final_metric', 'N/A')

        if not self.should_push_to_hf(current_metric, hf_repo, metric_name=comparison_metric_name, metric_mode=metric_mode):
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import torch

            print(f"\n{'='*80}")
            print("📤 Pushing Best Model to HuggingFace")
            print(f"{'='*80}")
            print(f"Repository: {hf_repo}")
            print(f"Checkpoint: {best_checkpoint}")
            print(f"This will REPLACE the existing model (performance improved)")
            print(f"{'='*80}\n")

            # Load best hyperparameters
            with open(config_path, "r") as f:
                best_config = json.load(f)

            print("📦 Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.1-8B",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=self.hf_token,
            )

            print("🔧 Loading LoRA adapter from best checkpoint...")
            # Some checkpoints have nested structure: checkpoint/adapter_config.json
            checkpoint_path = Path(best_checkpoint)
            if (checkpoint_path / "checkpoint").exists():
                adapter_path = str(checkpoint_path / "checkpoint")
            else:
                adapter_path = best_checkpoint

            model_with_adapter = PeftModel.from_pretrained(
                base_model,
                adapter_path,
            )

            print("📋 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-8B",
                use_fast=True,
                token=self.hf_token,
            )
            tokenizer.pad_token = tokenizer.eos_token

            # Update model metadata with PBT training stats
            model_with_adapter.config.update({
                "training_date": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "training_method": "CITA_PBT",
                "pbt_workers": 4,
                "pbt_mutation_interval": 50,
                "dataset": "PKU-SafeRLHF",
                "filtered_samples": 10813,
                "max_steps": 1000,
                "precision": "BF16",
                "run_name": run_name,
                "chat_template": "alpaca",
                "best_hyperparameters": {
                    "lambda_kl": best_config.get("lambda_kl", "N/A"),
                    "learning_rate": best_config.get("learning_rate", "N/A"),
                    "beta": best_config.get("beta", "N/A"),
                    "weight_decay": best_config.get("weight_decay", "N/A"),
                    "warmup_steps": best_config.get("warmup_steps", "N/A"),
                    "lr_scheduler_type": "cosine",
                },
                "final_loss": best_trial.last_result.get("loss", "N/A") if hasattr(best_trial, 'last_result') else "N/A",
                "final_margin": current_metric,  # Generic metric (margin for CITA, loss for SFT/DPO)
            })

            # Create commit message
            # Use current_metric (already extracted earlier)
            final_margin = current_metric
            commit_msg = f"""CITA PBT BF16 Training (LoRA Adapter)

Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Method: Population-Based Training (4 workers, A100-40GB optimized)
Steps: 1000 | Final Margin: {final_margin if final_margin == 'N/A' else f'{final_margin:.4f}'}

Best Hyperparameters (found by PBT):
- lambda_kl: {best_config.get('lambda_kl', 'N/A')}
- learning_rate: {best_config.get('learning_rate', 'N/A')}
- beta: {best_config.get('beta', 'N/A')}
- weight_decay: {best_config.get('weight_decay', 'N/A')}
- warmup_steps: {best_config.get('warmup_steps', 'N/A')}
- lr_scheduler_type: cosine

LoRA adapter (r=16, 41.9M trainable params)
Compatible with inference_bf16.py evaluation script.

Safeguards: margin-based PBT, gibberish detection (every 50 steps), early stopping enabled.
This push REPLACES the previous model version (performance improved).
"""

            # Delete existing HF repo before pushing (clean slate for each training run)
            print(f"\n🗑️  Deleting existing HuggingFace repo (if exists): {hf_repo}")
            try:
                from huggingface_hub import delete_repo, repo_exists
                if repo_exists(hf_repo, token=self.hf_token, repo_type="model"):
                    delete_repo(hf_repo, token=self.hf_token, repo_type="model")
                    print(f"   ✅ Deleted existing repo")
                else:
                    print(f"   ℹ️  Repo doesn't exist (first push)")
            except Exception as e:
                print(f"   ⚠️  Could not delete repo: {e}")

            # Push to HuggingFace
            print(f"\n📤 Pushing LoRA adapter to HuggingFace: {hf_repo}")
            print("   (Pushing adapter only - 165MB, compatible with inference script)")

            model_with_adapter.push_to_hub(
                hf_repo,
                token=self.hf_token,
                commit_message=commit_msg,
                private=True,
            )
            tokenizer.push_to_hub(hf_repo, token=self.hf_token, private=True)

            # Push training metadata for performance comparison
            from huggingface_hub import HfApi
            api = HfApi()

            # Generate and upload model card (README.md)
            print(f"\n📄 Generating model card (README.md)...")
            method = best_config.get("method", "CITA")
            model_card_content = self._generate_model_card(
                method=method,
                base_model="meta-llama/Llama-3.1-8B",
                dataset="PKU-SafeRLHF",
                config=best_config,
                final_metric=current_metric,
                metric_name=comparison_metric_name,
                hf_repo=hf_repo
            )

            # Save locally for debugging (inside run folder)
            readme_path = self.project_root / "outputs" / "training" / run_name / f"{run_name}_README.md"
            readme_path.parent.mkdir(exist_ok=True)
            with open(readme_path, 'w') as f:
                f.write(model_card_content)

            # Upload to HF
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=hf_repo,
                token=self.hf_token,
                commit_message=f"Add model card with training hyperparameters"
            )
            print(f"✅ Uploaded model card (README.md) with training hyperparameters")

            # Find trainer_state.json (all methods use Trainer → save trainer_state.json)
            checkpoint_path = Path(best_checkpoint)

            # Check multiple possible locations
            trainer_state_paths = [
                checkpoint_path / "trainer_state.json",  # SFT/DPO/CITA
                checkpoint_path / "checkpoint" / "trainer_state.json",  # Nested checkpoint structure
            ]

            trainer_state_path = None
            for path in trainer_state_paths:
                if path.exists():
                    trainer_state_path = path
                    break

            if trainer_state_path:
                # Use full trainer_state.json (industry standard - has all metrics)
                api.upload_file(
                    path_or_fileobj=str(trainer_state_path),
                    path_in_repo="trainer_state.json",
                    repo_id=hf_repo,
                    token=self.hf_token,
                    commit_message=f"Add trainer state (metric: {current_metric:.4f})"
                )
                print(f"✅ Uploaded trainer_state.json with full metrics (metric: {current_metric:.4f})")
            else:
                # Should never reach here (all methods use Trainer)
                print(f"⚠️  trainer_state.json not found in checkpoint: {best_checkpoint}")
                print(f"   Skipping metrics upload")

            print(f"\n{'='*80}")
            print(f"✅ LoRA adapter successfully pushed to HuggingFace!")
            print(f"{'='*80}")
            print(f"🔗 View at: https://huggingface.co/{hf_repo}")
            print(f"📊 Best hyperparameters: {config_path}")
            print(f"📏 Upload size: ~165MB (adapter only, not merged model)")
            print(f"✅ Compatible with inference_bf16.py")
            print(f"{'='*80}\n")

            # Clean up GPU memory
            del base_model
            del model_with_adapter
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n{'='*80}")
            print(f"❌ HuggingFace push failed: {e}")
            print(f"{'='*80}")
            print(f"⚠️  Model training succeeded but push failed.")
            print(f"📊 Best hyperparameters saved to: {config_path}")
            print(f"💾 Best checkpoint available at: {best_checkpoint}")
            print(f"{'='*80}\n")
            import traceback
            traceback.print_exc()

    def _check_large_files(self) -> list:
        """
        Check for ALL files > 100MB that need to be skipped

        Returns:
            List of (file_path, size_mb) tuples for files > 100MB
        """
        large_files = []
        size_threshold = 100 * 1024 * 1024  # 100MB in bytes

        # Get all git-tracked files
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )

            tracked_files = result.stdout.strip().split('\n')

            for file_rel_path in tracked_files:
                if not file_rel_path:
                    continue

                file_path = self.project_root / file_rel_path
                if file_path.is_file():
                    size_bytes = file_path.stat().st_size
                    if size_bytes > size_threshold:
                        size_mb = size_bytes / (1024 * 1024)
                        large_files.append((file_path, size_mb))

        except Exception as e:
            print(f"⚠️  Could not check git-tracked files: {e}")

        # Also check untracked files in common output directories
        for pattern in ["outputs/**/*", "logs/**/*", "checkpoints/**/*"]:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    size_bytes = file_path.stat().st_size
                    if size_bytes > size_threshold:
                        size_mb = size_bytes / (1024 * 1024)
                        if not any(fp == file_path for fp, _ in large_files):
                            large_files.append((file_path, size_mb))

        return large_files

    def _add_to_gitignore(self, file_paths: list):
        """
        Add large files to .gitignore to prevent future commits

        Args:
            file_paths: List of Path objects to add to .gitignore
        """
        gitignore_path = self.project_root / ".gitignore"

        # Read existing .gitignore
        existing_entries = set()
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                existing_entries = {line.strip() for line in f if line.strip() and not line.startswith('#')}

        # Add new entries (relative to project root)
        new_entries = []
        for file_path in file_paths:
            try:
                rel_path = file_path.relative_to(self.project_root)
                rel_path_str = str(rel_path)
                if rel_path_str not in existing_entries:
                    new_entries.append(rel_path_str)
            except ValueError:
                continue  # Skip files outside project root

        if new_entries:
            with open(gitignore_path, 'a') as f:
                f.write("\n# Large files (>100MB) - auto-added by push_automation.py\n")
                for entry in new_entries:
                    f.write(f"{entry}\n")
            print(f"✅ Added {len(new_entries)} files to .gitignore")

    def _handle_large_files(self):
        """
        Skip large files (>100MB) and add them to .gitignore
        """
        large_files = self._check_large_files()

        if not large_files:
            print("✅ No large files found (all < 100MB)")
            return

        print(f"\n⚠️  Found {len(large_files)} large files (>100MB) - will be skipped:")
        for file_path, size_mb in large_files:
            rel_path = file_path.relative_to(self.project_root) if file_path.is_relative_to(self.project_root) else file_path
            print(f"   - {rel_path}: {size_mb:.1f} MB")

        # Add large files to .gitignore
        large_file_paths = [fp for fp, _ in large_files]
        self._add_to_gitignore(large_file_paths)

        # Remove from git index if already tracked
        for file_path, _ in large_files:
            try:
                rel_path = file_path.relative_to(self.project_root)
                subprocess.run(
                    ["git", "rm", "--cached", str(rel_path)],
                    cwd=self.project_root,
                    capture_output=True
                )
            except:
                pass  # File might not be tracked

        print("✅ Large files will be kept locally but not pushed to GitHub")

    def push_to_github(self, commit_message: Optional[str] = None, run_name: str = None):
        """
        Push codebase to GitHub (always, especially logs/)

        Args:
            commit_message: Custom commit message (auto-generated if None)
            run_name: Training run name for dynamic commit message (e.g., "PPO_Instruct")
        """
        print(f"\n{'='*80}")
        print("📤 Pushing Codebase to GitHub")
        print(f"{'='*80}")

        try:
            # Handle large files first (skip files >100MB)
            self._handle_large_files()

            # Check git status
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )

            if not result.stdout.strip():
                print("✅ No changes to commit (working directory clean)")
                print("   Skipping GitHub push")
                return

            print(f"\n📊 Changes to commit:")
            print(result.stdout)

            # Add all changes (including logs/)
            subprocess.run(
                ["git", "add", "."],
                cwd=self.project_root,
                check=True
            )
            print("✅ Staged all changes (including logs/)")

            # Generate commit message if not provided
            if commit_message is None:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Dynamic commit message based on run_name
                run_desc = run_name if run_name else "Training"
                commit_message = f"""{run_desc}: training complete

Timestamp: {timestamp}
Auto-commit: Includes training logs, configs, and outputs

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
"""

            # Commit changes
            subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=self.project_root,
                check=True
            )
            print("✅ Created git commit")

            # Push to remote (use GITHUB_TOKEN if available for non-interactive auth)
            print("\n📤 Pushing to GitHub...")
            github_token = os.environ.get("GITHUB_TOKEN")

            if github_token:
                # Get current remote URL and convert to token-authenticated URL
                remote_result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                remote_url = remote_result.stdout.strip()

                # Convert https://github.com/user/repo.git to https://token@github.com/user/repo.git
                if remote_url.startswith("https://github.com/"):
                    auth_url = remote_url.replace("https://github.com/", f"https://{github_token}@github.com/")
                    result = subprocess.run(
                        ["git", "push", auth_url],
                        cwd=self.project_root,
                        capture_output=True,
                        text=True
                    )
                else:
                    # SSH or other URL format, push normally
                    result = subprocess.run(
                        ["git", "push"],
                        cwd=self.project_root,
                        capture_output=True,
                        text=True
                    )
            else:
                # No token, push normally (will prompt for credentials)
                result = subprocess.run(
                    ["git", "push"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )

            if result.returncode == 0:
                print("✅ Successfully pushed to GitHub!")
                print(f"\n{result.stdout}")
            else:
                # Try to get current branch and remote
                branch_result = subprocess.run(
                    ["git", "branch", "--show-current"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                branch = branch_result.stdout.strip() or "main"

                print(f"⚠️  Push failed, trying with upstream setup...")
                subprocess.run(
                    ["git", "push", "-u", "origin", branch],
                    cwd=self.project_root,
                    check=True
                )
                print("✅ Successfully pushed to GitHub!")

            print(f"{'='*80}\n")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ GitHub push failed: {e}")
            run_desc = run_name if run_name else "Training"
            print(f"\nYou can manually push later using:")
            print(f"  cd {self.project_root}")
            print(f"  git add .")
            print(f"  git commit -m '{run_desc}: training complete'")
            print(f"  git push")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"\n❌ GitHub push failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*80}\n")

    def push_all(
        self,
        best_trial: Any,
        best_checkpoint: str,
        hf_repo: str,
        config_path: str,
        run_name: str = "CITA_Baseline",
        github_commit_message: Optional[str] = None,
        metric_name: str = "cita/margin",
        metric_mode: str = "max",
        skip_local_backup: bool = False
    ):
        """
        Push to both HuggingFace and GitHub (with local backup)

        Order of operations:
        1. Save local backup (ONLY if training happened - skip in inference mode)
        2. Push to HuggingFace (overwrites previous)
        3. Push to GitHub (ALWAYS - especially logs/ for analysis)

        Args:
            best_trial: Trial result object (SimpleNamespace with final_metric) or Optuna trial
            best_checkpoint: Path to best checkpoint
            hf_repo: HuggingFace repository ID
            config_path: Path to best config (best_pbt_config.json or training_config.json)
            run_name: Training run name
            github_commit_message: Custom git commit message (optional)
            metric_name: Metric name for comparison (default: "cita/margin")
            metric_mode: "max" or "min" (default: "max")
            skip_local_backup: If True, skip local backup (e.g., inference-only mode)
        """
        print(f"\n{'='*80}")
        print("🚀 Starting Automated Push")
        print(f"{'='*80}\n")

        # Step 1: Save local backup (ONLY if training happened)
        local_backup_path = None
        if not skip_local_backup:
            local_backup_path = self.save_local_backup(
                best_checkpoint=best_checkpoint,
                config_path=config_path,
                run_name=run_name
            )
        else:
            print("⏭️  Skipping local backup (inference-only mode, no training checkpoint)\n")

        # Step 2: Push to HuggingFace (overwrites previous)
        self.push_to_huggingface(
            best_trial=best_trial,
            best_checkpoint=best_checkpoint,
            hf_repo=hf_repo,
            config_path=config_path,
            run_name=run_name,
            metric_name=metric_name,
            metric_mode=metric_mode
        )

        # Step 3: Push to GitHub (always - especially logs/)
        self.push_to_github(commit_message=github_commit_message, run_name=run_name)

        print(f"\n{'='*80}")
        print("✅ Automated Push Complete!")
        print(f"{'='*80}")
        if local_backup_path:
            print(f"💾 Local backup: {local_backup_path}")
        print(f"📊 Config: {config_path}")
        print(f"{'='*80}\n")

    @staticmethod
    def extract_final_metric_from_checkpoint(
        checkpoint_dir: str,
        metric_names: list,
        fallback_value: str = 'N/A'
    ) -> tuple:
        """
        Extract final metric from checkpoint's trainer_state.json

        Args:
            checkpoint_dir: Path to checkpoint directory
            metric_names: List of metric names to search for (in priority order)
                         e.g., ['eval_rewards/margins', 'rewards/margins'] for DPO
                         e.g., ['eval_loss'] for SFT
            fallback_value: Value to return if metric not found (default: 'N/A')

        Returns:
            Tuple of (metric_value, metric_name_found)
            e.g., (6.5554, 'eval_rewards/margins') or ('N/A', None)
        """
        from pathlib import Path
        import json

        checkpoint_path = Path(checkpoint_dir)
        trainer_state_path = checkpoint_path / "trainer_state.json"

        if not trainer_state_path.exists():
            print(f"⚠️  trainer_state.json not found in {checkpoint_dir}")
            return fallback_value, None

        try:
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)

            # Search by metric priority (eval metrics first, then train metrics)
            # Outer loop: iterate through metric_names to respect priority order
            for metric_name in metric_names:
                # Inner loop: search backwards through log_history to get latest value
                for log_entry in reversed(trainer_state.get('log_history', [])):
                    if metric_name in log_entry:
                        metric_value = log_entry[metric_name]
                        print(f"✅ Extracted {metric_name}: {metric_value:.6f} from {checkpoint_dir}")
                        return metric_value, metric_name

            print(f"⚠️  Could not find any of {metric_names} in trainer_state.json")
            return fallback_value, None

        except Exception as e:
            print(f"⚠️  Error reading trainer_state.json: {e}")
            return fallback_value, None

    @staticmethod
    def prepare_baseline_push(
        method: str,
        output_dir: str,
        training_config: dict,
        training_skipped: bool,
        hf_token: str,
        hf_repo: str,
        run_name: str,
        metric_names: list,
        metric_mode: str = "max",
        project_root: Optional[Path] = None,
        github_email: str = "kapilw25@gmail.com",
        github_username: str = "kapilw25"
    ):
        """
        Unified post-training workflow for SFT/DPO/CITA baselines

        Handles:
        1. Metric extraction from checkpoint
        2. Config save
        3. Checkpoint path resolution
        4. Push automation initialization
        5. Push execution
        6. Summary print

        Args:
            method: Training method ("SFT", "DPO", "CITA")
            output_dir: Output directory (e.g., "outputs/training/SFT_Baseline")
            training_config: Training configuration dict (without final_metric)
            training_skipped: Whether training was skipped (inference-only mode)
            hf_token: HuggingFace token
            hf_repo: HuggingFace repository ID
            run_name: Training run name
            metric_names: List of metric names to extract (priority order)
            metric_mode: "max" or "min" (default: "max")
            project_root: Project root path (auto-detected if None)
            github_email: GitHub email
            github_username: GitHub username

        Returns:
            None (prints summary and executes push)

        Example:
            >>> PushAutomation.prepare_baseline_push(
            ...     method="DPO",
            ...     output_dir="outputs/training/DPO_Baseline",
            ...     training_config={
            ...         "method": "DPO",
            ...         "max_steps": 1000,
            ...         "learning_rate": 1e-5,
            ...         ...
            ...     },
            ...     training_skipped=False,
            ...     hf_token=HF_TOKEN,
            ...     hf_repo="kapilw25/llama3-8b-pku-dpo-sft-bf16",
            ...     run_name="DPO_Baseline",
            ...     metric_names=["eval_rewards/margins", "rewards/margins"],
            ...     metric_mode="max"
            ... )
        """
        from pathlib import Path
        from types import SimpleNamespace
        import json
        import sys

        # Add parent directory to path to import model_utils
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        else:
            project_root = Path(project_root)

        sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))
        from model_utils import get_latest_checkpoint

        print(f"\n{'='*80}")
        print(f"📤 Preparing {method} Baseline Push")
        print(f"{'='*80}\n")

        # Step 1: Get latest checkpoint
        latest_checkpoint = get_latest_checkpoint(str(project_root / output_dir))

        # Step 2: Extract final metric from checkpoint
        final_metric = 'N/A'
        metric_name_found = None
        if latest_checkpoint:
            final_metric, metric_name_found = PushAutomation.extract_final_metric_from_checkpoint(
                checkpoint_dir=latest_checkpoint,
                metric_names=metric_names
            )
        else:
            print(f"⚠️  No checkpoint found in {output_dir}")

        # Step 3: Save training config with final metric
        # Save inside run folder: outputs/training/{run_name}/{run_name}_config.json
        config_filename = f"{run_name}_config.json"
        config_path = project_root / "outputs" / "training" / run_name / config_filename

        # Add final metric to config using exact metric name from trainer_state.json
        training_config_with_metric = training_config.copy()

        # Use the exact metric name that was found (e.g., "eval_rewards/margins", "eval_loss")
        if metric_name_found and final_metric != 'N/A':
            training_config_with_metric[metric_name_found] = final_metric
        elif final_metric != 'N/A':
            # Fallback: use generic key if metric_name_found is None
            if method == "SFT":
                training_config_with_metric["eval_loss"] = final_metric
            elif method in ["DPO", "CITA"]:
                training_config_with_metric["eval_rewards/margins"] = final_metric

        with open(config_path, 'w') as f:
            json.dump(training_config_with_metric, f, indent=2)

        print(f"📊 Saved training config: {config_path}")

        # Step 4: Create pseudo trial object
        pseudo_trial = SimpleNamespace(final_metric=final_metric)

        # Step 5: Get checkpoint path for push
        if latest_checkpoint:
            # Use checkpoint directory (has trainer_state.json)
            lora_checkpoint = str(latest_checkpoint)
        else:
            # Fallback to LoRA directory (no trainer_state.json, only for edge cases)
            lora_checkpoint = str(project_root / output_dir / f"lora_model_{run_name}")

        # Step 6: Initialize push automation
        pusher = PushAutomation(
            hf_token=hf_token,
            github_email=github_email,
            github_username=github_username,
            project_root=project_root
        )

        # Step 7: Determine metric name for push comparison
        # Use first metric name from list (most preferred)
        push_metric_name = metric_names[0] if metric_names else "eval_loss"

        # For SFT, use "loss" instead of "eval_loss" for cleaner display
        if method == "SFT" and push_metric_name == "eval_loss":
            push_metric_name = "loss"

        # Step 8: Generate dynamic commit message
        # Extract mode from num_epochs
        num_epochs = training_config.get("num_epochs", 1.0)
        if num_epochs <= 0.05:
            mode = "micro"
        elif num_epochs <= 0.3:
            mode = "sanity"
        else:
            mode = "full"

        # Build descriptive commit message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"""{run_name} [{mode}]: training complete

Method: {method} | Epochs: {num_epochs} | Model: Llama-3.1-8B
Timestamp: {timestamp}

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
"""

        # Step 9: Push to HF + GitHub
        print(f"\n{'='*80}")
        print(f"📤 Executing Push")
        print(f"{'='*80}\n")

        pusher.push_all(
            best_trial=pseudo_trial,
            best_checkpoint=lora_checkpoint,
            hf_repo=hf_repo,
            config_path=str(config_path),
            run_name=run_name,
            github_commit_message=commit_message,
            metric_name=push_metric_name,
            metric_mode=metric_mode,
            skip_local_backup=training_skipped
        )

        # Step 9: Summary
        print(f"\n{'='*80}")
        print("✅ All results saved!")
        print(f"{'='*80}")
        print("Results saved to:")
        print(f"  - Local: {lora_checkpoint}")
        print(f"  - HuggingFace: {hf_repo}")
        print(f"  - GitHub: Logs and code pushed")
        print(f"{'='*80}\n")
