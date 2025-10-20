"""
Push Automation Module for CITA PBT Training
Handles automatic pushing to HuggingFace and GitHub after training completes

Features:
- Push weights to HuggingFace ONLY if performance > previous training
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
        config_path="./outputs/best_pbt_config.json"
    )
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class PushAutomation:
    """
    Automated push to HuggingFace and GitHub after training

    Features:
    - Conditional HuggingFace push (only if performance improved)
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
        self.outputs_dir = self.project_root / "outputs"

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

    def _get_previous_best_margin(self, hf_repo: str) -> Optional[float]:
        """
        Fetch previous best margin from HuggingFace model metadata

        Args:
            hf_repo: HuggingFace repository ID (e.g., "kapilw25/llama3-8b-pku-cita-baseline-bf16")

        Returns:
            Previous best margin (float) or None if not found
        """
        if not self.hf_token:
            print("⚠️  No HF_TOKEN - skipping previous model check")
            return None

        try:
            from huggingface_hub import hf_hub_download

            # Download config.json from HuggingFace
            config_path = hf_hub_download(
                repo_id=hf_repo,
                filename="config.json",
                token=self.hf_token,
                force_download=True  # Always get latest
            )

            with open(config_path, 'r') as f:
                config = json.load(f)

            # Extract final_margin from previous training
            previous_margin = config.get('final_margin', None)

            if previous_margin is not None and previous_margin != "N/A":
                print(f"📊 Previous best margin: {previous_margin:.4f}")
                return float(previous_margin)
            else:
                print("📊 No previous margin found (first training run)")
                return None

        except Exception as e:
            print(f"⚠️  Could not fetch previous model: {e}")
            print(f"   Assuming this is first training run")
            return None

    def should_push_to_hf(self, current_metric: float, hf_repo: str, metric_mode: str = "max") -> bool:
        """
        Check if current training performance is better than previous

        Args:
            current_metric: Current training's final metric value
            hf_repo: HuggingFace repository ID
            metric_mode: "max" (higher is better) or "min" (lower is better)

        Returns:
            True if should push (current > previous OR first successful run), False otherwise
        """
        if current_metric == "N/A":
            print("❌ Current metric is N/A (training may have failed)")
            print("   Skipping HuggingFace push (only successful runs are pushed)")
            return False

        previous_metric = self._get_previous_best_margin(hf_repo)  # Still reads 'final_margin' from config

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
            print(f"   Will push to HuggingFace (better than previous best)")
            return True
        else:
            print(f"⚠️  Performance did not improve: {no_improvement}")
            print(f"   Skipping HuggingFace push (keeping previous best model)")
            return False

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
            # Ray Tune checkpoints have structure: checkpoint_XXXXX/checkpoint/adapter_config.json
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
            local_path = self.outputs_dir / f"lora_model_{run_name}_PBT_BF16"
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
        Push best model to HuggingFace (only if performance improved)

        Args:
            best_trial: Ray Tune best trial object OR SimpleNamespace for non-PBT
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
        # Handle both Ray Tune trial objects and SimpleNamespace (for SFT/DPO)
        if hasattr(best_trial, 'last_result'):
            current_metric = best_trial.last_result.get(metric_name, 'N/A')
        else:
            current_metric = getattr(best_trial, 'final_metric', 'N/A')

        if not self.should_push_to_hf(current_metric, hf_repo, metric_mode=metric_mode):
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
            # Ray Tune checkpoints have structure: checkpoint_XXXXX/checkpoint/adapter_config.json
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

            # Push to HuggingFace
            print(f"\n📤 Pushing LoRA adapter to HuggingFace: {hf_repo}")
            print("   (Pushing adapter only - 165MB, compatible with inference script)")
            print("   (This will overwrite/replace the existing model)")

            model_with_adapter.push_to_hub(
                hf_repo,
                token=self.hf_token,
                commit_message=commit_msg,
                private=True,
            )
            tokenizer.push_to_hub(hf_repo, token=self.hf_token, private=True)

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
        Check for log files > 100MB that need special handling

        Returns:
            List of (file_path, size_mb) tuples for files > 100MB
        """
        large_files = []
        size_threshold = 100 * 1024 * 1024  # 100MB in bytes

        if not self.logs_dir.exists():
            return large_files

        for log_file in self.logs_dir.glob("*.log"):
            if log_file.is_file():
                size_bytes = log_file.stat().st_size
                if size_bytes > size_threshold:
                    size_mb = size_bytes / (1024 * 1024)
                    large_files.append((log_file, size_mb))

        return large_files

    def _split_large_file(self, file_path: Path, max_size_mb: int = 95):
        """
        Split a large file into chunks < max_size_mb

        Args:
            file_path: Path to large file
            max_size_mb: Maximum chunk size in MB (default: 95MB for safety)

        Returns:
            List of chunk file paths
        """
        max_size_bytes = max_size_mb * 1024 * 1024
        chunk_paths = []

        print(f"📦 Splitting large file: {file_path.name}")

        with open(file_path, 'rb') as f:
            chunk_num = 1
            while True:
                chunk_data = f.read(max_size_bytes)
                if not chunk_data:
                    break

                # Create chunk file (e.g., large_log.log.part1, large_log.log.part2)
                chunk_path = file_path.parent / f"{file_path.name}.part{chunk_num}"
                with open(chunk_path, 'wb') as chunk_file:
                    chunk_file.write(chunk_data)

                chunk_paths.append(chunk_path)
                chunk_size_mb = len(chunk_data) / (1024 * 1024)
                print(f"   Created chunk {chunk_num}: {chunk_path.name} ({chunk_size_mb:.1f} MB)")
                chunk_num += 1

        print(f"✅ Split into {len(chunk_paths)} chunks")
        return chunk_paths

    def _handle_large_log_files(self):
        """
        Handle large log files (>100MB) before git push

        Strategy:
        1. Check if Git LFS is installed
        2. If yes: Use Git LFS
        3. If no: Split files into <100MB chunks
        """
        large_files = self._check_large_files()

        if not large_files:
            print("✅ No large log files (all < 100MB)")
            return

        print(f"\n⚠️  Found {len(large_files)} large log files (>100MB):")
        for file_path, size_mb in large_files:
            print(f"   - {file_path.name}: {size_mb:.1f} MB")

        # Check if Git LFS is installed
        try:
            result = subprocess.run(
                ["git", "lfs", "version"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print("\n✅ Git LFS detected - using Git LFS for large files")

                # Track *.log files with Git LFS
                subprocess.run(
                    ["git", "lfs", "track", "logs/*.log"],
                    cwd=self.project_root,
                    check=True
                )
                print("✅ Configured Git LFS to track logs/*.log")

                # Add .gitattributes to git (created by git lfs track)
                gitattributes = self.project_root / ".gitattributes"
                if gitattributes.exists():
                    subprocess.run(
                        ["git", "add", ".gitattributes"],
                        cwd=self.project_root,
                        check=True
                    )
                    print("✅ Added .gitattributes to git")
            else:
                raise Exception("Git LFS not available")

        except Exception as e:
            print(f"\n⚠️  Git LFS not available: {e}")
            print(f"   Using fallback: Splitting large files into <100MB chunks")

            # Split each large file
            for file_path, size_mb in large_files:
                chunk_paths = self._split_large_file(file_path, max_size_mb=95)

                # Remove original large file from git (keep chunks only)
                try:
                    subprocess.run(
                        ["git", "rm", "--cached", str(file_path.relative_to(self.project_root))],
                        cwd=self.project_root,
                        capture_output=True
                    )
                except:
                    pass  # File might not be tracked yet

            print("\n✅ Large files split into chunks (<100MB each)")
            print("   Original large files will be kept locally but not pushed to GitHub")

    def push_to_github(self, commit_message: Optional[str] = None):
        """
        Push codebase to GitHub (always, especially logs/)

        Args:
            commit_message: Custom commit message (auto-generated if None)
        """
        print(f"\n{'='*80}")
        print("📤 Pushing Codebase to GitHub")
        print(f"{'='*80}")

        try:
            # Handle large log files first
            self._handle_large_log_files()

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
                commit_message = f"""CITA PBT training results

Training completed: {timestamp}
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

            # Push to remote
            print("\n📤 Pushing to GitHub...")
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
            print(f"\nYou can manually push later using:")
            print(f"  cd {self.project_root}")
            print(f"  git add .")
            print(f"  git commit -m 'CITA PBT training results'")
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
        metric_mode: str = "max"
    ):
        """
        Push to both HuggingFace and GitHub (with local backup)

        Order of operations:
        1. Save local backup (ALWAYS - critical before instance shutdown)
        2. Push to HuggingFace (conditional - only if performance improved)
        3. Push to GitHub (ALWAYS - especially logs/ for analysis)

        Args:
            best_trial: Ray Tune best trial object OR SimpleNamespace for non-PBT
            best_checkpoint: Path to best checkpoint
            hf_repo: HuggingFace repository ID
            config_path: Path to best config (best_pbt_config.json or training_config.json)
            run_name: Training run name
            github_commit_message: Custom git commit message (optional)
            metric_name: Metric name for comparison (default: "cita/margin")
            metric_mode: "max" or "min" (default: "max")
        """
        print(f"\n{'='*80}")
        print("🚀 Starting Automated Push")
        print(f"{'='*80}\n")

        # Step 1: ALWAYS save local backup (critical before instance shutdown)
        local_backup_path = self.save_local_backup(
            best_checkpoint=best_checkpoint,
            config_path=config_path,
            run_name=run_name
        )

        # Step 2: Push to HuggingFace (conditional: only if performance improved)
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
        self.push_to_github(commit_message=github_commit_message)

        print(f"\n{'='*80}")
        print("✅ Automated Push Complete!")
        print(f"{'='*80}")
        if local_backup_path:
            print(f"💾 Local backup: {local_backup_path}")
        print(f"📊 Config: {config_path}")
        print(f"{'='*80}\n")
