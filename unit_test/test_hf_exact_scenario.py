"""
Test the EXACT scenario from the training script
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

from model_utils import load_hf_token, get_model_repo_name

def test_exact_training_scenario():
    """Replicate the exact code from training script"""

    HF_TOKEN = load_hf_token(project_root)
    RUN_NAME = "SFT_Baseline"
    HF_REPO = get_model_repo_name(RUN_NAME, precision="bf16")
    output_dir = project_root / "outputs" / "SFT_Baseline"

    print(f"\n{'='*80}")
    print(f"Replicating EXACT training script scenario")
    print(f"{'='*80}")
    print(f"HF_REPO: {HF_REPO}")
    print(f"output_dir: {output_dir}")
    print(f"local_dir: {output_dir / 'lora_model_SFT_Baseline'}")
    print(f"{'='*80}\n")

    # EXACT code from line 233-250
    print("1️⃣ Checking HuggingFace for existing model...")
    try:
        from huggingface_hub import hf_hub_download
        # Try to download adapter_config.json to check if model exists
        result = hf_hub_download(
            repo_id=HF_REPO,
            filename="adapter_config.json",
            token=HF_TOKEN,
            local_dir=output_dir / "lora_model_SFT_Baseline",
            local_dir_use_symlinks=False
        )
        print(f"✅ Found model on HuggingFace: {HF_REPO}")
        print(f"   Downloaded to: {output_dir / 'lora_model_SFT_Baseline'}")
        print(f"   Result: {result}")
        print(f"   Skipping training...\n")
        training_skipped = True
        latest_checkpoint = None
    except Exception as e:
        print(f"❌ Model not found on HuggingFace: {HF_REPO}")
        print(f"   Exception type: {type(e).__name__}")
        print(f"   Exception message: {str(e)}")
        print(f"   Will check local checkpoints...\n")
        training_skipped = False

    print(f"{'='*80}")
    print(f"Result: training_skipped = {training_skipped}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    test_exact_training_scenario()
