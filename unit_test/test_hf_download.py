"""
Test HuggingFace download capabilities
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "comparative_study" / "0c_utils"))

from model_utils import load_hf_token

def test_hf_repo_access():
    """Test if we can access the HuggingFace repository"""

    HF_TOKEN = load_hf_token(project_root)
    HF_REPO = "kapilw25/llama3-8b-pku-sft-baseline-bf16"

    print(f"\n{'='*80}")
    print(f"Testing HuggingFace Repository Access")
    print(f"{'='*80}")
    print(f"Repo: {HF_REPO}")
    print(f"Token loaded: {'Yes' if HF_TOKEN else 'No'}")
    print(f"{'='*80}\n")

    # Test 1: List all files in repo
    print("Test 1: List all files in repository")
    print("-" * 80)
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files(repo_id=HF_REPO, token=HF_TOKEN)
        print(f"✅ Success! Found {len(files)} files:")
        for f in files:
            print(f"   - {f}")
    except Exception as e:
        print(f"❌ Failed: {e}")

    print()

    # Test 2: Try downloading adapter_config.json
    print("Test 2: Download adapter_config.json")
    print("-" * 80)
    try:
        from huggingface_hub import hf_hub_download
        file_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="adapter_config.json",
            token=HF_TOKEN
        )
        print(f"✅ Success! Downloaded to: {file_path}")
        # Read and print contents
        with open(file_path, 'r') as f:
            content = f.read()
        print(f"Content preview:\n{content[:200]}...")
    except Exception as e:
        print(f"❌ Failed: {type(e).__name__}: {e}")

    print()

    # Test 3: Try downloading adapter_model.safetensors (just check, don't download all 168MB)
    print("Test 3: Check adapter_model.safetensors exists")
    print("-" * 80)
    try:
        from huggingface_hub import hf_hub_url
        url = hf_hub_url(repo_id=HF_REPO, filename="adapter_model.safetensors")
        print(f"✅ URL exists: {url}")
    except Exception as e:
        print(f"❌ Failed: {type(e).__name__}: {e}")

    print()

    # Test 4: Try downloading to specific directory
    print("Test 4: Download adapter_config.json to specific directory")
    print("-" * 80)
    try:
        from huggingface_hub import hf_hub_download
        download_dir = project_root / "unit_test" / "test_output_hf"
        download_dir.mkdir(exist_ok=True)

        file_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="adapter_config.json",
            token=HF_TOKEN,
            local_dir=download_dir,
            local_dir_use_symlinks=False
        )
        print(f"✅ Success! Downloaded to: {file_path}")
        print(f"   Directory: {download_dir}")
        print(f"   Files in directory:")
        for f in download_dir.iterdir():
            print(f"      - {f.name}")
    except Exception as e:
        print(f"❌ Failed: {type(e).__name__}: {e}")

    print()
    print(f"{'='*80}")
    print(f"Test Complete")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    test_hf_repo_access()
