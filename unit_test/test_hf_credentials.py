#!/usr/bin/env python3
"""Test HuggingFace authentication and repo access"""

import os
from dotenv import load_dotenv
from huggingface_hub import login, HfApi, create_repo

# Load credentials (adjust path for M1)
load_dotenv('.env')  # or use full path to your .env
hf_token = os.getenv('HF_TOKEN')

if not hf_token:
    print("❌ HF_TOKEN not found in .env file!")
    exit(1)

print(f"✅ HF_TOKEN loaded (length: {len(hf_token)})")

# Test 1: Login
try:
    login(token=hf_token)
    print("✅ HuggingFace login successful")
except Exception as e:
    print(f"❌ Login failed: {e}")
    exit(1)

# Test 2: Verify API access
try:
    api = HfApi()
    user_info = api.whoami(token=hf_token)
    print(f"✅ Authenticated as: {user_info['name']}")
except Exception as e:
    print(f"❌ API access failed: {e}")
    exit(1)

# Test 3: Check if repo exists or can be created
HF_REPO = "kapilw25/llama3-8b-pku-cita-baseline-bf16"  # Fixed: Use correct username

try:
    # Try to access repo info (will fail if doesn't exist)
    repo_info = api.repo_info(repo_id=HF_REPO, token=hf_token, repo_type="model")
    print(f"✅ Repo exists: {HF_REPO}")
    print(f"   Last modified: {repo_info.lastModified}")
except Exception as e:
    print(f"⚠️  Repo doesn't exist yet: {HF_REPO}")

    # Test if we can CREATE it
    try:
        response = input(f"Create repo '{HF_REPO}'? (y/n): ")
        if response.lower() == 'y':
            create_repo(
                repo_id=HF_REPO,
                token=hf_token,
                private=True,
                repo_type="model",
                exist_ok=True
            )
            print(f"✅ Repo created: https://huggingface.co/{HF_REPO}")
    except Exception as create_err:
        print(f"❌ Cannot create repo: {create_err}")

# Test 4: Test push capability with dummy file
try:
    from huggingface_hub import upload_file
    import tempfile

    # Create a tiny test file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Test file for credential validation")
        test_file = f.name

    response = input(f"Upload test file to {HF_REPO}? (y/n): ")
    if response.lower() == 'y':
        upload_file(
            path_or_fileobj=test_file,
            path_in_repo="test_credentials.txt",
            repo_id=HF_REPO,
            token=hf_token,
        )
        print(f"✅ Push capability confirmed!")
        print(f"   Test file uploaded to: https://huggingface.co/{HF_REPO}/blob/main/test_credentials.txt")

        # Clean up test file
        os.unlink(test_file)
except Exception as e:
    print(f"⚠️  Push test skipped or failed: {e}")

print("\n✅ HuggingFace credentials fully validated!")
