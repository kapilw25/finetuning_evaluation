#!/usr/bin/env python3
"""
Delete HuggingFace repositories to start fresh

Usage:
    python unit_test/delete_hf_repos.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, delete_repo

# Load environment variables
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("❌ HF_TOKEN not found in .env file")
    sys.exit(1)

# Initialize HF API
api = HfApi(token=HF_TOKEN)

# Repositories to delete
REPOS_TO_DELETE = [
    "kapilw25/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct",
    "kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct",
    "kapilw25/llama3-8b-pku-CITA-Instruct-DPO-Instruct",
]

print(f"\n{'='*80}")
print("⚠️  DELETE HUGGINGFACE REPOSITORIES")
print(f"{'='*80}")
print("The following repositories will be PERMANENTLY DELETED:")
for repo in REPOS_TO_DELETE:
    print(f"  - {repo}")
print(f"{'='*80}")
print("⚠️  WARNING: This action CANNOT be undone!")
print(f"{'='*80}\n")

# Confirmation
response = input("Type 'DELETE' to confirm permanent deletion: ").strip()
if response != "DELETE":
    print("❌ Deletion cancelled (you must type exactly 'DELETE' to confirm)")
    sys.exit(0)

print("\n")

# Delete each repository
deleted_count = 0
failed_count = 0
skipped_count = 0

for repo_id in REPOS_TO_DELETE:
    try:
        print(f"🗑️  Checking {repo_id}...")

        # Check if repo exists first
        try:
            api.repo_info(repo_id=repo_id, repo_type="model")
            repo_exists = True
        except Exception:
            repo_exists = False

        if not repo_exists:
            print(f"   ⏭️  Repository does not exist (already deleted or never created)")
            skipped_count += 1
            continue

        # Delete the repository
        print(f"   🗑️  Deleting repository...")
        delete_repo(
            repo_id=repo_id,
            token=HF_TOKEN,
            repo_type="model"
        )
        print(f"   ✅ Successfully deleted: {repo_id}")
        deleted_count += 1

    except Exception as e:
        print(f"   ❌ Failed to delete {repo_id}: {e}")
        failed_count += 1

# Summary
print(f"\n{'='*80}")
print("📊 DELETION SUMMARY")
print(f"{'='*80}")
print(f"✅ Deleted: {deleted_count} repositories")
print(f"⏭️  Skipped: {skipped_count} repositories (did not exist)")
print(f"❌ Failed: {failed_count} repositories")
print(f"{'='*80}\n")

if deleted_count > 0:
    print("✅ Repositories deleted successfully!")
    print("   You can now create fresh repositories with the same names")
elif skipped_count == len(REPOS_TO_DELETE):
    print("ℹ️  All repositories were already deleted or never existed")
else:
    print("⚠️  Some deletions failed - check error messages above")

print()
