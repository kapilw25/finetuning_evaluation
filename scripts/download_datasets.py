#!/usr/bin/env python3
"""
Download HuggingFace datasets to local storage following ML research best practices.
Datasets are stored in datasets/raw/ with proper organization.
Also exports Arrow files to CSV/JSON for easy analysis and visualization.
"""

import os
import json
from datasets import load_dataset, load_from_disk
from pathlib import Path

# Base directory
BASE_DIR = Path("/lambda/nfs/DiskUsEast1/finetuning_evaluation")
RAW_DATA_DIR = BASE_DIR / "datasets" / "raw"
EXPORT_DIR = BASE_DIR / "datasets" / "processed"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def export_to_csv_json(dataset, local_dir: str, split_name: str = "train"):
    """
    Export Arrow dataset to CSV and JSON formats for easy analysis.

    Dataset structures confirmed:
    - hasnat79_litmus: axiom, safety_label, source_dataset, input, target (20,439 examples)
    - utk6_safe_dataset_gretel: instruction, response, __index_level_0__ (8,361 examples)
    - utk6_unsafe_dataset_gretel: instruction, response, __index_level_0__ (8,361 examples)
    - vaibhaav_alignment_instructions: Accepted Response, Rejected Response, Instruction generated, Prompt (50,001 examples)
    """
    export_base = EXPORT_DIR / local_dir
    export_base.mkdir(parents=True, exist_ok=True)

    try:
        # Export full dataset to CSV
        csv_path = export_base / f"{split_name}.csv"
        dataset.to_csv(str(csv_path), index=False)
        csv_size = csv_path.stat().st_size / (1024 * 1024)  # MB
        print(f"   📄 CSV exported: {csv_path.name} ({csv_size:.2f} MB)")

        # Export sample to JSON for quick inspection (first 100 examples)
        json_path = export_base / f"{split_name}_sample.json"
        sample_size = min(100, len(dataset))
        dataset.select(range(sample_size)).to_json(
            str(json_path),
            orient="records",
            lines=False,
            indent=2,
            force_ascii=False
        )
        json_size = json_path.stat().st_size / (1024 * 1024)  # MB
        print(f"   📄 JSON sample: {json_path.name} ({sample_size} examples, {json_size:.2f} MB)")

        # Create dataset statistics
        stats_path = export_base / f"{split_name}_stats.json"
        stats = {
            "split": split_name,
            "num_examples": len(dataset),
            "columns": dataset.column_names,
            "column_types": {col: str(dataset.features[col]) for col in dataset.column_names},
            "sample_rows": dataset.select(range(min(3, len(dataset)))).to_dict()
        }
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"   📊 Statistics: {stats_path.name}")

        return True

    except Exception as e:
        print(f"   ❌ Export error: {str(e)}")
        return False

# Dataset configurations
DATASETS = [
    {
        "name": "utk6/safe_dataset_gretel",
        "local_dir": "utk6_safe_dataset_gretel",
        "description": "Safe instruction-response pairs from Gretel"
    },
    {
        "name": "utk6/unsafe_dataset_gretel",
        "local_dir": "utk6_unsafe_dataset_gretel",
        "description": "Unsafe instruction-response pairs from Gretel"
    },
    {
        "name": "Vaibhaav/alignment-instructions",
        "local_dir": "vaibhaav_alignment_instructions",
        "description": "Alignment instructions dataset"
    },
    {
        "name": "hasnat79/litmus",
        "local_dir": "hasnat79_litmus",
        "description": "Litmus safety evaluation dataset"
    }
]

def download_dataset(dataset_name: str, local_dir: str, description: str):
    """Download HuggingFace dataset only if not already exists in raw folder."""
    print(f"\n{'='*80}")
    print(f"📦 Dataset: {dataset_name}")
    print(f"📁 Description: {description}")
    print(f"💾 Local path: {RAW_DATA_DIR / local_dir}")
    print(f"{'='*80}\n")

    dataset_path = RAW_DATA_DIR / local_dir

    try:
        # Check if dataset already exists
        if dataset_path.exists():
            print(f"✅ Dataset already exists (skipping download)")
            # Load existing dataset
            dataset = load_from_disk(str(dataset_path))
        else:
            # Download dataset
            print(f"📥 Downloading from HuggingFace...")
            dataset = load_dataset(dataset_name)

            # Save to local directory (Arrow format)
            dataset.save_to_disk(str(dataset_path))
            print(f"✅ Downloaded and saved to disk")

        # Print dataset info
        print(f"📊 Dataset structure:")
        for split_name, split_data in dataset.items():
            print(f"   - {split_name}: {len(split_data):,} examples")
            if len(split_data) > 0:
                print(f"     Columns: {split_data.column_names}")

            # Export to CSV/JSON only if exports don't exist
            export_base = EXPORT_DIR / local_dir
            csv_path = export_base / f"{split_name}.csv"

            if csv_path.exists():
                print(f"\n   ✅ CSV/JSON exports already exist (skipping)")
            else:
                print(f"\n   🔄 Exporting {split_name} split to CSV/JSON...")
                export_to_csv_json(split_data, local_dir, split_name)

        return True

    except Exception as e:
        print(f"❌ Error processing {dataset_name}: {str(e)}")
        return False

def main():
    """
    Download datasets from HuggingFace (if not exist) and export to CSV/JSON (if not exported).
    """
    print("\n🚀 Starting dataset processing...")
    print(f"📂 Base directory: {BASE_DIR}")
    print(f"📂 Raw data directory: {RAW_DATA_DIR}")
    print(f"📂 Export directory: {EXPORT_DIR}\n")

    results = {}

    for dataset_config in DATASETS:
        success = download_dataset(
            dataset_config["name"],
            dataset_config["local_dir"],
            dataset_config["description"]
        )
        results[dataset_config["name"]] = success

    # Summary
    print(f"\n{'='*80}")
    print("📋 PROCESSING SUMMARY")
    print(f"{'='*80}")

    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful

    for dataset_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {dataset_name}")

    print(f"\n📊 Total: {len(results)} datasets")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"\n💾 Arrow files: {RAW_DATA_DIR}")
    print(f"📄 CSV/JSON exports: {EXPORT_DIR}")
    print(f"{'='*80}\n")

    return all(results.values())

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
