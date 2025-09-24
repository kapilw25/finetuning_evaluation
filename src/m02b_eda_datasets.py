#!/usr/bin/env python3
"""
EDA (Exploratory Data Analysis) for Downloaded Datasets
Analyze the datasets downloaded by m02_data_preparation.py
"""

import os
import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

class DatasetEDA:
    """Exploratory Data Analysis for fine-tuning datasets"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.cache_dir = self.project_root / "outputs" / "m02_data" / "cache"
        self.output_dir = self.project_root / "outputs" / "m02_data"

    def analyze_alignment_instructions(self):
        """Analyze Vaibhaav/alignment-instructions dataset"""
        print("=" * 60)
        print("ALIGNMENT INSTRUCTIONS DATASET ANALYSIS")
        print("=" * 60)

        try:
            # Load dataset
            dataset = load_dataset("Vaibhaav/alignment-instructions", cache_dir=str(self.cache_dir))
            data = dataset['train']

            print(f"📊 Dataset Size: {len(data):,} samples")
            print(f"📋 Columns: {data.column_names}")
            print(f"💾 Dataset Features:")
            for col, feature in data.features.items():
                print(f"  - {col}: {feature}")

            # Convert to pandas for easier analysis
            df = pd.DataFrame(data)

            # Basic statistics
            print("\n📈 BASIC STATISTICS:")
            print(f"Total samples: {len(df):,}")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

            # Text length analysis
            if 'Prompt' in df.columns:
                df['prompt_length'] = df['Prompt'].astype(str).str.len()
                print(f"\n📝 PROMPT LENGTH STATISTICS:")
                print(f"  - Mean: {df['prompt_length'].mean():.0f} characters")
                print(f"  - Median: {df['prompt_length'].median():.0f} characters")
                print(f"  - Min: {df['prompt_length'].min():.0f} characters")
                print(f"  - Max: {df['prompt_length'].max():.0f} characters")
                print(f"  - Std: {df['prompt_length'].std():.0f} characters")

            if 'Instruction generated' in df.columns:
                df['instruction_length'] = df['Instruction generated'].astype(str).str.len()
                print(f"\n📋 INSTRUCTION LENGTH STATISTICS:")
                print(f"  - Mean: {df['instruction_length'].mean():.0f} characters")
                print(f"  - Median: {df['instruction_length'].median():.0f} characters")
                print(f"  - Min: {df['instruction_length'].min():.0f} characters")
                print(f"  - Max: {df['instruction_length'].max():.0f} characters")
                print(f"  - Std: {df['instruction_length'].std():.0f} characters")

            # Sample data inspection
            print("\n🔍 SAMPLE DATA:")
            for i in range(min(3, len(data))):
                print(f"\n--- Sample {i+1} ---")
                sample = data[i]
                for key, value in sample.items():
                    value_str = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                    print(f"{key}: {value_str}")

            # Save detailed info
            alignment_info = {
                "dataset_name": "Vaibhaav/alignment-instructions",
                "total_samples": len(data),
                "columns": data.column_names,
                "sample_data": dict(data[0]) if len(data) > 0 else {},
                "statistics": {
                    "prompt_length": {
                        "mean": float(df['prompt_length'].mean()) if 'prompt_length' in df else None,
                        "median": float(df['prompt_length'].median()) if 'prompt_length' in df else None,
                        "std": float(df['prompt_length'].std()) if 'prompt_length' in df else None
                    },
                    "instruction_length": {
                        "mean": float(df['instruction_length'].mean()) if 'instruction_length' in df else None,
                        "median": float(df['instruction_length'].median()) if 'instruction_length' in df else None,
                        "std": float(df['instruction_length'].std()) if 'instruction_length' in df else None
                    }
                }
            }

            with open(self.output_dir / "alignment_dataset_analysis.json", 'w') as f:
                json.dump(alignment_info, f, indent=2, default=str)

            return df

        except Exception as e:
            print(f"❌ Error analyzing alignment dataset: {e}")
            return None

    def analyze_accd_safety(self):
        """Analyze hasnat79/ACCD safety dataset"""
        print("\n" + "=" * 60)
        print("ACCD SAFETY DATASET ANALYSIS")
        print("=" * 60)

        try:
            # Load dataset
            dataset = load_dataset("hasnat79/ACCD", cache_dir=str(self.cache_dir))
            data = dataset['train']

            print(f"📊 Dataset Size: {len(data):,} samples")
            print(f"📋 Columns: {data.column_names}")
            print(f"💾 Dataset Features:")
            for col, feature in data.features.items():
                print(f"  - {col}: {feature}")

            # Convert to pandas for easier analysis
            df = pd.DataFrame(data)

            # Basic statistics
            print("\n📈 BASIC STATISTICS:")
            print(f"Total samples: {len(df):,}")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

            # Identify text column
            text_cols = [col for col in df.columns if 'text' in col.lower() or 'prompt' in col.lower() or 'instruction' in col.lower()]
            if text_cols:
                text_col = text_cols[0]
                df['text_length'] = df[text_col].astype(str).str.len()
                print(f"\n📝 TEXT LENGTH STATISTICS (Column: {text_col}):")
                print(f"  - Mean: {df['text_length'].mean():.0f} characters")
                print(f"  - Median: {df['text_length'].median():.0f} characters")
                print(f"  - Min: {df['text_length'].min():.0f} characters")
                print(f"  - Max: {df['text_length'].max():.0f} characters")
                print(f"  - Std: {df['text_length'].std():.0f} characters")

            # Safety label analysis
            safety_cols = [col for col in df.columns if 'label' in col.lower() or 'safety' in col.lower() or 'safe' in col.lower()]
            if safety_cols:
                safety_col = safety_cols[0]
                print(f"\n🛡️ SAFETY LABEL DISTRIBUTION (Column: {safety_col}):")
                value_counts = df[safety_col].value_counts()
                for value, count in value_counts.items():
                    percentage = (count / len(df)) * 100
                    print(f"  - {value}: {count:,} ({percentage:.1f}%)")
            else:
                print("\n🛡️ SAFETY LABELS: No clear safety column found")
                print("Available columns:", df.columns.tolist())

            # Sample data inspection
            print("\n🔍 SAMPLE DATA:")
            for i in range(min(3, len(data))):
                print(f"\n--- Sample {i+1} ---")
                sample = data[i]
                for key, value in sample.items():
                    value_str = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                    print(f"{key}: {value_str}")

            # Save detailed info
            accd_info = {
                "dataset_name": "hasnat79/ACCD",
                "total_samples": len(data),
                "columns": data.column_names,
                "sample_data": dict(data[0]) if len(data) > 0 else {},
                "statistics": {
                    "text_length": {
                        "mean": float(df['text_length'].mean()) if 'text_length' in df else None,
                        "median": float(df['text_length'].median()) if 'text_length' in df else None,
                        "std": float(df['text_length'].std()) if 'text_length' in df else None
                    },
                    "safety_distribution": df[safety_cols[0]].value_counts().to_dict() if safety_cols else None
                }
            }

            with open(self.output_dir / "accd_dataset_analysis.json", 'w') as f:
                json.dump(accd_info, f, indent=2, default=str)

            return df

        except Exception as e:
            print(f"❌ Error analyzing ACCD dataset: {e}")
            return None

    def analyze_processed_data(self):
        """Analyze the processed data in centralized database"""
        print("\n" + "=" * 60)
        print("PROCESSED DATA ANALYSIS")
        print("=" * 60)

        try:
            from m00_centralized_db import CentralizedDB
            db = CentralizedDB()

            # Get recent processing results
            recent_data = db.get_latest_results('data', limit=5)
            if recent_data:
                print("📊 RECENT DATA PREPARATION RESULTS:")
                for result in recent_data:
                    print(f"  - Dataset: {result.get('dataset_name', 'Unknown')}")
                    print(f"  - Total samples: {result.get('dataset_size', 0):,}")
                    print(f"  - Train/Val: {result.get('train_samples', 0):,}/{result.get('val_samples', 0):,}")
                    print(f"  - Safe/Unsafe: {result.get('safe_prompts', 0):,}/{result.get('unsafe_prompts', 0):,}")
                    print(f"  - Status: {result.get('preprocessing_status', 'Unknown')}")
                    print(f"  - Time: {result.get('created_at', 'Unknown')}")
                    print()

            # Check processing reports
            reports_dir = self.output_dir
            if (reports_dir / "processing_report.json").exists():
                with open(reports_dir / "processing_report.json", 'r') as f:
                    report = json.load(f)

                print("📋 PROCESSING REPORT SUMMARY:")
                summary = report.get('processing_summary', {})
                print(f"  - Processing time: {summary.get('processing_time_seconds', 0):.1f} seconds")
                print(f"  - Overall status: {summary.get('overall_status', 'Unknown')}")

                alignment = report.get('alignment_dataset', {})
                safety = report.get('safety_dataset', {})

                print(f"  - Alignment dataset: {alignment.get('total_samples', 0):,} samples")
                print(f"  - Safety dataset: {safety.get('total_samples', 0):,} samples")

        except Exception as e:
            print(f"❌ Error analyzing processed data: {e}")

    def generate_visualizations(self, alignment_df, accd_df):
        """Generate visualizations for the datasets"""
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Alignment dataset - prompt length distribution
            if alignment_df is not None and 'prompt_length' in alignment_df.columns:
                axes[0, 0].hist(alignment_df['prompt_length'], bins=50, alpha=0.7, color='blue')
                axes[0, 0].set_title('Alignment Dataset - Prompt Length Distribution')
                axes[0, 0].set_xlabel('Characters')
                axes[0, 0].set_ylabel('Frequency')

            # Alignment dataset - instruction length distribution
            if alignment_df is not None and 'instruction_length' in alignment_df.columns:
                axes[0, 1].hist(alignment_df['instruction_length'], bins=50, alpha=0.7, color='green')
                axes[0, 1].set_title('Alignment Dataset - Instruction Length Distribution')
                axes[0, 1].set_xlabel('Characters')
                axes[0, 1].set_ylabel('Frequency')

            # ACCD dataset - text length distribution
            if accd_df is not None and 'text_length' in accd_df.columns:
                axes[1, 0].hist(accd_df['text_length'], bins=50, alpha=0.7, color='orange')
                axes[1, 0].set_title('ACCD Dataset - Text Length Distribution')
                axes[1, 0].set_xlabel('Characters')
                axes[1, 0].set_ylabel('Frequency')

            # ACCD dataset - safety distribution (if available)
            if accd_df is not None:
                safety_cols = [col for col in accd_df.columns if 'label' in col.lower() or 'safety' in col.lower()]
                if safety_cols:
                    safety_counts = accd_df[safety_cols[0]].value_counts()
                    axes[1, 1].pie(safety_counts.values, labels=safety_counts.index, autopct='%1.1f%%')
                    axes[1, 1].set_title('ACCD Dataset - Safety Label Distribution')
                else:
                    axes[1, 1].text(0.5, 0.5, 'No safety labels\nfound in dataset',
                                   ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('ACCD Dataset - Safety Distribution')

            plt.tight_layout()
            plt.savefig(self.output_dir / 'dataset_analysis_visualizations.png', dpi=300, bbox_inches='tight')
            plt.close()

            print("📊 Visualizations saved to: dataset_analysis_visualizations.png")

        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")

    def run_complete_eda(self):
        """Run complete exploratory data analysis"""
        print("🔍 STARTING COMPLETE EDA OF DOWNLOADED DATASETS")
        print("=" * 80)

        # Analyze individual datasets
        alignment_df = self.analyze_alignment_instructions()
        accd_df = self.analyze_accd_safety()

        # Analyze processed data
        self.analyze_processed_data()

        # Generate visualizations
        self.generate_visualizations(alignment_df, accd_df)

        print("\n" + "=" * 80)
        print("🎉 EDA COMPLETE!")
        print("=" * 80)
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📊 Analysis files:")
        print(f"  - alignment_dataset_analysis.json")
        print(f"  - accd_dataset_analysis.json")
        print(f"  - dataset_analysis_visualizations.png")
        print(f"  - processing_report.json")

def main():
    """Run EDA analysis"""
    eda = DatasetEDA()
    eda.run_complete_eda()

if __name__ == "__main__":
    main()