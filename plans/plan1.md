# 5-Step Execution Plan: Replicating Fine-Tuning & AQI Evaluation Pipeline

## Overview

Transform the monolithic Legacy code into a modular, production-ready pipeline with proper dependency management, data versioning, and comprehensive evaluation.

## Module Structure

```
src/
├── m01_environment_setup.py    # Environment setup & requirements
├── m02_data_preparation.py     # Dataset loading & preprocessing
├── m03_model_training.py       # QLoRA fine-tuning pipeline
├── m04_model_evaluation.py     # AQI evaluation & comparison
├── m05_results_analysis.py     # Visualization & reporting

output/
├── m01_environment/           # Virtual environment logs
├── m02_data/                 # Processed datasets & metadata
├── m03_training/             # Training logs & checkpoints
├── m04_evaluation/           # AQI scores & embeddings
├── m05_analysis/             # Visualization & reports
└── centralized.db           # SQLite database for all results
```

## Step 1: Environment Setup (m01_environment_setup.py)

**Purpose**: Create isolated environment with exact dependency versions
- Create virtual environment venv_FntngEval
- Generate requirements.txt with pinned versions from Legacy analysis
- Setup CUDA/PyTorch compatibility
- Configure HuggingFace authentication
- Initialize SQLite database schema
- **Output**: Environment logs, dependency manifest, database schema

## Step 2: Data Preparation (m02_data_preparation.py)

**Purpose**: Download, process & validate training data
- Download "Vaibhaav/alignment-instructions" dataset
- Apply Llama-3 chat template formatting: `<s>[INST] {prompt} [/INST] {instruction}</s>`
- Create train/validation splits (90/10, seed=42)
- Generate safety categorization using keyword heuristics
- Store processed data in /data with metadata
- **Output**: Formatted datasets, data statistics, safety labels

## Step 3: Model Training (m03_model_training.py)

**Purpose**: Implement QLoRA fine-tuning pipeline
- Load Meta-Llama-3-8B-Instruct with 4-bit quantization
- Configure LoRA: r=16, alpha=32, dropout=0.05
- Setup SFTTrainer with optimized hyperparameters
- Train for 3 epochs (max 250 steps) with monitoring
- Save LoRA adapter to /model directory
- **Output**: Fine-tuned adapter, training logs, model metadata

## Step 4: Model Evaluation (m04_model_evaluation.py)

**Purpose**: Comprehensive AQI evaluation & comparison
- Load base model vs fine-tuned model
- Extract hidden states for safe/unsafe prompts (64+ each)
- Calculate AQI scores using XBI + Calinski-Harabasz metrics
- Compute Delta-AQI for alignment quality assessment
- Generate G-Eval scores for behavioral evaluation
- **Output**: AQI scores, embeddings, evaluation metrics

## Step 5: Results Analysis (m05_results_analysis.py)

**Purpose**: Generate comprehensive reports & visualizations
- Create comparative visualizations (AQI, G-Eval charts)
- Generate alignment quality interpretation reports
- Compare base vs fine-tuned model performance
- Export results to centralized database
- Create executive summary with recommendations
- **Output**: Charts, reports, database records, final analysis

## Database Schema (centralized.db)

- `experiments`: Experiment metadata & configurations
- `training_metrics`: Loss, learning rate, epoch data
- `aqi_scores`: Base vs fine-tuned AQI comparisons
- `model_responses`: Generated text samples for analysis
- `evaluation_summary`: Final scores & interpretations

## Key Features

✅ **Modular Design**: Each module has single responsibility
✅ **Progress Tracking**: Output from previous module feeds into next
✅ **Error Handling**: Robust error handling with graceful fallbacks
✅ **Memory Management**: GPU memory cleanup between modules
✅ **Reproducibility**: Fixed seeds, versioned dependencies
✅ **Comprehensive Logging**: Detailed logs for debugging
✅ **Data Persistence**: All results stored in structured format