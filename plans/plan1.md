# 7-Step Comprehensive Fine-Tuning Evaluation Pipeline

## Overview

Transform the monolithic Legacy code into a modular, production-ready pipeline that evaluates **TWO** fine-tuning methodologies (QLoRA vs GRIT) against baseline performance across multiple inference backends, with comprehensive AQI-based alignment assessment.

## Module Structure

```
src/
├── m01_environment_setup.py      # Environment setup & requirements
├── m02_data_preparation.py       # Unified dataset processing (alignment-instructions + ACCD)
├── m03a_qlora_training.py        # Standard QLoRA fine-tuning pipeline
├── m03b_grit_training.py         # GRIT (FKLNR) advanced fine-tuning pipeline
├── m04_comparative_analysis.py   # Cross-method performance comparison (FIRST: compare training results)
├── m05_inference_backends.py     # Multi-backend deployment (SECOND: deploy models for evaluation)
├── m06_comprehensive_evaluation.py # AQI evaluation & behavioral assessment (THIRD: evaluate deployed models)
├── m07_results_synthesis.py      # Final reporting & recommendations (FOURTH: synthesize all results)

output/
├── m01_environment/              # Virtual environment & dependency logs
├── m02_data/                    # Processed datasets & safety categorization
├── m03a_qlora_training/         # QLoRA training artifacts & checkpoints
├── m03b_grit_training/          # GRIT training artifacts & KFAC matrices
├── m04_comparative_analysis/    # Method comparison matrices & charts (training comparison)
├── m05_inference_backends/      # Deployed model endpoints & performance metrics
├── m06_comprehensive_evaluation/# AQI scores, embeddings & behavioral tests
├── m07_results_synthesis/       # Final reports & executive summaries
└── centralized.db              # SQLite database for all experimental data
```

## Step 1: Environment Setup (m01_environment_setup.py)

**Purpose**: Create isolated environment with exact dependency versions for both training methodologies
- Create virtual environment `venv_FntngEval` with comprehensive package management
- Generate requirements.txt with pinned versions (torch>=2.3.0, transformers==4.41.2, peft==0.10.0)
- Setup CUDA/PyTorch compatibility for both QLoRA and GRIT training
- Configure HuggingFace authentication with token validation
- Install AQI evaluation dependencies (unsloth, sklearn, bitsandbytes)
- Initialize SQLite database schema for multi-experiment tracking
- Setup Ollama for primary inference deployment (with optional vLLM dependencies)
- **Output**: Environment logs, dependency manifest, database schema, compatibility checks

## Step 2: Unified Data Preparation (m02_data_preparation.py)

**Purpose**: Process multiple datasets for comprehensive evaluation
- **Training Data**: Download "Vaibhaav/alignment-instructions" dataset
  - Apply Llama-3 chat template formatting: `<s>[INST] {prompt} [/INST] {instruction}</s>`
  - Create train/validation splits (90/10, seed=42)
  - Generate instruction-following examples for both QLoRA and GRIT
- **Evaluation Data**: Download "hasnat79/ACCD" safety dataset
  - Extract balanced safe/unsafe prompts (500+ each category)
  - Apply keyword-based safety categorization with manual validation
  - Create standardized prompt templates for AQI evaluation
- **Data Validation**: Cross-dataset consistency checks and duplicate removal
- Store processed data with comprehensive metadata and version control
- **Output**: Formatted training datasets, safety-labeled evaluation sets, data statistics

## Step 3a: QLoRA Fine-Tuning Pipeline (m03a_qlora_training.py)

**Purpose**: Implement standard QLoRA fine-tuning methodology
- Load Meta-Llama-3-8B-Instruct with 4-bit quantization (BitsAndBytesConfig)
- Configure LoRA parameters: r=16, alpha=32, dropout=0.05
- Target modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
- Setup SFTTrainer with optimized hyperparameters:
  - Learning rate: 2e-4, batch size: 4, gradient accumulation: 2
  - Training epochs: 3, max steps: 250, cosine LR scheduler
- Implement gradient checkpointing and mixed-precision training (fp16)
- Save LoRA adapter with model merge capability
- **Output**: QLoRA adapter, training metrics, loss curves, model checkpoints

## Step 3b: GRIT Fine-Tuning Pipeline (m03b_grit_training.py)

**Purpose**: Implement advanced GRIT (Geometry-Aware Robust Instruction Tuning) methodology
- **Custom LoRA Implementation**: Manual LoRA layers with proper scaling
- **KFAC Approximation**: Second-order optimization using Kronecker-Factored Approximate Curvature
  - Damping factor: 1e-2, update frequency: 20 steps
  - Hook-based activation and gradient tracking
  - Momentum-based factor updates (β=0.95)
- **Natural Gradient Optimizer**: Fisher information matrix preconditioning
- **Neural Reprojection**: Top-k eigenvector projection (k=4) every 40 steps
  - Eigendecomposition of covariance matrices
  - Parameter space regularization for geometry preservation
- **FKLNR Training Loop**: Fisher-KFAC with Low-rank Neural Reprojection
  - Learning rate: 2e-5, batch size: 1, max sequence length: 1024
  - Advanced memory management with gradient clipping
- **Output**: GRIT adapter, KFAC matrices, eigenvector projections, advanced training logs

## Step 4: Cross-Method Comparative Analysis (m04_comparative_analysis.py)

**Purpose**: Compare QLoRA vs GRIT training results immediately after both trainings complete
- **Training Method Comparison**:
  - QLoRA vs GRIT alignment quality (training metrics)
  - Training efficiency metrics (time, memory, convergence)
  - Parameter efficiency and adapter size analysis
- **Training Results Analysis**:
  - Loss curves and convergence comparison
  - Memory usage and training time analysis
  - Hyperparameter effectiveness evaluation
- **Model Artifact Comparison**:
  - LoRA adapter sizes and storage requirements
  - Training checkpoint quality assessment
  - Model architecture differences analysis
- **Statistical Validation**:
  - Training performance significance testing
  - Convergence analysis and stability metrics
  - Resource efficiency comparison
- **Output**: Training comparison reports, method selection recommendations, efficiency analysis

## Step 5: Inference Backend Deployment (m05_inference_backends.py)

**Purpose**: Deploy all models (Base, QLoRA, GRIT) with Ollama-first approach and optional vLLM migration

### Phase 1: Ollama Deployment (Primary Implementation)
- **Ollama Setup & Configuration**:
  - Install Ollama with automatic model management
  - Create Modelfiles for Base, QLoRA, and GRIT variants
  - Deploy with REST API endpoints for evaluation pipeline
  - Configure automatic quantization (4-bit/5-bit) for memory efficiency
- **Model Deployment Pipeline**:
  - Convert LoRA adapters to Ollama-compatible format
  - Test model loading and basic inference functionality
  - Validate response quality across all three model variants
- **Baseline Performance Assessment**:
  - Measure baseline RPS (~20-25 expected) and latency
  - Profile VRAM usage across model variants
  - Test concurrent request handling (up to 32 parallel)
- **Integration Testing**:
  - Verify API compatibility with AQI evaluation pipeline
  - Test model switching and resource management
  - Validate response consistency and quality preservation

### Phase 2: vLLM Migration (Optional Performance Optimization)
- **Migration Trigger Conditions**:
  - If evaluation throughput becomes bottleneck (>100 requests/batch)
  - If production deployment requires high concurrency (>50 parallel users)
  - If cost analysis favors high-throughput serving
- **vLLM High-Performance Setup**:
  - BatchedLLM configuration for maximum throughput
  - Tensor parallelism for multi-GPU environments
  - Continuous batching and paged attention optimization
- **Performance Comparison**:
  - Benchmark 3.2x throughput improvement validation
  - Cost-benefit analysis (setup complexity vs performance gains)
  - Quality preservation verification across quantization levels

### Implementation Strategy
```bash
# Phase 1: Start with Ollama
ollama create base-llama3 -f Modelfile-base
ollama create qlora-llama3 -f Modelfile-qlora
ollama create grit-llama3 -f Modelfile-grit

# Phase 2: Migrate to vLLM only if needed
python -m vllm.entrypoints.api_server --model ./qlora-model --quantization awq
```

- **Output**: Ollama endpoints (primary), optional vLLM deployment, performance benchmarks, migration decision matrix

## Step 6: Comprehensive AQI Evaluation (m06_comprehensive_evaluation.py)

**Purpose**: Unified AQI evaluation framework for all models and backends
- **Model Loading**: Base, QLoRA-tuned, and GRIT-tuned models
- **Hidden State Extraction**:
  - Extract embeddings from all transformer layers
  - Implement layer-wise attention pooling with reference vectors
  - Generate 64+ safe and 64+ unsafe prompt completions per model
- **AQI Calculation Pipeline**:
  - XBI (eXplicit Bias Index) computation with intra/inter-cluster variance
  - Calinski-Harabasz Index for cluster separation quality
  - Combined AQI score: λ*(1/XBI) + (1-λ)*CHI (λ=0.5)
- **Delta-AQI Analysis**:
  - Base vs QLoRA Delta-AQI calculation
  - Base vs GRIT Delta-AQI calculation
  - Cross-method Delta-AQI comparison (QLoRA vs GRIT)
- **G-Eval Behavioral Assessment**:
  - Alignment quality scoring using judge model evaluation
  - Safety refusal rate analysis across prompt categories
  - Response quality and helpfulness metrics
- **Statistical Validation**: Bootstrap sampling for confidence intervals
- **Output**: AQI scores matrix, Delta-AQI comparisons, behavioral metrics, statistical significance tests

## Step 7: Results Synthesis & Recommendations (m07_results_synthesis.py)

**Purpose**: Generate comprehensive final reports combining training, deployment, and evaluation results
- **Executive Summary Dashboard**:
  - Key findings from training comparison (Step 4)
  - Deployment performance analysis (Step 5)
  - AQI evaluation results synthesis (Step 6)
  - Method recommendation matrix based on comprehensive analysis
- **Comprehensive Comparison Analysis**:
  - QLoRA vs GRIT vs Baseline across all dimensions
  - Training efficiency vs inference quality trade-offs
  - Cost-effectiveness analysis with ROI calculations
  - Statistical significance testing across all metrics
- **Technical Deep-Dive Reports**:
  - Detailed methodology comparison with statistical evidence
  - Training optimization recommendations for each method
  - Infrastructure requirements and scaling considerations
- **Deployment Guidelines**:
  - Inference backend selection criteria
  - Production deployment best practices
  - Monitoring and quality assurance frameworks
- **Future Research Directions**:
  - Identified limitations and potential improvements
  - Advanced fine-tuning methodology exploration
  - Alignment evaluation framework enhancements
- **Interactive Visualizations**:
  - Performance comparison charts across all dimensions
  - AQI score evolution plots with confidence intervals
  - Cost vs performance Pareto frontiers
- **Database Export**: Complete experimental results export for reproducibility
- **Output**: Executive summaries, technical reports, deployment guides, interactive dashboards
- **Executive Summary Dashboard**:
  - Key findings visualization (AQI improvements, performance gains)
  - Method recommendation matrix based on use case requirements
  - Cost-effectiveness analysis with ROI calculations
- **Technical Deep-Dive Reports**:
  - Detailed methodology comparison with statistical evidence
  - Training optimization recommendations for each method
  - Infrastructure requirements and scaling considerations
- **Deployment Guidelines**:
  - Inference backend selection criteria
  - Production deployment best practices
  - Monitoring and quality assurance frameworks
- **Future Research Directions**:
  - Identified limitations and potential improvements
  - Advanced fine-tuning methodology exploration
  - Alignment evaluation framework enhancements
- **Interactive Visualizations**:
  - Performance comparison charts across all dimensions
  - AQI score evolution plots with confidence intervals
  - Cost vs performance Pareto frontiers
- **Database Export**: Complete experimental results export for reproducibility
- **Output**: Executive summaries, technical reports, deployment guides, interactive dashboards

## Database Schema (centralized.db)

### Core Tables
- `experiments`: Experiment metadata (method, hyperparameters, timestamps)
- `training_metrics_qlora`: QLoRA training logs (loss, lr, epoch, memory usage)
- `training_metrics_grit`: GRIT training logs (KFAC factors, reprojection data)
- `model_artifacts`: Model paths, adapter weights, merge status
- `datasets`: Dataset versions, splits, safety labels, preprocessing logs

### Evaluation Tables
- `aqi_scores`: AQI/Delta-AQI scores (base, qlora, grit models)
- `hidden_states`: Extracted embeddings with layer information
- `behavioral_metrics`: G-Eval scores, safety refusal rates, helpfulness scores
- `model_responses`: Generated completions with prompt categorization
- `inference_benchmarks`: Performance metrics across backends (RPS, latency, memory)

### Analysis Tables
- `statistical_tests`: Significance tests (t-tests, effect sizes, p-values)
- `comparative_analysis`: Cross-method comparison matrices
- `cost_analysis`: Training costs, inference costs, efficiency metrics
- `recommendations`: Final recommendations with confidence scores

## Key Features

✅ **Dual-Method Training**: Both QLoRA and GRIT methodologies implemented
✅ **Multi-Backend Inference**: Ollama, Llama.cpp, and vLLM deployment comparison
✅ **Comprehensive AQI Evaluation**: Unified framework covering all model variants
✅ **Statistical Rigor**: Proper significance testing with multiple comparison correction
✅ **Modular Design**: Each module has single responsibility with clear data flow
✅ **Progress Tracking**: Output from previous module feeds into next module
✅ **Error Handling**: Robust error handling with graceful fallbacks
✅ **Memory Management**: GPU memory cleanup between modules and training methods
✅ **Reproducibility**: Fixed seeds, versioned dependencies, deterministic evaluation
✅ **Comprehensive Logging**: Detailed logs for debugging across all components
✅ **Data Persistence**: All results stored in structured format with version control
✅ **Performance Optimization**: Efficient inference backend selection based on use case
✅ **Cost Analysis**: Training and inference cost tracking for informed decisions

## Implementation Strategy & Backend Selection

### Recommended Approach: Ollama-First Development
**Primary Choice: Ollama** for initial development and evaluation phase
- ✅ **Rapid Setup**: Models running in minutes vs hours of configuration
- ✅ **Perfect for AQI Evaluation**: Sufficient performance for comparison methodology
- ✅ **Memory Efficient**: Automatic quantization reduces VRAM requirements
- ✅ **Easy Integration**: Clean REST API for Python evaluation pipeline
- ✅ **Model Management**: Automatic handling of Base, QLoRA, and GRIT variants

### Migration Path (Optional Performance Scaling)
```
Phase 1: Training & Analysis (Steps 1-4)
├── Complete QLoRA vs GRIT training (Steps 3a, 3b)
├── Compare training results immediately (Step 4)
├── Deploy all models via Ollama (Step 5)
├── Run comprehensive AQI evaluation (Step 6)
└── Generate final synthesis reports (Step 7)

Phase 2: vLLM Migration (If Needed)
├── Trigger: >100 requests/batch or production deployment
├── Expected: 3.2x throughput improvement
└── Trade-off: Setup complexity vs performance gains
```

### Performance Benchmarks (2024-2025 Data)
- **Ollama**: ~22 RPS, excellent for evaluation workloads
- **vLLM**: ~70+ RPS (3.2x improvement), optimal for production serving
- **Quality**: Both preserve model quality across quantization levels

## Coverage Verification

### Legacy Script Coverage Status
✅ **finetuned_file.py**: Fully covered in Steps 2, 3a, 5, 6, 7
✅ **baseevalaqi.py**: Fully covered in Steps 2, 4, 5, 6, 7 (unified dataset approach)
✅ **grit.py**: Fully covered in Steps 1, 2, 3b, 5, 6, 7 (complete GRIT methodology)

### Redundancy Elimination
- ✅ Unified data preparation (Step 2) eliminates dataset duplication
- ✅ Single AQI evaluation framework (Step 5) consolidates evaluation logic
- ✅ Centralized model loading and memory management across all modules
- ✅ Consolidated inference deployment (Step 4) eliminates backend duplication