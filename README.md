# Fine-Tuning Evaluation Pipeline

A modular pipeline for fine-tuning large language models and evaluating their alignment quality using the Alignment Quality Index (AQI) methodology.

## Project Structure

```
├── Legacy_code/           # Original monolithic implementation
├── src/                   # Modular pipeline implementation
├── data/                  # Dataset storage
├── model/                 # Model checkpoints and adapters
├── output/                # Results and analysis
├── plans/                 # Project planning documentation
└── requirements.txt       # Python dependencies
```

## Overview

This project transforms monolithic fine-tuning code into a production-ready modular pipeline with:

- **Modular Design**: 5 separate modules with single responsibilities
- **Comprehensive Evaluation**: AQI-based alignment quality assessment
- **Reproducibility**: Fixed seeds and versioned dependencies
- **Data Persistence**: SQLite database for all results
- **Memory Management**: Efficient GPU memory handling

## Modules

1. **m01_environment_setup.py** - Environment setup & dependencies
2. **m02_data_preparation.py** - Dataset processing & formatting
3. **m03_model_training.py** - QLoRA fine-tuning pipeline
4. **m04_model_evaluation.py** - AQI evaluation & comparison
5. **m05_results_analysis.py** - Visualization & reporting

## Quick Start

1. Clone the repository
2. Set up the environment: `python src/m01_environment_setup.py`
3. Prepare data: `python src/m02_data_preparation.py`
4. Train model: `python src/m03_model_training.py`
5. Evaluate: `python src/m04_model_evaluation.py`
6. Analyze results: `python src/m05_results_analysis.py`

## Key Features

- QLoRA (Quantized Low-Rank Adaptation) fine-tuning
- Alignment Quality Index (AQI) evaluation
- Delta-AQI analysis for alignment quality assessment
- Comprehensive logging and error handling
- GPU memory optimization

## Dependencies

- PyTorch 2.3.0+
- Transformers 4.41.2+
- PEFT 0.10.0+
- BitsAndBytesConfig for 4-bit quantization
- scikit-learn for evaluation metrics

## License

MIT License