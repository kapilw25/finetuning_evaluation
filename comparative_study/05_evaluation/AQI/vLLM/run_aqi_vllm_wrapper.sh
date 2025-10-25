#!/usr/bin/env bash
#
# vLLM Wrapper Script - Fixes CUDA Multiprocessing Issues
#
# This wrapper sets required environment variables BEFORE Python starts
# to prevent "Cannot re-initialize CUDA in forked subprocess" errors

# Set multiprocessing start method to 'spawn' (required by vLLM with CUDA)
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Optional: Disable CUDA fork warnings
export CUDA_LAUNCH_BLOCKING=0

# Run the Python script with all arguments passed through
python3 "$(dirname "$0")/run_aqi_vllm.py" "$@"
