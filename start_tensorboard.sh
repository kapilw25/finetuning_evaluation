#!/bin/bash
# ============================================================================
# TensorBoard Launcher - View All Training Logs
# ============================================================================
#
# STEP 1: On your LOCAL terminal, create SSH tunnel:
#   ssh -L 6006:localhost:6006 ubuntu@129.213.150.225
#
# STEP 2: On SERVER terminal, run this script:
#   cd ~/DiskUsEast1/finetuning_evaluation
#   ./start_tensorboard.sh
#
# STEP 3: Open in your browser:
#   http://localhost:6006
#
# This will show ALL runs: SFT_Baseline, SFT_GRIT, DPO_Baseline, CITA_Baseline
# ============================================================================

source venv_aqi/bin/activate
tensorboard --logdir=/home/ubuntu/DiskUsEast1/finetuning_evaluation/tensorboard_logs/ --port 6006
