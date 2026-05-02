#!/bin/bash
# =============================================================================
# AUTOMATED EVALUATION RUNNER - ALL 5 EVALS
# =============================================================================
# Runs: ISD, TruthfulQA, Conditional Safety, Length Control, AQI
# Uses cached inference, regenerates plots only
# =============================================================================
#
# USAGE:
#     # Option 1: Background (recommended)
#   `nohup bash comparative_study/05_evaluation/run_all_evals_sequentially.sh &`
#   `tail -f logs/eval_runner_*.log`

#   # Option 2: Foreground
#   `bash comparative_study/05_evaluation/run_all_evals_sequentially.sh`
#
# =============================================================================

set -e  # Exit on error

LOG_DIR="/workspace/finetuning_evaluation/logs"
EVAL_DIR="/workspace/finetuning_evaluation/comparative_study/05_evaluation"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/eval_runner_${TIMESTAMP}.log"

# Redirect all output to log file AND terminal
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=============================================================================="
echo "AUTOMATED EVALUATION RUNNER - ALL 5 EVALS"
echo "=============================================================================="
echo "Started at: $(date)"
echo "Master log: $MASTER_LOG"
echo "=============================================================================="

# =============================================================================
# INPUT SEQUENCES PER EVAL:
# =============================================================================
# Menu 1 - Cached data: 1 = Regenerate plots only
# Menu 2 - Mode: 3 = Max, 2 = Full
# Menu 3+ - Per-checkpoint: 1 = Use cached
#
# Checkpoint counts (10 models):
#   ISD:               1 variant  × 10 = 10 prompts
#   TruthfulQA:        2 variants × 10 = 20 prompts (honest, confident)
#   Conditional Safety: 2 variants × 10 = 20 prompts (strict, permissive)
#   Length Control:    2 variants × 10 = 20 prompts (concise, detailed)
#   AQI:               1 variant  × 10 = 10 prompts
# =============================================================================

# ISD: 1 + 3 + 10×1 = 12 inputs
ISD_INPUT="1\n3\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n"

# TruthfulQA: 1 + 3 + 20×1 = 22 inputs
TQA_INPUT="1\n3\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n"

# Conditional Safety: 1 + 3 + 20×1 = 22 inputs
CSAFE_INPUT="1\n3\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n"

# Length Control: 1 + 3 + 20×1 = 22 inputs
LCTRL_INPUT="1\n3\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n"

# AQI: 1 + 2 (Full, NOT Max) + 10×1 = 12 inputs
AQI_INPUT="1\n2\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n"

cd /workspace/finetuning_evaluation

# =============================================================================
# 1. ISD (Max mode) - 10 checkpoints
# =============================================================================
echo ""
echo "=============================================================================="
echo "[1/5] ISD EVALUATION (Max mode)"
echo "=============================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ISD..."

printf "$ISD_INPUT" | python -u "$EVAL_DIR/isd/evaluation.py" \
    --batch_size 8 \
    2>&1 | tee "$LOG_DIR/isd_auto_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ISD COMPLETED"
sleep 5

# =============================================================================
# 2. TRUTHFULQA (Max mode) - 20 checkpoints (honest + confident)
# =============================================================================
echo ""
echo "=============================================================================="
echo "[2/5] TRUTHFULQA EVALUATION (Max mode)"
echo "=============================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting TruthfulQA..."

printf "$TQA_INPUT" | python -u "$EVAL_DIR/truthfulqa/evaluation.py" \
    --batch_size 16 \
    2>&1 | tee "$LOG_DIR/truthfulqa_auto_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] TruthfulQA COMPLETED"
sleep 5

# =============================================================================
# 3. CONDITIONAL SAFETY (Max mode) - 20 checkpoints (strict + permissive)
# =============================================================================
echo ""
echo "=============================================================================="
echo "[3/5] CONDITIONAL SAFETY EVALUATION (Max mode)"
echo "=============================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Conditional Safety..."

printf "$CSAFE_INPUT" | python -u "$EVAL_DIR/conditional_safety/evaluation.py" \
    --batch_size 32 \
    2>&1 | tee "$LOG_DIR/conditional_safety_auto_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Conditional Safety COMPLETED"
sleep 5

# =============================================================================
# 4. LENGTH CONTROL (Max mode) - 20 checkpoints (concise + detailed)
# =============================================================================
echo ""
echo "=============================================================================="
echo "[4/5] LENGTH CONTROL EVALUATION (Max mode)"
echo "=============================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Length Control..."

printf "$LCTRL_INPUT" | python -u "$EVAL_DIR/length_control/evaluation.py" \
    --batch_size 32 \
    2>&1 | tee "$LOG_DIR/length_control_auto_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Length Control COMPLETED"
sleep 5

# =============================================================================
# 5. AQI (Full mode - NOT Max) - 10 checkpoints
# =============================================================================
echo ""
echo "=============================================================================="
echo "[5/5] AQI EVALUATION (Full mode)"
echo "=============================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting AQI..."

printf "$AQI_INPUT" | python -u "$EVAL_DIR/AQI/evaluation.py" \
    --batch_size 16 \
    2>&1 | tee "$LOG_DIR/aqi_auto_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] AQI COMPLETED"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=============================================================================="
echo "ALL 5 EVALUATIONS COMPLETE!"
echo "=============================================================================="
echo "Finished at: $(date)"
echo ""
echo "Logs saved to:"
echo "  - $MASTER_LOG (this log)"
echo "  - $LOG_DIR/isd_auto_${TIMESTAMP}.log"
echo "  - $LOG_DIR/truthfulqa_auto_${TIMESTAMP}.log"
echo "  - $LOG_DIR/conditional_safety_auto_${TIMESTAMP}.log"
echo "  - $LOG_DIR/length_control_auto_${TIMESTAMP}.log"
echo "  - $LOG_DIR/aqi_auto_${TIMESTAMP}.log"
echo "=============================================================================="
