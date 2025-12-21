#!/bin/bash
# =============================================================================
# AUTOMATED EVALUATION RUNNER
# =============================================================================
# Runs: Conditional Safety, Length Control, AQI
# Waits for ISD to complete first if it's running
# Models: PPO_NoInstruct, PPO_Instruct, GRPO_NoInstruct (+ 6 cached models)
# =============================================================================
#
# USAGE:
#   nohup /workspace/finetuning_evaluation/comparative_study/05_evaluation/run_remaining_evals.sh &
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
echo "AUTOMATED EVALUATION RUNNER"
echo "=============================================================================="
echo "Started at: $(date)"
echo "Master log: $MASTER_LOG"
echo "=============================================================================="

# =============================================================================
# WAIT FOR ISD TO COMPLETE (if running)
# =============================================================================
if pgrep -f "isd/evaluation.py" > /dev/null; then
    echo ""
    echo "=============================================================================="
    echo "ISD EVALUATION DETECTED - WAITING FOR COMPLETION"
    echo "=============================================================================="

    while pgrep -f "isd/evaluation.py" > /dev/null; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ISD still running... checking again in 60s"
        sleep 60
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ISD evaluation completed!"
    echo "Waiting 10 seconds before starting next evaluation..."
    sleep 10
else
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No ISD evaluation running. Proceeding immediately."
fi

# =============================================================================
# INPUT SEQUENCES EXPLANATION:
# =============================================================================
# Menu 1 - Cached data: 2 = Re-calculate metrics + plots (keep inference)
# Menu 2 - Mode: 3 = Max, 2 = Full
# Menu 3 - Missing model: 1 = Continue (skip GRPO_Instruct)
# Menu 4-9 - Per-model checkpoint: 1 = Use cached (for 6 existing models)
#
# Sequence for Max mode: 2, 3, 1, 1, 1, 1, 1, 1, 1
# Sequence for Full mode: 2, 2, 1, 1, 1, 1, 1, 1, 1
# =============================================================================

# Generate input sequence
# Need: 1 (cached) + 1 (mode) + 1 (missing) + up to 12 (for 6 models × 2 variants) = 15 minimum
# Adding extra 1s for safety (some evals may have more prompts)
MAX_INPUT="2\n3\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n"
FULL_INPUT="2\n2\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n"

# =============================================================================
# 1. CONDITIONAL SAFETY (Max mode)
# =============================================================================
echo ""
echo "=============================================================================="
echo "[1/3] CONDITIONAL SAFETY EVALUATION (Max mode)"
echo "=============================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Conditional Safety..."

cd /workspace/finetuning_evaluation
printf "$MAX_INPUT" | python -u "$EVAL_DIR/conditional_safety/evaluation.py" \
    --batch_size 32 \
    2>&1 | tee "$LOG_DIR/conditional_safety_auto_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Conditional Safety COMPLETED"
sleep 5

# =============================================================================
# 2. LENGTH CONTROL (Max mode)
# =============================================================================
echo ""
echo "=============================================================================="
echo "[2/3] LENGTH CONTROL EVALUATION (Max mode)"
echo "=============================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Length Control..."

printf "$MAX_INPUT" | python -u "$EVAL_DIR/length_control/evaluation.py" \
    --batch_size 32 \
    2>&1 | tee "$LOG_DIR/length_control_auto_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Length Control COMPLETED"
sleep 5

# =============================================================================
# 3. AQI (Full mode - NOT Max)
# =============================================================================
echo ""
echo "=============================================================================="
echo "[3/3] AQI EVALUATION (Full mode)"
echo "=============================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting AQI..."

printf "$FULL_INPUT" | python -u "$EVAL_DIR/AQI/evaluation.py" \
    --batch_size 16 \
    2>&1 | tee "$LOG_DIR/aqi_auto_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] AQI COMPLETED"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=============================================================================="
echo "ALL EVALUATIONS COMPLETE!"
echo "=============================================================================="
echo "Finished at: $(date)"
echo ""
echo "Logs saved to:"
echo "  - $MASTER_LOG (this log)"
echo "  - $LOG_DIR/conditional_safety_auto_${TIMESTAMP}.log"
echo "  - $LOG_DIR/length_control_auto_${TIMESTAMP}.log"
echo "  - $LOG_DIR/aqi_auto_${TIMESTAMP}.log"
echo "=============================================================================="
