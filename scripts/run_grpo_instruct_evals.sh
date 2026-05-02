#!/bin/bash
# =============================================================================
# GRPO_Instruct MISSING EVALUATIONS
# =============================================================================
# Runs 4 missing evaluations for GRPO_Instruct:
#   1. TruthfulQA (HONEST + CONFIDENT)
#   2. Conditional Safety (STRICT + PERMISSIVE)
#   3. ISD (Instruction-Switch Dataset)
#   4. Length Control (CONCISE + DETAILED)
#
# AQI already completed via run_remaining_evals.sh
# =============================================================================
#
# USAGE:
#   nohup /workspace/finetuning_evaluation/comparative_study/05_evaluation/run_grpo_instruct_evals.sh &
#
# =============================================================================

set -e  # Exit on error

LOG_DIR="/workspace/finetuning_evaluation/logs"
EVAL_DIR="/workspace/cita_ecliptica/src/eval"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/grpo_instruct_evals_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

# Redirect all output to log file AND terminal
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=============================================================================="
echo "GRPO_Instruct MISSING EVALUATIONS"
echo "=============================================================================="
echo "Started at: $(date)"
echo "Master log: $MASTER_LOG"
echo "Model: GRPO_Instruct"
echo "Evaluations: TruthfulQA, Conditional Safety, ISD, Length Control"
echo "=============================================================================="

cd /workspace/finetuning_evaluation

# =============================================================================
# INPUT SEQUENCE FOR INTERACTIVE MENUS
# =============================================================================
# Menu 1 - Cached data: 2 = Re-calculate metrics + plots (keep inference)
# Menu 2 - Mode: 3 = Max
# Extra 1s for any checkpoint resume prompts (safety buffer)
# =============================================================================
MAX_INPUT="2\n3\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n1\n"

# =============================================================================
# 1. TruthfulQA
# =============================================================================
echo ""
echo "=============================================================================="
echo "[1/4] TRUTHFULQA EVALUATION"
echo "=============================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting TruthfulQA..."

printf "$MAX_INPUT" | python -u "$EVAL_DIR/truthfulqa.py" \
    --models GRPO_Instruct \
    --batch_size 32 \
    2>&1 | tee "$LOG_DIR/truthfulqa_grpo_instruct_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] TruthfulQA COMPLETED"
sleep 5

# =============================================================================
# 2. Conditional Safety
# =============================================================================
echo ""
echo "=============================================================================="
echo "[2/4] CONDITIONAL SAFETY EVALUATION"
echo "=============================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Conditional Safety..."

printf "$MAX_INPUT" | python -u "$EVAL_DIR/conditional_safety.py" \
    --models GRPO_Instruct \
    --batch_size 32 \
    2>&1 | tee "$LOG_DIR/conditional_safety_grpo_instruct_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Conditional Safety COMPLETED"
sleep 5

# =============================================================================
# 3. ISD
# =============================================================================
echo ""
echo "=============================================================================="
echo "[3/4] ISD EVALUATION"
echo "=============================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ISD..."

printf "$MAX_INPUT" | python -u "$EVAL_DIR/isd.py" \
    --models GRPO_Instruct \
    --batch_size 32 \
    2>&1 | tee "$LOG_DIR/isd_grpo_instruct_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ISD COMPLETED"
sleep 5

# =============================================================================
# 4. Length Control
# =============================================================================
echo ""
echo "=============================================================================="
echo "[4/4] LENGTH CONTROL EVALUATION"
echo "=============================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Length Control..."

printf "$MAX_INPUT" | python -u "$EVAL_DIR/length_control.py" \
    --models GRPO_Instruct \
    --batch_size 32 \
    2>&1 | tee "$LOG_DIR/length_control_grpo_instruct_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Length Control COMPLETED"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=============================================================================="
echo "ALL GRPO_Instruct EVALUATIONS COMPLETE!"
echo "=============================================================================="
echo "Finished at: $(date)"
echo ""
echo "Logs saved to:"
echo "  - $MASTER_LOG (master log)"
echo "  - $LOG_DIR/truthfulqa_grpo_instruct_${TIMESTAMP}.log"
echo "  - $LOG_DIR/conditional_safety_grpo_instruct_${TIMESTAMP}.log"
echo "  - $LOG_DIR/isd_grpo_instruct_${TIMESTAMP}.log"
echo "  - $LOG_DIR/length_control_grpo_instruct_${TIMESTAMP}.log"
echo "=============================================================================="
