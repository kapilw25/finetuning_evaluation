#!/bin/bash
# ============================================================================
# Train all 5 methods × {Instruct, NoInstruct} sequentially.
# Skips methods that already have a final checkpoint in outputs/training/.
# ============================================================================
# Usage:
#   mkdir -p logs && ./scripts/train_all.sh sanity 2>&1 | tee logs/train_all_sanity.log
#   mkdir -p logs && ./scripts/train_all.sh full   2>&1 | tee logs/train_all_full.log
# ============================================================================

set -e
MODE="${1:-sanity}"
if [ "$MODE" != "sanity" ] && [ "$MODE" != "full" ]; then
  echo "Usage: $0 {sanity|full}"; exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source venv_CITA/bin/activate

# Order matters: SFT first (DPO/PPO/GRPO/CITA all init from SFT LoRA)
METHODS=(sft dpo ppo grpo cita)

for METHOD in "${METHODS[@]}"; do
  for INSTR in true false; do
    LABEL="${METHOD^^}_$([ "$INSTR" = "true" ] && echo "Instruct" || echo "NoInstruct")"
    echo "============================================"
    echo "[$MODE] Training $LABEL..."
    echo "============================================"
    python -u "src/train/${METHOD}.py" --mode "$MODE" --use-instruction "$INSTR" \
      2>&1 | tee "logs/train_${METHOD}_${INSTR}_${MODE}.log"
  done
done

echo "All training complete."
