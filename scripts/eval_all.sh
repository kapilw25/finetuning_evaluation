#!/bin/bash
# ============================================================================
# Run all 5 benchmarks × all 10 trained checkpoints sequentially.
# Wrapper around src/eval/{aqi,conditional_safety,isd,length_control,truthfulqa}.py.
# ============================================================================
# Usage:
#   mkdir -p logs && ./scripts/eval_all.sh sanity 2>&1 | tee logs/eval_all_sanity.log
#   mkdir -p logs && ./scripts/eval_all.sh full   2>&1 | tee logs/eval_all_full.log
# ============================================================================

set -e
MODE="${1:-sanity}"
if [ "$MODE" != "sanity" ] && [ "$MODE" != "full" ]; then
  echo "Usage: $0 {sanity|full}"; exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source venv_CITA/bin/activate

BENCHMARKS=(aqi conditional_safety isd length_control truthfulqa)

for BENCH in "${BENCHMARKS[@]}"; do
  echo "============================================"
  echo "[$MODE] Running $BENCH evaluation..."
  echo "============================================"
  python -u "src/eval/${BENCH}.py" --mode "$MODE" \
    2>&1 | tee "logs/eval_${BENCH}_${MODE}.log"
done

echo "All evaluations complete."
