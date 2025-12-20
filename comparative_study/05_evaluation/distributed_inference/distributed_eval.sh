#!/bin/bash
#
# Distributed Evaluation Runner for 4x A100-40GB Nodes
#
# Splits evaluation dataset across nodes, each node processes its shard.
# After all nodes complete, merge results.
#
# Usage:
#   # On each node (0-3):
#   ./distributed_eval.sh --eval truthfulqa --nodes 4 --node-id 0 --mode full
#   ./distributed_eval.sh --eval truthfulqa --nodes 4 --node-id 1 --mode full
#   ./distributed_eval.sh --eval truthfulqa --nodes 4 --node-id 2 --mode full
#   ./distributed_eval.sh --eval truthfulqa --nodes 4 --node-id 3 --mode full
#
#   # After all complete, merge on any node:
#   ./distributed_eval.sh --eval truthfulqa --nodes 4 --merge
#
# Evaluations: truthfulqa, conditional_safety, length_control, aqi, isd
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV_PATH="$PROJECT_ROOT/venv_CITA"

# Defaults
EVAL_TYPE=""
TOTAL_NODES=4
NODE_ID=""
MODE="sanity"
MODELS=""
MERGE_ONLY=false
USE_LLM=false

print_usage() {
    cat << EOF
Distributed Evaluation for 4x A100-40GB Nodes

Usage: $0 --eval <type> --nodes <n> --node-id <id> [options]

Required:
  --eval <type>       truthfulqa | conditional_safety | length_control | aqi | isd
  --nodes <n>         Total nodes (default: 4)
  --node-id <id>      This node's ID (0 to n-1)

Options:
  --mode <mode>       sanity | full (default: sanity)
  --models <list>     Comma-separated models (default: all)
  --use-llm           Enable LLM judge
  --merge             Merge shards after all nodes complete

Examples:
  # Run on 4 nodes:
  Node 0: $0 --eval truthfulqa --nodes 4 --node-id 0 --mode full
  Node 1: $0 --eval truthfulqa --nodes 4 --node-id 1 --mode full
  Node 2: $0 --eval truthfulqa --nodes 4 --node-id 2 --mode full
  Node 3: $0 --eval truthfulqa --nodes 4 --node-id 3 --mode full

  # Merge after all complete:
  $0 --eval truthfulqa --nodes 4 --merge
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --eval)      EVAL_TYPE="$2"; shift 2 ;;
        --nodes)     TOTAL_NODES="$2"; shift 2 ;;
        --node-id)   NODE_ID="$2"; shift 2 ;;
        --mode)      MODE="$2"; shift 2 ;;
        --models)    MODELS="$2"; shift 2 ;;
        --use-llm)   USE_LLM=true; shift ;;
        --merge)     MERGE_ONLY=true; shift ;;
        -h|--help)   print_usage; exit 0 ;;
        *)           echo "Unknown: $1"; print_usage; exit 1 ;;
    esac
done

# Validate
if [[ -z "$EVAL_TYPE" ]]; then
    echo "Error: --eval required"
    print_usage
    exit 1
fi

# Map eval type to script
case $EVAL_TYPE in
    truthfulqa)         EVAL_SCRIPT="$EVAL_DIR/truthfulqa/evaluation.py" ;;
    conditional_safety) EVAL_SCRIPT="$EVAL_DIR/conditional_safety/evaluation.py" ;;
    length_control)     EVAL_SCRIPT="$EVAL_DIR/length_control/evaluation.py" ;;
    aqi)                EVAL_SCRIPT="$EVAL_DIR/AQI/evaluation.py" ;;
    isd)                EVAL_SCRIPT="$EVAL_DIR/isd/evaluation_embedding.py" ;;
    *)                  echo "Error: Unknown eval: $EVAL_TYPE"; exit 1 ;;
esac

# =============================================================================
# Merge Mode
# =============================================================================
if [[ "$MERGE_ONLY" == true ]]; then
    echo "=========================================="
    echo "  Merging $EVAL_TYPE Results ($TOTAL_NODES shards)"
    echo "=========================================="

    source "$VENV_PATH/bin/activate" 2>/dev/null || true

    python3 << 'PYEOF'
import sys
import json
from pathlib import Path
from collections import defaultdict

eval_type = "$EVAL_TYPE"
total_nodes = $TOTAL_NODES
project_root = Path("$PROJECT_ROOT")

output_dirs = {
    "truthfulqa": "TruthfulQA_Evaluation",
    "conditional_safety": "Conditional_Safety_Evaluation",
    "length_control": "Length_Control_Evaluation",
    "aqi": "AQI_Evaluation",
    "isd": "ISD_Evaluation",
}

output_dir = project_root / "outputs" / "evaluation" / output_dirs.get(eval_type, eval_type)
if not output_dir.exists():
    print(f"Error: {output_dir} not found")
    sys.exit(1)

shard_files = list(output_dir.glob("*_shard_*_of_*.json"))
print(f"Found {len(shard_files)} shard files")

if not shard_files:
    print("No shards to merge")
    sys.exit(0)

# Group by base name
groups = defaultdict(list)
for f in shard_files:
    base = f.stem.rsplit("_shard_", 1)[0]
    groups[base].append(f)

for base, files in groups.items():
    print(f"\nMerging: {base} ({len(files)} shards)")

    all_responses = []
    metadata = None

    for f in sorted(files):
        with open(f) as fp:
            data = json.load(fp)
            if metadata is None:
                metadata = {k: v for k, v in data.items() if k != "responses"}
            all_responses.extend(data.get("responses", []))

    merged = metadata or {}
    merged["responses"] = all_responses
    merged["total_samples"] = len(all_responses)

    out_file = output_dir / f"{base}_merged.json"
    with open(out_file, "w") as fp:
        json.dump(merged, fp, indent=2)
    print(f"  Saved: {out_file.name} ({len(all_responses)} samples)")

print("\n✅ Merge complete!")
PYEOF
    exit 0
fi

# =============================================================================
# Run Evaluation
# =============================================================================
if [[ -z "$NODE_ID" ]]; then
    echo "Error: --node-id required"
    exit 1
fi

echo "=========================================="
echo "  Node $NODE_ID/$TOTAL_NODES - $EVAL_TYPE"
echo "=========================================="
echo "Mode: $MODE"
echo "Script: $EVAL_SCRIPT"

source "$VENV_PATH/bin/activate" 2>/dev/null || true

# Build command
CMD="python $EVAL_SCRIPT --mode $MODE --shard $NODE_ID --total-shards $TOTAL_NODES"
[[ -n "$MODELS" ]] && CMD="$CMD --models $MODELS"
[[ "$USE_LLM" == true ]] && CMD="$CMD --use_llm"

echo "Running: $CMD"
echo "=========================================="
exec $CMD
