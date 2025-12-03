#!/bin/bash
# Auto-terminate Lambda Cloud instances when all 3 evaluations complete
# Run with: nohup bash unit_test/auto_terminate_on_complete.sh > auto_terminate.log 2>&1 &

# Load API key from .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${PROJECT_ROOT}/.env"

if [[ -f "$ENV_FILE" ]]; then
    export $(grep -E "^LAMBDA_API_KEY=" "$ENV_FILE" | xargs)
else
    echo "ERROR: .env file not found at $ENV_FILE"
    exit 1
fi

if [[ -z "$LAMBDA_API_KEY" ]]; then
    echo "ERROR: LAMBDA_API_KEY not found in .env"
    exit 1
fi

# Instance IDs:
# Instance 1 (ISD + Claude session): 330e1ebf0d1e4c42ab7d28461509b7b8 - DO NOT TERMINATE (has Claude session)
# Instance 2 (AQI):                   TERMINATED
# Instance 3 (Length Control):        0217e4b5c0c7405eb9e55097741fc5e7
INSTANCE3_ID="0217e4b5c0c7405eb9e55097741fc5e7"
LOG_DIR="${PROJECT_ROOT}/logs"

echo "$(date): Starting auto-terminate for Instance3 only..."
echo "Checking: Length Control completion"

# Check for completion marker
LC_DONE=$(grep -c "Results saved to:" ${LOG_DIR}/length_control_evaluation_training_20251203_000621.log 2>/dev/null | tr -d '\n' || echo 0)
LC_DONE=${LC_DONE:-0}

echo "$(date): LC=${LC_DONE}"

if [[ "${LC_DONE}" -ge 1 ]]; then
    echo "$(date): ✅ LENGTH CONTROL COMPLETE! Terminating Instance3..."
    curl -s -X POST "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate" \
      -H "Authorization: Bearer ${LAMBDA_API_KEY}" \
      -H "Content-Type: application/json" \
      -d "{\"instance_ids\": [\"${INSTANCE3_ID}\"]}"
    echo ""
    echo "$(date): Instance3 terminate request sent. Done!"
    exit 0
else
    echo "$(date): Length Control not yet complete. Exiting."
    exit 1
fi
