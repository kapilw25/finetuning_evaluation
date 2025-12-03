"""
Evaluation Progress Monitor
Extracts ETAs and progress metrics from evaluation log files.

Usage:
    python unit_test/eval_progress_monitor.py
"""

import re
import os
from pathlib import Path
from datetime import datetime, timedelta

LOGS_DIR = Path(__file__).parent.parent / "logs"

# Log files to monitor
LOG_FILES = {
    "ISD": "ISD_evaluation_embedding_training_20251202_170136.log",
    "TruthfulQA": "truthfulqa_evaluation_training_20251202_170613.log",
    "Conditional Safety": "conditional_safety_evaluation_training_20251202_171039.log",
    "Length Control": "length_control_evaluation_training_*.log",  # wildcard
    "AQI": "aqi_evaluation_training_20251202_222133.log",
}

# Expected totals per eval
EVAL_CONFIG = {
    "ISD": {"batches_per_model": 375, "models": 6, "variants": 1, "samples": 3000},
    "TruthfulQA": {"batches_per_model": 103, "models": 6, "variants": 2, "samples": 1634},
    "Conditional Safety": {"batches_per_model": 153, "models": 6, "variants": 2, "samples": 2444},
    "Length Control": {"batches_per_model": 101, "models": 6, "variants": 2, "samples": 1610},
    "AQI": {"batches_per_model": 350, "models": 6, "variants": 1, "samples": 2800},  # Full mode
}


def find_log_file(pattern: str) -> Path | None:
    """Find log file matching pattern (supports wildcards)."""
    if "*" in pattern:
        import glob
        matches = sorted(glob.glob(str(LOGS_DIR / pattern)), reverse=True)
        return Path(matches[0]) if matches else None
    else:
        path = LOGS_DIR / pattern
        return path if path.exists() else None


def parse_log_file(log_path: Path) -> dict:
    """Parse evaluation log file and extract progress metrics."""
    if not log_path or not log_path.exists():
        return {"status": "NOT_FOUND", "file": str(log_path)}

    with open(log_path, 'r', errors='ignore') as f:
        content = f.read()

    result = {
        "file": log_path.name,
        "status": "RUNNING",
        "current_model": None,
        "current_variant": None,
        "batch_progress": None,
        "batch_total": None,
        "batch_speed": None,
        "eta_remaining": None,
        "checkpoints_saved": 0,
        "models_completed": [],
        "last_update": None,
    }

    # Check if evaluation is complete (different evals use different markers)
    if "EVALUATION COMPLETE" in content or "Results saved to:" in content or "Evaluation Complete" in content:
        result["status"] = "COMPLETED"

    # Find all checkpoint saves (handles both model/variant and model-only formats)
    # Pattern 1: ✅ Checkpoint COMPLETED: Model/Variant (count/total)
    checkpoint_pattern_variant = r"✅ Checkpoint COMPLETED: (\w+)/(\w+)"
    # Pattern 2: ✅ Checkpoint COMPLETED: Model (count/total) - ISD/AQI format
    checkpoint_pattern_model = r"✅ Checkpoint COMPLETED: (\w+) \(\d+/\d+\)"
    # Pattern 3: ✅ INFERENCE ALREADY COMPLETED for Model/Variant - cached/resumed
    checkpoint_pattern_cached = r"✅ INFERENCE ALREADY COMPLETED for (\w+)/(\w+)"

    checkpoints_variant = re.findall(checkpoint_pattern_variant, content)
    checkpoints_cached = re.findall(checkpoint_pattern_cached, content)
    checkpoints_model = re.findall(checkpoint_pattern_model, content)

    # Combine variant checkpoints (both new and cached)
    all_variant_checkpoints = list(set(checkpoints_variant + checkpoints_cached))
    result["models_completed"] = all_variant_checkpoints
    result["models_completed_count"] = len(set(m for m, v in all_variant_checkpoints)) + len(set(checkpoints_model))

    # Count checkpoint saves
    result["checkpoints_saved"] = content.count("💾 Checkpoint saved:")

    # Find latest progress bar (last line with progress)
    # Pattern: ModelName/Variant: XX%|...| batch/total [time<eta, speed]
    progress_patterns = [
        # Standard format: Model/Variant: 45%|...| 69/153 [17:25<21:13, 15.16s/it]
        (r"(\w+)/(\w+):\s+(\d+)%\|[^|]+\|\s*(\d+)/(\d+)\s+\[[\d:]+<([\d:]+),\s*([\d.]+)s/it\]", "variant"),
        # ISD format: Evaluating Model: 45%|...| 153/375 [1:05:12<1:33:57, 25.39s/it]
        (r"Evaluating (\w+(?:\s+\w+)?):\s+(\d+)%\|[^|]+\|\s*(\d+)/(\d+)\s+\[[\d:]+<([\d:]+),\s*([\d.]+)s/it\]", "isd"),
        # AQI format: Generating (SFT_NoInstruct): 4%|...|  29/700 [04:04<1:34:16, 8.43s/it]
        (r"Generating \((\w+)\):\s+(\d+)%\|[^|]+\|\s*(\d+)/(\d+)\s+\[[\d:]+<([\d:]+),\s*([\d.]+)s/it\]", "aqi"),
    ]

    lines = content.split('\n')
    for line in reversed(lines):
        for pattern, fmt_type in progress_patterns:
            match = re.search(pattern, line)
            if match:
                if fmt_type == "variant":
                    result["current_model"] = match.group(1)
                    result["current_variant"] = match.group(2)
                    result["batch_progress"] = int(match.group(4))
                    result["batch_total"] = int(match.group(5))
                    result["eta_remaining"] = match.group(6)
                    result["batch_speed"] = float(match.group(7))
                    result["percent"] = int(match.group(3))
                elif fmt_type == "isd":
                    result["current_model"] = match.group(1).replace(" ", "_")
                    result["current_variant"] = None
                    result["batch_progress"] = int(match.group(3))
                    result["batch_total"] = int(match.group(4))
                    result["eta_remaining"] = match.group(5)
                    result["batch_speed"] = float(match.group(6))
                    result["percent"] = int(match.group(2))
                elif fmt_type == "aqi":
                    result["current_model"] = match.group(1)
                    result["current_variant"] = None
                    result["batch_progress"] = int(match.group(3))
                    result["batch_total"] = int(match.group(4))
                    result["eta_remaining"] = match.group(5)
                    result["batch_speed"] = float(match.group(6))
                    result["percent"] = int(match.group(2))
                break
        if result["batch_progress"] is not None:
            break

    # Get file modification time
    result["last_update"] = datetime.fromtimestamp(log_path.stat().st_mtime)

    return result


def parse_eta(eta_str: str) -> timedelta:
    """Parse ETA string like '1:33:57' or '21:13' to timedelta."""
    if not eta_str:
        return timedelta(0)

    parts = eta_str.split(':')
    if len(parts) == 3:
        return timedelta(hours=int(parts[0]), minutes=int(parts[1]), seconds=int(parts[2]))
    elif len(parts) == 2:
        return timedelta(minutes=int(parts[0]), seconds=int(parts[1]))
    return timedelta(0)


def calculate_total_eta(result: dict, config: dict) -> str:
    """Calculate total ETA based on current progress and config."""
    if result["status"] == "COMPLETED":
        return "DONE"
    if result["status"] == "NOT_FOUND":
        return "N/A"
    if not result["batch_progress"] or not result["batch_speed"]:
        return "Unknown"

    # Current model progress
    current_batch = result["batch_progress"]
    total_batches = result["batch_total"]
    remaining_current = total_batches - current_batch

    # If current batch == total, this model is done - check if it's the last model
    if remaining_current <= 0:
        # Model inference done, likely computing metrics
        remaining_current = 0

    # Remaining models
    total_models = config["models"]
    variants = config["variants"]
    current_model = result.get("current_model")

    # If we have variant info (TruthfulQA, Conditional Safety, Length Control)
    if result["current_variant"]:
        # Count how many variants of current model are already complete
        variants_completed_for_current = sum(
            1 for m, v in result.get("models_completed", []) if m == current_model
        )
        remaining_variants_this_model = variants - variants_completed_for_current - 1  # -1 for current

        # Count fully completed models (all variants done)
        # A model is fully complete if it has `variants` number of completed checkpoints
        model_variant_counts = {}
        for m, v in result.get("models_completed", []):
            model_variant_counts[m] = model_variant_counts.get(m, 0) + 1

        fully_completed_models = sum(1 for m, count in model_variant_counts.items() if count >= variants)
        remaining_models = max(0, total_models - fully_completed_models - 1)  # -1 for current model, clamp to 0
    else:
        # ISD/AQI format - no variants
        remaining_variants_this_model = 0
        models_done = result.get("models_completed_count", 0)
        remaining_models = max(0, total_models - models_done - 1)  # -1 for current model, clamp to 0

    # Calculate remaining time
    speed = result["batch_speed"]  # seconds per batch

    # Time for current variant
    time_current = remaining_current * speed

    # Time for remaining variants of current model
    time_remaining_variants = remaining_variants_this_model * total_batches * speed

    # Time for remaining models
    time_remaining_models = remaining_models * variants * total_batches * speed

    total_seconds = time_current + time_remaining_variants + time_remaining_models

    # If all inference done but status not COMPLETED, it's computing metrics
    if total_seconds <= 0:
        return "~Computing metrics"

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)

    return f"~{hours}h {minutes}m"


def print_progress_report():
    """Print progress report for all evaluations."""
    print("\n" + "=" * 80)
    print(f"EVALUATION PROGRESS MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    results = []

    for eval_name, log_pattern in LOG_FILES.items():
        log_path = find_log_file(log_pattern)
        result = parse_log_file(log_path)
        result["eval_name"] = eval_name
        result["config"] = EVAL_CONFIG.get(eval_name, {})
        results.append(result)

    # Print table header
    print(f"\n{'Eval':<20} {'Status':<12} {'Model':<20} {'Progress':<15} {'Speed':<12} {'ETA Total':<12}")
    print("-" * 95)

    for r in results:
        eval_name = r["eval_name"]
        status = r["status"]

        if status == "NOT_FOUND":
            print(f"{eval_name:<20} {'NOT FOUND':<12} {'-':<20} {'-':<15} {'-':<12} {'N/A':<12}")
            continue

        if status == "COMPLETED":
            print(f"{eval_name:<20} {'✅ DONE':<12} {'-':<20} {'100%':<15} {'-':<12} {'DONE':<12}")
            continue

        model = r.get("current_model", "?")
        variant = r.get("current_variant", "")
        model_str = f"{model}/{variant}" if variant else model

        batch_prog = r.get("batch_progress", 0)
        batch_total = r.get("batch_total", 0)
        percent = r.get("percent", 0)
        progress_str = f"{batch_prog}/{batch_total} ({percent}%)"

        speed = r.get("batch_speed")
        speed_str = f"{speed:.1f}s/it" if speed else "-"

        total_eta = calculate_total_eta(r, r["config"])
        total_eta = total_eta if total_eta else "Unknown"

        print(f"{eval_name:<20} {'🔄 RUNNING':<12} {model_str:<20} {progress_str:<15} {speed_str:<12} {total_eta:<12}")

    # Print details
    print("\n" + "-" * 80)
    print("DETAILED PROGRESS:")
    print("-" * 80)

    for r in results:
        if r["status"] in ["NOT_FOUND", "COMPLETED"]:
            continue

        eval_name = r["eval_name"]
        config = r["config"]

        models_completed = r.get("models_completed_count", len(set(m for m, v in r.get("models_completed", []))))
        total_models = config.get("models", 6)

        print(f"\n{eval_name}:")
        print(f"  File: {r.get('file', 'N/A')}")
        print(f"  Last Update: {r.get('last_update', 'N/A')}")
        print(f"  Models Completed: {models_completed}/{total_models}")
        print(f"  Checkpoints Saved: {r.get('checkpoints_saved', 0)}")
        print(f"  Current: {r.get('current_model', '?')}/{r.get('current_variant', '?')}")
        print(f"  Batch: {r.get('batch_progress', '?')}/{r.get('batch_total', '?')}")
        print(f"  ETA (current variant): {r.get('eta_remaining', 'N/A')}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)

    running = [r for r in results if r["status"] == "RUNNING"]
    completed = [r for r in results if r["status"] == "COMPLETED"]
    not_found = [r for r in results if r["status"] == "NOT_FOUND"]

    print(f"  Running: {len(running)}")
    print(f"  Completed: {len(completed)}")
    print(f"  Not Found: {len(not_found)}")

    if running:
        # Find bottleneck
        max_eta = timedelta(0)
        bottleneck = None
        for r in running:
            eta = calculate_total_eta(r, r["config"])
            if "h" in eta:
                hours = int(eta.split("h")[0].replace("~", ""))
                mins = int(eta.split("h")[1].replace("m", "").strip())
                td = timedelta(hours=hours, minutes=mins)
                if td > max_eta:
                    max_eta = td
                    bottleneck = r["eval_name"]

        if bottleneck:
            print(f"\n  🚨 BOTTLENECK: {bottleneck} (ETA: {calculate_total_eta([r for r in running if r['eval_name'] == bottleneck][0], EVAL_CONFIG[bottleneck])})")


if __name__ == "__main__":
    print_progress_report()
