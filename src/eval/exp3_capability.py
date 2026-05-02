"""Capability preservation eval (MMLU / GSM8K / HumanEval) via lm-eval-harness. GPU-only.

    # Sanity (24 GB GPU, ~10 min): MMLU only, 50 items, 2 methods
    python -u src/eval/exp3_capability.py --mode sanity 2>&1 | tee logs/rebuttal/exp3_sanity.log

    # Full (96 GB GPU, ~10h): 3 tasks × 11 methods, full datasets
    python -u src/eval/exp3_capability.py --mode full 2>&1 | tee logs/rebuttal/exp3_full.log

Tests whether LoRA alignment training degraded base Llama-3.1-8B capabilities.
Loops methods × tasks; saves per-cell JSON; aggregates a summary CSV + bar plot.
"""

import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.eval.model_loader import MODELS  # noqa: E402


def load_lm_eval():
    """Lazy-import lm-eval (slow import) only when actually needed."""
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    return evaluator, HFLM


def get_hf_repo(method_key: str) -> str | None:
    """Resolve method key → HF repo (None for base)."""
    if method_key == "base":
        return None
    if method_key not in MODELS:
        raise ValueError(f"Unknown method: {method_key}. Valid: {list(MODELS.keys())}")
    return MODELS[method_key]["hf_repo"]


def run_one(method: str, tasks: list, num_fewshot: int, limit, batch_size: int,
            base_model: str, output_dir: Path, evaluator, HFLM):
    """Run lm-eval on one method × task list. Saves per-task JSON; returns summary dict."""
    import torch

    hf_repo = get_hf_repo(method)
    print(f"\n{'='*80}")
    print(f"📦 Loading {method}  ({hf_repo or 'base, no adapter'})")
    print(f"{'='*80}")

    t0 = time.time()
    if hf_repo is None:
        model = HFLM(pretrained=base_model, dtype="bfloat16",
                     batch_size=batch_size, attn_implementation="flash_attention_2")
    else:
        model = HFLM(pretrained=base_model, peft=hf_repo, dtype="bfloat16",
                     batch_size=batch_size, attn_implementation="flash_attention_2")
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    method_results = {}
    for task in tasks:
        t1 = time.time()
        print(f"\n  → {method} on {task} (n_few={num_fewshot}, limit={limit})...")
        try:
            res = evaluator.simple_evaluate(
                model=model,
                tasks=[task],
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                limit=limit,
            )
            elapsed = time.time() - t1
            print(f"  ✓ {task} done in {elapsed:.1f}s")

            # Save per-cell JSON
            cell_path = output_dir / f"{method}_{task}.json"
            with open(cell_path, "w") as f:
                # lm-eval results object has nested numpy/torch types — extract scalar metrics
                summary = {
                    "method": method,
                    "task": task,
                    "num_fewshot": num_fewshot,
                    "limit": limit,
                    "elapsed_s": elapsed,
                    "results": {k: {kk: float(vv) if hasattr(vv, "item") else vv
                                    for kk, vv in v.items()}
                                for k, v in res["results"].items()},
                    "n_samples": res.get("n-samples", {}),
                }
                json.dump(summary, f, indent=2, default=str)
            method_results[task] = summary["results"]
        except Exception as e:
            print(f"  ❌ {method} on {task} FAILED: {type(e).__name__}: {e}")
            method_results[task] = {"error": str(e)}

    # Free GPU before next method
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(2)

    return method_results


def write_summary_table(all_results: dict, output_dir: Path, mode: str):
    """Aggregate (method, task) → primary metric into a summary CSV + Markdown table."""
    import csv

    # lm-eval primary metric per task (the "headline" number)
    primary_metric = {
        "mmlu": "acc,none",
        "gsm8k": "exact_match,strict-match",
        "humaneval": "pass@1,create_test",
    }

    methods = list(all_results.keys())
    tasks = sorted({t for m in methods for t in all_results[m].keys()})

    # Try to find an actually-present metric per task (lm-eval names vary by version)
    def get_metric(task_results, task_name):
        if "error" in task_results:
            return None
        # First key inside task_results is usually the per-subset key (e.g. "mmlu")
        # or directly contains numeric metrics
        for subkey, metrics in task_results.items():
            if not isinstance(metrics, dict):
                continue
            preferred = primary_metric.get(task_name, "acc,none")
            if preferred in metrics:
                return metrics[preferred]
            # Fallback: first numeric metric that doesn't end in _stderr
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and "stderr" not in k:
                    return v
        return None

    # Write CSV
    csv_path = output_dir / f"summary_{mode}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + tasks)
        for method in methods:
            row = [method]
            for task in tasks:
                val = get_metric(all_results[method].get(task, {}), task)
                row.append(f"{val:.4f}" if val is not None else "—")
            writer.writerow(row)

    # Write Markdown
    md_lines = ["| Method | " + " | ".join(tasks) + " |",
                "|" + "|".join(["---"] * (len(tasks) + 1)) + "|"]
    for method in methods:
        row = [method]
        for task in tasks:
            val = get_metric(all_results[method].get(task, {}), task)
            row.append(f"{val*100:.1f}" if val is not None else "—")
        md_lines.append("| " + " | ".join(row) + " |")
    md_path = output_dir / f"rebuttal_table_{mode}.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"\n📋 Summary CSV → {csv_path}")
    print(f"📋 Markdown   → {md_path}")
    print("\n" + "\n".join(md_lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sanity", "full"], required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load((project_root / "configs/rebuttal/exp3.yaml").read_text())
    mode_cfg = cfg[args.mode]
    output_dir = project_root / cfg["output_dir"] / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎯 Mode: {args.mode}")
    print(f"   tasks:        {mode_cfg['tasks']}")
    print(f"   num_fewshot:  {mode_cfg['num_fewshot']}")
    print(f"   limit:        {mode_cfg['limit']}")
    print(f"   methods:      {mode_cfg['methods']}")
    print(f"   batch_size:   {mode_cfg['batch_size']}")
    print(f"   output_dir:   {output_dir}")

    # Late import so --help is fast
    print("\n📦 Importing lm_eval (slow first-time)...")
    evaluator, HFLM = load_lm_eval()

    all_results = {}
    for method in mode_cfg["methods"]:
        all_results[method] = run_one(
            method=method,
            tasks=mode_cfg["tasks"],
            num_fewshot=mode_cfg["num_fewshot"],
            limit=mode_cfg["limit"],
            batch_size=mode_cfg["batch_size"],
            base_model=cfg["base_model"],
            output_dir=output_dir,
            evaluator=evaluator,
            HFLM=HFLM,
        )

    # Aggregate
    write_summary_table(all_results, output_dir, args.mode)

    print(f"\n✅ Exp 3 [{args.mode}] complete. Outputs in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            print(f"   {f.name:48s} {f.stat().st_size / 1024:>8.1f} KB")


if __name__ == "__main__":
    main()
