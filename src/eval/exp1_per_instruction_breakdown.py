"""Per-instruction breakdown across 10 methods × 10 instructions with 95% BCa bootstrap CI. CPU-only.

    python -u src/eval/exp1_per_instruction_breakdown.py 2>&1 | tee logs/rebuttal/exp1.log

Re-scores each prompt's fidelity from cached responses (outputs/evaluation/ISD_Evaluation_Embedding/<METHOD>/<METHOD>_isd_responses.csv)
using sentence-transformer embeddings (matches src/utils/eval/isd_metrics.py logic), groups
by (method, instruction_type), computes BCa 95% CI per cell (CLAUDE.md §7.4), and emits:
JSON + CSV + Markdown table + grouped-bar plot (PNG + PDF).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.eval.isd_dataset import ISD_INSTRUCTIONS  # noqa: E402
from src.utils.eval.eclipta_scoring import (  # noqa: E402
    CHAR_DESCRIPTIONS,  # noqa: F401  (kept for backward-compat, used by helpers below)
    build_prototype_embeddings,
    bca_ci,
)


def score_method_responses(method: str, csv_path: Path, embedder, prototypes: dict) -> pd.DataFrame:
    """Re-score per-prompt fidelity for one method's responses."""
    df = pd.read_csv(csv_path)
    responses = df["response"].fillna("").astype(str).tolist()

    response_embs = embedder.encode(responses, batch_size=64, show_progress_bar=False)
    norms = np.linalg.norm(response_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    response_embs = response_embs / norms

    fidelities = np.empty(len(df), dtype=np.float32)
    for i, inst_type in enumerate(df["instruction_type"]):
        if inst_type not in prototypes:
            fidelities[i] = np.nan
            continue
        sim = float(np.dot(response_embs[i], prototypes[inst_type]))
        fidelities[i] = (sim + 1.0) / 2.0  # [-1,1] → [0,1]

    out = df[["prompt_id", "instruction_type"]].copy()
    out["fidelity"] = fidelities
    out["method"] = method
    return out[["method", "prompt_id", "instruction_type", "fidelity"]]


def emit_outputs(table: list, methods: list, plot_methods: list, output_dir: Path) -> None:
    """Write JSON, CSV, Markdown table, and grouped-bar plot."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON (CLAUDE.md schema)
    (output_dir / "per_instruction_breakdown.json").write_text(json.dumps(table, indent=2))

    # 2. CSV (paste-friendly)
    csv_rows = [{
        "method": r["method"],
        "instruction_type": r["instruction_type"],
        "n": r["n"],
        "mean": r["ci"]["mean"],
        "ci_lo": r["ci"]["ci_lo"],
        "ci_hi": r["ci"]["ci_hi"],
        "ci_half": r["ci"]["ci_half"],
    } for r in table]
    df_flat = pd.DataFrame(csv_rows)
    df_flat.to_csv(output_dir / "per_instruction_breakdown.csv", index=False)

    # 3. Pivot tables (instruction × method)
    inst_order = list(ISD_INSTRUCTIONS.keys())
    pivot_mean = df_flat.pivot(index="instruction_type", columns="method", values="mean").reindex(inst_order)
    pivot_ci   = df_flat.pivot(index="instruction_type", columns="method", values="ci_half").reindex(inst_order)

    # 4. Markdown table — Instruct family only (rebuttal-paste-ready)
    instruct_methods = [m for m in plot_methods if m in pivot_mean.columns]
    md_lines = ["| Instruction | " + " | ".join(instruct_methods) + " |"]
    md_lines.append("|" + "|".join(["---"] * (len(instruct_methods) + 1)) + "|")
    for inst in inst_order:
        if inst not in pivot_mean.index:
            continue
        row = [inst]
        for m in instruct_methods:
            mean = pivot_mean.loc[inst, m]
            ci   = pivot_ci.loc[inst, m]
            if pd.isna(mean):
                row.append("—")
            else:
                row.append(f"{mean:.3f}±{ci:.3f}")
        md_lines.append("| " + " | ".join(row) + " |")
    (output_dir / "rebuttal_table.md").write_text("\n".join(md_lines) + "\n")

    # 5. Grouped bar plot — dual format (PNG + PDF) per CLAUDE.md §7
    fig, ax = plt.subplots(figsize=(15, 5))
    x = np.arange(len(inst_order))
    width = 0.8 / max(len(instruct_methods), 1)
    colors = plt.get_cmap("tab10").colors

    for i, m in enumerate(instruct_methods):
        means = [pivot_mean.loc[inst, m] if inst in pivot_mean.index and not pd.isna(pivot_mean.loc[inst, m]) else 0
                 for inst in inst_order]
        cis   = [pivot_ci.loc[inst, m] if inst in pivot_ci.index and not pd.isna(pivot_ci.loc[inst, m]) else 0
                 for inst in inst_order]
        offset = (i - (len(instruct_methods) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=cis, capsize=2, label=m, color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(inst_order, rotation=30, ha="right")
    ax.set_ylabel("ECLIPTICA Fidelity (95% bootstrap CI)")
    ax.set_title("Per-Instruction Performance Breakdown — uniformity test (Instruct family)")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / "per_instruction_bars.png", dpi=200, bbox_inches="tight")
    plt.savefig(output_dir / "per_instruction_bars.pdf", bbox_inches="tight")
    plt.close()

    # Return pivots for the uniformity check
    return pivot_mean


def uniformity_check(pivot_mean: pd.DataFrame, ref: str, challenger: str, output_dir: Path) -> dict:
    """Reference vs challenger: how many instructions does ref beat challenger on?"""
    if ref not in pivot_mean.columns or challenger not in pivot_mean.columns:
        return {"ref": ref, "challenger": challenger, "wins": None, "total": None,
                "msg": f"missing column ({ref} or {challenger})"}
    df2 = pivot_mean[[ref, challenger]].dropna()
    wins = int((df2[ref] > df2[challenger]).sum())
    total = int(len(df2))
    summary = {"ref": ref, "challenger": challenger, "wins": wins, "total": total,
               "msg": f"{ref} beats {challenger} on {wins}/{total} instruction types"}
    (output_dir / "uniformity_check.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    cfg_path = project_root / "configs" / "rebuttal" / "exp1.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    methods = cfg["methods"]
    plot_methods = cfg["plot_methods"]
    data_dir = project_root / cfg["data_dir"]
    output_dir = project_root / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading embedder: {cfg['embedding_model']}")
    embedder = SentenceTransformer(cfg["embedding_model"])
    prototypes = build_prototype_embeddings(embedder, ISD_INSTRUCTIONS)
    print(f"✅ Built {len(prototypes)} prototype embeddings")

    # 1. Score every method's responses
    all_dfs = []
    for method in tqdm(methods, desc="methods"):
        csv_path = data_dir / method / cfg["responses_filename_pattern"].format(method=method)
        if not csv_path.exists():
            print(f"⚠️  {csv_path} missing — skipping {method}")
            continue
        df = score_method_responses(method, csv_path, embedder, prototypes)
        all_dfs.append(df)
        print(f"  scored {method}: {len(df)} rows, mean fidelity={df['fidelity'].mean():.3f}")

    if not all_dfs:
        print("❌ No methods scored. Aborting.")
        sys.exit(1)

    results = pd.concat(all_dfs, ignore_index=True)
    results.to_csv(output_dir / "per_prompt_fidelities.csv", index=False)

    # 2. Bootstrap BCa CI per cell
    print(f"\n🎲 Bootstrap BCa CI (n={cfg['n_bootstrap']}, conf={cfg['confidence_level']})...")
    table = []
    for method in methods:
        for inst_type in ISD_INSTRUCTIONS.keys():
            sub = results[(results.method == method) & (results.instruction_type == inst_type)]
            if len(sub) == 0:
                continue
            ci = bca_ci(sub["fidelity"].values,
                        n_bootstrap=cfg["n_bootstrap"],
                        confidence=cfg["confidence_level"],
                        seed=cfg["seed"])
            table.append({"method": method, "instruction_type": inst_type, "n": int(len(sub)), "ci": ci})

    # 3. Emit outputs
    pivot_mean = emit_outputs(table, methods, plot_methods, output_dir)

    # 4. Uniformity check
    summary = uniformity_check(pivot_mean,
                               ref=cfg["reference_method"],
                               challenger=cfg["challenger_method"],
                               output_dir=output_dir)
    print(f"\n🎯 {summary['msg']}")

    print(f"\n✅ Exp 1 complete. Outputs in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"   {f.name:48s} {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()
