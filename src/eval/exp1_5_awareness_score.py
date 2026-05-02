"""Per-instruction instruction_awareness_score = fidelity × shift, with 95% BCa CI. CPU-only.

    python -u src/eval/exp1_5_awareness_score.py 2>&1 | tee logs/rebuttal/exp1_5.log

This is the metric where CITA's headline win lives (paper: CITA 0.37 vs DPO 0.25).
Per-prompt awareness(P, X) = fidelity(P, X) × shift(P, X), where shift is the cosine
distance of response embedding from the per-prompt centroid (deviation across instructions).
Reuses fidelities from Exp 1; computes shift fresh from cached responses.
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
from src.utils.eval.eclipta_scoring import bca_ci  # noqa: E402


def compute_shift_per_row(method: str, csv_path: Path, embedder, save_dir: Path = None) -> pd.DataFrame:
    """Compute per-(prompt, instruction) shift = 1 - cos(response_emb, prompt_centroid).

    Returns DataFrame with columns: method, prompt_id, instruction_type, shift.
    """
    df = pd.read_csv(csv_path)
    responses = df["response"].fillna("").astype(str).tolist()

    # Embed all 3000 responses once
    embs = embedder.encode(responses, batch_size=64, show_progress_bar=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms  # (3000, dim) normalized

    # Optional: cache embeddings for downstream experiments
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / f"embeddings_{method}.npy", embs)

    # Per-prompt centroid (vectorized)
    prompt_ids = df["prompt_id"].values
    unique_prompts = np.sort(np.unique(prompt_ids))
    p2idx = {p: i for i, p in enumerate(unique_prompts)}

    centroids = np.zeros((len(unique_prompts), embs.shape[1]), dtype=np.float32)
    counts = np.zeros(len(unique_prompts), dtype=np.int32)
    for i, p in enumerate(prompt_ids):
        ci = p2idx[p]
        centroids[ci] += embs[i]
        counts[ci] += 1
    centroids = centroids / counts[:, None]
    cent_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    cent_norms[cent_norms == 0] = 1.0
    centroids = centroids / cent_norms  # normalize

    # shift(P, X) = 1 - cos(emb, centroid_P)
    row_centroids = centroids[[p2idx[p] for p in prompt_ids]]  # (3000, dim)
    cos_sims = np.sum(embs * row_centroids, axis=1)  # (3000,)
    shifts = 1.0 - cos_sims

    out = df[["prompt_id", "instruction_type"]].copy()
    out["shift"] = shifts.astype(np.float32)
    out["method"] = method
    return out[["method", "prompt_id", "instruction_type", "shift"]]


def emit_outputs(table: list, methods: list, plot_methods: list, output_dir: Path,
                 metric_name: str, metric_label: str) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / f"{metric_name}_breakdown.json").write_text(json.dumps(table, indent=2))

    rows = [{
        "method": r["method"],
        "instruction_type": r["instruction_type"],
        "n": r["n"],
        "mean": r["ci"]["mean"],
        "ci_lo": r["ci"]["ci_lo"],
        "ci_hi": r["ci"]["ci_hi"],
        "ci_half": r["ci"]["ci_half"],
    } for r in table]
    df_flat = pd.DataFrame(rows)
    df_flat.to_csv(output_dir / f"{metric_name}_breakdown.csv", index=False)

    inst_order = list(ISD_INSTRUCTIONS.keys())
    pivot_mean = df_flat.pivot(index="instruction_type", columns="method", values="mean").reindex(inst_order)
    pivot_ci   = df_flat.pivot(index="instruction_type", columns="method", values="ci_half").reindex(inst_order)

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
            row.append("—" if pd.isna(mean) else f"{mean:.3f}±{ci:.3f}")
        md_lines.append("| " + " | ".join(row) + " |")
    (output_dir / f"{metric_name}_rebuttal_table.md").write_text("\n".join(md_lines) + "\n")

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
    ax.set_ylabel(metric_label + " (95% bootstrap CI)")
    ax.set_title(f"Per-Instruction {metric_label} — Instruct family")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric_name}_bars.png", dpi=200, bbox_inches="tight")
    plt.savefig(output_dir / f"{metric_name}_bars.pdf", bbox_inches="tight")
    plt.close()

    return pivot_mean


def uniformity_check(pivot_mean: pd.DataFrame, ref: str, challenger: str, output_dir: Path,
                     metric_name: str) -> dict:
    if ref not in pivot_mean.columns or challenger not in pivot_mean.columns:
        return {"ref": ref, "challenger": challenger, "wins": None, "total": None,
                "msg": f"missing column ({ref} or {challenger})"}
    df2 = pivot_mean[[ref, challenger]].dropna()
    wins = int((df2[ref] > df2[challenger]).sum())
    total = int(len(df2))
    summary = {"metric": metric_name, "ref": ref, "challenger": challenger,
               "wins": wins, "total": total,
               "msg": f"{ref} beats {challenger} on {wins}/{total} instructions ({metric_name})"}
    (output_dir / f"{metric_name}_uniformity_check.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    cfg_path = project_root / "configs" / "rebuttal" / "exp1_5.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    methods = cfg["methods"]
    plot_methods = cfg["plot_methods"]
    data_dir = project_root / cfg["data_dir"]
    output_dir = project_root / cfg["output_dir"]
    embeddings_dir = output_dir / "embeddings" if cfg.get("save_embeddings") else None
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Exp 1's per-prompt fidelities (must exist)
    fid_csv = project_root / cfg["exp1_fidelities_csv"]
    if not fid_csv.exists():
        print(f"❌ Missing {fid_csv} — run Exp 1 first")
        sys.exit(1)
    fid_df = pd.read_csv(fid_csv)
    print(f"📂 Loaded Exp 1 fidelities: {len(fid_df)} rows")

    print(f"📂 Loading embedder: {cfg['embedding_model']}")
    embedder = SentenceTransformer(cfg["embedding_model"])

    # Compute shift per method
    all_shift_dfs = []
    for method in tqdm(methods, desc="methods"):
        csv_path = data_dir / method / cfg["responses_filename_pattern"].format(method=method)
        if not csv_path.exists():
            print(f"⚠️  {csv_path} missing — skipping {method}")
            continue
        sdf = compute_shift_per_row(method, csv_path, embedder, embeddings_dir)
        all_shift_dfs.append(sdf)
        print(f"  shift {method}: mean={sdf['shift'].mean():.4f}")

    if not all_shift_dfs:
        print("❌ No methods processed.")
        sys.exit(1)

    shift_df = pd.concat(all_shift_dfs, ignore_index=True)
    shift_df.to_csv(output_dir / "per_prompt_shifts.csv", index=False)

    # Join: awareness = fidelity * shift (per prompt × instruction × method)
    merged = shift_df.merge(
        fid_df[["method", "prompt_id", "instruction_type", "fidelity"]],
        on=["method", "prompt_id", "instruction_type"],
        how="inner",
    )
    merged["awareness"] = merged["fidelity"] * merged["shift"]
    merged.to_csv(output_dir / "per_prompt_awareness.csv", index=False)
    print(f"\n📊 Merged: {len(merged)} (method × prompt × instruction) cells")

    # Bootstrap BCa CI per (method, instruction) for BOTH metrics
    print(f"\n🎲 Bootstrap BCa CI (n={cfg['n_bootstrap']})...")

    table_shift = []
    table_awareness = []
    for method in methods:
        for inst_type in ISD_INSTRUCTIONS.keys():
            sub = merged[(merged.method == method) & (merged.instruction_type == inst_type)]
            if len(sub) == 0:
                continue
            for label, col, target in [("shift", "shift", table_shift),
                                       ("awareness", "awareness", table_awareness)]:
                ci = bca_ci(sub[col].values,
                            n_bootstrap=cfg["n_bootstrap"],
                            confidence=cfg["confidence_level"],
                            seed=cfg["seed"])
                target.append({"method": method, "instruction_type": inst_type,
                               "n": int(len(sub)), "ci": ci})

    pivot_shift = emit_outputs(table_shift, methods, plot_methods, output_dir,
                               metric_name="shift", metric_label="Semantic Shift")
    pivot_aware = emit_outputs(table_awareness, methods, plot_methods, output_dir,
                               metric_name="awareness", metric_label="Awareness Score")

    s_shift = uniformity_check(pivot_shift,
                               ref=cfg["reference_method"],
                               challenger=cfg["challenger_method"],
                               output_dir=output_dir, metric_name="shift")
    s_aware = uniformity_check(pivot_aware,
                               ref=cfg["reference_method"],
                               challenger=cfg["challenger_method"],
                               output_dir=output_dir, metric_name="awareness")
    print(f"\n🎯 {s_shift['msg']}")
    print(f"🎯 {s_aware['msg']}")

    print(f"\n✅ Exp 1.5 complete. Outputs in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"   {f.name:48s} {size_kb:>8.1f} KB")
    if embeddings_dir and embeddings_dir.exists():
        print(f"   embeddings/ ({len(list(embeddings_dir.glob('*.npy')))} .npy files)")


if __name__ == "__main__":
    main()
