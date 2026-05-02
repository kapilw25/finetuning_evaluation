"""Query-scrambled evaluation: CITA full vs scrambled vs i_only vs x_only. GPU-only.

    python -u src/eval/exp2_query_scrambled.py 2>&1 | tee logs/rebuttal/exp2.log

Tests whether CITA_Instruct learns joint (I, X) interaction (full ≫ scrambled) or
a shortcut "I → style" mapping (full ≈ scrambled). The "full" responses are reused
from cached ISD eval; only 3 new conditions get fresh inference (~10-30 min).
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.eval.isd_dataset import ISD_INSTRUCTIONS  # noqa: E402,F401
from src.utils.eval.model_loader import load_model_for_eval, unload_model  # noqa: E402
from src.utils.eval.generation import batch_generate, format_chat_messages  # noqa: E402
from src.utils.eval.eclipta_scoring import (  # noqa: E402
    CHAR_DESCRIPTIONS,  # noqa: F401  (re-exported for any downstream callers)
    build_prototype_embeddings,
    score_fidelities,
    bca_ci,
)


def build_message_lists(condition, sub_df, all_prompts_by_id, i_only_placeholder, rng):
    """Build the [(messages, instruction_type), ...] list for a given condition."""
    messages_list = []
    metas = []  # parallel list of {prompt_id, instruction_type, instruction, query}

    for _, row in sub_df.iterrows():
        pid = row["prompt_id"]
        inst = row["instruction"]
        query = row["prompt"]
        inst_type = row["instruction_type"]

        if condition == "scrambled":
            # Replace query with a random different prompt's text
            other_pids = [p for p in all_prompts_by_id.keys() if p != pid]
            other_pid = rng.choice(other_pids)
            scrambled_query = all_prompts_by_id[other_pid]
            messages = [
                {"role": "system", "content": inst},
                {"role": "user", "content": scrambled_query},
            ]
            used_query = scrambled_query
        elif condition == "i_only":
            messages = [
                {"role": "system", "content": inst},
                {"role": "user", "content": i_only_placeholder},
            ]
            used_query = i_only_placeholder
        elif condition == "x_only":
            messages = [{"role": "user", "content": query}]
            used_query = query
        else:
            raise ValueError(f"Unknown condition: {condition}")

        messages_list.append(messages)
        metas.append({"prompt_id": pid, "instruction_type": inst_type,
                      "instruction": inst, "query_used": used_query})

    return messages_list, metas


def main():
    cfg_path = project_root / "configs" / "rebuttal" / "exp2.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    output_dir = project_root / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg["seed"])
    py_random = random.Random(cfg["seed"])  # noqa: F841  # for any future use

    # 1. Load existing CITA_Instruct responses (this gives us 'full' condition + prompt set)
    existing_csv = project_root / cfg["existing_responses_csv"]
    if not existing_csv.exists():
        print(f"❌ Missing {existing_csv}")
        sys.exit(1)
    full_df = pd.read_csv(existing_csv)
    print(f"📂 Loaded {len(full_df)} cached CITA_Instruct responses (full condition)")

    # Build prompt-id → prompt-text map (any single instruction works since prompt is the same)
    pids_unique = sorted(full_df["prompt_id"].unique())
    prompts_by_id = {}
    for pid in pids_unique:
        first_row = full_df[full_df["prompt_id"] == pid].iloc[0]
        prompts_by_id[pid] = first_row["prompt"]

    # 2. Sample n_prompts of the 300 unique prompts (deterministic via seed)
    n_prompts = cfg["n_prompts"]
    sampled_pids = list(rng.choice(pids_unique, size=min(n_prompts, len(pids_unique)),
                                   replace=False))
    sampled_pids = sorted(int(p) for p in sampled_pids)
    sub_df = full_df[full_df["prompt_id"].isin(sampled_pids)].copy()
    print(f"🎯 Sampled {len(sampled_pids)} prompts × 10 instructions = {len(sub_df)} (I,X) pairs")

    # 3. Load model
    print(f"\n📦 Loading {cfg['model']}...")
    model, tokenizer = load_model_for_eval(cfg["model"])

    all_results = []  # rows: condition, prompt_id, instruction_type, response, fidelity

    # ---- Condition: full (cached) ----
    print("\n=== Condition: full (cached) ===")
    for _, row in sub_df.iterrows():
        all_results.append({
            "condition": "full",
            "prompt_id": int(row["prompt_id"]),
            "instruction_type": row["instruction_type"],
            "instruction": row["instruction"],
            "query_used": row["prompt"],
            "response": row["response"],
        })

    # ---- Conditions: scrambled, i_only, x_only (fresh inference) ----
    for condition in ["scrambled", "i_only", "x_only"]:
        print(f"\n=== Condition: {condition} (running inference) ===")
        t0 = time.time()
        # Fresh RNG for each condition (deterministic, but independent)
        cond_rng = np.random.default_rng(cfg["seed"] + hash(condition) % 1000)
        messages_list, metas = build_message_lists(
            condition, sub_df, prompts_by_id, cfg["i_only_placeholder"], cond_rng
        )
        prompts_text = format_chat_messages(tokenizer, messages_list)

        responses = batch_generate(
            model=model, tokenizer=tokenizer, prompts=prompts_text,
            max_new_tokens=cfg["max_new_tokens"],
            batch_size=cfg["batch_size"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            do_sample=cfg["do_sample"],
            checkpoint_callback=None,
            checkpoint_interval=200,
        )
        print(f"  [{condition}] {len(responses)} responses in {time.time()-t0:.1f}s")

        for meta, resp in zip(metas, responses):
            all_results.append({
                "condition": condition,
                "prompt_id": meta["prompt_id"],
                "instruction_type": meta["instruction_type"],
                "instruction": meta["instruction"],
                "query_used": meta["query_used"],
                "response": resp,
            })

    # 4. Free GPU before scoring
    print("\n🔓 Unloading model before fidelity scoring...")
    unload_model(model)

    # 5. Score per-prompt fidelity for ALL conditions
    print(f"\n📂 Loading embedder: {cfg['embedding_model']}")
    embedder = SentenceTransformer(cfg["embedding_model"])
    prototypes = build_prototype_embeddings(embedder)

    df_all = pd.DataFrame(all_results)
    print(f"\n🎲 Scoring fidelity for {len(df_all)} responses...")
    df_all["fidelity"] = score_fidelities(
        df_all["response"].fillna("").astype(str).tolist(),
        df_all["instruction_type"].tolist(),
        embedder, prototypes,
    )
    df_all.to_csv(output_dir / "all_responses_with_fidelity.csv", index=False)

    # 6. Bootstrap CI per condition
    print(f"\n🎲 Bootstrap BCa CI (n={cfg['n_bootstrap']})...")
    summary = []
    for cond in ["full", "scrambled", "i_only", "x_only"]:
        sub = df_all[df_all.condition == cond]
        ci = bca_ci(sub["fidelity"].values, cfg["n_bootstrap"], cfg["confidence_level"], cfg["seed"])
        summary.append({"condition": cond, "n": int(len(sub)), "ci": ci})
        print(f"  {cond:11s} n={len(sub):4d}  mean={ci['mean']:.3f}  ±{ci['ci_half']:.3f}")

    # 7. Bootstrap CI on deltas (full − each_other_condition) — paired by prompt+instruction
    print("\n🎲 Bootstrap BCa CI on deltas (paired)...")
    deltas = []
    full_scores = df_all[df_all.condition == "full"].set_index(["prompt_id", "instruction_type"])["fidelity"]
    for cond in ["scrambled", "i_only", "x_only"]:
        cond_scores = df_all[df_all.condition == cond].set_index(["prompt_id", "instruction_type"])["fidelity"]
        joined = full_scores.to_frame("full").join(cond_scores.to_frame(cond), how="inner")
        diff = (joined["full"] - joined[cond]).values
        ci = bca_ci(diff, cfg["n_bootstrap"], cfg["confidence_level"], cfg["seed"])
        deltas.append({"delta": f"full−{cond}", "n": int(len(diff)), "ci": ci})
        print(f"  full−{cond:11s} n={len(diff):4d}  Δmean={ci['mean']:+.3f}  ±{ci['ci_half']:.3f}  "
              f"[{ci['ci_lo']:+.3f}, {ci['ci_hi']:+.3f}]")

    # 8. Emit JSON + CSV + Markdown table + plot
    (output_dir / "summary.json").write_text(json.dumps(
        {"per_condition": summary, "deltas": deltas}, indent=2))

    # Markdown table for rebuttal text
    md = ["| Condition | Score (CITA_I) | Δ vs Full |",
          "|---|---|---|"]
    md.append(f"| Full (I, X)        | {summary[0]['ci']['mean']:.3f}±{summary[0]['ci']['ci_half']:.3f} | — |")
    for d, s in zip(deltas, summary[1:]):
        md.append(f"| {s['condition']:18s} | {s['ci']['mean']:.3f}±{s['ci']['ci_half']:.3f} | "
                  f"{d['ci']['mean']:+.3f}±{d['ci']['ci_half']:.3f} |")
    (output_dir / "rebuttal_table.md").write_text("\n".join(md) + "\n")

    # Bar plot
    conditions = [s["condition"] for s in summary]
    means = [s["ci"]["mean"] for s in summary]
    cis = [s["ci"]["ci_half"] for s in summary]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2E7D32", "#C62828", "#F57C00", "#1565C0"]
    bars = ax.bar(conditions, means, yerr=cis, capsize=5, color=colors, alpha=0.85,
                  edgecolor="black")
    ax.set_ylabel("ECLIPTICA Fidelity (95% bootstrap CI)")
    ax.set_title(f"Exp 2: Query-Scrambled Eval — {cfg['model']} (n={cfg['n_prompts']} prompts × 10 instr)")
    ax.set_ylim(0, max(0.7, max(means) + 0.1))
    ax.grid(axis="y", alpha=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{m:.3f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "exp2_4condition_bars.png", dpi=200, bbox_inches="tight")
    plt.savefig(output_dir / "exp2_4condition_bars.pdf", bbox_inches="tight")
    plt.close()

    # 9. Print verdict
    delta_full_scrambled = deltas[0]["ci"]
    if delta_full_scrambled["ci_lo"] > 0.05:
        verdict = "✅ JOINT learning (full > scrambled, CI excludes 0)"
    elif delta_full_scrambled["ci_hi"] < 0.05:
        verdict = "⚠️  SHORTCUT learning (full ≈ scrambled, CI near 0)"
    else:
        verdict = "🟡 INCONCLUSIVE (CI overlaps 0)"
    (output_dir / "verdict.txt").write_text(verdict + "\n")

    print(f"\n{verdict}")
    print(f"\n✅ Exp 2 complete. Outputs in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"   {f.name:48s} {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()
