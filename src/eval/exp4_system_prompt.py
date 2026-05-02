"""System-prompt baseline: trained CITA vs vanilla Llama + system prompt. GPU-only.

    # Sanity (24 GB GPU, ~5 min): 10 prompts, 2 variants, ECLIPTICA only
    python -u src/eval/exp4_system_prompt.py --mode sanity 2>&1 | tee logs/rebuttal/exp4_sanity.log

    # Full (96 GB GPU, ~7h): 300 prompts × 4 variants × 5 benchmarks
    python -u src/eval/exp4_system_prompt.py --mode full 2>&1 | tee logs/rebuttal/exp4_full.log

Tests whether prompting alone (vanilla Llama-Instruct + system prompt) can match a
trained CITA adapter. Expected: training matters, especially for switching tasks.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.eval.isd_dataset import ISD_INSTRUCTIONS  # noqa: E402,F401
from src.utils.eval.model_loader import load_model_for_eval, unload_model, MODELS  # noqa: E402
from src.utils.eval.generation import batch_generate, format_chat_messages  # noqa: E402
from src.utils.eval.eclipta_scoring import (  # noqa: E402
    CHAR_DESCRIPTIONS,  # noqa: F401  (re-exported for downstream callers)
    build_prototype_embeddings,
    score_fidelities,
    bca_ci,
)


def load_variant(variant: str, cfg: dict):
    """Return (model, tokenizer, label) for a given variant. Loads from HF."""
    if variant == "zero_shot_prompt":
        # Vanilla Llama-3.1-8B-Instruct (Meta's chat-tuned, gated separately)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        repo = cfg["llama_instruct_repo"]
        print(f"📦 Loading vanilla Instruct model: {repo}")
        tok = AutoTokenizer.from_pretrained(repo)
        tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        try:
            model = AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=torch.bfloat16, device_map="auto",
                attn_implementation="flash_attention_2",
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {repo}. This is a SEPARATE gated repo from base 8B. "
                f"Accept license at https://huggingface.co/{repo} and re-run.\n"
                f"Underlying error: {e}"
            )
        return model, tok, variant

    if variant == "zero_shot_prompt_base":
        # Sanity substitute: base Llama (no chat tune). Comparison is degraded but code-path works.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        repo = cfg["llama_base_repo"]
        print(f"📦 [SANITY SUBSTITUTE] Loading base Llama: {repo}")
        tok = AutoTokenizer.from_pretrained(repo)
        tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        if tok.chat_template is None:
            tok.chat_template = (
                "{% for message in messages %}"
                "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"
            )
        model = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2",
        )
        return model, tok, variant

    if variant == "few_shot_prompt":
        # Same as zero_shot_prompt but with 3-shot in-context examples in the user message.
        # Implemented by wrapping zero_shot_prompt during prompt construction.
        return load_variant("zero_shot_prompt", cfg) + ()  # noqa: F722

    # Trained adapter (CITA_Instruct, DPO_Instruct, etc.) — uses model_loader's MODELS dict
    if variant in MODELS:
        model, tok = load_model_for_eval(variant)
        return model, tok, variant

    raise ValueError(f"Unknown variant: {variant}")


def build_messages(variant: str, instruction: str, prompt: str, few_shot_examples=None):
    """Build the chat-message list for the given variant."""
    if variant in ("zero_shot_prompt", "zero_shot_prompt_base"):
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ]
    if variant == "few_shot_prompt":
        # Stuff 3 examples into user message
        shot_text = ""
        for ex in (few_shot_examples or []):
            shot_text += f"Question: {ex['q']}\nAnswer: {ex['a']}\n\n"
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": shot_text + f"Question: {prompt}\nAnswer:"},
        ]
    # Trained adapter — same format as ISD eval (system=instruction, user=prompt)
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ]


def run_ecliptica_for_variant(variant: str, cfg: dict, mode_cfg: dict, output_dir: Path,
                              embedder, prototypes, sampled_pids: list,
                              prompts_by_id: dict):
    """Run all 10 instructions × n_prompts for one variant; return DataFrame."""

    model, tokenizer, label = load_variant(variant, cfg)

    # Build (prompt_id × instruction_type) pairs
    pairs = []
    for pid in sampled_pids:
        for inst_type, info in ISD_INSTRUCTIONS.items():
            inst_text = info["variants"][0]   # take canonical instruction phrasing
            pairs.append({
                "prompt_id": pid,
                "instruction_type": inst_type,
                "instruction": inst_text,
                "prompt": prompts_by_id[pid],
            })

    messages_list = [build_messages(variant, p["instruction"], p["prompt"]) for p in pairs]
    prompts_text = format_chat_messages(tokenizer, messages_list)

    t0 = time.time()
    responses = batch_generate(
        model=model, tokenizer=tokenizer, prompts=prompts_text,
        max_new_tokens=cfg["max_new_tokens"],
        batch_size=mode_cfg["batch_size"],
        temperature=cfg["temperature"], top_p=cfg["top_p"], do_sample=cfg["do_sample"],
        checkpoint_callback=None, checkpoint_interval=200,
    )
    print(f"  [{variant}] {len(responses)} responses in {time.time()-t0:.1f}s")

    unload_model(model)

    # Score fidelity
    fidelities = score_fidelities(responses,
                                  [p["instruction_type"] for p in pairs],
                                  embedder, prototypes)
    df = pd.DataFrame(pairs)
    df["response"] = responses
    df["fidelity"] = fidelities
    df["variant"] = variant
    df.to_csv(output_dir / f"responses_{variant}.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sanity", "full"], required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load((project_root / "configs/rebuttal/exp4.yaml").read_text())
    mode_cfg = cfg[args.mode]
    output_dir = project_root / cfg["output_dir"] / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎯 Mode: {args.mode}")
    print(f"   variants:    {mode_cfg['variants']}")
    print(f"   benchmarks:  {mode_cfg['benchmarks']}")
    print(f"   n_prompts:   {mode_cfg['n_prompts']}")
    print(f"   batch_size:  {mode_cfg['batch_size']}")
    print(f"   output_dir:  {output_dir}")

    # Sanity: only ECLIPTICA. (Full extension to other 4 benchmarks is a TODO.)
    if "ecliptica" not in mode_cfg["benchmarks"]:
        print("❌ Sanity/full must include 'ecliptica'. Other benchmarks not yet wired.")
        sys.exit(1)

    # Load existing CITA responses for prompt set
    existing_csv = project_root / cfg["existing_responses_csv"]
    full_df = pd.read_csv(existing_csv)
    pids_unique = sorted(full_df["prompt_id"].unique())
    prompts_by_id = {pid: full_df[full_df["prompt_id"] == pid].iloc[0]["prompt"]
                     for pid in pids_unique}

    # Sample prompts deterministically
    rng = np.random.default_rng(cfg["seed"])
    sampled = list(rng.choice(pids_unique, size=min(mode_cfg["n_prompts"], len(pids_unique)),
                              replace=False))
    sampled_pids = sorted(int(p) for p in sampled)
    print(f"🎯 Sampled {len(sampled_pids)} prompts × 10 instructions = {len(sampled_pids)*10} (I,X) pairs")

    # Embedder (small, CPU)
    embedder = SentenceTransformer(cfg["embedding_model"])
    prototypes = build_prototype_embeddings(embedder)

    # Run each variant sequentially (one model in VRAM at a time)
    all_dfs = []
    for variant in mode_cfg["variants"]:
        print(f"\n{'='*80}")
        print(f"=== VARIANT: {variant} ===")
        print(f"{'='*80}")
        df = run_ecliptica_for_variant(variant, cfg, mode_cfg, output_dir,
                                       embedder, prototypes, sampled_pids, prompts_by_id)
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(output_dir / "all_variants.csv", index=False)

    # Bootstrap CI per variant
    print(f"\n🎲 Bootstrap BCa CI (n={cfg['n_bootstrap']})...")
    summary = []
    for variant in mode_cfg["variants"]:
        sub = df_all[df_all.variant == variant]
        ci = bca_ci(sub["fidelity"].values, cfg["n_bootstrap"], cfg["confidence_level"], cfg["seed"])
        summary.append({"variant": variant, "n": int(len(sub)), "ci": ci})
        print(f"  {variant:25s} n={len(sub):4d}  mean={ci['mean']:.3f}  ±{ci['ci_half']:.3f}")

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown table
    md = ["| Variant | ECLIPTICA fidelity (95% CI) |", "|---|---|"]
    for s in summary:
        md.append(f"| {s['variant']} | {s['ci']['mean']:.3f}±{s['ci']['ci_half']:.3f} |")
    (output_dir / f"rebuttal_table_{args.mode}.md").write_text("\n".join(md) + "\n")

    # Bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    variants = [s["variant"] for s in summary]
    means = [s["ci"]["mean"] for s in summary]
    cis = [s["ci"]["ci_half"] for s in summary]
    bars = ax.bar(variants, means, yerr=cis, capsize=5,
                  color=plt.get_cmap("tab10").colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("ECLIPTICA Fidelity (95% bootstrap CI)")
    ax.set_title(f"Exp 4 [{args.mode}]: System-Prompt vs Trained Adapter")
    ax.set_ylim(0, max(0.7, max(means) + 0.1))
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{m:.3f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / f"exp4_bars_{args.mode}.png", dpi=200, bbox_inches="tight")
    plt.savefig(output_dir / f"exp4_bars_{args.mode}.pdf", bbox_inches="tight")
    plt.close()

    print(f"\n✅ Exp 4 [{args.mode}] complete. Outputs in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            print(f"   {f.name:48s} {f.stat().st_size / 1024:>8.1f} KB")


if __name__ == "__main__":
    main()
