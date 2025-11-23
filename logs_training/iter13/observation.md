# Evaluation Metrics Comparison

## LLM-as-Judge
- **Provider**: Fireworks AI
- **Primary Model**: Llama-3.3-70B-Instruct
- **Fallback Model**: Llama-3.1-70B-Instruct

## Comparison Table

| Aspect | ISD | Toxicity | TruthfulQA | Conditional Safety | Style Transfer |
|--------|-----|----------|------------|-------------------|----------------|
| **Dataset** | HF: kapilw25/ISD (500 prompts x 10 types) | PKU-SafeRLHF test (both-unsafe) | TruthfulQA validation (817 Qs) | PKU-SafeRLHF test (borderline) | AlpacaEval (805 prompts) |
| **Sanity Mode** | 50 prompts × 10 = 500 | 150 prompts | 50 Qs × 2 = 100 | 100 prompts × 2 = 200 | 100 prompts × 2 = 200 |
| **Variants** | 10 instruction types (see below) | N/A (harm-category instruction) | HONEST, CONFIDENT | STRICT, PERMISSIVE | CONCISE, DETAILED |
| **LLM Judge** | No (embedding similarity) | Llama-3.3-70B | No (heuristic markers) | No (heuristic refusal) | No (word count) |
| **Metric Formula** | `Fidelity x Semantic_Shift` | `safe_refusals / total` | `HONEST_uncertainty - CONFIDENT_uncertainty` | `abs(STRICT_refusal - PERMISSIVE_refusal)` | `DETAILED_words / CONCISE_words` |
| **Score Type** | Product (0-1) | Rate (0-100%) | Difference (directional) | Difference (magnitude) | Ratio (>=1) |
| **Perfect Score** | 1.0 | 100% | Positive (higher=better) | 1.0 | >4.0 |
| **Good Result** | Higher = better instruction following | Higher = safer | 0 or positive = correct direction | Higher = more adaptive | >1 = follows length instruction |
| **Bad Result** | Low = ignores instruction variant | Low = unsafe outputs | Negative = wrong direction | 0 = ignores instruction | ~1 = ignores instruction |

**ISD instruction types**: Neutral, Conservative, Liberal, Regulatory, Empathetic, Safety, Educational, Concise, Professional, Creative

## Key Insights

- **ISD & Toxicity**: Absolute metrics (rate/product) - higher is simply better
- **TruthfulQA**: Directional metric - sign matters (negative = wrong adaptation direction)
- **Conditional Safety & Style Transfer**: Magnitude metrics - measures how much the model adapts

This explains why DPO can "win" Toxicity (100% safe) but "lose" TruthfulQA (negative score means it adapted in the wrong direction - more uncertain when told to be confident).

## Results Comparison (Overall Scores)

| Model | ISD | Toxicity | TruthfulQA | Conditional Safety | Style Transfer |
|-------|-----|----------|------------|-------------------|----------------|
| **CITA_NoInstruct** | 0.215 | 36.5% | -0.040 🥈 | 0.010 | 1.11 🥉 |
| **CITA_Instruct** | 0.439 🥈 | 58.7% 🥉 | 0.111 🥇 | 0.390 🥇 | 1.14 🥈 |
| **DPO_NoInstruct** | 0.246 🥉 | 79.7% 🥈 | -0.260 🥉 | 0.030 🥉 | 1.02 |
| **DPO_Instruct** | 0.453 🥇 | 100.0% 🥇 | -0.300 | 0.370 🥈 | 1.26 🥇 |

**Winner by Metric:**
- ISD: DPO_Instruct (0.453)
- Toxicity: DPO_Instruct (100%)
- TruthfulQA: CITA_Instruct (0.111) - only positive score
- Conditional Safety: CITA_Instruct (0.390)
- Style Transfer: DPO_Instruct (1.26)

**Final Score: DPO 3 - CITA 2**

**Key Observations:**
- All Instruct variants >> NoInstruct variants (validates instruction-alignment hypothesis)
- DPO wins on absolute safety (Toxicity) and style adaptation
- CITA wins on correct directional adaptation (TruthfulQA, Conditional Safety)
- TruthfulQA: DPO shows negative scores (wrong adaptation direction - more uncertain when told to be confident)
