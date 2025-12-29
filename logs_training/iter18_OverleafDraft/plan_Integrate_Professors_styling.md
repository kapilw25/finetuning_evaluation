# Merge Plan: ECLIPTICA Branding + Correct Numbers

## Goal

Merge professor's ECLIPTICA branding/styling (from first 3 sections) with our correct evaluation numbers (95.6%, all 5 methods).

## Scope

**Professor modified:** `0_main.tex`, `1_introduction.tex`, `2_methodology.tex` only
- Added styled visuals, ECLIPTICA branding
- BUT used OLD narrative: "CITA wins 3/5, DPO wins 2/5"

**Our task:** Update these 3 files with correct numbers while keeping professor's styling

**Remaining files:** Already have correct numbers - NO CHANGES NEEDED

---

## Files to Modify (3 only)

### 1. `0_main.tex`

**Keep from Professor:**
- Entire preamble (packages, colors, commands)
- Banner title: `\title{\includegraphics[width=\textwidth]{banner.pdf}}`
- tcolorbox with "ECLIPTICA at-a-glance"
- FA icon-based contributions box structure

**Update Numbers:**
```
OLD: "CITA improves instruction sensitivity and outperforms DPO in 3/5 metrics"

NEW: "ECLIPTICA + CITA achieves 95.6% instruction-alignment efficiency, outperforming
     DPO (70.5%) by 25.1 pp, GRPO (59.3%) by 36.3 pp, PPO (50.4%) by 45.2 pp,
     and SFT (15.9%) by 79.7 pp"
```

---

### 2. `1_introduction.tex`

**Keep from Professor:**
- ALL color definitions and commands
- Section title: "Why Instruction-Based Alignment?"
- Visual row-based layout with colored boxes
- Elaborate styling with icons

**Update Numbers:**
- Find any "3/5" or "outperforms DPO in 3/5" text
- Replace with 95.6% narrative
- Ensure ECLIPTICA + CITA naming is consistent

---

### 3. `2_methodology.tex`

**Keep from Professor:**
- Table formatting improvements
- Any styling changes

**Verify:**
- 5-column table with ALL methods (SFT, PPO, GRPO, DPO, CITA)
- Correct hyperparameter values

---

## Files to Keep Unchanged

| File | Status |
|------|--------|
| `3_unified_loss.tex` | Already correct |
| `4_cita_framework.tex` | Already correct |
| `5_isd_dataset.tex` | Already correct |
| `6_experiments.tex` | Already correct |
| `7_results.tex` | Already correct (95.6% narrative) |
| `8_conclusion.tex` | Already correct |
| `9_related_work.tex` | Already correct |
| `10_faq.tex` | Already correct |
| `11_appendix.tex` | Already correct |

---

## Key Numbers to Preserve

| Metric            | Value  |
|-------------------|--------|
| CITA efficiency   | 95.6%  |
| DPO efficiency    | 70.5%  |
| GRPO efficiency   | 59.3%  |
| PPO efficiency    | 50.4%  |
| SFT efficiency    | 15.9%  |
| AQI delta SFT     | -23.4  |
| AQI delta DPO     | +22.4  |
| AQI delta CITA    | +41.7  |
| TruthfulQA ratio  | 54x    |

---

## Execution Order

1. Copy professor's `0_main.tex` → update numbers
2. Copy professor's `1_introduction.tex` → update numbers
3. Review professor's `2_methodology.tex` → verify 5-method table exists
4. Done - other files stay as-is

---

## Files Required

- `banner.pdf` - must exist in `figures/` directory
- FontAwesome5 package available in LaTeX
