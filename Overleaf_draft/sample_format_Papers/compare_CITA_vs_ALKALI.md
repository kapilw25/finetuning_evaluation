# Format Comparison: CITA vs ALKALI

## High-Level Overview

| Aspect | CITA (Your Paper) | ALKALI (Sample Format) |
|--------|-------------------|------------------------|
| Total Pages | 10 | 52 (10 main + 42 appendix) |
| Column Layout | Single-column | Two-column (ACL format) |
| Template | Basic LaTeX article | ACL/EMNLP conference |
| Authors | 6 | 7 |
| References | 6 citations | 60+ citations |
| Appendix | None | 42 pages (Sections A-M) |

## Document Structure Comparison

| Component | CITA | ALKALI |
|-----------|------|--------|
| Main Sections | 8 | 10 |
| Subsections | ~12 | ~40+ |
| Appendix Sections | 0 | 13 (A through M) |
| FAQ Section | No | Yes (10 questions) |

## Visual Elements Count

| Element | CITA | ALKALI |
|---------|------|--------|
| Figures | 13 | 11+ (main) + many in appendix |
| Tables | 10 | 8+ (main) + many in appendix |
| Equations | 7 numbered | 20+ numbered |
| Algorithm Boxes | 1 (pipeline diagram) | Multiple formal algorithms |
| Heatmaps | 1 | 3+ |
| Bar Charts | 2 | 4+ |
| Radar Charts | 1 | 0 |

## Section Page Distribution (Estimated)

### CITA (10 pages)
| Section | Pages |
|---------|-------|
| Abstract | ~0.3 |
| Introduction + Related Work | ~1.5 |
| Methodology (Sections 2-4) | ~3 |
| Dataset (Section 5) | ~1 |
| Experiments (Section 6) | ~1.5 |
| Results (Section 7) | ~1.5 |
| Conclusion + Limitations | ~0.7 |
| References | ~0.5 |

### ALKALI (10 main + 42 appendix)
| Section | Pages |
|---------|-------|
| Abstract + Contributions | ~0.5 |
| Categories of Attacks (1-2) | ~1.5 |
| Benchmark + Evaluation (3-4) | ~2 |
| DPO Baseline (5) | ~0.5 |
| Latent Pooling (6) | ~1 |
| GRACE Framework (7) | ~1.5 |
| Conclusion (8) | ~0.5 |
| Discussion + Limitations (9) | ~1.5 |
| References | ~1 |
| Appendix A-M | ~42 |

## Appendix Content (ALKALI only)

| Section | Content | Pages |
|---------|---------|-------|
| A | Overview | ~1 |
| B | Attack Categories | ~1 |
| C | Defense Analysis | ~1 |
| D | Logits to Latents | ~1 |
| E | Layerwise Pooling | ~3 |
| F | GRACE Loss Formulation | ~3 |
| G | Performance Analysis | ~2 |
| H | AVQI Derivation | ~2 |
| I | Implementation Details | ~2 |
| J | ASR Protocol | ~2 |
| K | Visualizations | ~4 |
| L | Ablation Studies | ~2 |
| M | Extended Attack Discussion | ~6 |
| FAQ | 10 detailed Q&As | ~14 |

## Citation Density

| Metric | CITA | ALKALI |
|--------|------|--------|
| Total Citations | 6 | 60+ |
| Citations per Page (main) | 0.6 | 6+ |
| Self-citations | 0 | 0 |
| Recent Papers (2023-2024) | 4 | 40+ |

## Key Structural Differences

1. **Column Format**: CITA single-column vs ALKALI two-column (conference standard)
2. **Appendix**: CITA has none; ALKALI has 42 pages (4.2× main content)
3. **Scale**: ALKALI evaluates 3.5× more models (21 vs 6)
4. **Dataset**: ALKALI benchmark is 3× larger (9,000 vs 3,000)
5. **Citations**: ALKALI has 10× more references
6. **FAQ**: ALKALI includes 10 detailed FAQs with mathematical formulations

## Recommendations for CITA (Priority Order)

### P0: Rejection Risk (fix first)
- [ ] Switch to two-column ACL/EMNLP format
- [ ] Increase citations from 6 to 30+ (6 citations = immediate red flag)

### P1: Missing Standard Elements
- [ ] Add "Contributions at-a-glance" bullet list after abstract
- [ ] Add algorithm pseudocode box for CITA training loop
- [ ] Add formal loss function equations with clear notation

### P2: Appendix Content
- [ ] Full hyperparameter tables (currently partial)
- [ ] Additional ablation studies
- [ ] Extended dataset statistics
- [ ] More qualitative examples (good/bad responses)

### P3: Reviewer Questions (FAQ)
- [ ] Why CITA over standard DPO?
- [ ] What is instruction-alignment vs instruction-following?
- [ ] How does KL regularization prevent collapse?
- [ ] Why Llama-3.1-8B specifically?

### P4: Visual Polish
- [ ] Add more detailed heatmaps
- [ ] Include attention weight visualizations
- [ ] 3D latent space visualizations (if available)
