---
name: Full LaTeX compilation required
description: Always run full pdflatex+bibtex cycle when compiling LaTeX papers, never single-pass
type: feedback
---

When compiling LaTeX papers (especially in Overleaf_draft/), ALWAYS run the full compilation cycle:

```
pdflatex -interaction=nonstopmode 0_main.tex && bibtex 0_main && pdflatex -interaction=nonstopmode 0_main.tex && pdflatex -interaction=nonstopmode 0_main.tex
```

NEVER run a single `pdflatex` pass — it causes `(???)` for citations and broken cross-references.
Reason: User was frustrated by repeated partial compilations showing `(???)` instead of actual references.
