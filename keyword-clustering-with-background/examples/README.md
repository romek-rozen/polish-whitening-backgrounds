# examples/

Committed sample outputs, so the numbers in the parent README can be checked
without re-running anything.

| file | contents |
|---|---|
| `leiden_sweep.csv` | kNN + Leiden sweep, 432 configurations |
| `threshold_sweep.csv` | threshold + union-find sweep, 90 configurations |
| `umap_sweep.csv` | UMAP + HDBSCAN sweep, 36 configurations |
| `sweep_comparison.xlsx` | all three merged, pivot-ready, with a legend sheet |

These hold **aggregate statistics only** — cluster counts, noise shares, size
distributions. No keywords.

**Cluster dumps are deliberately not committed here.** `cluster_keywords.py`
prints and exports the keywords themselves, and the runs behind these numbers
used a private commercial keyword export. Those outputs stay in the git-ignored
`work/`. If you need a shareable example of what clusters look like, generate it
from a public keyword list rather than copying one out of `work/`.
