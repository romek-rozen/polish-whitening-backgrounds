# AGENTS.md — keyword-clustering-with-background

Instructions for any coding agent working here. Read this first.

Communication with the user: **Polish**. All files in repo: **English**.

## What this is

Clustering Polish keyword lists with the repo's CPU-servable models, with and
without a whitening background, comparing three clustering families at real
scale (~20 k keywords).

It is **not** the model-selection benchmark — that is
`../pl-keyword-embedding-cpu-bench`, which has hand-labelled ground truth, stays
strictly CPU-only, and must not be loosened. This project imports from it
(`sys.path` bootstrap at the top of each script). That coupling is deliberate;
the user explicitly said self-containment is not required.

## Method inventory and their equivalences

Five methods, and two of them are the same algorithm in disguise — knowing that
prevents re-deriving it:

| script | method | equivalent to |
|---|---|---|
| `cluster_keywords.py` | kNN + Leiden | — |
| `sweep_leiden.py` | kNN + Leiden + floor + min size | "sharpened Leiden" (D91) |
| `cluster_agglomerative.py` | agglomerative, **average** linkage | — |
| `cluster_bdos_method.py` / `sweep_threshold.py` | threshold + union-find | **single linkage** cut at that threshold |
| `sweep_umap_hdbscan.py` | UMAP + HDBSCAN | — (reference only) |

**Threshold + union-find is single linkage.** Connecting every pair above a
cutoff and taking connected components is exactly what single linkage does when
cut there. That is why it chains: one bridging keyword merges two unrelated
groups, which put 18 040 of 19 801 keywords in one cluster on raw embeddings.
`cluster_agglomerative.py --linkage single` reproduces it; `average` (the
default) is what removes the chaining while keeping the same "merge what is
similar, stop at a floor" idea.

## UMAP + HDBSCAN is a reference, not a candidate

The user rejects it for production, with a sound reason: projecting a 1024-dim
embedding to 30 dims discards what the embedding was computed for, and UMAP is
stochastic. Do not propose it as a recommended method.

Keep running it anyway. It is the only method here whose noise is **not a knob
we set** — HDBSCAN decides noise from density. That independent ~22 % is what
lets us say a chosen `min_similarity` is sane: agglomerative at floor 0.5
produced 20.2 % noise, two points away, from completely different machinery.
Lose the reference and every noise figure becomes self-justifying.

## Hard constraints

0. **A threshold's noise share is arithmetic — say so.** `analyze_similarity_distribution.py`
   shows the 0.8 cutoff discards exactly the keywords whose nearest neighbour is
   below 0.8: predicted 60.6 % vs measured 60.63 % on bge-m3. Never report a
   threshold method's noise as a judgement about keyword quality without
   checking the top-1 distribution first. Median top-1 in whitened space is
   ~0.75, so 0.8 sits above the median by construction.
   Corollary: a similarity floor belongs on a **percentile of that list's own
   top-1 distribution**, not on a constant carried over from another dataset.

1. **Never compare absolute similarity floors across spaces.** Whitening
   rescales every cosine: mean kNN edge similarity drops from ~0.75-0.84 raw to
   ~0.51-0.56 whitened. A `sim_floor=0.5` means something different in each. Use
   percentile floors for cross-space comparison. This is the single easiest way
   to produce a confidently wrong conclusion here.
2. **Do not present 0 % noise as a good result.** Plain Leiden labels every
   node, so 0 % noise is a property of the algorithm, not the data. Any noise
   figure must say which knob produced it.
3. **The clustering itself stays on CPU.** `--device` affects embedding only.
   pynndescent, igraph/leidenalg, UMAP and HDBSCAN have no GPU path here, so the
   scale-test timings are genuine CPU numbers — keep them that way.
4. **Sweeps write incrementally.** `IncrementalCsv` flushes every row. Do not
   "optimise" it back into a buffered write at the end: a UMAP or threshold
   sweep that dies at the last model would otherwise leave nothing on disk.
5. **Report cluster *sizes* alongside counts.** The threshold method produced
   150 clusters where one held 18 040 of 19 801 keywords. A cluster count on its
   own hides that completely.

## Layout

```
README.md                 # findings, discussion, limitations
AGENTS.md                 # this file
cluster_keywords.py       # cluster + print groups + per-keyword CSV export
sweep_leiden.py           # kNN + Leiden × floors × resolutions × min sizes
sweep_threshold.py        # keyword_cluster method: threshold + connected components
sweep_umap_hdbscan.py     # UMAP + HDBSCAN
make_xlsx.py              # merge sweeps -> pivot-ready workbook
scale_test_cpu.py         # 100k keywords on CPU: time + RSS per stage
embed_cache.py            # embed once, reuse across scripts
incremental_csv.py        # crash-safe row-by-row CSV writer
examples/                 # committed sample outputs
work/                     # git-ignored: keyword data, caches, sweep outputs
```

**`work/` holds the user's commercial keyword exports.** It is git-ignored and
must stay that way — never commit anything from it, and never copy its contents
into a committed file or a chat summary beyond aggregate statistics.

## Adding a clustering method

Add a `sweep_<method>.py` emitting the same columns as the others plus a
`method` value, then wire it into `make_xlsx.py`. Keep `summarise()` from
`sweep_leiden.py` as the shared stats function so noise and size definitions
stay identical across families — that comparability is the whole point.

## Embedding cache

`embed_cache.encode_cached()` keys on model + a hash of the exact keyword list,
content **and order** (every downstream array is indexed by row). It stores raw
L2-normalised vectors — not whitened — so one entry serves every background.

A cached call returns `throughput=None` on purpose: a disk read is not an
embedding benchmark and must never be reported as one.

## Honesty rules

The methods measured here disagree wildly, and each has a shape of failure that
a summary table hides:

- threshold + union-find can put 91 % of a list in one cluster while reporting a
  plausible cluster count,
- Leiden without a floor reports zero noise on a list that visibly contains junk,
- UMAP + HDBSCAN looks cleanest partly because it discards a fifth of the input.

When reporting, state the failure mode, not just the metric. If a run was
skipped, a model missing, or a background absent, say so rather than presenting
a partial table as complete.
