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

## Hard constraints

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
