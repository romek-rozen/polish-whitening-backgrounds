# backgrounds/

Pre-fitted ZCA whitening artefacts. One subdirectory per background.

## What's in each subdirectory

Each `<name>/` contains exactly four files:

| File | Shape | dtype | Purpose |
|---|---|---|---|
| `W_A.npy` | `(dim, dim)` | float32 | The whitening matrix. `x_white = (x - mu) @ W`. |
| `mu_A.npy` | `(dim,)` | float32 | Mean vector of the (L2-renormalised) corpus embeddings. |
| `eigvals_A.npy` | `(dim,)` | float32 | Eigenvalues of Σ in descending order. Use to inspect rank / anisotropy. |
| `<name>.meta.json` | — | JSON | Provenance: model, corpus fingerprint, diagnostics, cost report. |

The four-file layout is stable across versions — `loader.py` in the
repo root looks for exactly these names.

## Naming convention

```
<model>_<corpus>_<granularity>_mrl<dim>/
   │       │          │            │
   │       │          │            └─ mrl<dim>     MRL refit at dim N
   │       │          └─ doc | chunks | segments | kw   embedding granularity
   │       └─ pl_mixed50k                          language + corpus tag
   └─ qwen3_4b | qwen3_8b | te3small | te3large    embedding model
```

Example: `qwen3_4b_pl_mixed50k_doc_mrl1024/` is the ZCA refit for
Qwen3-Embedding-4B at MRL dim 1024, fitted on the `pl_mixed50k`
corpus at **document** granularity (one embedding per whole doc).

Model first means `ls backgrounds | grep qwen3_4b` lists every
variant of a model in one shot.

## The `pl_mixed50k` corpus

The current corpus is **50 042 documents**: 22 500 Wikipedia +
27 500 FineWeb-2 PL + 42 oasst threads, ~46 M tokens, every doc
≥500 chars and truncated to 30 000 tokens with the Qwen3 tokenizer
before embedding. The oasst tier was originally targeted at 5 000
threads but the public oasst-1 dump yields only 42 Polish-tagged
ones, so the FineWeb tier was extended by 5 000 to get back to a
genuine 50k.

Four granularities are shipped:

- `_doc_` — one embedding per whole document (50 042 vectors).
- `_chunks_` — one embedding per 512-token chunk with 64-token
  overlap, produced by `scripts/lib/chunker.py`
  (`RecursiveCharacterTextSplitter` with `merge_tiny` floor=100 chars
  + `strip_overlap_fragments`). Yields **129 181 chunks** from the
  same 50 042 docs.
- `_segments_` — one embedding per article **section** (up to 1024
  tokens, no overlap), produced by `scripts/lib/segmenter.py`
  (heading-first separators, `merge_tiny` floor=300 chars). Yields
  **73 692 segments**; Qwen models only. Built for internal-linking
  retrieval (segment→segment, targets represented by their segments).
- `_kw_` — one embedding per short keyword-like phrase (1–5 words),
  50 000 phrases mined by `scripts/build_corpus_keywords.py`. For
  keyword grouping / clustering.

None is a drop-in replacement for another — the background's
fit-time granularity MUST match your index-time granularity; see
[`../GOTCHAS.md`](../GOTCHAS.md#1-background-granularity-must-match-index-granularity).

## Picking the right one for your pipeline

All 80 variants are shipped. Pick `_doc_` if you whiten whole
documents, `_chunks_` if you whiten 512-token chunks, `_segments_`
if you whiten article sections (internal linking), `_kw_` if you
whiten short keyword phrases.

| Your model | Granularities | Effective dims (after MRL slice + L2 renorm) |
|---|---|---|
| Qwen3-Embedding-4B | doc, chunks, segments, kw | 2560 (native), 1536, 1024, 768, 512 |
| Qwen3-Embedding-8B | doc, chunks, segments, kw | 4096 (native), 3072, 2048, 1024, 768, 512 |
| text-embedding-3-small | doc, chunks, kw | 1536 (native), 1024, 768, 512, 256 |
| text-embedding-3-large | doc, chunks, kw | 3072 (native), 2048, 1536, 1024, 768, 512, 256 |

Directory name pattern: `<model>_pl_mixed50k_<granularity>_mrl<dim>/`,
e.g. `qwen3_8b_pl_mixed50k_segments_mrl1024/`.

If you slice to a dim we don't ship (e.g. 256, or 2048 against 4B),
refit yourself — see [Rebuild from scratch](../README.md#rebuild-from-scratch-or-fit-your-own-model)
in the root README. The chunks the fits run against
(`data/chunks_<model_slug>/`) are *not* in git, but the embed step
is deterministic and rerunable.

## Diagnostics you should check before trusting a background

Every `<name>.meta.json` carries a `diagnostics` block. Sanity
thresholds:

- `rank_deficient_eigvals` (count of eigvals < 1e-7) should be
  **well under ~100**. Anything higher means SVD found an
  unhealthy low-rank structure — usually a corpus problem.
  Shipped backgrounds top out at 59 (4096-dim 8B kw).
- `top_ev_ratio_pre` is the anisotropy ratio (top eigval ÷ mean).
  Values in the **tens to low hundreds** are normal for modern
  embeddings — shipped Qwen3 backgrounds run 20.4 (8B chunks @512)
  to 157.6 (8B doc @4096) across the MRL ladder. A ratio near 1
  would mean the model was already isotropic and whitening
  wouldn't do anything.
- `fit_s` is just the wall-clock for SVD — sanity-check it scaled
  ~`dim²` against the other refits.

## What's *not* here

- The raw embedding chunks (`data/chunks_<model_slug>/`). Reproducible
  via `scripts/embed_via_openrouter.py`; not worth the ~700 MB in git.
- The corpus parquet itself. Reproducible via `scripts/build_corpus.py`;
  the fingerprint in every meta.json pins which corpus the
  background saw.
- Backgrounds for models other than the four shipped (Qwen3-Embedding
  4B/8B, text-embedding-3 small/large). The pipeline is generic —
  extend the model↔tokenizer mappings in
  [`scripts/lib/tokenizer.py`](../scripts/lib/tokenizer.py)
  and re-run.

See [`../REGISTRY.md`](../REGISTRY.md) for the live table of
backgrounds available right now (autogenerated by
`scripts/index_backgrounds.py`).
