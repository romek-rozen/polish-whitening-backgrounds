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
   │       │          └─ doc | chunks              embedding granularity
   │       └─ pl_mixed50k                          language + corpus tag
   └─ qwen3_4b | qwen3_8b                          embedding model
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

Both granularities are shipped:

- `_doc_` — one embedding per whole document (50 042 vectors).
- `_chunks_` — one embedding per 512-token chunk with 64-token
  overlap, produced by `scripts/lib/chunker.py`
  (`RecursiveCharacterTextSplitter` with `merge_tiny` floor=100 chars
  + `strip_overlap_fragments`). Yields **129 181 chunks** from the
  same 50 042 docs. Not a drop-in replacement for `_doc_` — the
  background's fit-time granularity MUST match your index-time
  granularity; see
  [`../GOTCHAS.md`](../GOTCHAS.md#1-background-granularity-must-match-index-granularity).

## Picking the right one for your pipeline

All 22 variants are shipped. Pick `_doc_` if you whiten whole
documents, `_chunks_` if you whiten 512-token chunks.

| Your model | Your effective dim (after MRL slice + L2 renorm) | Use |
|---|---:|---|
| Qwen3-Embedding-4B | 2560 (native) | `qwen3_4b_pl_mixed50k_{doc,chunks}_mrl2560/` |
| Qwen3-Embedding-4B | 1536 | `qwen3_4b_pl_mixed50k_{doc,chunks}_mrl1536/` |
| Qwen3-Embedding-4B | 1024 | `qwen3_4b_pl_mixed50k_{doc,chunks}_mrl1024/` |
| Qwen3-Embedding-4B | 768 | `qwen3_4b_pl_mixed50k_{doc,chunks}_mrl768/` |
| Qwen3-Embedding-4B | 512 | `qwen3_4b_pl_mixed50k_{doc,chunks}_mrl512/` |
| Qwen3-Embedding-8B | 4096 (native) | `qwen3_8b_pl_mixed50k_{doc,chunks}_mrl4096/` |
| Qwen3-Embedding-8B | 3072 | `qwen3_8b_pl_mixed50k_{doc,chunks}_mrl3072/` |
| Qwen3-Embedding-8B | 2048 | `qwen3_8b_pl_mixed50k_{doc,chunks}_mrl2048/` |
| Qwen3-Embedding-8B | 1024 | `qwen3_8b_pl_mixed50k_{doc,chunks}_mrl1024/` |
| Qwen3-Embedding-8B | 768 | `qwen3_8b_pl_mixed50k_{doc,chunks}_mrl768/` |
| Qwen3-Embedding-8B | 512 | `qwen3_8b_pl_mixed50k_{doc,chunks}_mrl512/` |

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
  Shipped backgrounds top out at 24 (4096-dim 8B chunks).
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
- Backgrounds for models other than Qwen3-Embedding-4B and -8B.
  The pipeline is generic — extend
  [`scripts/lib/tokenizer.OPENROUTER_TO_HF_TOKENIZER`](../scripts/lib/tokenizer.py)
  with the model's HF repo id and re-run.

See [`../REGISTRY.md`](../REGISTRY.md) for the live table of
backgrounds available right now (autogenerated by
`scripts/index_backgrounds.py`).
