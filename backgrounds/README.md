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
   │       │          └─ doc | chunks | segments | kw | paragraphs   embedding granularity
   │       └─ pl_mixed50k | med_pl                 language + corpus tag
   └─ qwen3_4b | qwen3_8b | te3small | te3large    embedding model
```

Example: `qwen3_4b_pl_mixed50k_doc_mrl1024/` is the ZCA refit for
Qwen3-Embedding-4B at MRL dim 1024, fitted on the `pl_mixed50k`
corpus at **document** granularity (one embedding per whole doc).

Model first means `ls backgrounds | grep qwen3_4b` lists every
variant of a model in one shot.

## Two corpora: `pl_mixed50k` (general) and `med_pl` (medical)

Backgrounds come in two independent corpus families, distinguished by
the corpus tag in the directory name:

- **`pl_mixed50k`** — general Polish (Wikipedia + FineWeb-2 + oasst).
  The original release; four models, up to five granularities.
- **`med_pl`** — professional Polish **medical** text (ChPL drug
  labels, incl. OCR of scanned labels). A separate distribution for
  medical retrieval; see [Medical backgrounds](#the-med_pl-medical-corpus)
  below and the root README's [Medical backgrounds](../README.md#medical-backgrounds-med_pl).

A `med_pl` background is **not** interchangeable with a `pl_mixed50k`
one — they whiten against different embedding distributions. Match the
corpus to your text domain.

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

## The `med_pl` (medical) corpus

Professional Polish medical text: **14 392 ChPL documents**
(*Charakterystyki Produktów Leczniczych* — drug labels), **475 M
chars**, median ~27 k chars/doc. This is **13 514 scraped labels +
878 OCR'd image-only scans** (the scans have no text layer;
`scripts/build_med_pl_corpus_chpl_ocr.py` OCRs them with a local
Qwen3-VL model). A separate PES exam-question source (170 950 board
questions) is **held out** for a future short-text background — it is
not part of these fits.

Three granularities are shipped, all fit on the **same 14 392-doc,
OCR-inclusive corpus**:

- `_doc_` — one embedding per whole ChPL document (14 392 vectors).
  Qwen only — te3 caps input at 8 191 tokens and 60 % of these docs
  exceed that (median ~10.6 k tokens), so a te3 `doc` fit would see
  mostly truncated document heads; deliberately not shipped.
- `_paragraphs_` — one embedding per blank-line paragraph. The docs
  split into ~1.18 M paragraphs; the fit runs on a **~150 000-row
  random sample (seed 42)** — a whitening Σ needs a representative
  sample, not every row.
- `_chunks_` — one embedding per **300-token** window with **50-token**
  overlap (tighter than the general corpus's 512/64). The docs split
  into ~0.84 M chunks; fit on a **~150 000-row random sample (seed 42)**.

`segments` and `kw` are **not** shipped for `med_pl`. All four models
ship `paragraphs` + `chunks`; `doc` is Qwen-only (see above).

## Picking the right one for your pipeline

**160 variants** are shipped — **103 general** (`pl_mixed50k`) +
**57 medical** (`med_pl`). Pick `_doc_` if you whiten whole
documents, `_chunks_` if you whiten fixed-size chunks, `_segments_`
if you whiten article sections (internal linking), `_paragraphs_` if
you whiten blank-line paragraphs, `_kw_` if you whiten short keyword
phrases.

### `pl_mixed50k` (general)

| Your model | Granularities | Effective dims (after MRL slice + L2 renorm) |
|---|---|---|
| Qwen3-Embedding-4B | doc, chunks, segments, kw, paragraphs | 2560 (native), 1536, 1024, 768, 512 |
| Qwen3-Embedding-8B | doc, chunks, segments, kw, paragraphs | 4096 (native), 3072, 2048, 1024, 768, 512 |
| text-embedding-3-small | doc, chunks, kw, paragraphs | 1536 (native), 1024, 768, 512, 256 |
| text-embedding-3-large | doc, chunks, kw, paragraphs | 3072 (native), 2048, 1536, 1024, 768, 512, 256 |

Pattern: `<model>_pl_mixed50k_<granularity>_mrl<dim>/`,
e.g. `qwen3_8b_pl_mixed50k_segments_mrl1024/`.

### `med_pl` (medical)

| Your model | Granularities | Effective dims (after MRL slice + L2 renorm) |
|---|---|---|
| Qwen3-Embedding-4B | doc, paragraphs, chunks | 2560 (native), 1536, 1024, 768, 512 |
| Qwen3-Embedding-8B | doc, paragraphs, chunks | 4096 (native), 3072, 2048, 1024, 768, 512 |
| text-embedding-3-small | paragraphs, chunks | 1536 (native), 1024, 768, 512, 256 |
| text-embedding-3-large | paragraphs, chunks | 3072 (native), 2048, 1536, 1024, 768, 512, 256 |

Pattern: `<model>_med_pl_<granularity>_mrl<dim>/`,
e.g. `te3large_med_pl_paragraphs_mrl1024/`. Note: **no te3 `doc`**
(input-length truncation — see the corpus note above).

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
- The corpus parquet itself. Reproducible via `scripts/build_corpus.py`
  (general) or `scripts/build_med_pl_corpus.py` (medical); the
  fingerprint in every meta.json pins which corpus the background saw.
- `med_pl` at `doc` granularity for the **OpenAI** models
  (`te3small`/`te3large`) — deliberately skipped, the docs exceed the
  8 191-token te3 input cap. And `med_pl` `segments` / `kw` for any
  model — not built yet.
- Backgrounds for models other than the four shipped (Qwen3-Embedding
  4B/8B, text-embedding-3 small/large). The pipeline is generic —
  extend the model↔tokenizer mappings in
  [`scripts/lib/tokenizer.py`](../scripts/lib/tokenizer.py)
  and re-run.

See [`../REGISTRY.md`](../REGISTRY.md) for the live table of
backgrounds available right now (autogenerated by
`scripts/index_backgrounds.py`).
