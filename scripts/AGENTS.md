# AGENTS.md — scripts/

End-to-end pipeline that turns a Polish corpus into a folder full of
ZCA whitening backgrounds.  Top-level files here are **CLI entry
points**; the heavy lifting lives under `scripts/lib/` (see its own
[AGENTS.md](./lib/AGENTS.md)).

Read order for a fresh agent:

1. Repo-root [AGENTS.md](../AGENTS.md) — what the repo ships, what's
   running, what's deliberately empty right now.
2. This file — pipeline shape, conventions for CLI scripts.
3. [`lib/AGENTS.md`](./lib/AGENTS.md) — rules for the reusable
   helpers underneath the CLI layer.

## Pipeline shape

```
build_corpus.py  →  data/corpus.parquet
                         │
                         ├─ v2 (doc-level, shipped)
                         │       │
                         │       ▼
                         │     embed_via_openrouter.py
                         │
                         ├─ v3 (chunk-level, shipped)
                         │       │
                         │       ▼
                         │  build_corpus_chunks.py
                         │       │   data/corpus_chunks_<S>_<O>.parquet
                         │       ▼
                         │  embed_via_openrouter.py  (same script,
                         │       │   different --corpus)
                         │       ▼
                         │       …
                         │
                         ├─ segments (section-level, shipped)
                         │       │
                         │       ▼
                         │  build_corpus_segments.py
                         │       │   data/corpus_segments_<size>.parquet
                         │       ▼
                         │  embed_via_openrouter.py  (--corpus +
                         │       │   --out data/segments_corpus)
                         │       ▼
                         │       …
                         │
                         └─ sentences (sentence-level, shipped, LOCAL)
                                 │
                                 ▼
                            build_corpus_sentences.py
                                 │   data/corpus_sentences.parquet
                                 ▼
                            embed_local_sentences.py   ← local vLLM,
                                 │   NOT embed_via_openrouter.py
                                 ▼
                            fit_zca.py
                                 │
                                 ▼
                            finalize_sentence_meta.py  ← required
                                 │
                                 ▼
                            index_backgrounds.py

embed_via_openrouter.py  →  data/chunks_<slug>/chunk_NNNN.npy
                         │   data/manifest_<slug>.jsonl
                         │   data/cost_report_<slug>.json
                         ▼
fit_zca.py               →  backgrounds/<name>/{W_A,mu_A,eigvals_A}.npy
                         │   backgrounds/<name>/<name>.meta.json
                         ▼
index_backgrounds.py     →  REGISTRY.md, registry.json
                         ▲
                         │
run_full.sh ─────────────┘  orchestrates all four
```

The v2 and v3 branches share **everything** downstream of the corpus
parquet — embed, fit, index, registry.  The only difference is whether
the parquet rows are docs or chunks.  See [GOTCHAS.md §1](../GOTCHAS.md)
for why those backgrounds are not interchangeable.

Each step is **idempotent**:

- `build_corpus.py` skips when `data/corpus.parquet` already exists.
- `embed_via_openrouter.py` resumes from the highest `chunk_NNNN.npy`
  on disk, with skipped docs replaced by zero-vector placeholders so
  chunk row N always maps to corpus row N.
- `fit_zca.py` overwrites the target `backgrounds/<name>/` — re-run
  to refresh.
- `index_backgrounds.py` regenerates `REGISTRY.md` + `registry.json`
  from whatever is currently in `backgrounds/`.

That property is what makes the pipeline survive killed runs, network
flakes, and refactors: you can stop it anywhere and re-launch
`run_full.sh` without losing work or double-billing.

## What each script owns

| Script | Owns |
|---|---|
| `build_corpus.py` | HF dataset streaming (wiki + FineWeb-2 PL + oasst), per-source filters, `MIN_DOC_CHARS=500` floor, manifest write, corpus fingerprint. |
| `build_corpus_chunks.py` | v3 only: read `data/corpus.parquet`, sentence-aware chunk via `lib/chunker.py`, write `data/corpus_chunks_<size>_<overlap>.parquet`.  Output schema is a superset of `corpus.parquet` (adds `doc_sha`, `chunk_idx`) so the rest of the pipeline runs unchanged. |
| `build_corpus_segments.py` | segments only: read `data/corpus.parquet`, section-level split via `lib/segmenter.py` (1024-token cap, no overlap, heading-first separators), write `data/corpus_segments_<size>.parquet`.  Same schema contract as the chunks parquet (adds `doc_sha`, `segment_idx`). |
| `build_corpus_keywords.py` | kw only: mine 50 000 keyword-like phrases (1–5-word n-grams, stopwords-pl edge filter, df ≥ 3, stratified word-count mix) → `data/corpus_keywords.parquet`. |
| `build_corpus_paragraphs.py` | paragraphs only: read a corpus parquet, blank-line split via `lib/paragrapher.py`, write `data/corpus_<tag>_paragraphs.parquet` (adds `doc_sha`, `paragraph_idx`). Used for both `pl_mixed50k` and `med_pl`. |
| `build_corpus_sentences.py` | sentences only: reservoir-sample 130 000 sentences (`--sample-size`, `--seed 42`) out of `data/corpus.parquet` → `data/corpus_sentences.parquet` + `.stats.json` (the stats file records the *complete* generated population, not just the sample). Sentence splitting comes from `entropy_lab.features.chunk.split_sentences` — external, see below. |
| `embed_local_sentences.py` | sentences only: embed against a **local vLLM** OpenAI-compatible endpoint (`--endpoint http://127.0.0.1:8002/v1/embeddings`, `--model Qwen/Qwen3-Embedding-4B`), MRL-slice native 2560-d → `--mrl-dim 1024`, L2-renormalize, write fp16 `chunk_NNNN.npy` under `data/chunks_qwen_qwen3-embedding-4b_sentences/`. No API key, no cost report. |
| `embed_local_st.py` | `pl_kwmix900k` family: embed **in-process** with sentence-transformers (`--device cuda`), no server at all. This is the path for models nobody serves as an embedding API — `BAAI/bge-m3`, `Qwen/Qwen3-Embedding-0.6B`, `google/embeddinggemma-300m`. Writes the same fp16 `chunk_NNNN.npy` layout `fit_zca.py` expects, and is resumable (completed chunks are skipped). Prompts default to off: only embeddinggemma defines them, and its `Clustering` prompt measurably *hurts* short phrases. |
| `build_mixed_kw_corpus.py` | `pl_kwmix900k` only: assemble 900 000 **lowercase** Polish phrases from three public sources — Wikipedia PL article titles (58 %), web n-grams from `build_corpus_keywords.py` (27 %), `clarin-knext/msmarco-pl` queries (15 %) — deduped, same 1–5-word length mix as `kw`. Lowercase throughout: the tokenisers are case-sensitive, so mixing cased titles with lowercase n-grams would fit a bimodal distribution matching neither at inference. |
| `finalize_sentence_meta.py` | sentences only: patch fields into `backgrounds/<name>/<name>.meta.json` that `fit_zca.py` cannot know — `purpose`, `granularity: sentences`, `embedding_endpoint`, `provider: local_vllm`, `build_script`, notes. Must run **before** `index_backgrounds.py`. |
| `validate_sentence_background.py` | sentences only: sanity-check raw vs whitened on real website sentences pulled from a SQLite page cache (`--database … pages.sqlite`, `--domain rozenberger.com`), and compare the `sentences` background against the `chunks` one on a held-out split. Diagnostic, not part of the build. |
| `run_kw_fits.sh` | Fit every `kw` background whose embeddings are complete under `data/kw_corpus/` (all four models) → index.  Refuses partial embeds. |
| `run_oai_fits.sh` | Fit the OpenAI `doc` + `chunks` backgrounds from `data/chunks_<model>/` and `data/chunks_corpus/chunks_<model>/` → index.  Refuses partial embeds. |
| `run_paragraphs.sh` | All four models' `paragraphs` fits in one launch (general corpus). |
| `embed_via_openrouter.py` | The adaptive-batch retry loop. Imports HTTP, tokenizer, persistence from `lib/`. |
| `fit_zca.py` | Argparse + `lib.zca.fit` + `lib.zca.write_meta`. ~110 lines. |
| `index_backgrounds.py` | Walk `backgrounds/`, read every `*.meta.json`, write `REGISTRY.md` + `registry.json`. Deterministic — depends only on what's on disk. |
| `run_full.sh` | Orchestrator: `.env` load, defaults (`MODELS`, `NAME_PREFIX`, `CORPUS`, `OUT_ROOT`, `DIMS_<MODEL>`, `PROVIDER_ORDER`), loops embed + N×fit per model, final index.  For derived corpora (chunks / segments) set `CORPUS` + a dedicated `OUT_ROOT` — a hard guard refuses `OUT_ROOT=data` with a non-default `CORPUS` (it would resume from the doc-level embeddings and poison Σ). |

## Medical (`med_pl`) scripts

A parallel build chain for the Polish medical corpus (ChPL drug
labels). Same embed/fit/index hydraulics, different corpus. Assemble
→ embed → fit is one orchestrator, `run_med.sh`.

| Script | Owns |
|---|---|
| `build_med_pl_corpus_chpl.py` | Scrape ChPL labels per-ID from the Polish drug registry API (no bulk export — list endpoint is access-denied) → `data/med_pl/chpl.parquet` + `chpl.jsonl` (13 514 docs). |
| `build_med_pl_corpus_chpl_ocr.py` | OCR the image-only ChPL scans (no text layer) → `data/med_pl/chpl_ocr.parquet`. Local Qwen3-VL (or cloud vision), Markdown output, `max_tokens=4096`. 878 scans. |
| `build_med_pl_corpus_pes.py` | Build the PES board-exam question set → `data/med_pl/pes.parquet` (170 950 rows). Held separate from the doc corpus (length-band mismatch — see GOTCHAS §1). |
| `build_med_pl_corpus.py` | Assemble the shipped doc corpus from scrape + OCR → `data/corpus_med_pl.parquet` (14 392 docs). |
| `run_med.sh` | End-to-end `med_pl` orchestrator: derive granularity → embed (per model) → fit ZCA (full MRL grid) → index, for all four models × `doc paragraphs chunks` (te3 minus `doc`). ⚠️ paid (OpenRouter + OpenAI). `GRANS` / `MODELS_FILTER` / `PARA_SAMPLE` / `CHUNK_SAMPLE` scope the run. |

OCR benchmarking / one-off helpers (not part of the shipped pipeline —
used while tuning the OCR step): `ocr_bench.py`, `run_ocr_bench.sh`,
`run_3way_bench.sh` (compare OCR models/prompts), `download_med_pdf.py`,
`dump_med_txt.py`, `dl_serializer.sh`, `finish_med_doc.sh`.

## Sentences (local vLLM)

The `sentences` granularity is the odd one out — it is the only path that
**does not** go through `embed_via_openrouter.py`. Sentences are short
and there are 130 000 of them; paying per-token for that made no sense,
so they are embedded against a local vLLM server instead. The run is
free, but it has preconditions the other paths don't:

- **A vLLM server must be up on `127.0.0.1:8002`** serving
  `Qwen/Qwen3-Embedding-4B` over the OpenAI-compatible
  `/v1/embeddings` route. Not ollama — the Q4 quant there gives
  different vectors, and a background fit on one is invalid for the
  other.
- **`entropy_lab` must be importable.** `build_corpus_sentences.py` and
  `validate_sentence_background.py` do `sys.path.insert` on
  `/home/spark001/Workspaces/entropy_001/src` to reach
  `entropy_lab.features.chunk.split_sentences`. That package lives
  outside this repo; without it the scripts fail at import time. If the
  sentence splitter ever changes, every `sentences` background is stale
  — the unit of fit changed.
- **There is no `run_sentences.sh`.** The four scripts are run by hand,
  in order. `finalize_sentence_meta.py` is not optional: `fit_zca.py`
  writes a generic meta, and without the patch the registry entry lacks
  `granularity`, so nobody downstream can tell the background apart from
  a doc-level one.

Native vectors are 2560-d and get sliced to 1024 then L2-renormalized
**before** both storage and fit — the fit never sees the full-width
vectors. Shipped background: `qwen3_4b_pl_mixed50k_sentences_mrl1024`.

## Conventions

1. **Top-level scripts are thin.**  If a function has any chance of
   being useful from another script (or a test), it goes under
   `lib/`.  See [`lib/AGENTS.md`](./lib/AGENTS.md) for the cut rules.

2. **Argparse, not env vars, for per-run options.**  Env vars are
   reserved for **secrets** (`OPENROUTER_API_KEY`) and for
   `run_full.sh` defaults that the user might want to override
   without editing argv (`MODELS`, `DIMS_4B`, `PROVIDER_ORDER`,
   `NAME_PREFIX`, `MAX_CHARS`).

3. **Exit codes:** `0` success, `2` user error (missing API key, no
   corpus, bad CLI arg).  Reserve `1` for genuinely unexpected
   Python exceptions — argparse default.

4. **Never print the API key.**  It comes in via env or `.env`;
   never as a CLI argument; never logged at any level.

5. **Logging format is fixed:**
   ```python
   logging.basicConfig(
       level=logging.INFO,
       format="%(asctime)s %(levelname)s %(name)s %(message)s",
   )
   ```
   …configured once in `main()` of the script, not in `lib/`.
   `lib/` modules each do `logger = logging.getLogger(__name__)` so
   their messages show up tagged `lib.foo` automatically.

6. **`if __name__ == "__main__": sys.exit(main())`** at the bottom of
   every CLI script.  Makes them importable for tests without firing
   side effects.

7. **`meta.json` is written by `lib.zca.write_meta`** — one writer, so
   every background's metadata has the same shape.
   `finalize_sentence_meta.py` breaks this rule: it patches the file
   after the fact. That's known debt from bringing the `sentences` path
   up quickly. The fix is to teach `write_meta` about
   `granularity`/`provider`/`endpoint` and delete the patch script —
   don't add a second post-hoc patcher for the next granularity.

## When you edit a script

- Bump `AGENTS.md` (this file or the root one) if you've changed:
  - The pipeline shape diagram above.
  - Defaults that someone tuning a re-run cares about (mix sizes,
    `MIN_DOC_CHARS`, `DIMS_<MODEL>`, `PROVIDER_ORDER`, token cap).
  - The exit-code contract.
- Don't introduce new top-level dependencies — every package in
  `requirements.txt` already pays for itself.  If you need a new
  one, justify it in the commit message.

## Common operations

```bash
# Full rebuild from scratch (corpus → embed × 2 models × {doc, chunks} → fit × 22 MRL → index).
PY=/path/to/venv/bin/python bash scripts/run_full.sh

# Single model.
MODELS="qwen/qwen3-embedding-4b" bash scripts/run_full.sh

# Just one MRL dim for 4B (e.g. you only care about 1024).
DIMS_4B="1024" MODELS="qwen/qwen3-embedding-4b" bash scripts/run_full.sh

# Provider routing: skip siliconflow (~4× more expensive than the cheapest).
PROVIDER_ORDER="--ignore-providers siliconflow" bash scripts/run_full.sh

# Segments granularity end-to-end (one model): build the derived
# corpus, then run the orchestrator against it.
python scripts/build_corpus_segments.py
CORPUS=data/corpus_segments_1024.parquet OUT_ROOT=data/segments_corpus \
  NAME_PREFIX=pl_mixed50k_segments MODELS="qwen/qwen3-embedding-4b" \
  bash scripts/run_full.sh

# Sentences granularity end-to-end (free — local vLLM on :8002, no API key).
# No orchestrator; run the four steps in order.
python scripts/build_corpus_sentences.py
python scripts/embed_local_sentences.py
python scripts/fit_zca.py \
  --chunks data/chunks_qwen_qwen3-embedding-4b_sentences \
  --name qwen3_4b_pl_mixed50k_sentences_mrl1024 \
  --model Qwen/Qwen3-Embedding-4B \
  --corpus data/corpus_sentences.parquet   # else the fingerprint is the doc corpus'
# no --truncate-to: embed_local_sentences.py already sliced to 1024 and renormalized
python scripts/finalize_sentence_meta.py     # required before indexing
python scripts/validate_sentence_background.py   # optional sanity check
python scripts/index_backgrounds.py

# Re-index after manually placing artefacts in backgrounds/.
python scripts/index_backgrounds.py
```

For the running-process tooling (tmux session, monitor filter, kill
+ resume) see the root [AGENTS.md](../AGENTS.md).
