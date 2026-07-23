# scripts/

CLI entry points for the whole pipeline: **Polish corpus → embeddings
→ ZCA whitening backgrounds**. The reusable helpers live under
[`lib/`](lib/); these top-level scripts are thin wrappers around them.

For the full pipeline shape, conventions, and per-script contracts see
[`AGENTS.md`](AGENTS.md). This file is the short human-facing map.

## Two corpora

| Corpus | Assemble | Fit family |
|---|---|---|
| **`pl_mixed50k`** (general: wiki + FineWeb-2 + oasst) | `build_corpus.py` | `run_full.sh`, `run_paragraphs.sh`, `run_oai_fits.sh`, `run_kw_fits.sh` |
| **`med_pl`** (medical: ChPL drug labels + OCR) | `build_med_pl_corpus*.py` | `run_med.sh` |

The `sentences` granularity is derived from `pl_mixed50k` but has **no
orchestrator** — it runs locally against vLLM, step by step (see below).

## The stages

```
build_corpus*.py        →  data/*.parquet          (corpus, one row per unit)
build_corpus_<gran>.py  →  data/*_<gran>.parquet   (derive paragraphs/chunks/segments/kw)
embed_via_openrouter.py →  data/.../chunks_<slug>/ (embeddings + manifest + cost report)
embed_local_sentences.py→  data/chunks_..._sentences/ (sentences only: local
                                                       vLLM, free, no manifest)
fit_zca.py              →  backgrounds/<name>/      (W_A, mu_A, eigvals_A, meta.json)
index_backgrounds.py    →  REGISTRY.md, registry.json
```

Every stage is **idempotent** — stop anywhere and re-launch the
orchestrator without losing work or double-billing.

## Common runs

```bash
# General corpus, full rebuild (doc + chunks, both Qwen models):
PY=/path/to/venv/bin/python bash scripts/run_full.sh

# General paragraphs (all four models):
bash scripts/run_paragraphs.sh

# Medical family end-to-end (all four models, doc/paragraphs/chunks;
# ⚠️ paid — OpenRouter for Qwen, OpenAI for te3):
bash scripts/run_med.sh
# scope it: cheap Qwen-only, one granularity
GRANS="doc" MODELS_FILTER="qwen3_4b qwen3_8b" bash scripts/run_med.sh

# Sentences — free, but needs a local vLLM serving Qwen3-Embedding-4B
# on 127.0.0.1:8002 (not ollama). Run the steps in order; there is no
# orchestrator, and finalize_sentence_meta.py is mandatory.
python scripts/build_corpus_sentences.py
python scripts/embed_local_sentences.py
python scripts/fit_zca.py \
  --chunks data/chunks_qwen_qwen3-embedding-4b_sentences \
  --name qwen3_4b_pl_mixed50k_sentences_mrl1024 \
  --model Qwen/Qwen3-Embedding-4B \
  --corpus data/corpus_sentences.parquet
python scripts/finalize_sentence_meta.py
python scripts/index_backgrounds.py

# Re-index after placing/refitting artefacts by hand:
python scripts/index_backgrounds.py
```

## Granularities

`doc` (whole document) · `paragraphs` (blank-line) · `chunks`
(fixed token windows) · `segments` (article sections, general only) ·
`kw` (keyword phrases, general only) · `sentences` (one sentence per
row, general only, embedded locally). A background's fit granularity
**must** match your index granularity — see
[`../GOTCHAS.md`](../GOTCHAS.md#1-background-granularity-must-match-index-granularity).

## OCR helpers (medical)

`build_med_pl_corpus_chpl_ocr.py` OCRs image-only ChPL scans to
Markdown. The `*_bench*` / `download_med_pdf.py` / `dump_med_txt.py`
scripts are one-off tools used while tuning the OCR step, not part of
the shipped pipeline.

## Secrets

API keys come from `.env` (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`) —
never as CLI args, never logged. See [`.env.example`](../.env.example).

The `sentences` path needs no key at all — it talks to a local vLLM
endpoint. It does, however, need `entropy_lab` (the sentence splitter)
importable from `/home/spark001/Workspaces/entropy_001/src`, which is
outside this repo.
