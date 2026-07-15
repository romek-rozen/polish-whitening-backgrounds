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

## The stages

```
build_corpus*.py        →  data/*.parquet          (corpus, one row per unit)
build_corpus_<gran>.py  →  data/*_<gran>.parquet   (derive paragraphs/chunks/segments/kw)
embed_via_openrouter.py →  data/.../chunks_<slug>/ (embeddings + manifest + cost report)
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

# Re-index after placing/refitting artefacts by hand:
python scripts/index_backgrounds.py
```

## Granularities

`doc` (whole document) · `paragraphs` (blank-line) · `chunks`
(fixed token windows) · `segments` (article sections, general only) ·
`kw` (keyword phrases, general only). A background's fit granularity
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
