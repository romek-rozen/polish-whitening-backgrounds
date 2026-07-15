# data/

Working directory for the pipeline: corpora, embeddings, cost reports.

> **Not in git.** This whole directory is `.gitignore`d (~25 GB) — only
> this README is force-tracked. Everything here is **reproducible** from
> the scripts; the corpus fingerprint in each `backgrounds/*/meta.json`
> pins which corpus a shipped background saw. Nothing here is a source
> of truth you can't rebuild.

## Layout

### Corpus parquets (one row per unit)

| File | Corpus / granularity |
|---|---|
| `corpus.parquet` | `pl_mixed50k` documents (wiki + FineWeb-2 + oasst) |
| `corpus_chunks_512_64.parquet` | `pl_mixed50k` chunks (512-tok / 64 overlap) |
| `corpus_segments_1024.parquet` | `pl_mixed50k` segments (≤1024 tok) |
| `corpus_paragraphs.parquet` | `pl_mixed50k` paragraphs |
| `corpus_keywords.parquet` | `pl_mixed50k` keyword phrases |
| `corpus_med_pl.parquet` | `med_pl` documents (14 392 ChPL: scrape + OCR) |
| `corpus_med_pl_paragraphs.parquet` (+ `_150000` sample) | `med_pl` paragraphs |
| `corpus_med_pl_chunks_300_50.parquet` (+ `_150000` sample) | `med_pl` chunks (300-tok / 50 overlap) |

`_<N>` suffixes are random samples (seed 42) — the `paragraphs` /
`chunks` fits run on a 150 000-row sample, not every row.

### Embedding output (one subdir per model, `chunks_<model_slug>/`)

| Dir | Granularity |
|---|---|
| `chunks_<model>/` (repo root of `data/`) | `pl_mixed50k` doc |
| `chunks_corpus/` | `pl_mixed50k` chunks |
| `segments_corpus/` | `pl_mixed50k` segments |
| `paragraphs_corpus/` | `pl_mixed50k` paragraphs |
| `kw_corpus/` | `pl_mixed50k` keywords |
| `med_corpus/{doc,paragraphs,chunks}/` | `med_pl` (all grans) |

Each holds `chunk_NNNN.npy` vectors + `manifest_<slug>.jsonl` +
`cost_report_<slug>.json`.

### Medical sources — `med_pl/`

`chpl.parquet` (13 514 scraped labels) · `chpl_ocr.parquet` (878 OCR'd
scans) · `pes.parquet` (170 950 board-exam questions, held out of the
doc corpus).

## Rebuilding

See [`../scripts/README.md`](../scripts/README.md). General:
`run_full.sh` / `run_paragraphs.sh`. Medical: `run_med.sh`.
