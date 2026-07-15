# AGENTS.md — polish-whitening-backgrounds

Instructions for any coding agent (Claude Code, Codex, Cursor, Gemini, …)
working in this repo. Read this file FIRST on every new session.

Communication with the user (chat, PR comments): **Polish**.
All files in repo: **English** (incl. this one).

## What this repo is

Public, drop-in artefact repo:
[https://github.com/romek-rozen/polish-whitening-backgrounds](https://github.com/romek-rozen/polish-whitening-backgrounds)

Pre-fitted ZCA whitening backgrounds (`W_A.npy`, `mu_A.npy`,
`eigvals_A.npy`) for the Qwen3-Embedding family **and OpenAI
text-embedding-3-small/-large** on Polish text. Goal:
**colleagues don't recompute μ and Σ on a 50k-doc Polish corpus** —
they clone, load via `loader.py`, apply. Provenance + diagnostics in
each `backgrounds/<name>/<name>.meta.json`.

The Polish corpus is sampled live from public HuggingFace datasets
(Wikipedia PL + FineWeb-2 PL + oasst-1 PL) with a fixed seed, into
this repo's `data/` (git-ignored). Never read or write to
`/home/spark001/Spark-testy/llm-extraction-embedding-dgx-final-setup/`
or any of its `data/` / `results/` / `models/` subtrees — that's the
user's main project, not this one.

## Current state (2026-07-15)

**Shipped on GitHub `main`**: **160 backgrounds** — 103 general
(`pl_mixed50k`) + 57 medical (`med_pl`). `registry.json` is the source
of truth for what is actually committed right now.

General corpus `pl_mixed50k` (103):

| Background dirs | Dim range | Status |
|---|---:|---|
| `qwen3_4b_pl_mixed50k_{doc,chunks,segments,kw,paragraphs}_mrl{2560,1536,1024,768,512}/` | 2560…512 | shipped |
| `qwen3_8b_pl_mixed50k_{doc,chunks,segments,kw,paragraphs}_mrl{4096,3072,2048,1024,768,512}/` | 4096…512 | shipped |
| `te3small_pl_mixed50k_{doc,chunks,kw,paragraphs}_mrl{1536,1024,768,512,256}/` | 1536…256 | shipped |
| `te3large_pl_mixed50k_{doc,chunks,kw,paragraphs}_mrl{3072,2048,1536,1024,768,512,256}/` | 3072…256 | shipped |

Medical corpus `med_pl` (57; `doc`/`paragraphs`/`chunks` all fit on the
same 14 392-doc OCR-inclusive corpus, `paragraphs`/`chunks` on a 150k
sample):

| Background dirs | Dim range | Status |
|---|---:|---|
| `qwen3_4b_med_pl_{doc,paragraphs,chunks}_mrl{2560,1536,1024,768,512}/` | 2560…512 | shipped (15) |
| `qwen3_8b_med_pl_{doc,paragraphs,chunks}_mrl{4096,3072,2048,1024,768,512}/` | 4096…512 | shipped (18) |
| `te3small_med_pl_{paragraphs,chunks}_mrl{1536,1024,768,512,256}/` | 1536…256 | shipped (10) |
| `te3large_med_pl_{paragraphs,chunks}_mrl{3072,2048,1536,1024,768,512,256}/` | 3072…256 | shipped (14) |

No **te3 `doc`**: text-embedding-3 caps input at 8 191 tokens and 60 %
of the ChPL docs exceed it (median ~10.6 k tokens), so a te3 `doc` fit
would whiten mostly truncated document heads — Qwen (longer context)
ships `doc`, te3 ships only `paragraphs` + `chunks`.

`med_pl` is a **separate corpus** (drug labels), not a granularity of
`pl_mixed50k`. Source = **ChPL** (*Charakterystyka Produktu
Leczniczego*) scraped per-ID from the Polish drug registry (*Rejestr
Produktów Leczniczych*): **14 392 docs** (13 514 scraped + 878 OCR'd
image-only scans, merged), median ~27 k chars, 475 M chars. `doc` fits
on all 14 392 docs; `paragraphs`/`chunks` fit on a 150 000-row random
sample (seed 42) of the ~1.18 M paragraphs / ~0.84 M chunks (300-tok
windows, 50-tok overlap) they split into. A second source, **PES**
(170 950 board-exam questions from HF
`amu-cai/medical-exams-PES-PL-2007-2024`), is built but deliberately
held out of the doc corpus (~370-char questions are a different length
band — would violate the §1 granularity contract); reserved for a
future short-text medical background. The `segments`/`kw` medical
granularities are not built.

The `paragraphs` granularity (all four models) on the general corpus is
23 dirs (`<model>_pl_mixed50k_paragraphs_mrl<dim>`): 4b × 5 dims,
8b × 6, te3small × 5, te3large × 7.

Spend: Qwen doc+chunks ~$2.77 via OpenRouter (4b doc $0.92, 8b doc
$0.46, 4b chunks $0.95, 8b chunks $0.48); Qwen segments ~$1.39
(4b $0.93, 8b $0.46 — 46.3 M tokens each); OpenAI doc+chunks ~$14 via
api.openai.com (~95 M tokens; 3-small $0.02/M, 3-large $0.13/M); the
four `kw` families cost pennies (~0.4 M tokens each). The medical
`med_pl` build (`run_med.sh`) adds Qwen para+chunks (~$1–2) + OpenAI
te3 para+chunks (~$8). Orchestrators:
`scripts/run_full.sh` (Qwen doc; chunks with
`NAME_PREFIX=pl_mixed50k_chunks` + `CORPUS=data/corpus_chunks_512_64.parquet`;
segments with `NAME_PREFIX=pl_mixed50k_segments` +
`CORPUS=data/corpus_segments_1024.parquet` + `OUT_ROOT=data/segments_corpus`),
`scripts/run_kw_fits.sh` (all kw fits), `scripts/run_oai_fits.sh`
(OpenAI doc+chunks fits), `scripts/run_paragraphs.sh` (all four
`paragraphs` fits in one launch), `scripts/run_med.sh` (the whole
`med_pl` family — build/embed/fit/index for all four models ×
`doc paragraphs chunks`, te3 minus `doc`). The `kw` granularity exists for
keyword grouping / clustering (Google Ads use case) — whole-doc
backgrounds misfit on 1–5-word phrases.  The `segments` granularity
(Qwen models only) exists for internal-linking retrieval —
segment→segment matching with target docs represented by their own
segments.  The `paragraphs` granularity (all four models) is one
embedding per blank-line paragraph — a distinct length band between
`kw` and `chunks` (196 759 paragraphs, median 488 chars, ~32 M
tokens): whitening paragraph embeddings with a `_chunks_` background
leaves ~7× the residual anisotropy (top_ev/mean ≈ 13.5 vs ≈ 2.0 with
a proper `_paragraphs_` background on Qwen3-4B), so the §1 granularity
contract applies to it too.

**Retired** (still in git history, removed from `main` working tree):
- `polish_mixed_50k_v1{,_mrl1024,_mrl1536}/`
- `corpus205_n3155/`
- `polish_smoke_1500/`

Replaced because: (a) the v1 mix used noisier mC4 + sentence-only KLEJ;
(b) the new naming scheme is model-first with an explicit `_doc_` or
`_chunks_` granularity tag (see Naming convention below).

**Local-only, not in git** (under `data/`, git-ignored):
- `data/corpus.parquet` — the 50 042-doc current corpus.
  Fingerprint `6e9e965ffbb6dbe6…`.  Don't delete without a copy —
  it costs ~10 min of HF streaming to rebuild.
- `data/corpus_45k_backup.parquet` — pre-fineweb_more snapshot from
  before the corpus enlargement.  Safe to delete once we're confident
  in the 50k run.
- `data/corpus_chunks_512_64.parquet` — 129 181 chunks from the
  splitter (`lib.chunker`, post merge_tiny + strip_overlap_fragments).
  Used for the shipped chunk-level fits.
- `data/corpus_keywords.parquet` — 50 000 keyword-like phrases mined
  by `build_corpus_keywords.py` (needs `data/stopwords-pl.txt`,
  downloaded from stopwords-iso/stopwords-pl).  Used for the `kw` fits.
- `data/corpus_segments_1024.parquet` — 73 692 section-level segments
  from `lib.segmenter` (1024-token cap, no overlap, merge_tiny
  floor=300 chars).  Used for the shipped `segments` fits.
- `data/corpus_paragraphs.parquet` — 196 759 blank-line paragraphs
  from `lib.paragrapher` (strict `\n\n` split, oversize >512-token
  paragraphs subdivided at sentence boundaries, merge_tiny floor=120
  chars).  Used for the shipped `paragraphs` fits.
- `data/chunks_qwen_qwen3-embedding-{4b,8b}/`,
  `data/chunks_text-embedding-3-{small,large}/` — doc-level embedding
  output of the embed step.  Resumable; up to ~700 MB each.
- `data/chunks_corpus/chunks_<model>/` — chunk-level embedding output
  (129 181 rows each, all four models).
- `data/kw_corpus/chunks_<model>/` — keyword-level embedding output
  (50 000 rows each, all four models).
- `data/segments_corpus/chunks_<model>/` — segment-level embedding
  output (73 692 rows each, both Qwen models).
- `data/paragraphs_corpus/chunks_<model>/` — paragraph-level embedding
  output (196 759 rows each, all four models).
- `data/med_pl/` — raw medical sources: `chpl.parquet` + `chpl.jsonl`
  (13 514 ChPL drug labels scraped per-ID), `chpl_ocr.parquet` (878
  OCR'd image-only scans), `pes.parquet` (170 950 PES board-exam
  questions, held separate from the doc corpus).
- `data/corpus_med_pl.parquet` — the assembled ChPL medical doc corpus
  (**14 392 docs** = 13 514 scraped + 878 OCR'd) built by
  `build_med_pl_corpus.py`.  Used for all shipped `med_pl` fits.
- `data/med_corpus/{doc,paragraphs,chunks}/chunks_<model>/` — medical
  embedding output (Qwen for all three grans; te3 for
  `paragraphs`/`chunks`); `paragraphs`/`chunks` are 150 000-row random
  samples (seed 42) of the docs' ~1.18 M paragraphs / ~0.84 M chunks.

## Naming convention

```
<model>_<corpus>_<granularity>_mrl<dim>/
   │       │          │            │
   │       │          │            └─ mrl<dim>     MRL refit at dim N
   │       │          └─ doc | chunks | segments | kw | paragraphs   embedding granularity
   │       └─ pl_mixed50k | med_pl                 language + corpus tag
   └─ qwen3_4b | qwen3_8b | te3small | te3large    embedding model
```

Model first means `ls backgrounds | grep qwen3_4b` lists every
variant of that model in one shot.  `te3small` / `te3large` =
OpenAI `text-embedding-3-small` / `-large`.

The corpus tag slot takes `pl_mixed50k` (general web/wiki/oasst mix)
**or `med_pl`** (Polish medical / ChPL drug labels — a separate corpus,
not a granularity).  `med_pl` ships all four models: Qwen at
`doc`/`paragraphs`/`chunks`, te3 at `paragraphs`/`chunks` (no te3
`doc`, input-length cap — e.g. `te3large_med_pl_paragraphs_mrl1024`).

- `_doc_` — one embedding per whole document.
- `_chunks_` — one embedding per ~512-token chunk with 64-token
  overlap, produced by `scripts/lib/chunker.py`.
- `_segments_` — one embedding per article **section** (up to 1024
  tokens, no overlap), produced by `scripts/lib/segmenter.py`.
  Built for internal linking: segment→segment matching with targets
  aggregated per doc (see GOTCHAS.md §1).
- `_paragraphs_` — one embedding per **blank-line paragraph**, produced
  by `scripts/lib/paragrapher.py` (strict `\n\n` split, one paragraph
  per row, oversize >512-token paragraphs subdivided at sentence
  boundaries, `merge_tiny` floor 120 chars). A distinct length band
  between `kw` and `chunks` (median ~490 chars).
- `_kw_` — one embedding per keyword-like phrase (1–5 words), mined
  by `scripts/build_corpus_keywords.py`.

Granularity contract: a background's fit-time granularity MUST match
its inference-time granularity.  Don't whiten paragraphs with a
doc-level background, or keyword lists with anything but a `kw`
background — see [`GOTCHAS.md`](GOTCHAS.md) §1.

`run_full.sh` builds names as
`${MODEL_SHORT}_${NAME_PREFIX}_mrl${DIM}` where `MODEL_SHORT` is the
last segment of the OpenRouter id with `-` → `_` (so
`qwen/qwen3-embedding-4b` → `qwen3_4b`).  Default `NAME_PREFIX` is
`pl_mixed50k_doc`; for chunk fits set
`NAME_PREFIX=pl_mixed50k_chunks` before launching, for segment fits
`NAME_PREFIX=pl_mixed50k_segments` (plus `CORPUS` + `OUT_ROOT`, see
below).

## Pipeline shape

`scripts/run_full.sh` orchestrates four phases:

1. `build_corpus.py` → `data/corpus.parquet`  (skip if exists)
2. `embed_via_openrouter.py` per model → `data/chunks_<slug>/chunk_*.npy`  (resume by chunk file)
3. `fit_zca.py` per (model, MRL dim) → `backgrounds/<name>/{W_A,mu_A,eigvals_A}.npy + <name>.meta.json`
4. `index_backgrounds.py` → `REGISTRY.md` + `registry.json`

For chunk-level fits an extra phase 1.5 runs first:
`build_corpus_chunks.py --chunk-size 512 --chunk-overlap 64`
producing `data/corpus_chunks_512_64.parquet` which the embed step
then consumes via `--corpus` instead of `corpus.parquet`.  For
keyword-level fits the analogue is `build_corpus_keywords.py` →
`data/corpus_keywords.parquet` (embed with `--out data/kw_corpus/`),
then `run_kw_fits.sh`.  For segment-level fits:
`build_corpus_segments.py` → `data/corpus_segments_1024.parquet`,
then `run_full.sh` with `CORPUS=data/corpus_segments_1024.parquet
OUT_ROOT=data/segments_corpus NAME_PREFIX=pl_mixed50k_segments` —
`run_full.sh` hard-refuses a derived `CORPUS` with the default
`OUT_ROOT=data` (it would resume from the doc-level embeddings and
poison Σ).  For paragraph-level fits: `build_corpus_paragraphs.py` →
`data/corpus_paragraphs.parquet`, then `run_paragraphs.sh`, which
builds/embeds/fits all four models (paragraphs is the only
granularity covering all four in a single orchestrator).

For the **medical `med_pl`** corpus the sources are built by
`build_med_pl_corpus_chpl.py` (ChPL scrape → `data/med_pl/chpl.parquet`;
no bulk export exists — the registry list/export endpoint is
access-denied, so it scrapes per-ID) and `build_med_pl_corpus_pes.py`
(PES questions → `data/med_pl/pes.parquet`, held separate);
`build_med_pl_corpus_chpl_ocr.py` OCRs the 878 image-only scans →
`data/med_pl/chpl_ocr.parquet` (local Qwen3-VL, Markdown output).
`build_med_pl_corpus.py` assembles the ChPL doc corpus (scrape + OCR)
→ `data/corpus_med_pl.parquet` (14 392 docs).  `run_med.sh` then embeds
+ fits the `med_pl` family into `data/med_corpus/{doc,paragraphs,chunks}/`
and `backgrounds/`; it ships all four models × `doc paragraphs chunks`
(te3 minus `doc`, input-length cap).  The `paragraphs`/`chunks` medical
fits use a ~150 000-row random sample (seed 42) rather than all ~1.18 M
paragraphs / ~0.84 M chunks — a representative Σ needs no more.

OpenAI models run through the **same** embed script against
`api.openai.com`: add `--base-url https://api.openai.com/v1/embeddings
--api-key-env OPENAI_API_KEY`, and use `--max-tokens-per-doc 8191`
for the doc corpus (OpenAI's hard input cap; tokenised via tiktoken
`cl100k_base` — see `OPENAI_TO_TIKTOKEN` in `scripts/lib/tokenizer.py`).
Their doc/chunks fits are orchestrated by `run_oai_fits.sh`.  Both
fit runners refuse to fit on incomplete embeddings (manifest row
count vs corpus row count).

Every script is idempotent — re-running `run_full.sh` resumes from
disk state without losing work or double-billing OpenRouter.  See
[`scripts/AGENTS.md`](scripts/AGENTS.md) for the CLI conventions and
[`scripts/lib/AGENTS.md`](scripts/lib/AGENTS.md) for the helper rules.

## Constants worth knowing

In `scripts/build_corpus.py`:
- `DEFAULT_MIX = {"wikipedia": 22500, "fineweb": 22500, "oasst": 5000, "fineweb_more": 5000}`
  (oasst yields only 42 / 5000 Polish threads, `fineweb_more` was
  appended at the end to get back to ~50k actual — order is
  load-bearing, see the comment on `DEFAULT_MIX`).
- `MIN_DOC_CHARS = 500` enforced on every source — paragraph not sentence.
- `seed = 42`, no per-doc upper cap by default.
- KLEJ + mC4 retired (sentence-only / noisy boilerplate).

In `scripts/embed_via_openrouter.py`:
- Default `--max-tokens-per-doc 30000` (Qwen3's context is 32k; we
  pre-flight truncate via Qwen3's own `tokenizer.json` pulled from HF).
  For OpenAI models use `8191` (their hard cap) — truncation then
  routes through tiktoken automatically.
- Adaptive batch: start 16, max 32, min 1; halves on 429 / 5xx /
  200-but-no-data, grows back after a clean streak.  For short-phrase
  corpora crank it (`--start-batch 64 --max-batch 256`) — the default
  batch wastes 50 min on what takes 5.
- `--ignore-providers siliconflow` by default (it's ~4× the price of
  Nebius / DeepInfra on Qwen3 embeddings); the provider block is only
  sent to OpenRouter, never to other `--base-url` backends.
- Resume: by chunk_NNNN.npy file count.  Skipped docs get a
  zero-vector placeholder so chunk row N maps to corpus row N.
- `cost_usd` in cost reports is OpenRouter-only — the OpenAI API
  doesn't return a `cost` field, so those reports show tokens but
  $0.00; compute spend from tokens × list price.

In `scripts/run_full.sh`:
- `MODELS="qwen/qwen3-embedding-4b qwen/qwen3-embedding-8b"`
- `NAME_PREFIX="pl_mixed50k_doc"`
- `DIMS_4B="2560 1536 1024 768 512"`  (5 fits, includes native 2560)
- `DIMS_8B="4096 3072 2048 1024 768 512"`  (6 fits, includes native 4096;
  2560 / 1536 dropped — 8B wasn't MRL-trained at those off-grid dims)

In `scripts/lib/chunker.py`:
- `chunk_size=512` Qwen3 tokens, `chunk_overlap=64`.
- `merge_tiny(chunks, min_chars=100)` forward-merges sub-100-char
  chunks into their next neighbour — fixes LangChain's "tiny
  paragraph between two \n\n separators" wart.
- `strip_overlap_fragments(chunks)` strips leading `[\.\,\;\:\!\?]+\s+`
  fragments dragged in by token-aligned overlap (chunk 2..N only).

## Operating notes (long-running embeds)

The two embeds (4B + 8B) run in **separate tmux sessions** so they can
proceed in parallel:

```bash
tmux ls   # whiten / w4b / w8b sessions if active
tmux capture-pane -t w8b -p | tail -20   # peek at progress
tmux attach -t w8b                       # attach to watch live
```

Monitoring is via a `tail -F data/run_4b.log data/run_8b.log` piped
through a grep filter for `Traceback|ERROR|FAILED|exception|SKIP|HTTP [45]|...`
events.  Use the Monitor tool for that — don't poll the logs by hand.

Killing the embed cleanly: `tmux send-keys -t <name> C-c` then verify
with `pgrep -af embed_via_openrouter`.  Chunks already on disk stay
valid; resume picks up from the highest chunk number.

## Safety rules

- **Never commit `.env`.**  In `.gitignore` but always double-check
  with `git status --short` before staging.
- **Never push to `main` without "ok push" from the user.**  This is
  public — rollback is messy.  Docs / scripts / GOTCHAS updates that
  don't touch `backgrounds/` are lower risk and the user has
  approved pushing those without asking each time, but artefact
  pushes always need explicit ok.
- **Never touch the user's main project at**
  `/home/spark001/Spark-testy/llm-extraction-embedding-dgx-final-setup/`.
  Long-running batch over real data lives there.  Read-only access to
  copy templates is OK; do not kill its processes (tmux sessions
  `waterfall_*`, `cbstress*`, anything matching
  `run_full_pipeline_waterfall.sh`).  Notably its `.venv` is what we
  use to run *our* python — installing or upgrading packages there
  affects the user's main project, so we created a local `.venv` in
  this repo for chunker-specific deps (`langchain-text-splitters`).
- **Never delete the user's data without explicit instruction.**  Also
  applies to this repo's `data/corpus.parquet` and
  `data/corpus_45k_backup.parquet` — rebuildable but costly.

## Update this AGENTS.md when

- Backgrounds get added or retired.
- `scripts/` defaults change (mix, cap, NAME_PREFIX, dims).
- The naming convention changes.
- The publication strategy changes (private repo, LFS, etc.).
- A non-obvious gotcha lands that future-you should know about
  (those tend to belong in [`GOTCHAS.md`](GOTCHAS.md) instead — link
  them from here rather than duplicate).
