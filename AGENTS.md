# AGENTS.md — polish-whitening-backgrounds

Instructions for any coding agent (Claude Code, Codex, Cursor, Gemini, …)
working in this repo. Read this file FIRST on every new session.

Communication with the user (chat, PR comments): **Polish**.
All files in repo: **English** (incl. this one).

## What this repo is

Public, drop-in artefact repo:
[https://github.com/romek-rozen/polish-whitening-backgrounds](https://github.com/romek-rozen/polish-whitening-backgrounds)

Pre-fitted ZCA whitening backgrounds (`W_A.npy`, `mu_A.npy`,
`eigvals_A.npy`) for the Qwen3-Embedding family on Polish text. Goal:
**colleagues don't recompute μ and Σ on a 50k-doc Polish corpus** —
they clone, load via `loader.py`, apply. Provenance + diagnostics in
each `backgrounds/<name>/<name>.meta.json`.

The Polish corpus is sampled live from public HuggingFace datasets
(Wikipedia PL + FineWeb-2 PL + oasst-1 PL) with a fixed seed, into
this repo's `data/` (git-ignored). Never read or write to
`/home/spark001/Spark-testy/llm-extraction-embedding-dgx-final-setup/`
or any of its `data/` / `results/` / `models/` subtrees — that's the
user's main project, not this one.

## Current state (2026-06-09)

**Shipped on GitHub `main`** (11 target backgrounds, 5/11 done):

| Background dir | Dim | Status |
|---|---:|---|
| `qwen3_4b_pl_mixed50k_doc_mrl2560/` | 2560 | ✅ |
| `qwen3_4b_pl_mixed50k_doc_mrl1536/` | 1536 | ✅ |
| `qwen3_4b_pl_mixed50k_doc_mrl1024/` | 1024 | ✅ |
| `qwen3_4b_pl_mixed50k_doc_mrl768/` | 768 | ✅ |
| `qwen3_4b_pl_mixed50k_doc_mrl512/` | 512 | ✅ |
| `qwen3_8b_pl_mixed50k_doc_mrl{4096,3072,2048,1024,768,512}/` | 4096…512 | ⏳ |

8B embed is mid-run in tmux session `w8b` against the new 50k corpus;
fits land automatically when its embed completes (~3h ETA from start
at 13:48).  The orchestrator is `scripts/run_full.sh` with
`MODELS=qwen/qwen3-embedding-8b` set in the launching tmux env.

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
- `data/corpus_chunks_512_64.parquet` — 129 181 chunks from the v3
  splitter (`lib.chunker`, post merge_tiny + strip_overlap_fragments).
  Used for the planned v3 chunk-level fits.
- `data/chunks_qwen_qwen3-embedding-{4b,8b}/` — embedding output of
  the embed step.  Resumable; ~700 MB each at completion.

## Naming convention

```
<model>_<corpus>_<granularity>_mrl<dim>/
   │       │          │            │
   │       │          │            └─ mrl<dim>     MRL refit at dim N
   │       │          └─ doc | chunks              embedding granularity
   │       └─ pl_mixed50k                          language + corpus tag
   └─ qwen3_4b | qwen3_8b                          embedding model
```

Model first means `ls backgrounds | grep qwen3_4b` lists every
variant of that model in one shot.

- `_doc_` — one embedding per whole document.  Used for the v2
  shipped fits.
- `_chunks_` — one embedding per overlapping ~512-token chunk
  (target overlap 64 tokens).  Planned v3 fits — corpus parquet
  already on disk, embed + fit pending after 8B doc-level finishes.

`run_full.sh` builds names as
`${MODEL_SHORT}_${NAME_PREFIX}_mrl${DIM}` where `MODEL_SHORT` is the
last segment of the OpenRouter id with `-` → `_` (so
`qwen/qwen3-embedding-4b` → `qwen3_4b`).  Default `NAME_PREFIX` is
`pl_mixed50k_doc`; for v3 chunk fits set
`NAME_PREFIX=pl_mixed50k_chunks` before launching.

## Pipeline shape

`scripts/run_full.sh` orchestrates four phases:

1. `build_corpus.py` → `data/corpus.parquet`  (skip if exists)
2. `embed_via_openrouter.py` per model → `data/chunks_<slug>/chunk_*.npy`  (resume by chunk file)
3. `fit_zca.py` per (model, MRL dim) → `backgrounds/<name>/{W_A,mu_A,eigvals_A}.npy + <name>.meta.json`
4. `index_backgrounds.py` → `REGISTRY.md` + `registry.json`

For v3 chunks, an extra phase 1.5 runs first:
`build_corpus_chunks.py --chunk-size 512 --chunk-overlap 64`
producing `data/corpus_chunks_512_64.parquet` which the embed step
then consumes via `--corpus` instead of `corpus.parquet`.

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
- Adaptive batch: start 16, max 32, min 1; halves on 429 / 5xx /
  200-but-no-data, grows back to 32 after a clean streak.
- `--ignore-providers siliconflow` by default (it's ~4× the price of
  Nebius / DeepInfra on Qwen3 embeddings).
- Resume: by chunk_NNNN.npy file count.  Skipped docs get a
  zero-vector placeholder so chunk row N maps to corpus row N.

In `scripts/run_full.sh`:
- `MODELS="qwen/qwen3-embedding-4b qwen/qwen3-embedding-8b"`
- `NAME_PREFIX="pl_mixed50k_doc"`
- `DIMS_4B="2560 1536 1024 768 512"`  (5 fits, includes native 2560)
- `DIMS_8B="4096 3072 2048 1024 768 512"`  (6 fits, includes native 4096;
  2560 / 1536 dropped — 8B wasn't MRL-trained at those off-grid dims)

In `scripts/lib/chunker.py` (v3):
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
