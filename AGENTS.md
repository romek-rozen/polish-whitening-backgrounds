# AGENTS.md — polish-whitening-backgrounds

Instructions for any coding agent (Claude Code, Codex, Cursor, Gemini, …)
working in this repo. Read this file FIRST on every new session.

Communication with the user (chat, PR comments): **Polish**.
All files in repo: **English** (incl. this one).

## What this repo is

A public, drop-in artefact repo:
[https://github.com/romek-rozen/polish-whitening-backgrounds](https://github.com/romek-rozen/polish-whitening-backgrounds)
Pre-fitted ZCA whitening backgrounds (`W_A.npy`, `mu_A.npy`,
`eigvals_A.npy`) for Qwen3-Embedding family on Polish text. Goal:
**colleagues don't recompute μ and Σ on a 50k-doc Polish corpus** —
they clone, load via `loader.py`, apply. Provenance + diagnostics in
each `backgrounds/<name>/<name>.meta.json`.

We are NOT working on user's private extraction data. The Polish
training corpus we use is sampled live from public HuggingFace
datasets (Wikipedia PL + mC4 PL + KLEJ + OASST PL) with a fixed seed,
inside this repo's `data/` (git-ignored). Never read or write to
`/home/spark001/Spark-testy/llm-extraction-embedding-dgx-final-setup/`
or any of its `data/` / `results/` / `models/` subtrees — that's the
user's main project, not this one.

## Current state (snapshot at handover)

Committed in the GitHub repo, **but wiped from the working tree** on
2026-06-09 to make room for a clean rebuild (user request):
- 5 legacy backgrounds — `polish_mixed_50k_v1{,_mrl1024,_mrl1536}`,
  `corpus205_n3155`, `polish_smoke_1500`. Recoverable from `git
  checkout HEAD -- backgrounds/` if anyone changes their mind before
  the push. The GitHub `main` branch still has them.

Still committed and untouched:
- Numpy-only `loader.py`, `REGISTRY.md`, `registry.json`, EN+PL README,
  `LICENSE` (CC-BY-4.0).
- `scripts/` pipeline for **rebuilding from scratch via OpenRouter**
  (no GPU needed): `build_corpus.py`, `embed_via_openrouter.py`,
  `fit_zca.py`, `index_backgrounds.py`, `run_full.sh`.
- `.env.example`, `requirements.txt`, `.gitignore` covering
  `data/`, `.env`, `__pycache__`.

Done locally but NOT YET committed:
- `data/corpus.parquet` — **v2** mix (wikipedia 22 500 + fineweb 22 500 +
  oasst ~42 = 45 042 docs, 112.8 MB chars, 41.1 M tokens, fingerprint
  `8e4549ffdbb7a406…`). The earlier v1 (with mc4 + KLEJ + a different
  fingerprint `1f8e9b1…`) was wiped on 2026-06-09 after spotting that
  KLEJ items are single sentences and mc4 text carries menu/breadcrumb
  boilerplate. Lives in `data/` which is git-ignored. Safe to keep.
- Qwen3 tokenizer cached in `~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-{4B,8B}/`
  (~7 MB each, but tokenizer.json is byte-identical between 4B and 8B,
  sha256 `83cdf8c3a34f6886…`). First call to `_load_tokenizer()` pulls
  it via `huggingface_hub`.

## What was paused mid-run and why

We started the full rebuild pipeline (target: add `_qwen3-4b-nocap` and
`_qwen3-8b-nocap` backgrounds via OpenRouter API). The user paused it
to clarify scope. The pause was clean — nothing committed, no broken
state.

Key learnings that are already baked into the scripts:

1. **Per-doc token cap = 30 000** is the safe default in
   `embed_via_openrouter.py --max-tokens-per-doc 30000`. Why: Qwen3
   embedders have a 32k context. We pull `tokenizer.json` from HF
   (`Qwen/Qwen3-Embedding-4B`, 4B & 8B share an identical tokenizer)
   and tokenize all 45 156 docs in ~3.5 s via the Rust `tokenizers`
   crate. 25 docs exceed the 30k cap (max 100 554 tok). Pre-flight
   truncation matches the provider's count exactly, so the previous
   char-cap heuristic ("3 chars/tok") is gone. Providers may still
   return HTTP 200 with an error body on edge cases; the client detects
   this ("200-but-no-data") and adaptively shrinks the batch / skips
   the doc.
2. **Skipped docs are NOT dropped** — they get a zero-vector
   placeholder so chunk row N still maps to corpus row N. ZCA is
   robust (L2-renorm guards against div-by-zero, zeros don't pull μ).
   The skip log goes to `data/skipped_<model-slug>.jsonl`.
3. **Provider routing**: default `PROVIDER_ORDER=nebius,deepinfra`
   (cheap, ~$0.01–0.02/M tokens). Without it OpenRouter sometimes
   routed to SiliconFlow at $0.04/M which is 4× the price for 8B.
4. **Adaptive batch**: starts at 16, halves on 429/5xx/"200-no-data",
   grows back to 32 after 16 clean successes.
5. **Idempotent resume**: each chunk is persisted as soon as it's full
   (1000 rows). Re-running `run_full.sh` continues from the highest
   chunk.

## What the next session needs to do

The user wants to actually complete the rebuild — fit μ and Σ for
Qwen3-Embedding-4B (no cap) and Qwen3-Embedding-8B (no cap) so they
can be added to the public repo and never have to be computed again.

Required from user before you run anything:
- A fresh OpenRouter API key with budget for ~$1–2 (the prior key was
  to be revoked after the planned run; assume it's gone).
- Explicit "go" — this writes ~$1 of API spend.

Then:

```bash
cd /home/spark001/Spark-testy/polish-whitening-backgrounds
cp .env.example .env
$EDITOR .env   # paste OPENROUTER_API_KEY=sk-or-...
# Already-built data/corpus.parquet is reused (45156 docs, fingerprint
# 1f8e9b...). If you want a fresh sample, delete it first.
PY=/home/spark001/Spark-testy/llm-extraction-embedding-dgx-final-setup/.venv/bin/python \
    nohup bash scripts/run_full.sh > data/run.log 2>&1 &
disown
```

Expected behaviour:
- Phase 1 `build_corpus.py` — instant skip ("corpus exists").
- Phase 2 `embed_via_openrouter.py --model qwen/qwen3-embedding-4b`.
  First pulls tokenizer.json from HF (~7 MB, cached), tokenizes the
  whole corpus in ~3.5 s, truncates ~25 outlier docs to 30k tokens,
  then embeds. About **90 minutes** at ~8 doc/s, batch 16. Watch
  `data/cost_report_qwen_qwen3-embedding-4b.json`. Total corpus is
  ~38.1 M tokens → ~$0.38 at $0.01/M (Nebius/DeepInfra).
- Phase 3 same for `qwen/qwen3-embedding-8b`. Slower model — about
  **2–3 hours**. Same token count, same per-M price → ~$0.38. Don't
  let it fall back to SiliconFlow (4× the price); the order is pinned.
- Phase 4 `fit_zca.py` once per model — each takes 20-60 s.
- Phase 5 `index_backgrounds.py` regenerates `REGISTRY.md` +
  `registry.json`.

Output backgrounds will land in:
- `backgrounds/polish_mixed_50k_v1_qwen3-4b-nocap/` (2560-D)
- `backgrounds/polish_mixed_50k_v1_qwen3-8b-nocap/` (4096-D)

After completion:
1. Read both `*.meta.json` files. Check `rank_deficient_eigvals` is
   small (say <100) and `top_ev_ratio_pre` is somewhere in 30–100.
   Anything wildly worse than the existing
   `polish_mixed_50k_v1.meta.json` (rank def 16, ratio 83) means the
   embed pass had problems.
2. Show the user a summary: artefact sizes, costs, diagnostics.
3. **DO NOT push without explicit "ok push" from the user.** This is a
   public repo; rollback after push is messy.
4. When approved:
   ```bash
   cd /home/spark001/Spark-testy/polish-whitening-backgrounds
   git add backgrounds REGISTRY.md registry.json
   # Only if the scripts/ side hasn't been committed yet:
   git add scripts/ requirements.txt README.md README.pl.md .env.example .gitignore AGENTS.md
   git commit -m "add OpenRouter rebuild pipeline + 2 backgrounds (4B & 8B no-cap)"
   git push origin main
   ```
5. Update memory in the main project so we know the next-revision
   backgrounds are live.

## File map

```
/home/spark001/Spark-testy/polish-whitening-backgrounds/
├── backgrounds/                                  # committed artefacts
│   ├── polish_mixed_50k_v1/
│   ├── polish_mixed_50k_v1_mrl1024/
│   ├── polish_mixed_50k_v1_mrl1536/
│   ├── corpus205_n3155/
│   └── polish_smoke_1500/
├── scripts/                                      # rebuild pipeline
│   ├── build_corpus.py
│   ├── embed_via_openrouter.py
│   ├── fit_zca.py
│   ├── index_backgrounds.py
│   └── run_full.sh
├── data/                                         # git-ignored (large, rebuildable)
│   ├── corpus.parquet                            # 45156 docs, no cap, fp=1f8e9b...
│   ├── corpus_manifest.json
│   ├── chunks_qwen_qwen3-embedding-4b/           # (empty — pending the run)
│   ├── chunks_qwen_qwen3-embedding-8b/           # (empty — pending the run)
│   ├── cost_report_<slug>.json                   # (created during the run)
│   ├── skipped_<slug>.jsonl                      # (created if any docs are skipped)
│   └── run.log                                   # (created at run start)
├── loader.py                                     # numpy-only consumer API
├── REGISTRY.md  registry.json                    # auto-generated
├── README.md  README.pl.md                       # EN + PL
├── LICENSE                                       # CC-BY-4.0
├── requirements.txt
├── .env.example                                  # OPENROUTER_API_KEY=...
├── .env                                          # git-ignored; per session
├── .gitignore                                    # ignores data/, .env, __pycache__
└── AGENTS.md                                     # this file
```

## Key constants worth knowing

In `scripts/build_corpus.py` (v2):
- `DEFAULT_MIX = {"wikipedia": 22500, "fineweb": 22500, "oasst": 5000}`
- `MIN_DOC_CHARS = 500` enforced on every source — paragraph not sentence.
- `seed = 42`, no per-doc upper cap by default.
- mC4 dropped in favour of `HuggingFaceFW/fineweb-2` config `pol_Latn`
  (already extracted with trafilatura by HF, language/quality filtered,
  minhash-deduped). Trafilatura is in `requirements.txt` but we don't
  invoke it ourselves — fineweb-2 ships the output.
- KLEJ dropped (single-sentence items skew the paragraph-level
  distribution we want for retrieval whitening).
- OASST PL yields only ~42 docs under the 500-char floor (was ~156
  before the floor; OASST conversations are mostly short).

In `scripts/embed_via_openrouter.py`:
- `--max-tokens-per-doc 30000` (default) — pre-flight truncation via
  Qwen3 tokenizer pulled from HF (`tokenizers` + `huggingface_hub`,
  no `transformers` needed). Override per-model with `--tokenizer-repo`.
- `OPENROUTER_TO_HF_TOKENIZER` maps OR ids to HF repo ids
  (qwen3-{0.6,4,8}b covered).
- `OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"`.
- `TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504, 529}`.
- Adaptive batch: start 16, max 32, min 1.

In `scripts/run_full.sh`:
- `MODELS="qwen/qwen3-embedding-4b qwen/qwen3-embedding-8b"`.
- `NAME_PREFIX="polish_mixed_50k_v1"`.
- `PROVIDER_ORDER="nebius,deepinfra"`.
- `DIMS_4B="2560 1536 1024 768 512"` (5 fits, includes native 2560).
- `DIMS_8B="4096 3072 2048 1024 768 512"` (6 fits, includes native
  4096; 2560 / 1536 dropped because 8B was not MRL-trained at those
  off-grid dims).

Output naming: `<NAME_PREFIX>_<model-short>_mrl<DIM>`, where
`<model-short>` strips `qwen/qwen3-embedding-` (e.g. `qwen3-4b`).
The `_mrl<DIM>` suffix applies even when `<DIM>` equals the native
dim — the fit is still produced by `fit_zca.py --truncate-to` and the
naming convention stays uniform.

In `scripts/fit_zca.py`:
- `--truncate-to N` slices each chunk to the first N columns and
  L2-renormalises row-wise before fitting ZCA. Used for MRL refits.
  Without it the native dim is used.

## Safety rules

- **Never commit `.env`.** It's in `.gitignore` but always double-check
  with `git status --short` before staging.
- **Never push to `main` without "ok push" from the user.** This is
  public — rollback is messy.
- **Never touch the user's main project at**
  `/home/spark001/Spark-testy/llm-extraction-embedding-dgx-final-setup/`.
  The waterfall pipeline there is a long-running batch over real data;
  read-only access to copy templates is OK, but do not kill its
  processes (tmux sessions `waterfall_*`, `cbstress*`, anything
  matching `run_full_pipeline_waterfall.sh`).
- **Never delete the user's data without explicit instruction.** That
  also applies to this repo's `data/corpus.parquet` — even though it's
  rebuildable, it cost ~10 min of HF streaming to produce.

## Update this AGENTS.md when

- You change `scripts/` defaults (cap, batch, provider order).
- You add a new background.
- The user changes the publication strategy (private repo, LFS, etc.).
- You discover a non-obvious gotcha future-you should know about.
