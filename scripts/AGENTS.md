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
                         ├─ v2 (doc-level, currently shipped)
                         │       │
                         │       ▼
                         │     embed_via_openrouter.py
                         │
                         └─ v3 (paragraph-level, planned)
                                 │
                                 ▼
                            build_corpus_chunks.py
                                 │   data/corpus_chunks_<S>_<O>.parquet
                                 ▼
                            embed_via_openrouter.py  (same script,
                                 │   different --corpus)
                                 ▼
                                 …

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
| `embed_via_openrouter.py` | The adaptive-batch retry loop. Imports HTTP, tokenizer, persistence from `lib/`. |
| `fit_zca.py` | Argparse + `lib.zca.fit` + `lib.zca.write_meta`. ~110 lines. |
| `index_backgrounds.py` | Walk `backgrounds/`, read every `*.meta.json`, write `REGISTRY.md` + `registry.json`. Deterministic — depends only on what's on disk. |
| `run_full.sh` | Orchestrator: `.env` load, defaults (`MODELS`, `NAME_PREFIX`, `DIMS_<MODEL>`, `PROVIDER_ORDER`), loops embed + N×fit per model, final index. |

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
# Full rebuild from scratch (corpus → embed × 2 models → fit × 11 MRL → index).
PY=/path/to/venv/bin/python bash scripts/run_full.sh

# Single model.
MODELS="qwen/qwen3-embedding-4b" bash scripts/run_full.sh

# Just one MRL dim for 4B (e.g. you only care about 1024).
DIMS_4B="1024" MODELS="qwen/qwen3-embedding-4b" bash scripts/run_full.sh

# Re-index after manually placing artefacts in backgrounds/.
python scripts/index_backgrounds.py
```

For the running-process tooling (tmux session, monitor filter, kill
+ resume) see the root [AGENTS.md](../AGENTS.md).
