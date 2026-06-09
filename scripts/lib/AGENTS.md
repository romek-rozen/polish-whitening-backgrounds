# AGENTS.md — scripts/lib

Helpers for the corpus → embed → fit pipeline.  Each module here owns
**one** concern.  The top-level scripts (`scripts/build_corpus.py`,
`scripts/embed_via_openrouter.py`, `scripts/fit_zca.py`,
`scripts/index_backgrounds.py`) should be thin CLI wrappers that wire
these modules together — never re-implement the same logic inline.

## What lives here

| Module | Owns |
|---|---|
| `dotenv.py` | Tiny `KEY=VALUE` loader for `.env`.  No quoting, no interpolation. |
| `tokenizer.py` | OpenRouter↔HF id mapping, tokenizer pull, in-place token-precise truncation. |
| `openrouter_client.py` | Single embeddings POST, transient-status classification, 200-but-no-data unwrap, provider routing payload. |
| `chunk_store.py` | `chunk_NNNN.npy` persistence, resume detection, cost-report I/O, skip-log append. |
| `zca.py` | Streaming μ / Σ + SVD with optional MRL truncation, meta-json writer. |

The top-level scripts (`embed_via_openrouter.py`, `fit_zca.py`) are
still inlined at the time of writing.  The rewrite to thin wrappers
is planned for after the live v2 embed + fit run completes — see
the AGENTS.md at the repo root for the schedule.

## Rules of the road

1. **One concern per module.** If a function needs both HTTP and disk
   I/O, it belongs in the top-level script — not here.  The whole
   point of this folder is that each file is small enough to load in
   your head in one sitting.

2. **No hidden globals.** Module-level constants (`OPENROUTER_URL`,
   `OPENROUTER_TO_HF_TOKENIZER`, etc.) are fine — they're the
   immutable contract.  Module-level mutable state is not.  Pass it
   in, return it back.

3. **Logger per module: `logger = logging.getLogger(__name__)`.**
   That way the top-level script can configure `logging.basicConfig`
   once and every module's messages show up tagged with `lib.foo`.

4. **No CLI.** Argparse stays in `scripts/*.py`.  These modules are
   imported, not run.

5. **Type-annotate everything public.**  We're on Python 3.11+ — use
   `list[str]`, `dict[str, Any]`, `tuple[int, int]` directly, no
   `typing.List`.  `from __future__ import annotations` at the top of
   every file so forward refs are free.

6. **Docstrings explain WHY.**  The function name + signature already
   tells you what.  The docstring's job is to capture the constraint
   or gotcha that made the implementation look the way it does
   (e.g. "OpenRouter sometimes returns HTTP 200 with an error body; we
   unwrap that here so the retry path can see the real status code").

7. **Don't import sibling modules unless you need to.**  Cross-module
   deps inside `lib/` should be shallow — `tokenizer.py` should not
   import from `openrouter_client.py` or vice versa.  If you find
   yourself needing to, the cut is probably in the wrong place — talk
   to the user before doing it.

## Adding a new module

Before creating a new file, check whether the new code would naturally
live inside an existing module.  Empty-shell modules with one function
in them are a smell.

If the module is justified, also:

- Add it to the table above.
- Re-export anything the top-level scripts actually need via plain
  imports (`from scripts.lib.foo import bar`), not via `__init__.py`
  star-exports.

## Style nits

- 4-space indent, double-quoted strings (matches the rest of the
  repo).
- No emojis in code or docstrings.
- Keep lines ≤ 80 chars in docstrings and comments; 100 max for code.
- If you use a regex, leave a one-line comment above it spelling out
  what it matches — future-you will thank you.
