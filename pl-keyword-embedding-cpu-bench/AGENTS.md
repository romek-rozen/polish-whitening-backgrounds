# AGENTS.md — pl-keyword-embedding-cpu-bench

Instructions for any coding agent working in this subproject. Read this before
touching anything here.

Communication with the user: **Polish**. All files in repo: **English**
(including this one) — same rule as the parent repo.

## What this is

A self-contained benchmark that answers **one** question:

> Which embedding model best clusters Polish search keywords when it has to run
> on a CPU-only VPS?

It lives inside `polish-whitening-backgrounds/` because the answer decides which
model gets a keyword-granularity whitening background fitted next. It is
**not** part of the shipped artefact set — nothing here is loaded by
`loader.py` and nothing here belongs in `registry.json`.

## Hard constraints

These are requirements, not preferences. Do not "optimise" them away.

1. **CPU only.** Every model is loaded with `device="cpu"`. Do not add CUDA
   paths, do not add vLLM, do not auto-detect GPU. The entire point is to
   measure what a VPS without a GPU can do. The host has a GPU and it is
   usually busy with another run — using it would both invalidate the
   benchmark and collide with that work.
2. **No cosine threshold for graph construction.** Clustering is cosine kNN +
   Leiden. A fixed similarity floor empties a high-dim anisotropic graph,
   isolated nodes each count as a "cluster", and modularity climbs to ~1.0 —
   scores that look excellent and mean nothing. See README "Clustering method".
3. **Report the sweep, not the best cell.** Metrics are reported across all
   resolutions. Do not silently report only a model's best AMI; that
   cherry-picks the knob and is how benchmarks lie.
4. **Keep the TF-IDF baseline in every run.** It is the floor the embeddings
   must beat. A result table without it is not interpretable.
5. **Task prompts stay off by default.** Only embeddinggemma defines them.
   Prompting one model and not the others measures different things. Measured
   here: the `Clustering` prompt *lowered* embeddinggemma's AMI (0.924 → 0.874
   at r=4) and cost throughput. Do not re-enable it as a default because the
   model card recommends it.

## Corpus naming is load-bearing — never reuse `pl_mixed50k_kw`

Backgrounds fitted here use the `pl_kwmix900k` corpus: 900 000 lowercase
phrases from Wikipedia PL titles + web n-grams + `clarin-knext/msmarco-pl`
queries (`build_mixed_corpus.py`).

The parent repo **already ships** `*_pl_mixed50k_kw_*` backgrounds fitted on a
completely different corpus — 50 000 web-mined n-grams, no titles, no queries.
The two are **not comparable**: different sources, different size, different
phrase quality.

Never name a background fitted here `*_pl_mixed50k_kw_*`, and never compare its
diagnostics against the shipped ones as if they were the same series. A
background's name is the only thing carrying its provenance once the artefacts
are copied around.

## GPU is allowed for fitting, never for the benchmark

Fitting runs on GPU (`run_kwmix_backgrounds.sh`, `.venv-cuda`) because it is a
one-off offline job over ~900 k phrases. The **benchmark** stays CPU-only — see
"Hard constraints" above. This is not a contradiction: the same model produces
the same vectors on either device, so a GPU-fitted background applies cleanly
to CPU-produced embeddings. What must never move to GPU is the *measurement* of
what a VPS can serve.

## Work in `work/` until a result is final

`work/` (git-ignored) holds corpora, embedding chunks, fitted backgrounds and
logs while an experiment is in flight. Nothing is promoted to the parent repo's
`data/` or `backgrounds/` until it has been validated — that promotion is a
separate, deliberate decision. The parent repo is a public artefact repo and
frequently has a second session running against it; writing into its shared
directories mid-experiment risks clobbering someone else's work.

## Layout

```
AGENTS.md                    # this file
README.md                    # user-facing: what/why/how, limitations
requirements.txt             # CPU torch + sentence-transformers + igraph/leidenalg
bench.py                     # the whole benchmark (single file on purpose)
test_dataset/keywords_pl.jsonl    # 150 keywords, 15 hand-labelled groups
results/                     # results.json (full) + results.md (table)
.venv/                       # git-ignored
```

`test_dataset/` is deliberately **not** named `data/`: the parent repo's
`.gitignore` ignores `data/` at any depth, which would silently drop the
ground truth from git.

`bench.py` is one file by design — it is a measurement script, not a library.
Split it only if it genuinely outgrows readability.

## Adding a model

1. Download it to `~/models/<slug>` (`hf download <repo> --local-dir ...`).
2. Add a `ModelSpec` to `MODELS` in `bench.py`: key, path, `prompt_name`, note.
3. `prompt_name` must be a prompt the model actually defines in its
   sentence-transformers config. `encode()` checks this and falls back to raw
   text with a warning rather than crashing — but check the warning, because
   silently encoding without the intended task prompt changes the result.
4. Add a row to the model table in `README.md`.

Candidates worth adding if the question comes up again:
`sdadas/mmlw-retrieval-roberta-large-v2` (Polish specialist, 1024d, needs the
`[sts]:` prefix for symmetric tasks), `snowflake-arctic-embed-l-v2.0`,
`intfloat/multilingual-e5-large` (needs `query:` / `passage:` prefixes).

Prefix-requiring models are the main correctness trap here: feeding them raw
text quietly costs accuracy and would make the comparison unfair.

## Editing the dataset

The cross-group **lexical traps** are load-bearing: `warszawa`, `cena`,
`ranking`, `kalkulator`, `online`, `opinie` deliberately appear in several
unrelated groups. They are what stops a bag-of-words method from scoring well.
If you extend the dataset, preserve that property and keep groups balanced.

Keep Polish diacritics correct. Never transliterate (`ą` → `a` etc.) — the
tokenizer's handling of accented characters is part of what is being tested.

## Honesty rules

The README's "Limitations" section is not decoration. 150 keywords labelled by
one person cannot resolve small differences between models. When reporting
results to the user:

- state cluster counts alongside AMI (a model can win AMI while badly
  over-splitting),
- do not declare a winner on a gap this dataset cannot support,
- if a model was skipped (not downloaded) or fell back to raw-text encoding,
  say so explicitly rather than presenting a partial table as complete.
