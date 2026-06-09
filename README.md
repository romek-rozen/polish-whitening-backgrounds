# Polish ZCA whitening backgrounds for Qwen3-Embedding (4B & 8B)

🇵🇱 **Polski:** [README.pl.md](./README.pl.md)

Pre-fitted whitening artefacts (`W_A.npy`, `mu_A.npy`, `eigvals_A.npy`)
ready to drop into siteFocus / any retrieval pipeline that uses
[`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
or [`Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
on Polish text. Skip the corpus sampling, the 45k embeddings, and the
ZCA SVD — clone, load, apply.

Backgrounds in this repo: **2**  ·  License: [CC-BY-4.0](LICENSE)

> **Heads-up (2026-06-09):** the repo was wiped and rebuilt on a fresh
> 45 156-doc Polish mix with token-precise (not char-cap) truncation
> under Qwen3's 32k context. The previous 5 backgrounds
> (`polish_mixed_50k_v1{,_mrl1024,_mrl1536}`, `corpus205_n3155`,
> `polish_smoke_1500`) are gone from `main`. Use git history if you
> need them.

## Quick start

```bash
git clone https://github.com/romek-rozen/polish-whitening-backgrounds.git
cd polish-whitening-backgrounds
```

```python
from loader import load_background, list_backgrounds

print(list_backgrounds())
# ['polish_mixed_50k_v1_qwen3-4b_nocap', 'polish_mixed_50k_v1_qwen3-8b_nocap']

bg = load_background("polish_mixed_50k_v1_qwen3-4b_nocap")
print(bg.dim, bg.W.shape, bg.mu.shape)
# 2560 (2560, 2560) (2560,)

# Whiten a batch of L2-normalised Qwen3 embeddings.
import numpy as np
x = np.random.randn(8, bg.dim).astype("float32")
x /= np.linalg.norm(x, axis=1, keepdims=True)
x_white = bg.apply(x)         # equivalent to (x - bg.mu) @ bg.W
```

The only runtime dependency is `numpy`. No `git lfs`, no external
downloads — every artefact is committed to the repo.

## Picking a background

| When | Use | Dim |
|---|---|---:|
| Qwen3-Embedding-4B, native | `polish_mixed_50k_v1_qwen3-4b_nocap` | 2560 |
| Qwen3-Embedding-8B, native | `polish_mixed_50k_v1_qwen3-8b_nocap` | 4096 |

See [`REGISTRY.md`](REGISTRY.md) for the full table with `n_fit`, rank
deficiency, eigenvalue ratios, and build timestamps. The same data is
in [`registry.json`](registry.json) for programmatic consumption.

Need an MRL-truncated refit (e.g. 1024 or 1536 dims for the 4B model)?
Re-run `scripts/fit_zca.py` against the embedding chunks — see
[Rebuild from scratch](#rebuild-from-scratch-or-fit-your-own-model)
below.

## MRL-truncated refits

Both Qwen3-Embedding-4B and 8B are Matryoshka Representation Learning
models — the first `N < D` components of every vector form a valid
embedding by themselves (after L2 renorm). MRL-truncated backgrounds
are ZCA refits on those truncated + renormalised vectors, so the
whitening transform matches what your pipeline actually sees at
inference. **None ship in this repo by default**; produce one in
seconds from the stored embedding chunks:

```bash
python scripts/fit_zca.py \
  --chunks data/chunks_qwen_qwen3-embedding-4b \
  --name polish_mixed_50k_v1_qwen3-4b_mrl1024 \
  --model qwen/qwen3-embedding-4b \
  --truncate-to 1024
```

Then pair the resulting background only with vectors sliced +
renormalised the same way:

```python
x_full = embed("...")                     # (2560,) from Qwen3-4B
x_1024 = x_full[:1024]
x_1024 /= np.linalg.norm(x_1024)
bg = load_background("polish_mixed_50k_v1_qwen3-4b_mrl1024")
x_white = bg.apply(x_1024[None])[0]       # whitened in MRL-1024 space
```

Mixing MRL-1024 vectors with a full-dim background is undefined — the
means / covariance are not compatible.

## Provenance

All backgrounds were fitted on a balanced Polish text mix (v2 —
sentence-only KLEJ replaced with more paragraph content, noisier mC4
swapped for the pre-cleaned FineWeb-2):

| Source | Docs | Notes |
|---|---:|---|
| Wikipedia PL | 22 500 | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) config `20231101.pl` |
| FineWeb-2 PL | 22 500 | [`HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) config `pol_Latn` — Polish web crawl extracted with trafilatura + language/quality filtered + minhash-deduped at source |
| OASST PL | ~156 | [`OpenAssistant/oasst1`](https://huggingface.co/datasets/OpenAssistant/oasst1) filtered `lang == 'pl'` (target 5 000, yields ~156 in practice) |

All sources enforce a 500-char minimum per doc (paragraph, not
sentence). Seed = 42, streaming shuffle, deterministic.

Earlier builds (now in git history) also included **KLEJ** (NKJP-NER +
DYK + CDSC-R) and used **mC4** instead of FineWeb-2. KLEJ was dropped
because its median item is 78 characters — single sentences skew the
embedding distribution away from the paragraph-level retrieval target.
mC4 was swapped because its raw text carries menu / breadcrumb /
timestamp boilerplate from a naive HTML→text extraction that we can't
fix downstream (the HTML is gone). FineWeb-2 ships text already
extracted with [trafilatura](https://trafilatura.readthedocs.io).

Each background's `*.meta.json` records the exact `sample_size_actual`,
`corpus_fingerprint_sha256`, seed, and diagnostic eigenvalues.

## Repo layout

```
backgrounds/<name>/
  W_A.npy           # (dim, dim) float32  — apply: (x - mu) @ W
  mu_A.npy          # (dim,)    float32
  eigvals_A.npy     # (dim,)    float32   — diagnostic, not needed at apply time
  <name>.meta.json  # provenance + diagnostics
REGISTRY.md         # human-readable index
registry.json       # same, machine-readable
loader.py           # numpy-only loader (see Quick start)
LICENSE             # CC-BY-4.0
README.md           # this file
README.pl.md        # Polish version
```

## How they were built

Sample the corpus mix above (seed=42), embed each doc via
OpenRouter against `Qwen/Qwen3-Embedding-{4B,8B}`, fit ZCA via two
streaming passes over the embedding chunks (`μ = E[x]`,
`Σ = E[(x-μ)(x-μ)ᵀ]`), then
`W = U · diag(1/√(S + ε)) · Uᵀ` from `SVD(Σ)` with `ε=1e-6`. No GPU
needed; total API spend was ~$1.2 for both models on a 45 156-doc mix.

The `_nocap` suffix marks the absence of a hard char cap at corpus
build time. Per-doc context is enforced precisely at embed time: each
doc is run through the model's own tokenizer (pulled from HF — same
`tokenizer.json` for 4B and 8B, sha256 `83cdf8c3a34f6886…`) and
truncated to **30 000 tokens** if needed (~2k margin under Qwen3's 32k
context window). Only ~25 of 45 156 docs hit the cap; the rest pass
through untouched. See the next section for the full recipe.

## Rebuild from scratch (or fit your own model)

The `scripts/` directory contains a complete pipeline you can run with
any OpenRouter API key, on any embedding model OpenRouter supports.
Expected wall time is ~1–3 hours per model and ~$0.5–1 in API spend
per model for the 45k-doc Polish mix (~38 M tokens at $0.01–0.02 / M
depending on which provider OpenRouter routes to).

```bash
git clone https://github.com/romek-rozen/polish-whitening-backgrounds.git
cd polish-whitening-backgrounds

# 1. Install minimal deps
pip install -r requirements.txt

# 2. Provide your OpenRouter API key (https://openrouter.ai/keys)
cp .env.example .env
$EDITOR .env             # paste OPENROUTER_API_KEY=sk-or-...

# 3. End-to-end: corpus → embed (both 4B + 8B) → fit → index
bash scripts/run_full.sh
```

What each script does:

| Script | Purpose |
|---|---|
| `scripts/build_corpus.py` | Sample the Polish mix (wiki + mc4 + klej + oasst) with seed=42. Writes `data/corpus.parquet`. Default: no per-doc cap. |
| `scripts/embed_via_openrouter.py` | Embed `corpus.parquet` via OpenRouter. Pre-flight token-precise truncation under the model's context window (default 30 000 tokens via the Qwen3 tokenizer pulled from HF — overridable with `--max-tokens-per-doc` and `--tokenizer-repo`). Adaptive batch (starts at 16, halves on 429/5xx, grows back after success streaks). Idempotent: resumes from the highest existing chunk. Writes `data/chunks_<slug>/*.npy` and a per-call `cost_report_<slug>.json`. |
| `scripts/fit_zca.py` | Two streaming passes (μ, Σ) over chunks + SVD. Writes `backgrounds/<name>/{W_A.npy, mu_A.npy, eigvals_A.npy, *.meta.json}`. |
| `scripts/index_backgrounds.py` | Regenerate `REGISTRY.md` + `registry.json`. Called by `run_full.sh`. |
| `scripts/run_full.sh` | Orchestrator. Idempotent — safe to re-run. |

`data/` is git-ignored (corpus + chunks are rebuildable). Only the
fitted `backgrounds/<name>/` artefacts ship in this repo.

To fit on a single model only:

```bash
MODELS="qwen/qwen3-embedding-8b" bash scripts/run_full.sh
```

To keep a corpus-level char cap (e.g. for repro of the legacy 1800-char
build), pass it to `build_corpus.py`:

```bash
MAX_CHARS=1800 NAME_PREFIX=polish_mixed_50k_cap1800 bash scripts/run_full.sh
```

To tighten or relax the per-doc token cap on the embed step (default
30 000, ~2k margin under Qwen3's 32k context):

```bash
python scripts/embed_via_openrouter.py \
  --model qwen/qwen3-embedding-4b \
  --max-tokens-per-doc 28000
```

Set `--max-tokens-per-doc 0` to disable the cap; documents that exceed
the model's context will then trigger an HTTP 200 + error body from the
provider and be skipped (with a zero-vector placeholder, so chunk row N
still maps to corpus row N).

## License

[CC-BY-4.0](LICENSE). Free to use, share, and adapt with attribution.
No warranty.

## Citation

If you use these in a paper or write-up, please cite Qwen3-Embedding-4B
and link back to this repo so others can find the artefacts:

```
@misc{polish-whitening-backgrounds,
  author = {Rozenberger, Roman},
  title  = {Polish ZCA whitening backgrounds for Qwen3-Embedding-4B},
  year   = {2026},
  url    = {https://github.com/romek-rozen/polish-whitening-backgrounds}
}
```
