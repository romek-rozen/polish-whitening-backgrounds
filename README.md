# Polish ZCA whitening backgrounds for Qwen3-Embedding (4B & 8B)

🇵🇱 **Polski:** [README.pl.md](./README.pl.md)

Pre-fitted whitening artefacts (`W_A.npy`, `mu_A.npy`, `eigvals_A.npy`)
ready to drop into siteFocus / any retrieval pipeline that uses
[`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
or [`Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
on Polish text. Skip the corpus sampling, the 45k embeddings, and the
ZCA SVD — clone, load, apply.

Backgrounds in this repo: **11**  ·  License: [CC-BY-4.0](LICENSE)

> **Heads-up (2026-06-09):** corpus rebuilt as **v2** — wiki 22.5k +
> FineWeb-2 PL 22.5k + oasst ~42 = 45 042 docs, paragraph-only (≥500
> chars), token-precise truncation under Qwen3's 32k context. The
> earlier `polish_mixed_50k_v1{,_mrl1024,_mrl1536}`, `corpus205_n3155`
> and `polish_smoke_1500` are gone from `main`. Use git history if you
> need them.

## Why whitening?

Modern embeddings (Qwen3 included) are **anisotropic**: similarity
scores are biased toward a few dominant directions in the vector
space, which makes cosine distance crowded — most pairs look
"similar" even when they aren't. Concretely, on this corpus the
ratio of the top eigenvalue of the embedding covariance to the mean
eigenvalue is **~50–100×** (vs. ~1× for an ideal isotropic
distribution).

A **ZCA whitening transform** rebalances the space:

```
x_white = (x - μ) @ W       where  Σ = U S Uᵀ,
                                   W = U · diag(1 / √(S + ε)) · Uᵀ
```

After applying it, every direction carries comparable variance and
cosine distance behaves much closer to the textbook ideal. In
retrieval that typically translates into:

- meaningfully better **recall@k** on hard polysemy / topic-cluster
  queries, especially with short queries against longer documents,
- much cleaner **clustering / dedup** signals — the "top eigenvalue
  monoculture" stops pulling unrelated docs together,
- a fix for the well-known **"all cosines look like 0.7"** problem.

You only need this once per (model, corpus, language) combination
— hence pre-fitting and shipping it as a static artefact.

## Quick start

```bash
git clone https://github.com/romek-rozen/polish-whitening-backgrounds.git
cd polish-whitening-backgrounds
```

```python
from loader import load_background, list_backgrounds

print(list_backgrounds())
# ['polish_mixed_50k_v2_qwen3-4b_mrl2560',
#  'polish_mixed_50k_v2_qwen3-4b_mrl1536',  '..._mrl1024', '..._mrl768', '..._mrl512',
#  'polish_mixed_50k_v2_qwen3-8b_mrl4096',
#  'polish_mixed_50k_v2_qwen3-8b_mrl3072',  '..._mrl2048', '..._mrl1024', '..._mrl768', '..._mrl512']

# Pair the background with the model + slice dimension you actually use.
bg = load_background("polish_mixed_50k_v2_qwen3-4b_mrl1024")
print(bg.dim, bg.W.shape, bg.mu.shape)
# 1024 (1024, 1024) (1024,)

# Whiten a batch of L2-normalised Qwen3 embeddings.
import numpy as np
x = np.random.randn(8, bg.dim).astype("float32")
x /= np.linalg.norm(x, axis=1, keepdims=True)
x_white = bg.apply(x)         # equivalent to (x - bg.mu) @ bg.W
```

The only runtime dependency is `numpy`. No `git lfs`, no external
downloads — every artefact is committed to the repo.

## End-to-end: use in a retrieval pipeline

This is the actual cosine-retrieval flow you'd run in production
against a Qwen3-4B index. The whitening step slots in **right after
the L2 renorm, before the dot product** — nothing else changes.

```python
import numpy as np
from loader import load_background
# Whatever you already use to call Qwen3 — locally, vLLM, OpenRouter, etc.
from your_pipeline import embed_qwen3_4b

# 1. Load once at startup.
bg = load_background("polish_mixed_50k_v2_qwen3-4b_mrl1024")

def encode(texts):
    """Embed → MRL slice → L2 renorm → ZCA whiten."""
    x = embed_qwen3_4b(texts)             # (n, 2560) float32
    x = x[:, :bg.dim]                     # MRL slice to 1024
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return bg.apply(x)                    # (n, 1024) whitened

# 2. Index your documents once.
doc_vecs = encode(documents)              # (N, 1024)

# 3. At query time, encode the query the same way.
q_vec = encode([query])                   # (1, 1024)
scores = q_vec @ doc_vecs.T               # (1, N) cosine, post-whitening
topk = np.argpartition(-scores[0], 10)[:10]
```

What matters in this pattern:

- **Whiten both sides identically** — query vectors and doc vectors must
  go through the same `bg.apply`. Mixing whitened and raw vectors gives
  meaningless scores.
- **Pair (model, dim, background)** — `mrl1024` from the 4B background
  only matches 4B embeddings sliced to 1024. The 8B's `mrl1024` looks
  the same in shape but the statistics behind μ and Σ are different.
- **The transform is exact and lossless at full dim** — `bg.apply` is a
  rotation + per-axis scaling; it doesn't drop information, it just
  redistributes variance across axes.

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

## MRL-truncated backgrounds

Both Qwen3-Embedding-4B and 8B are Matryoshka Representation Learning
models — the first `N < D` components of every vector form a valid
embedding by themselves (after L2 renorm). For each model this repo
ships a separate ZCA refit at every commonly-used `N`, so the
whitening matches what your pipeline actually feeds the index at
inference:

| Model | Native dim | MRL refits shipped |
|---|---:|---|
| Qwen3-Embedding-4B | 2560 | `mrl{2560, 1536, 1024, 768, 512}` |
| Qwen3-Embedding-8B | 4096 | `mrl{4096, 3072, 2048, 1024, 768, 512}` |

Pair each one only with vectors sliced + renormalised the same way:

```python
x_full = embed("...")                     # (2560,) from Qwen3-4B
x_1024 = x_full[:1024]                    # MRL slice
x_1024 /= np.linalg.norm(x_1024)          # renorm to unit L2
bg = load_background("polish_mixed_50k_v2_qwen3-4b_mrl1024")
x_white = bg.apply(x_1024[None])[0]       # whitened in MRL-1024 space
```

Mixing MRL-1024 vectors with a full-dim background is undefined — the
means / covariance are not compatible. Likewise `mrl1024` from the 4B
background does **not** interchange with `mrl1024` from the 8B
background even though the shapes match — the underlying statistics
differ.

Need a non-shipping dim (e.g. 256, or 2048 against 4B)? Refit in
seconds against the stored embedding chunks — the recipe is in
[Rebuild from scratch](#rebuild-from-scratch-or-fit-your-own-model)
below.

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
