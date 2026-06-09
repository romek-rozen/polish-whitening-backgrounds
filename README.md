# Polish ZCA whitening backgrounds for Qwen3-Embedding-4B

🇵🇱 **Polski:** [README.pl.md](./README.pl.md)

Pre-fitted whitening artefacts (`W_A.npy`, `mu_A.npy`, `eigvals_A.npy`)
ready to drop into siteFocus / any retrieval pipeline that uses
[`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
on Polish text. Skip the ~hour of corpus sampling, 50k embeddings, and
ZCA SVD — clone, load, apply.

Backgrounds in this repo: **5**  ·  License: [CC-BY-4.0](LICENSE)

## Quick start

```bash
git clone https://github.com/romek-rozen/polish-whitening-backgrounds.git
cd polish-whitening-backgrounds
```

```python
from loader import load_background, list_backgrounds

print(list_backgrounds())
# ['corpus205_n3155', 'polish_mixed_50k_v1', 'polish_mixed_50k_v1_mrl1024',
#  'polish_mixed_50k_v1_mrl1536', 'polish_smoke_1500']

bg = load_background("polish_mixed_50k_v1_mrl1024")
print(bg.dim, bg.W.shape, bg.mu.shape)
# 1024 (1024, 1024) (1024,)

# Whiten a batch of L2-normalised Qwen3 embeddings (sliced to bg.dim if MRL).
import numpy as np
x = np.random.randn(8, bg.dim).astype("float32")
x /= np.linalg.norm(x, axis=1, keepdims=True)
x_white = bg.apply(x)         # equivalent to (x - bg.mu) @ bg.W
```

The only runtime dependency is `numpy`. No `git lfs`, no external
downloads — every artefact is committed to the repo (largest file
~25 MB, total ~88 MB).

## Picking a background

| When | Use |
|---|---|
| Production, full Qwen3 dim (2560) | `polish_mixed_50k_v1` |
| MRL-truncated to 1024 dims | `polish_mixed_50k_v1_mrl1024` |
| MRL-truncated to 1536 dims | `polish_mixed_50k_v1_mrl1536` |
| Tiny smoke / unit tests | `polish_smoke_1500` (do **NOT** use in prod — rank-deficient) |
| Bootstrap (legacy) | `corpus205_n3155` — kept for repro of older runs |

See [`REGISTRY.md`](REGISTRY.md) for the full table with `n_fit`, rank
deficiency, eigenvalue ratios, and build timestamps. The same data is
in [`registry.json`](registry.json) for programmatic consumption.

## What's an MRL-truncated background?

[`Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
is a Matryoshka Representation Learning model — the first `N < 2560`
components of every vector form a valid embedding by themselves (after
L2 renorm). The `_mrlN` backgrounds in this repo are ZCA refits on
those truncated + renormalised vectors, so the whitening transform
matches what your pipeline actually sees at inference. Pair them only
with vectors sliced + renormalised the same way:

```python
x_full = embed("...")                     # (2560,) from Qwen3
x_1024 = x_full[:1024]
x_1024 /= np.linalg.norm(x_1024)
bg = load_background("polish_mixed_50k_v1_mrl1024")
x_white = bg.apply(x_1024[None])[0]       # whitened in MRL-1024 space
```

Mixing MRL-1024 vectors with the 2560-D `polish_mixed_50k_v1`
background is undefined — the means / covariance are not compatible.

## Provenance

All backgrounds were fitted on a balanced Polish text mix:

| Source | Docs | Notes |
|---|---:|---|
| Wikipedia PL | 20 000 | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) config `20231101.pl` |
| mC4 PL | 20 000 | [`allenai/c4`](https://huggingface.co/datasets/allenai/c4) config `pl` |
| KLEJ | 5 000 | NKJP-NER, DYK, CDSC-R subsets |
| OASST PL | 156 | [`OpenAssistant/oasst1`](https://huggingface.co/datasets/OpenAssistant/oasst1) filtered `lang == 'pl'` |

Each background's `*.meta.json` records the exact `sample_size_actual`,
`corpus_fingerprint_sha256`, seed, and diagnostic eigenvalues.

The full sampling + embedding + fit recipe is open-sourced (script
names are referenced in the meta files for traceability).

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

`polish_mixed_50k_v1` (the 2560-D parent) was fitted from scratch:
sample the corpus mix above (seed=42), embed each doc with Qwen3 at
`max_chars_per_doc=1800`, fit ZCA via two passes over the chunks
(`μ = E[x]`, `Σ = E[(x-μ)(x-μ)ᵀ]`), then
`W = U · diag(1/√(S + ε)) · Uᵀ` from `SVD(Σ)` with `ε=1e-6`.

The `_mrl*` children were refit in seconds from the parent's stored
embedding chunks — no re-embedding. Slice each chunk to the first
`N < 2560` columns, L2-renormalise row-wise, re-fit ZCA on the
truncated set. The result is deterministic given the parent.

The `_qwen3-*-nocap` backgrounds were rebuilt via OpenRouter API
(no local GPU needed) and **without** the 1800-char cap — see the next
section for the recipe. They use a token-precise pre-flight truncation
(default 30 000 tokens, ~2k margin under Qwen3's 32k context), enforced
with the model's own tokenizer pulled from HuggingFace. About 25 docs
in the 45 156-doc Polish mix exceed that cap; everything else passes
through untouched.

## Rebuild from scratch (or fit your own model)

The `scripts/` directory contains a complete pipeline you can run with
any OpenRouter API key, on any embedding model OpenRouter supports.
Expected wall time is ~2–4 hours per model and ~$2–5 in API spend for
the 45k-doc Polish mix (depending on doc-length distribution).

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
