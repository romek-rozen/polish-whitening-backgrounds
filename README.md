# Polish ZCA whitening backgrounds for Qwen3-Embedding & OpenAI text-embedding-3

🇵🇱 **Polski:** [README.pl.md](./README.pl.md)

Pre-fitted whitening artefacts (`W_A.npy`, `mu_A.npy`, `eigvals_A.npy`)
ready to drop into siteFocus / any retrieval or clustering pipeline
that uses
[`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B),
[`Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B),
`text-embedding-3-small` or `text-embedding-3-large`
on Polish text. Skip the corpus sampling, the 50k embeddings, and the
ZCA SVD — clone, load, apply.

License: [CC-BY-4.0](LICENSE)

> **Status (2026-07-02):** **69 backgrounds shipped** — four models ×
> up to three granularities × the full MRL/dimensions grid. The corpus
> is `pl_mixed50k` — 22 500 Wikipedia + 27 500 FineWeb-2 PL + 42 oasst
> threads = **50 042 docs**, full paragraphs ≥500 chars. The `chunks`
> granularity splits the same docs into **129 181 chunks** of 512
> tokens with 64-token overlap (`lib.chunker`). The `kw` granularity
> is **50 000 keyword-like Polish phrases** (1–5 words) mined from the
> same corpus — fitted for grouping / clustering short search queries
> (e.g. Google Ads keyword lists), where whole-document backgrounds
> silently misfit.
>
> | Model | Granularity | Dim → name |
> |---|---|---|
> | Qwen3-Embedding-4B | doc, chunks, kw | `qwen3_4b_pl_mixed50k_{doc,chunks,kw}_mrl{2560, 1536, 1024, 768, 512}` |
> | Qwen3-Embedding-8B | doc, chunks, kw | `qwen3_8b_pl_mixed50k_{doc,chunks,kw}_mrl{4096, 3072, 2048, 1024, 768, 512}` |
> | text-embedding-3-small | doc, chunks, kw | `te3small_pl_mixed50k_{doc,chunks,kw}_mrl{1536, 1024, 768, 512, 256}` |
> | text-embedding-3-large | doc, chunks, kw | `te3large_pl_mixed50k_{doc,chunks,kw}_mrl{3072, 2048, 1536, 1024, 768, 512, 256}` |
>
> Earlier `polish_mixed_50k_v1{,_mrl1024,_mrl1536}`, `corpus205_n3155`
> and `polish_smoke_1500` were retired (different corpus, no
> granularity tag in the name) — use git history if you need them.
> See [`REGISTRY.md`](REGISTRY.md) for the full per-background table
> (sizes, costs, anisotropy diagnostics).

> ⚠️ **Granularity matters.** Pick the variant that matches your
> index: `_doc_` backgrounds are fitted on one embedding per whole
> document (FineWeb-2 / wiki / oasst), `_chunks_` backgrounds on one
> embedding per 512-token chunk (64-token overlap), `_kw_` backgrounds
> on one embedding per short keyword-like phrase (1–5 words).
> Whitening keyword lists with a doc-level background (or vice versa)
> silently degrades recall / cluster quality — see
> [GOTCHAS.md §1](GOTCHAS.md#1-background-granularity-must-match-index-granularity).

## Why whitening?

Modern embeddings (Qwen3 and OpenAI's included) are **anisotropic**: similarity
scores are biased toward a few dominant directions in the vector
space, which makes cosine distance crowded — most pairs look
"similar" even when they aren't. On the Polish corpus used here the
ratio of the top eigenvalue of the embedding covariance to the mean
eigenvalue runs in the tens (vs. ~1× for an ideal isotropic
distribution) — and for **short keyword phrases it's far worse**:
81× for Qwen3-4B, 150× for Qwen3-8B, 40× for text-embedding-3-small.
That's exactly the "every keyword looks similar to every other"
failure mode that wrecks keyword grouping.

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

You only need this once per (model, corpus, language) combination —
hence pre-fitting it once and shipping it as a static artefact.

## Quick start

```bash
git clone https://github.com/romek-rozen/polish-whitening-backgrounds.git
cd polish-whitening-backgrounds
```

```python
from loader import load_background, list_backgrounds

print(list_backgrounds())
# Returns 69 names — 4 models × {doc, chunks, kw} × the MRL grid, e.g.:
# ['qwen3_4b_pl_mixed50k_doc_mrl2560',  … , 'qwen3_4b_pl_mixed50k_kw_mrl512',
#  'qwen3_8b_pl_mixed50k_doc_mrl4096',  … , 'qwen3_8b_pl_mixed50k_kw_mrl512',
#  'te3small_pl_mixed50k_doc_mrl1536',  … , 'te3small_pl_mixed50k_kw_mrl256',
#  'te3large_pl_mixed50k_doc_mrl3072',  … , 'te3large_pl_mixed50k_kw_mrl256']

# Pair the background with the model + granularity + slice dim you actually use.
bg = load_background("qwen3_4b_pl_mixed50k_doc_mrl1024")
print(bg.dim, bg.W.shape, bg.mu.shape)
# 1024 (1024, 1024) (1024,)

# Whiten a batch of L2-normalised Qwen3 embeddings.
import numpy as np
x = np.random.randn(8, bg.dim).astype("float32")
x /= np.linalg.norm(x, axis=1, keepdims=True)
x_white = bg.apply(x)         # equivalent to (x - bg.mu) @ bg.W
```

The only runtime dependency is `numpy`. No `git lfs`, no external
downloads — every artefact is committed directly to the repo.

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
bg = load_background("qwen3_4b_pl_mixed50k_doc_mrl1024")

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
  only matches 4B embeddings sliced to 1024. The 8B's `mrl1024` has the
  same shape but the statistics behind μ and Σ are different — not
  interchangeable.
- **The transform is exact and lossless** — `bg.apply` is a rotation +
  per-axis scaling; it doesn't drop information, it just redistributes
  variance across axes.

## Keyword grouping / clustering (`kw` granularity)

The `_kw_` backgrounds exist for a different job than retrieval:
**grouping and clustering short search phrases** — Google Ads keyword
lists, GSC queries, search-term reports. Whole-document backgrounds
misfit there because the embedding distribution of a 3-word phrase is
nothing like that of a 2 000-char paragraph (and the anisotropy is
much worse — see the eigenvalue ratios above).

```python
import numpy as np
from loader import load_background
from sklearn.cluster import AgglomerativeClustering

bg = load_background("te3small_pl_mixed50k_kw_mrl1536")

keywords = ["buty do biegania", "buty biegowe damskie",
            "kredyt hipoteczny kalkulator", ...]
x = embed_openai(keywords)                # (n, 1536), L2-normalised
x_white = bg.apply(x)
x_white /= np.linalg.norm(x_white, axis=1, keepdims=True) + 1e-12

# Cosine clustering on whitened vectors — clusters stop being glued
# together by the dominant-direction monoculture.
labels = AgglomerativeClustering(
    n_clusters=None, distance_threshold=0.55,
    metric="cosine", linkage="average",
).fit_predict(x_white)
```

The `kw` fit corpus is 50 000 keyword-like Polish phrases (1–5 words)
mined from `pl_mixed50k` — see [Provenance](#provenance). It
approximates the *shape* of short-phrase embedding space, not any
specific niche; it works for keywords from any industry. If you want
the fit distribution even closer to your accounts, mix your own
exported keywords into `data/corpus_keywords.parquet` and refit
(`bash scripts/run_kw_fits.sh`, minutes of work, cents of spend).

## MRL-truncated backgrounds

Qwen3-Embedding-4B/8B are Matryoshka Representation Learning models —
the first `N < D` components of every vector form a valid embedding by
themselves (after L2 renorm). OpenAI's text-embedding-3 models expose
the same mechanism as the `dimensions` API parameter (shorten + L2
renormalise). For each model this repo ships a separate ZCA refit at
every commonly-used `N`, so the whitening matches what your pipeline
actually feeds the index at inference:

| Model | Native dim | MRL refits shipped |
|---|---:|---|
| Qwen3-Embedding-4B | 2560 | `mrl{2560, 1536, 1024, 768, 512}` |
| Qwen3-Embedding-8B | 4096 | `mrl{4096, 3072, 2048, 1024, 768, 512}` |
| text-embedding-3-small | 1536 | `mrl{1536, 1024, 768, 512, 256}` |
| text-embedding-3-large | 3072 | `mrl{3072, 2048, 1536, 1024, 768, 512, 256}` |

The 8B dim list follows the canonical Qwen3 MRL targets (powers of
two plus 768 and 3072); off-grid sizes like 2560 / 1536 are skipped
for 8B because the model was not MRL-trained at those — slicing still
works mathematically but recall would be worse than at the trained
dims. For the OpenAI models, `mrl<N>` matches vectors requested with
`dimensions=N` **or** sliced to `N` + L2-renormalised locally — the
two are equivalent per OpenAI's docs.

Pair each one only with vectors sliced + renormalised the same way:

```python
x_full = embed("...")                     # (2560,) from Qwen3-4B
x_1024 = x_full[:1024]                    # MRL slice
x_1024 /= np.linalg.norm(x_1024)          # renorm to unit L2
bg = load_background("qwen3_4b_pl_mixed50k_doc_mrl1024")
x_white = bg.apply(x_1024[None])[0]       # whitened in MRL-1024 space
```

Need a non-shipping dim (e.g. 256, or 2048 against 4B)? Refit in
seconds against the stored embedding chunks — the recipe is in
[Rebuild from scratch](#rebuild-from-scratch-or-fit-your-own-model)
below.

## Provenance

The `pl_mixed50k` corpus is a balanced Polish text mix (sentence-only
KLEJ replaced with more paragraph content, noisier mC4 swapped for the
pre-cleaned FineWeb-2):

| Source | Docs | Notes |
|---|---:|---|
| Wikipedia PL | 22 500 | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) config `20231101.pl` |
| FineWeb-2 PL | 27 500 | [`HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) config `pol_Latn` — Polish web crawl extracted with trafilatura + language/quality filtered + minhash-deduped at source |
| OASST PL | 42 | [`OpenAssistant/oasst1`](https://huggingface.co/datasets/OpenAssistant/oasst1) filtered `lang == 'pl'` (target 5 000; only 42 docs clear the 500-char floor in the public dump) |

Actual corpus: **50 042 docs, ~46 M tokens, fingerprint
`6e9e965ffbb6dbe6…`**.  All sources enforce a 500-char minimum per
doc (paragraph, not sentence).  Seed = 42, streaming shuffle,
deterministic.  For the `chunks` granularity the same docs are passed
through `lib.chunker` (RecursiveCharacterTextSplitter, 512 tokens with
64-token overlap, `merge_tiny` floor=100 chars,
`strip_overlap_fragments`) yielding **129 181 chunks**.

For the `kw` granularity, `scripts/build_corpus_keywords.py` mines
**50 000 keyword-like phrases** from the same corpus: lowercase
n-grams (1–5 words) that don't start/end with a stopword (the 328-word
[stopwords-iso/stopwords-pl](https://github.com/stopwords-iso/stopwords-pl)
list) or a bare number, 3–60 chars, document frequency ≥ 3, sampled
uniformly within each word-count bucket at a realistic keyword-list
mix (10% 1-word / 35% 2-word / 30% 3-word / 15% 4-word / 10% 5-word),
seed = 42.  Uniform-within-bucket sampling (not frequency-weighted)
avoids over-selecting web boilerplate.

Earlier builds (now in git history) also included **KLEJ** (NKJP-NER +
DYK + CDSC-R) and used **mC4** instead of FineWeb-2.  KLEJ was
dropped because its median item is 78 characters — single sentences
skew the embedding distribution away from the paragraph-level
retrieval target.  mC4 was swapped because its raw text carries menu /
breadcrumb / timestamp boilerplate from a naive HTML→text extraction
that we can't fix downstream (the HTML is gone).  FineWeb-2 ships text
already extracted with [trafilatura](https://trafilatura.readthedocs.io).

Each background's `*.meta.json` records the exact
`sample_size_actual`, `corpus_fingerprint_sha256`, seed, OpenRouter
cost report (prompt tokens, USD spend, n_calls, n_429), and
diagnostic eigenvalues.

## Repo layout

```
backgrounds/<name>/                   # one dir per shipped background (69 total)
  W_A.npy           # (dim, dim) float32  — apply: (x - mu) @ W
  mu_A.npy          # (dim,)    float32
  eigvals_A.npy     # (dim,)    float32   — diagnostic, not needed at apply time
  <name>.meta.json  # provenance + diagnostics
REGISTRY.md         # human-readable index, autogenerated
registry.json       # same, machine-readable
loader.py           # numpy-only loader (see Quick start)
lib/chunker.py      # RecursiveCharacterTextSplitter used by build_corpus_chunks.py
scripts/            # corpus + embed + fit + index pipeline
LICENSE             # CC-BY-4.0
README.md           # this file
README.pl.md        # Polish version
```

## How they were built

Sample the corpus mix above (seed=42), embed each doc / chunk / phrase
— Qwen models via OpenRouter, OpenAI models via `api.openai.com`
(same script, `--base-url` + `--api-key-env`) — then fit ZCA via two
streaming passes over the embedding chunks (`μ = E[x]`,
`Σ = E[(x-μ)(x-μ)ᵀ]`), then `W = U · diag(1/√(S + ε)) · Uᵀ` from
`SVD(Σ)` with `ε=1e-6`. No GPU needed. Spend: the Qwen doc+chunks
families cost **~$2.77** via OpenRouter (4B doc $0.92 / 8B doc $0.46 /
4B chunks $0.95 / 8B chunks $0.48); the OpenAI doc+chunks families
**~$14** via the OpenAI API (~95 M tokens × $0.02/M for 3-small and
$0.13/M for 3-large); the four `kw` families are **pennies** (~0.4 M
tokens each).

Per-doc context is enforced precisely at embed time: each doc is run
through the model's own tokenizer — the Qwen3 `tokenizer.json` pulled
from HF (identical for 4B and 8B, sha256 `83cdf8c3a34f6886…`), or
`tiktoken` `cl100k_base` for the OpenAI models — and truncated to
**30 000 tokens** for Qwen (~2k margin under the 32k context) or
**8 191 tokens** for OpenAI (their hard input cap; 326 of 50 042 docs
were truncated). Chunks and keyword phrases never approach either
limit.

The same embedding chunks are then fit once per MRL dim by slicing
each embedding to `N` columns, L2-renormalising row-wise, **refitting
μ and Σ from scratch**, and re-running the SVD. (Slicing the full-dim
`W` directly would be wrong.) The whole MRL grid for one model +
granularity takes under two minutes on CPU once the embed pass is
done.

## Rebuild from scratch (or fit your own model)

The `scripts/` directory contains a complete pipeline you can run with
any OpenRouter API key (any embedding model OpenRouter supports) or an
OpenAI API key (`--base-url https://api.openai.com/v1/embeddings
--api-key-env OPENAI_API_KEY`). Expected wall time is ~0.5–3 hours per
model+granularity; API spend for the 50k-doc Polish mix is
**$0.46–0.95 per Qwen model+granularity** via OpenRouter
(`--ignore-providers siliconflow` is set by default because that route
is ~4× more expensive), **$0.9–6.4 per OpenAI model+granularity**
($0.02/M for 3-small, $0.13/M for 3-large), and **cents** for any `kw`
family (~0.4 M tokens).

```bash
git clone https://github.com/romek-rozen/polish-whitening-backgrounds.git
cd polish-whitening-backgrounds

# 1. Install minimal deps (numpy + pyarrow + datasets + requests + tokenizers + tiktoken + trafilatura).
pip install -r requirements.txt

# 2. Provide your API key(s).
cp .env.example .env
$EDITOR .env             # paste OPENROUTER_API_KEY=sk-or-... and/or OPENAI_API_KEY=sk-...

# 3. End-to-end for the Qwen families: corpus → embed (4B + 8B, doc +
#    chunks) → fit → index.
bash scripts/run_full.sh

# 4. Keyword-granularity families (all four models):
python scripts/build_corpus_keywords.py
python scripts/embed_via_openrouter.py --model qwen/qwen3-embedding-4b \
    --corpus data/corpus_keywords.parquet --out data/kw_corpus/
python scripts/embed_via_openrouter.py --model text-embedding-3-small \
    --corpus data/corpus_keywords.parquet --out data/kw_corpus/ \
    --base-url https://api.openai.com/v1/embeddings --api-key-env OPENAI_API_KEY
# … same for qwen3-embedding-8b / text-embedding-3-large, then:
bash scripts/run_kw_fits.sh

# 5. OpenAI doc+chunks families:
bash scripts/run_oai_fits.sh   # after embedding corpus.parquet and
                               # corpus_chunks_512_64.parquet with both models
```

What each script does:

| Script | Purpose |
|---|---|
| `scripts/build_corpus.py` | Sample the Polish mix (wiki + FineWeb-2 PL + oasst) with seed=42 and a 500-char paragraph floor. Writes `data/corpus.parquet`. Default: no upper cap. |
| `scripts/build_corpus_chunks.py` | Same corpus → run `lib.chunker` (512 tokens, 64-token overlap, merge_tiny floor=100 chars, strip_overlap_fragments) → writes `data/corpus_chunks.parquet` (129 181 rows). |
| `scripts/build_corpus_keywords.py` | Same corpus → mine 50 000 keyword-like phrases (1–5-word n-grams, stopwords-pl edge filter, df ≥ 3, stratified word-count mix) → writes `data/corpus_keywords.parquet`. |
| `scripts/embed_via_openrouter.py` | Embed any corpus parquet via an OpenAI-compatible `/v1/embeddings` endpoint — OpenRouter by default, `api.openai.com` via `--base-url` + `--api-key-env`. Pre-flight token-precise truncation under the model's context window (Qwen3 tokenizer from HF, or tiktoken for OpenAI models — `--max-tokens-per-doc` / `--tokenizer-repo` to override). Adaptive batch (halves on 429/5xx, grows back after success streaks). Idempotent: resumes from the highest existing chunk. Writes `chunks_<slug>/*.npy` and a per-call `cost_report_<slug>.json`. |
| `scripts/fit_zca.py` | Two streaming passes (μ, Σ) over chunks + SVD. Optional `--truncate-to N` slices each chunk to `N` columns and re-renormalises before fitting, for MRL refits. Writes `backgrounds/<name>/{W_A.npy, mu_A.npy, eigvals_A.npy, *.meta.json}`. |
| `scripts/index_backgrounds.py` | Regenerate `REGISTRY.md` + `registry.json`. Called by the runner scripts. |
| `scripts/run_full.sh` | Orchestrator for the Qwen families: corpus → embed each model → fit at every dim in `DIMS_<MODEL>` → index. Idempotent — safe to re-run. |
| `scripts/run_kw_fits.sh` | Fit every `kw` background whose embeddings are complete under `data/kw_corpus/` (all four models) → index. Skips partial embeds. |
| `scripts/run_oai_fits.sh` | Fit the OpenAI `doc` + `chunks` backgrounds from `data/chunks_<model>/` and `data/chunks_corpus/chunks_<model>/` → index. Skips partial embeds. |

`data/` is git-ignored (corpus + chunks are rebuildable). Only the
fitted `backgrounds/<name>/` artefacts ship in this repo.

To fit on a single model only:

```bash
MODELS="qwen/qwen3-embedding-8b" bash scripts/run_full.sh
```

To change the MRL dim list for a model (default: 4B = 2560/1536/1024/768/512,
8B = 4096/3072/2048/1024/768/512):

```bash
DIMS_4B="2560 1024" bash scripts/run_full.sh   # only two fits for 4B
```

To tighten or relax the per-doc token cap on the embed step:

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

If you use these in a paper or write-up, please cite the underlying
embedding model and link back to this repo so others can find the
artefacts:

```
@misc{polish-whitening-backgrounds,
  author = {Rozenberger, Roman},
  title  = {Polish ZCA whitening backgrounds for Qwen3-Embedding and OpenAI text-embedding-3},
  year   = {2026},
  url    = {https://github.com/romek-rozen/polish-whitening-backgrounds}
}
```
