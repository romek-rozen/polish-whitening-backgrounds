# pl-keyword-embedding-cpu-bench

Which embedding model groups **Polish search keywords** best when it has to run
on a **CPU-only VPS**?

This benchmark answers that with numbers instead of leaderboard folklore. Every
model is loaded through `sentence-transformers` on `device="cpu"` — the same
code path a FastAPI or [TEI](https://github.com/huggingface/text-embeddings-inference)
wrapper would use in production. **No vLLM, no GPU.**

## Why this exists

The parent repo ships ZCA whitening backgrounds for embedding models. Picking
*which* model to fit a keyword-granularity background for is a prerequisite
decision, and the public leaderboards do not answer it:

- **PL-MTEB** ranks whole-model averages dominated by >1B-parameter models
  (Qwen3-Embedding-4B/8B, BGE-Multilingual-Gemma2) that will not fit a plain
  VPS.
- **MTEB clustering** tasks cluster *documents*, not 2–5 word keywords. Short
  text is a different length band with different geometry.

So we measure the thing we actually care about: short Polish keywords, CPU
throughput, and cluster agreement with hand-labelled ad-group intent.

## What is measured

| Axis | Metric | Why |
|---|---|---|
| Quality | homogeneity, completeness, AMI, ARI vs. hand labels | do clusters match real ad-group intent? |
| Granularity | `#clusters` at each Leiden resolution | over- vs. under-splitting |
| Cost | keywords/s, ms/keyword on CPU | can a VPS actually serve it? |
| Startup | model load seconds | cold-start cost of the API process |

A **TF-IDF lexical baseline** is always reported as the floor. The dataset is
built with cross-group lexical traps — `warszawa`, `cena`, `ranking`,
`kalkulator`, `online` each appear in several unrelated groups — so a model
that only matches words cannot score well. Embeddings have to earn their cost.

## Dataset

`test_dataset/keywords_pl.jsonl` — 150 Polish keywords, 15 hand-labelled intent
groups (10 each), written in realistic SEO / Google Ads long-tail style:
running shoes, car insurance, English courses, cheesecake recipes, mortgages,
Zakopane hotels, washing-machine repair, dog food, photovoltaics, gaming
laptops, weight-loss diets, SEO services, Warsaw flat rentals, cheap flights,
dental implants.

One row per keyword: `{"group": "...", "keyword": "..."}`.

## The `pl_kwmix900k` corpus — a NEW corpus, not the shipped one

> ⚠️ **Do not confuse this with the corpus behind the shipped
> `*_pl_mixed50k_kw_*` backgrounds.** Those were fitted on
> `data/corpus_keywords.parquet`: 50 000 phrases mined as n-grams from Polish
> web text only. Backgrounds fitted here carry `pl_kwmix900k` in their name
> precisely so the two series can never be mistaken for each other. Their
> numbers are **not comparable**.

Built by `build_mixed_corpus.py`. 900 000 lowercase Polish phrases, deduped,
length mix 10/35/30/15/10 by word count (same `kw` granularity contract as the
parent repo):

| source | phrases | share | what it contributes |
|---|--:|--:|---|
| `wiki_title` | 523 992 | 58% | native Polish noun phrases — the canonical keyword *shape* |
| `ngram_web` | 243 935 | 27% | natural web language, long tail, realistic noise |
| `msmarco_query` | 132 073 | 15% | real search-query shape and intent |

Every source is public and reproducible: a Wikipedia dump
(`plwiki-latest-all-titles-in-ns0.gz`), the HuggingFace dataset
`clarin-knext/msmarco-pl`, and this repo's own miner. **No account-gated or
proprietary keyword data** — anyone can rebuild it.

### Why a mix

Web n-grams alone are a poor stand-in for search keywords. The miner's filter
only rejects phrases that *start or end* with a stopword, so grammatical
fragments survive: `operacja rozpoczęła`, `europejskiego i rady z dnia`,
`czas dostawy wynosi`. Nobody types those into a search box.

Wikipedia titles fix exactly that, and for free: an article title is by
construction a noun phrase with no finite verb — the same shape POS filtering
would produce, without running a tagger or inheriting its errors on
fragmentary web text.

### Casing: everything is lowercased, including titles

All three models are case-sensitive (SentencePiece tokenisers), so
`Warszawa` and `warszawa` are different token sequences with different vectors.
Mixing cased Wikipedia titles into lowercase n-grams would make the background
model a **bimodal** distribution that matches neither half at inference.

Lowercase is also what inference sees: Google Ads keyword lists are lowercase
by convention, and so is this benchmark's test set. **Matching the inference
distribution matters more than which casing you pick** — but the two must
agree.

### Known limitations

- `msmarco-pl` is **machine-translated from English**. The phrasing is
  translationese and the topics skew Anglo-American — Polish-specific
  commercial intents (`ubezpieczenie oc`) are underrepresented.
- Wikipedia titles are **encyclopedic entities**, not commercial intent.
- There is **no public dataset of real Polish ad keywords with volumes** —
  that data is proprietary to Google/Ahrefs/Semrush. Everything here is a
  reproducible substitute, closer than pure web n-grams but still a substitute.

## Clustering method

Cosine **kNN graph + Leiden** (`RBConfigurationVertexPartition`), swept over
resolutions `0.5 / 1 / 2 / 4`. This mirrors the production backend in the
sitefocus repo (`analysis/clustering_Louvain_Leiden_UmapHdbscan`).

**A cosine threshold is deliberately not used.** In high-dimensional
anisotropic embedding space a fixed similarity floor empties the graph: most
nodes end up isolated, each isolated node counts as its own "cluster", and
modularity climbs toward 1.0 — high scores that mean *nothing was clustered*.
kNN is adaptive and gives every keyword edges regardless of absolute cosine.

## Models compared

All must be downloaded to `~/models/` first.

| key | model | params | dim | task prompt available |
|---|---|--:|--:|---|
| `embeddinggemma` | `google/embeddinggemma-300m` | 308M | 768 (Matryoshka) | yes (`Clustering`) |
| `qwen3_06b` | `Qwen/Qwen3-Embedding-0.6B` | 0.6B | 1024 | no |
| `bge_m3` | `BAAI/bge-m3` | 568M | 1024 | no |

**Prompts are off by default.** Only embeddinggemma defines task prompts;
bge-m3 and Qwen3-Embedding take raw text. Using a prompt for one model and not
the others compares different things, so every model is measured on plain
embeddings. `--use-prompts` enables them for comparison.

Measured effect of embeddinggemma's `Clustering` prompt on this dataset — it
**hurts**, and it is also slower (an instruction is re-encoded with every
keyword):

| resolution | with prompt | without | Δ AMI |
|---|--:|--:|--:|
| r=2.0 | 0.885 | 0.914 | +0.029 |
| r=4.0 | 0.874 | 0.924 | +0.050 |

Plausible reason: that prompt targets clustering of *documents*. Prepending a
long instruction to a 3-word phrase makes the instruction most of the input and
dilutes the phrase's own signal. Do not assume a task prompt helps on short
text just because the model card offers one — measure it.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
```

The CPU torch index matters — the default index pulls ~2.5 GB of CUDA wheels
this benchmark never uses.

Download the models:

```bash
hf download BAAI/bge-m3                 --local-dir ~/models/bge-m3
hf download Qwen/Qwen3-Embedding-0.6B   --local-dir ~/models/qwen3-embedding-0.6b
hf download google/embeddinggemma-300m  --local-dir ~/models/embeddinggemma-300m   # gated
```

`embeddinggemma` is gated: accept the Gemma licence on the model page and run
`hf auth login` first.

## Run

```bash
.venv/bin/python bench.py                              # everything, default sweep
.venv/bin/python bench.py --models embeddinggemma      # one model
.venv/bin/python bench.py --whiten                     # add ZCA-whitened space
.venv/bin/python bench.py --threads 4                  # simulate a smaller VPS
.venv/bin/python bench.py --use-prompts                # opt into task prompts
```

Results land in `results/results.json` (full detail) and `results/results.md`
(the comparison table).

To see *what* a model got wrong rather than how much:

```bash
.venv/bin/python inspect_clusters.py --model bge_m3 --resolution 2.0
```

It prints every cluster with its keywords, flags **mixed clusters** (several
intents merged — hurts homogeneity) and **split groups** (one intent
fragmented — hurts completeness). Run it before believing any AMI number.

### `--threads`

Re-run with `--threads 2` / `--threads 4` to match the box you intend to rent.

Counter-intuitive measured result: **4 threads is faster than 20** for every
model here (bge-m3 40.4 vs 37.3 kw/s; embeddinggemma 53.4 vs 35.5). Keywords
are short and batches small, so thread synchronisation overhead outweighs the
parallelism. A small VPS pays no penalty for having few cores — do not assume
throughput scales with core count without measuring it.

### `--whiten`

Adds a ZCA-whitened variant of each embedding space. **Read it as an
indication only.** With 150 keywords and 768–1024 dimensions the covariance is
massively rank-deficient, so it is shrunk toward the identity before
inversion. A real fitted background (the parent repo's `kw` granularity,
fitted on 50 000 phrases) is a different and far better-conditioned object.

## Findings

Measured on 150 keywords / 15 groups, no prompts, `--threads 4`, Leiden on a
cosine kNN graph. **All of it in `raw+L2` space — no fitted whitening
background** (see "What this benchmark does not settle").

| model | best AMI | #clusters | mixed clusters | misplaced keywords | kw/s |
|---|--:|--:|--:|--:|--:|
| **bge-m3** | **0.989** (r=4) | **15** ✓ | **1 / 15** | **2 / 150** | 38.5 |
| embeddinggemma | 0.924 (r=4) | 18 | 5 / 14 | 14 / 150 | **63.3** |
| qwen3-0.6b | 0.897 (r=4) | 20 | 11 / 14 | 19 / 150 | 39.6 |
| tfidf (baseline) | 0.471 | 83 | — | — | — |

### AMI hides single-group collapse — always inspect the clusters

This is the most transferable lesson here. On AMI alone, embeddinggemma (0.914
at r=2) looks like a reasonable trade for being 1.6× faster. `inspect_clusters.py`
shows it is not:

```
[1] n=16  purity=0.56  ~naprawa_pralki  <== MIXED: naprawa_pralki×9, ubezpieczenie_samochodu×7
```

Car insurance merged into **washing-machine repair**. The whole
`ubezpieczenie_samochodu` group dissolved across four unrelated clusters; under
qwen3-0.6b it scattered across seven. Nothing semantic or lexical connects
those intents — as ad groups the output is unusable.

bge-m3's only error is defensible by comparison: two keywords carrying
`warszawa` (`catering dietetyczny warszawa`, `agencja seo warszawa`) were
pulled into the Warsaw-rental cluster. Those phrases genuinely share a
location. Its insurance group stayed at purity 1.00.

A single averaged agreement score cannot distinguish "slightly fuzzy
boundaries everywhere" from "one whole intent shredded". Report cluster counts
and inspect the members before trusting a headline metric.

### Geo modifiers are a systematic trap

Every model's residual errors cluster around city names. No embedding model
will fix this: `agencja seo warszawa` really is close to other Warsaw phrases.
The fix belongs upstream — strip geo modifiers before clustering and restore
them afterwards. In Google Ads location is campaign targeting anyway, not ad
group semantics.

### More CPU threads is not faster

4 threads beat 20 for every model (bge-m3 40.4 vs 37.3 kw/s; embeddinggemma
53.4 vs 35.5). Short keywords in small batches make thread synchronisation cost
more than the parallelism returns. A small VPS is not penalised here.

## Does a fitted background help? Measured.

Three backgrounds were fitted on `pl_kwmix900k` (one per model,
`run_kwmix_backgrounds.sh`) and applied to the same CPU embeddings. Δ AMI
against the same model's `raw+L2` space:

| model | r=0.5 | r=1.0 | r=2.0 | r=4.0 | best raw → best bg |
|---|--:|--:|--:|--:|---|
| **qwen3-0.6b** | **+0.152** | **+0.123** | **+0.086** | **+0.075** | 0.897 → **0.972** |
| embeddinggemma | +0.112 | −0.019 | −0.038 | −0.020 | 0.924 → 0.904 |
| bge-m3 | +0.049 | +0.057 | −0.023 | −0.011 | 0.989 → 0.978 |

**The background helps the model that needed it most.** qwen3-0.6b gains at
every resolution and nearly catches bge-m3 (its cluster count also corrects
from 20 to 16, against a true 15). bge-m3 was already at 0.989 — there is no
anisotropy damage left to undo, so whitening only blurs its strongest signal.

**One pattern holds for all three: at low resolution the background always
helps, substantially (+0.05 to +0.15).** Coarse cuts are where a shared
dominant direction glues unrelated groups together, and that is exactly what
whitening removes. At high resolution the cut is already fine-grained, so
there is little to rescue and decorrelation costs more than it returns.

**Anisotropy magnitude does not predict who benefits.** embeddinggemma has by
far the strongest anisotropy (`top_ev_ratio` 53.9 vs bge-m3's 29.3) and gains
the least. Do not use the eigenvalue ratio to decide whether to whiten —
measure on the actual task.

### The earlier "whitening destroys everything" result was an artefact

`--whiten` (batch, shrinkage-stabilised on 150 rows) dropped bge-m3 from 0.989
to 0.55. A properly fitted background does nothing of the sort — worst case
here is −0.038. Anyone reading a batch-whitening number as evidence about
backgrounds is reading a rank-deficient covariance estimate, not a background.

## What this benchmark does not settle

**The test set is too easy to show what a background is for.** Its 15 groups
are semantically distant (running shoes vs. cheesecake vs. dental implants), so
the topical signal is strong enough that anisotropy barely interferes. A
background removes a shared dominant direction; where that direction was not
doing damage, removing it shows as ~zero.

The scenario backgrounds actually serve is the opposite one: a large keyword
list from a *single* niche, where every phrase is close to every other and the
shared direction is what stops anything from separating. This benchmark cannot
measure that, and a near-ceiling score on it (bge-m3 at 0.989) is not evidence
that whitening is unnecessary at 100 k keywords from one domain.

**The scores are also saturated for bge-m3.** At AMI 0.989 there is no headroom
left to detect further improvement, so this dataset cannot rank anything above
that level.

## Interpreting the output

- **homogeneity** — do clusters stay inside one intent group? High = no mixing.
- **completeness** — is each intent group kept in one cluster? Low = split up.
- **AMI / ARI** — chance-corrected overall agreement. The headline numbers.
- **#clusters vs. 15** — far above 15 means over-splitting at that resolution;
  compare models at a resolution where cluster counts are comparable, not just
  at their individual best.

Do not pick a winner on a single best-resolution AMI: that cherry-picks the
knob. Look at whether a model is consistently ahead across the sweep.

## Limitations

Stated plainly, because they bound what this benchmark can conclude:

- **150 keywords, 15 groups** is small. It detects large differences between
  models reliably; it cannot resolve a 0.01 AMI gap.
- **Hand-labelled by one person.** Group boundaries are opinions; a few
  keywords are legitimately ambiguous.
- **Polish only, commercial/SEO domain.** Says nothing about other languages or
  about long-document clustering.
- **Throughput is host-specific.** Absolute keywords/s transfers to a VPS only
  after re-running with matching `--threads`.
- **Batch whitening is not a fitted background** — see `--whiten` above.
