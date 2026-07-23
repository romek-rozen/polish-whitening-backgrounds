# keyword-clustering-with-background

Clustering Polish keyword lists with the CPU-servable models from this repo,
with and without a ZCA whitening background — and a like-for-like comparison of
three clustering families on the same keywords, same models, same backgrounds.

Sibling project: [`../pl-keyword-embedding-cpu-bench`](../pl-keyword-embedding-cpu-bench)
answers *which model* (it has hand-labelled ground truth and stays CPU-only).
This project answers *how to cluster with it* at real scale. The scripts here
import from it rather than duplicating the model registry.

## The question

You have 20 000 or 100 000 Polish keywords. You want ad-group-shaped clusters.
Three things are actually in play:

1. **Which clustering family** — they behave completely differently at scale.
2. **Whitening or not** — a background removes the shared direction that makes
   every keyword look ~0.5 similar to every other.
3. **How much noise** — how many keywords genuinely belong to no group.

## Candidate methods vs. reference

**Candidates** work directly on the full embedding — 768 or 1024 dimensions,
cosine only, nothing discarded:

| method | how it groups | linkage equivalent |
|---|---|---|
| threshold + union-find | connected components of the threshold graph | **single** |
| agglomerative | merge the most similar pairs first, stop at a floor | **average** (default) |
| kNN + Leiden | modularity on a neighbour graph | — (graph, not linkage) |

**UMAP + HDBSCAN is kept as a reference only, not a candidate.** Projecting a
1024-dim embedding down to 30 dimensions throws away the information you just
paid to compute, and UMAP is stochastic — a different seed moves the clusters.

It stays in the comparison for one reason: it is the only method here whose
noise is **not a knob we set**. HDBSCAN calls a point noise when it lies in no
dense region, which makes its ~22 % the only independent estimate of how much of
a keyword list genuinely belongs nowhere. Every other noise figure in this
project is exactly as large as we chose `min_similarity` and `min_size` to make
it. Drop the reference and you lose the ability to say whether your chosen noise
level is sane — so quote it as a yardstick, never as a recommendation.

## Models and backgrounds

| model | dim | background |
|---|--:|---|
| `BAAI/bge-m3` | 1024 | `bgem3_pl_kwmix900k_mrl1024` |
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | `qwen3_06b_pl_kwmix900k_mrl1024` |
| `google/embeddinggemma-300m` | 768 | `embgemma_pl_kwmix900k_mrl768` |

All three are small enough to self-host on a GPU-less VPS. The backgrounds ship
in [`../backgrounds/`](../backgrounds) and were fitted on `pl_kwmix900k` — see
the parent [README](../README.md#cpu-servable-models-pl_kwmix900k).

## Results on a real 19 801-keyword list

One Polish site's keyword export, deduplicated and lowercased. **No ground
truth**: the export's URLs were 16 months stale, so scoring against them would
measure the annotations, not the clustering. What follows is structural —
cluster counts, sizes, noise — plus clusters printed for inspection.

### A threshold's "noise" is arithmetic, not a quality judgement

The single most important result here. `analyze_similarity_distribution.py`
measures, for every keyword, its **top-1 similarity** — how close its nearest
neighbour is. A threshold method cannot cluster a keyword whose best neighbour
sits below the cutoff, however sensible that keyword is.

In whitened space, with the fixed 0.8 the upstream tool uses:

| model | keywords with a neighbour ≥ 0.8 | ⇒ unclusterable | measured noise at 0.8 |
|---|--:|--:|--:|
| bge-m3 | 39.4 % | 60.6 % | **60.63 %** |
| qwen3-0.6b | 38.5 % | 61.5 % | **61.46 %** |
| embeddinggemma | 42.0 % | 58.0 % | **57.96 %** |

The prediction and the measurement agree to two decimals, because they are the
same quantity computed two ways. **That 60 % is not a verdict on the keywords —
it is where the cutoff falls on the distribution.** Median top-1 similarity in
whitened space is ~0.75, so 0.8 sits *above the median*: more than half the list
has no twin that close, by construction.

Coverage by cutoff (share of keywords with any neighbour at or above it):

| cutoff | 0.5 | 0.6 | 0.7 | 0.75 | 0.8 | 0.9 |
|---|--:|--:|--:|--:|--:|--:|
| bge-m3, whitened | 86.8 % | 75.6 % | 59.6 % | 49.6 % | 39.4 % | 18.1 % |

**Consequence: a similarity floor should be a percentile of the list's own
top-1 distribution, not a constant.** A value calibrated on lists of hundreds
will discard most of a list of twenty thousand — not because the second list is
worse, but because its distribution is different.

Whitening is also visible here: median similarity of a *random* pair drops to
0.006–0.009 while top-1 stays at ~0.75. That gap between "random pair" and
"nearest neighbour" is what makes a threshold meaningful at all; in raw space
top-1 medians are 0.87–0.92 and everything looks close to everything.

### Two independent methods agree on how much noise there is

This is the most useful thing the comparison produced.

| method | how noise is decided | noise |
|---|---|--:|
| UMAP + HDBSCAN | by the algorithm — no threshold to set | **~22 %** |
| agglomerative, similarity floor 0.5, min 2 | by our chosen floor | **20.2 %** |

Two methods sharing no machinery — one density-based on a projection, one
merging pairs by cosine in full dimension — land within two points of each
other. That makes ~20 % a defensible estimate of how much of this list belongs
to no group, rather than an artefact of one algorithm.

It also means a 0.5 floor is **not** an arbitrary knob on this data: it lands
where a method with no knob independently puts the boundary.

### The three families do not agree, at all

| method | clusters | noise | largest | median |
|---|--:|--:|--:|--:|
| kNN + Leiden, no cutoff | 101–120 | **0 %** | 417–778 | 152–188 |
| kNN + Leiden + cutoff | 103–126 | 6–9 % | 447–719 | 124–155 |
| **agglomerative, floor 0.5** | **2 883** | 20.2 % | **136** | **3** |
| UMAP + HDBSCAN *(reference)* | 391–454 | 20–26 % | 275–739 | 21–25 |
| threshold + union-find (bdos) | 150–1954 | 7–61 % | **239–18 040** | 2 |

(Leiden, HDBSCAN and threshold ranges span the three models, raw and whitened;
the agglomerative row is bge-m3 + background.)

**Cluster size is where the methods separate, not cluster count.** Leiden's
largest group holds 400–780 keywords — that is a campaign, not an ad group.
Union-find's largest holds up to 18 040 of 19 801. Agglomerative with average
linkage tops out at 136 with a median of 3, which is the size an ad group
actually is.

**Plain Leiden reports 0 % noise because it is not allowed to report anything
else.** Every node gets a label whether or not it belongs anywhere. A real
keyword list always contains junk — bare prices, punctuation-prefixed
fragments, stray numbers — so 0 % noise is an artefact of the method, not a
property of the data.

### How much noise is there really?

**About 20–25 %.** That figure comes from UMAP + HDBSCAN, and it is the most
defensible one available here because it is *not* a knob we set: HDBSCAN calls a
point noise when it lies in no dense region. The other families' noise is
whatever we chose the floor and `min_size` to be.

Treat it as an estimate of the same order, not a constant. It moves with
`min_cluster_size` (19.6 % at min=10 vs 23.8 % at min=3) and with the UMAP
projection.

### The threshold method breaks down at this scale

`keyword_cluster`'s semantic tier — cosine threshold + connected components —
produces one of two failure modes at 19 801 keywords, and its default 0.8 hits
both depending on the space:

| space | largest cluster | noise |
|---|--:|--:|
| raw, qwen3-0.6b | **18 040 / 19 801 (91 %)** | 7 % |
| raw, bge-m3 | 11 925 (60 %) | 21 % |
| background, bge-m3 | 239 | **61 %** |

Union-find takes the *transitive closure*: A~B and B~C merges A with C even when
A and C are unrelated. In a space where everything is ~0.8 similar, one chain
swallows most of the list. Whitening breaks the chains — and then the same fixed
0.8 sits far above the whitened similarity scale, so 61 % falls out as noise.
Median cluster size 2 means the survivors are mostly pairs.

**This is not a verdict on the tool.** `keyword_cluster` targets lists of
hundreds, where transitive chaining has no room to snowball and a hand-tuned
threshold is reasonable. It says the *method* does not transfer to 20 k+ without
recalibration, which is exactly what a threshold requires and a kNN graph does
not.

### What the background does

It is not a quality knob here — it is a **calibration** step:

| model | mean kNN edge similarity, raw | whitened |
|---|--:|--:|
| qwen3-0.6b | 0.835 | 0.507 |
| bge-m3 | 0.752 | 0.507 |
| embeddinggemma | 0.751 | 0.560 |

Raw, the three models sit at different similarity scales (0.75–0.84), so a
threshold tuned on one does not transfer to another. **Whitened, they converge
to ~0.51–0.56** — the same setting starts meaning the same thing across models.

On structure the effect is modest and consistent: slightly more clusters,
smaller largest cluster (qwen3-0.6b: 598 → 417), similar noise. On the
hand-labelled benchmark next door it is worth **+0.075 AMI for qwen3-0.6b** and
~nothing for bge-m3 — the background helps most where the raw space was worst.

### Cost on a CPU-only VPS

100 000 keywords, 4 threads (`scale_test_cpu.py`):

| stage | time | peak RSS |
|---|--:|--:|
| **apply background** | **0.4 s** | 1.29 GB |
| approximate kNN | 18.0 s | 1.29 GB |
| build graph | 1.3 s | 1.44 GB |
| Leiden | 9.4 s | 1.47 GB |
| **total** | **29.2 s** | **1.47 GB** |

Clustering is cheap. Embedding is the real cost (~40 kw/s per CPU core-set), and
it is paid once — hence the embedding cache.

## Recommendation

Everything above points at one configuration:

**bge-m3 + `bgem3_pl_kwmix900k_mrl1024` + agglomerative (average linkage) +
similarity floor 0.526 + min cluster size 2.**

Four independent reasons for bge-m3:

1. best on the labelled benchmark next door (AMI 0.989 vs 0.924 / 0.897),
2. fewest oversized groups — 2 clusters of 100+ keywords, against 9 for
   qwen3-0.6b and 16 for embeddinggemma,
3. smallest largest-cluster under agglomerative (136, against 462 and 523),
4. it does not glue by syntactic frame; embeddinggemma merges "kurs/opis/typy
   <term>" phrases into one group regardless of topic.

The floor comes from `sweep_agglomerative_fine.py` (flattest statistics in a
±0.010 window): 0.526 for bge-m3, 0.502 for qwen3-0.6b, 0.506 for embeddinggemma.

### Two things to do to the data first

Neither is a model defect — both are phrase types that belong in separate
campaigns, and no embedding will separate them for you:

- **Person names.** All three models pull them into a single "people" bucket.
  They are similar as a *kind of entity*, not as an intent.
- **Geographic modifiers.** A city name drags a phrase into the local cluster
  regardless of industry. In Google Ads location is campaign targeting anyway.

### Do not delete the noise

~20 % of the list lands outside every cluster, and qualitative review found
commercial offer keywords in there. "Noise" here means "no close twin", not
"worthless" — see the top-1 distribution section above.

## Scripts

| script | what it does |
|---|---|
| `cluster_keywords.py` | cluster a keyword CSV, print groups, export per-keyword CSV with cluster labels |
| `sweep_leiden.py` | kNN + Leiden across floors, resolutions, min sizes |
| `sweep_threshold.py` | the `keyword_cluster` method: threshold + connected components |
| `sweep_umap_hdbscan.py` | UMAP projection + density clustering |
| `make_xlsx.py` | merge all sweeps into one pivot-ready workbook |
| `scale_test_cpu.py` | 100 k keywords on CPU: time and memory per stage |
| `embed_cache.py` | embed once, reuse everywhere |
| `incremental_csv.py` | flush every row, so a crash costs one row |

### Typical run

```bash
V=../.venv-cuda/bin/python          # or ../pl-keyword-embedding-cpu-bench/.venv/bin/python for CPU

$V cluster_keywords.py --csv work/keywords.csv \
    --background ../backgrounds/bgem3_pl_kwmix900k_mrl1024 \
    --resolution 4.0 --out-csv work/clusters.csv

$V sweep_leiden.py        --csv work/keywords.csv --out work/leiden_sweep.csv
$V sweep_threshold.py     --csv work/keywords.csv --out work/threshold_sweep.csv
$V sweep_umap_hdbscan.py  --csv work/keywords.csv --out work/umap_sweep.csv
$V make_xlsx.py --leiden work/leiden_sweep.csv --threshold work/threshold_sweep.csv \
    --umap work/umap_sweep.csv --out work/klastrowanie.xlsx
```

Input: any CSV with a `KEYWORD` column (first column is used otherwise).
Keywords are lowercased, deduplicated, and stripped of invisible formatting
characters.

**The first run embeds; every later run reads the cache.** Cache key is the
model plus a hash of the exact keyword list, so editing the list produces a new
entry rather than a silently mismatched one. `--refresh-cache` forces re-embedding.

### GPU

Scripts default to `--device cuda` for embedding only — vectors are identical
on either device, so it changes wall-clock, not results. **All clustering runs
on CPU** regardless: pynndescent, igraph/leidenalg, UMAP and HDBSCAN have no GPU
path here. Timings in the scale test are therefore genuine CPU numbers. Pass
`--device cpu` to embed without a GPU.

## Reading the sweep workbook

`work/klastrowanie.xlsx`, sheet `sweep`, one row per configuration. Two traps —
also written into the `legend` sheet:

**Do not compare absolute floors across spaces.** Whitening rescales every
cosine. `floor_mode=abs` rows are comparable only within one space; use
`floor_mode=pct` (drop the weakest X % of edges) to compare raw against
whitened.

**Noise is not a defect to minimise.** `floor=0, min_size=1` gives 0 % noise by
forcing every keyword into a group, junk included. Some noise means the method
is being honest about the tail.

## Limitations

- **No ground truth at scale.** The structural numbers are solid; "which
  clustering is *better*" is not answered here. The labelled comparison lives in
  the sibling benchmark, on 150 keywords.
- **One site, one language.** 19 801 keywords from a single Polish domain in
  one vertical. A different vertical may behave differently.
- **UMAP is stochastic.** Fixed seed, but a different seed shifts cluster counts
  by a few percent. Do not read two-digit precision into them.
- **Coherence is not comparable across spaces.** Whitening shrinks all cosines,
  so a lower centroid coherence after whitening means nothing on its own.
