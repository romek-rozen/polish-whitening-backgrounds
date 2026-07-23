"""Evaluate clustering on a real keyword→URL export, using URL as ground truth.

The hand-labelled test set has 150 keywords across 15 semantically distant
topics — too easy, and too small, to show what a whitening background does.
A real site export is the opposite: thousands of keywords inside **one domain**,
where everything is close to everything and the shared dominant direction is
exactly what stops intents from separating.

Ground truth here is the URL a keyword maps to: keywords landing on the same
page are, in practice, the same intent. That label is **imperfect and this
matters when reading the scores**:

- A hub page (homepage, category index) ranks for hundreds of unrelated
  queries. It is not one intent, and counting it as one punishes every model
  unfairly — so URLs above ``--max-per-url`` are dropped.
- Two different pages can serve the same intent. A model that merges them is
  arguably right, and the metric penalises it anyway. Scores are therefore a
  *lower bound* on real quality.
- A keyword mapping to several URLs has no unambiguous label, so it is dropped.

Because of that, the number to read is the **difference** between raw and
whitened space, not the absolute AMI. Both suffer the same label noise.

Usage::

    python eval_real_keywords.py --csv work/keywords_url_2026-07-23.csv \\
        --model bge_m3 --background ../backgrounds/bgem3_pl_kwmix900k_mrl1024
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bench import MODELS, apply_background, encode, l2_normalize

ROOT = Path(__file__).resolve().parent


def load_pairs(
    path: Path, min_per_url: int, max_per_url: int
) -> tuple[list[str], np.ndarray, dict]:
    """Return (keywords, integer URL labels, stats) with ambiguous rows dropped."""
    by_keyword: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            keyword = (row.get("KEYWORD") or "").strip().lower()
            url = (row.get("URL") or "").strip()
            if keyword and url:
                by_keyword[keyword].add(url)

    raw_keywords = len(by_keyword)
    # A keyword on several URLs has no unambiguous label.
    unique = {k: next(iter(v)) for k, v in by_keyword.items() if len(v) == 1}
    dropped_ambiguous = raw_keywords - len(unique)

    per_url = Counter(unique.values())
    kept_urls = {
        url for url, count in per_url.items() if min_per_url <= count <= max_per_url
    }
    dropped_hub = sum(c for u, c in per_url.items() if c > max_per_url)
    dropped_tiny = sum(c for u, c in per_url.items() if c < min_per_url)

    keywords, labels = [], []
    url_index: dict[str, int] = {}
    for keyword, url in unique.items():
        if url not in kept_urls:
            continue
        keywords.append(keyword)
        labels.append(url_index.setdefault(url, len(url_index)))

    stats = {
        "raw_unique_keywords": raw_keywords,
        "dropped_ambiguous_keyword_on_many_urls": dropped_ambiguous,
        "dropped_hub_url_over_max": dropped_hub,
        "dropped_url_under_min": dropped_tiny,
        "kept_keywords": len(keywords),
        "kept_urls": len(url_index),
    }
    return keywords, np.asarray(labels), stats


def cluster_and_score(
    vectors: np.ndarray, truth: np.ndarray, knn: int, resolutions: list[float], seed: int
) -> list[dict]:
    """Approximate kNN + Leiden. Brute force is not an option at this size."""
    import igraph as ig
    import leidenalg
    from pynndescent import NNDescent
    from sklearn import metrics

    index = NNDescent(
        vectors.astype(np.float32), metric="cosine", n_neighbors=knn + 1,
        random_state=seed, n_jobs=4,
    )
    neighbours, distances = index.neighbor_graph

    n = vectors.shape[0]
    edges, weights = [], []
    for i in range(n):
        for j, dist in zip(neighbours[i], distances[i]):
            j = int(j)
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges.append((a, b))
            weights.append(max(1.0 - float(dist), 1e-6))
    graph = ig.Graph(n=n, edges=edges)
    graph.es["weight"] = weights
    graph.simplify(combine_edges="max")

    out = []
    for resolution in resolutions:
        partition = leidenalg.find_partition(
            graph, leidenalg.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=resolution, seed=seed,
        )
        pred = np.asarray(partition.membership)
        out.append({
            "resolution": resolution,
            "n_clusters": int(len(set(pred.tolist()))),
            "homogeneity": float(metrics.homogeneity_score(truth, pred)),
            "completeness": float(metrics.completeness_score(truth, pred)),
            "ami": float(metrics.adjusted_mutual_info_score(truth, pred)),
            "ari": float(metrics.adjusted_rand_score(truth, pred)),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--model", default="bge_m3", choices=list(MODELS))
    ap.add_argument("--background", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-per-url", type=int, default=5)
    ap.add_argument(
        "--max-per-url", type=int, default=300,
        help="URLs above this are hub pages, not one intent — dropped.",
    )
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--resolutions", type=float, nargs="*", default=[1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    keywords, truth, stats = load_pairs(args.csv, args.min_per_url, args.max_per_url)
    print(f"source: {args.csv}")
    for key, value in stats.items():
        print(f"  {key:42s} {value}")
    print(f"\n  -> {len(keywords)} keywords / {stats['kept_urls']} URL groups "
          f"(median {int(np.median(np.bincount(truth)))} kw per group)\n")

    started = time.perf_counter()
    encoded = encode(MODELS[args.model], keywords, args.batch_size)
    print(f"encoded on {args.device} in {time.perf_counter()-started:.0f}s "
          f"({encoded.throughput:.0f} kw/s, dim={encoded.dim})\n")

    spaces = {"raw+L2": l2_normalize(encoded.vectors)}
    if args.background:
        spaces[f"bg:{args.background.name}"] = apply_background(
            encoded.vectors, args.background
        )

    print(f"{'space':38s} {'r':>5} {'#clus':>6} {'homog':>7} {'compl':>7} "
          f"{'AMI':>7} {'ARI':>7}")
    print("-" * 82)
    results: dict[str, list[dict]] = {}
    for name, vectors in spaces.items():
        rows = cluster_and_score(vectors, truth, args.knn, args.resolutions, args.seed)
        results[name] = rows
        for row in rows:
            print(f"{name:38s} {row['resolution']:5.1f} {row['n_clusters']:6d} "
                  f"{row['homogeneity']:7.3f} {row['completeness']:7.3f} "
                  f"{row['ami']:7.3f} {row['ari']:7.3f}")

    if len(results) == 2:
        raw, bg = results["raw+L2"], results[f"bg:{args.background.name}"]
        print(f"\n{'resolution':>12} {'Δ AMI':>9} {'Δ ARI':>9}")
        for r, b in zip(raw, bg):
            print(f"{r['resolution']:12.1f} {b['ami']-r['ami']:+9.3f} "
                  f"{b['ari']-r['ari']:+9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
