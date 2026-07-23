"""Agglomerative clustering: merge the most similar pairs first, stop at a floor.

This is the method that matches how you would group keywords by hand — start
from the closest pairs and keep merging while the groups still resemble each
other, then stop. No graph resolution, no dimensionality reduction, no
stochastic projection: just cosine similarity and a cutoff.

**Linkage choice is the whole game here.** A cosine threshold plus union-find —
the ``keyword_cluster`` semantic tier — is mathematically identical to
**single linkage** cut at that threshold. Single linkage merges two groups when
*any* pair across them is close enough, so one bridging keyword chains unrelated
groups together; that is exactly why it put 18 040 of 19 801 keywords into one
cluster. The default here is therefore **average linkage**, which merges on the
mean similarity between groups and refuses that bridge.

Defaults: similarity floor 0.5, minimum cluster size 2, clusters below the
minimum become noise (-1).

Memory: agglomerative clustering needs the full pairwise distance matrix —
~1.6 GB at 20 k keywords, ~40 GB at 100 k. It does not scale like the kNN graph
does; ``--max-n`` guards against starting a run that cannot finish.

Usage::

    python cluster_agglomerative.py --csv work/keywords.csv \\
        --background ../backgrounds/bgem3_pl_kwmix900k_mrl1024 \\
        --min-similarity 0.5 --min-size 2 --out-csv work/agglo.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "pl-keyword-embedding-cpu-bench"))

from bench import MODELS, apply_background, l2_normalize  # noqa: E402
from cluster_keywords import cluster_labels, load_keywords  # noqa: E402
from embed_cache import encode_cached  # noqa: E402
from sweep_leiden import BACKGROUNDS  # noqa: E402

logger = logging.getLogger("agglomerative")


def agglomerative_tree(vectors: np.ndarray, linkage: str) -> np.ndarray:
    """Build one linkage tree that can be cut at multiple similarities."""
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    from scipy.spatial.distance import pdist

    started = time.perf_counter()
    # Vectors are unit-L2, so cosine distance = 1 - cosine similarity.
    distances = pdist(vectors.astype(np.float64), metric="cosine")
    logger.info("pairwise distances: %.0fs (%.1f GB)",
                time.perf_counter() - started, distances.nbytes / 1e9)

    started = time.perf_counter()
    tree = scipy_linkage(distances, method=linkage)
    logger.info("linkage (%s): %.0fs", linkage, time.perf_counter() - started)
    return tree


def cut_agglomerative_tree(tree: np.ndarray, min_similarity: float) -> np.ndarray:
    """Cut a precomputed linkage tree and return zero-based labels."""
    from scipy.cluster.hierarchy import fcluster

    # fcluster labels from 1; shift to 0-based for consistency with the rest.
    return fcluster(tree, t=1.0 - min_similarity, criterion="distance") - 1


def agglomerative(
    vectors: np.ndarray, min_similarity: float, linkage: str
) -> np.ndarray:
    """Cluster by merging closest pairs first; cut at 1 - min_similarity."""
    return cut_agglomerative_tree(
        agglomerative_tree(vectors, linkage), min_similarity
    )


def demote_small(labels: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return labels
    counts = np.bincount(labels)
    small = np.flatnonzero(counts < min_size)
    out = labels.copy()
    out[np.isin(labels, small)] = -1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--model", default="bge_m3", choices=list(MODELS))
    ap.add_argument("--background", type=Path, default=None)
    ap.add_argument("--backgrounds", type=Path, default=ROOT.parent / "backgrounds")
    ap.add_argument("--no-background", action="store_true")
    ap.add_argument("--min-similarity", type=float, default=0.5)
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument(
        "--linkage", default="average", choices=["average", "complete", "single"],
        help="single reproduces threshold+union-find, including its chaining.",
    )
    ap.add_argument("--max-n", type=int, default=40_000,
                    help="Refuse to start above this; the distance matrix is O(n^2).")
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--show", type=int, default=15)
    ap.add_argument("--members", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--refresh-cache", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    keywords = load_keywords(args.csv)
    n = len(keywords)
    if n > args.max_n:
        raise SystemExit(
            f"{n} keywords needs ~{n*(n-1)//2*8/1e9:.0f} GB for the distance "
            f"matrix. Raise --max-n only if that fits in RAM."
        )
    print(f"{n} unique keywords")

    encoded = encode_cached(args.model, keywords, batch_size=args.batch_size,
                            device=args.device, refresh=args.refresh_cache)
    if args.no_background:
        vectors, space = l2_normalize(encoded.vectors), "raw+L2"
    else:
        bg = args.background or (args.backgrounds / BACKGROUNDS[args.model])
        vectors, space = apply_background(encoded.vectors, bg), bg.name
    print(f"space: {space}   linkage: {args.linkage}   "
          f"min_similarity: {args.min_similarity}   min_size: {args.min_size}\n")

    labels = demote_small(agglomerative(vectors, args.min_similarity, args.linkage),
                          args.min_size)

    members: dict[int, list[int]] = defaultdict(list)
    for row, cluster_id in enumerate(labels):
        members[int(cluster_id)].append(row)
    noise = members.pop(-1, [])
    sizes = np.array([len(v) for v in members.values()]) if members else np.array([0])

    print(f"clusters: {len(members)}   noise: {len(noise)} "
          f"({100*len(noise)/n:.1f}%)   largest: {sizes.max()}   "
          f"median: {int(np.median(sizes))}\n")

    names = cluster_labels(keywords, vectors, labels)
    for rank, cluster_id in enumerate(
        sorted(members, key=lambda c: -len(members[c]))[:args.show], start=1
    ):
        rows = members[cluster_id]
        print(f"[{rank}] n={len(rows)}  ~{names.get(cluster_id, '')}")
        for row in rows[:args.members]:
            print(f"      {keywords[row]}")
        if len(rows) > args.members:
            print(f"      … +{len(rows)-args.members} more")
        print()

    if args.out_csv:
        import csv

        counts = Counter(labels.tolist())
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["keyword", "cluster", "label", "size"])
            order = sorted(range(n), key=lambda i: (-counts[int(labels[i])],
                                                    int(labels[i]), keywords[i]))
            for i in order:
                cid = int(labels[i])
                writer.writerow([keywords[i], cid,
                                 "(NOISE)" if cid == -1 else names.get(cid, ""),
                                 counts[cid]])
        print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
