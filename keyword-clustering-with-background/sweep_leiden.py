"""Sweep sharpened-Leiden settings across all models and dump a pivot-ready CSV.

Plain Leiden labels every node, so a keyword list comes back with zero noise —
which is a lie. Real lists contain phrases that belong to no group: brand
one-offs, typos, price fragments, stray numbers. "Sharpened" clustering (D91)
introduces noise deliberately, two ways:

- **similarity floor** — drop kNN edges below a cosine threshold. Nodes left
  with no edges cannot join anything.
- **minimum cluster size** — clusters smaller than N are relabelled as noise
  rather than shipped as one-keyword ad groups.

**A fixed floor is not portable between spaces.** Whitening shrinks every
cosine — mean pairwise similarity drops from ~0.37 raw to ~0.03 whitened — so
D91's `sim_floor=0.50` would delete the entire graph in whitened space and
report 100 % noise. Two floor modes are therefore swept:

- ``abs``  — absolute cosine, comparable to D91 but only within one space
- ``pct``  — drop the weakest X % of edges, comparable *across* spaces because
             it adapts to each space's own similarity scale

Read the ``pct`` rows when comparing raw against whitened. Read ``abs`` rows
only when comparing settings inside one space.

Output: one row per (model, space, floor, resolution, min_size) with cluster
counts and noise share — ready for a pivot table.

Usage::

    python sweep_noise.py --csv work/keywords_url_2026-07-23.csv \\
        --out work/noise_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pl-keyword-embedding-cpu-bench"))

from bench import MODELS, apply_background, encode, l2_normalize
from cluster_keywords import load_keywords
from embed_cache import encode_cached
from incremental_csv import IncrementalCsv

ROOT = Path(__file__).resolve().parent

# background dir name per model key; None = raw only
BACKGROUNDS = {
    "bge_m3": "bgem3_pl_kwmix900k_mrl1024",
    "qwen3_06b": "qwen3_06b_pl_kwmix900k_mrl1024",
    "embeddinggemma": "embgemma_pl_kwmix900k_mrl768",
}


def knn_edges(vectors: np.ndarray, knn: int, seed: int, jobs: int):
    """Approximate cosine kNN -> (i, j, similarity) arrays, deduped i<j."""
    from pynndescent import NNDescent

    index = NNDescent(
        vectors.astype(np.float32), metric="cosine", n_neighbors=knn + 1,
        random_state=seed, n_jobs=jobs,
    )
    neighbours, distances = index.neighbor_graph
    n = neighbours.shape[0]
    src = np.repeat(np.arange(n), neighbours.shape[1])
    dst = neighbours.reshape(-1).astype(np.int64)
    sim = 1.0 - distances.reshape(-1).astype(np.float64)
    keep = src != dst
    src, dst, sim = src[keep], dst[keep], sim[keep]
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    # Deduplicate mutual neighbours, keeping the stronger similarity.
    order = np.lexsort((hi, lo))
    lo, hi, sim = lo[order], hi[order], sim[order]
    unique = np.ones(len(lo), dtype=bool)
    unique[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
    return lo[unique], hi[unique], sim[unique]


def leiden(n: int, lo, hi, sim, resolution: float, seed: int) -> np.ndarray:
    import igraph as ig
    import leidenalg

    graph = ig.Graph(n=n, edges=list(zip(lo.tolist(), hi.tolist())))
    graph.es["weight"] = np.maximum(sim, 1e-6).tolist()
    partition = leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution, seed=seed,
    )
    return np.asarray(partition.membership)


def apply_min_size(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Relabel clusters below min_size as noise (-1).

    ``labels`` may already contain -1 for nodes isolated by the similarity
    floor, so cluster sizes are counted over the non-negative labels only —
    np.bincount rejects negatives.
    """
    if min_size <= 1:
        return labels
    out = labels.copy()
    real = labels[labels >= 0]
    if real.size == 0:
        return out
    counts = np.bincount(real)
    small = np.flatnonzero(counts < min_size)
    out[np.isin(labels, small) & (labels >= 0)] = -1
    return out


def summarise(labels: np.ndarray, n: int) -> dict:
    noise = int((labels == -1).sum())
    real = labels[labels >= 0]
    sizes = np.bincount(real) if real.size else np.array([0])
    sizes = sizes[sizes > 0]
    return {
        "n_clusters": int(sizes.size),
        "n_noise": noise,
        "noise_pct": round(100.0 * noise / n, 2),
        "largest": int(sizes.max()) if sizes.size else 0,
        "median_size": int(np.median(sizes)) if sizes.size else 0,
        "clusters_lt_10": int((sizes < 10).sum()) if sizes.size else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backgrounds", type=Path, default=ROOT.parent / "backgrounds")
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--resolutions", type=float, nargs="*", default=[1.0, 2.0, 4.0, 8.0])
    ap.add_argument(
        "--abs-floors", type=float, nargs="*", default=[0.0, 0.5, 0.7],
        help="Absolute cosine floors (D91 style). Only comparable within a space.",
    )
    ap.add_argument(
        "--pct-floors", type=float, nargs="*", default=[0.0, 25.0, 50.0],
        help="Drop the weakest X%% of edges. Comparable across spaces.",
    )
    ap.add_argument("--min-sizes", type=int, nargs="*", default=[1, 3, 10])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--refresh-cache", action="store_true",
                    help="Ignore cached vectors and re-embed.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    keywords = load_keywords(args.csv)
    n = len(keywords)
    print(f"{n} unique keywords from {args.csv.name}\n")

    writer = IncrementalCsv(args.out, ['method', 'model', 'space', 'floor_mode', 'floor_setting', 'floor_cosine', 'mean_edge_sim', 'edges_kept_pct', 'isolated_nodes', 'resolution', 'min_size', 'n_keywords', 'n_clusters', 'n_noise', 'noise_pct', 'largest', 'median_size', 'clusters_lt_10'])
    rows: list[dict] = []
    for model_key in args.models:
        started = time.perf_counter()
        encoded = encode_cached(model_key, keywords, batch_size=args.batch_size,
                                device=args.device, refresh=args.refresh_cache)
        print(f"{model_key}: encoded in {time.perf_counter()-started:.0f}s "
              f"(dim={encoded.dim})")

        spaces = {"raw": l2_normalize(encoded.vectors)}
        bg_dir = args.backgrounds / BACKGROUNDS[model_key]
        if bg_dir.exists():
            spaces["background"] = apply_background(encoded.vectors, bg_dir)
        else:
            print(f"  WARNING: {bg_dir} missing — raw only")

        for space, vectors in spaces.items():
            lo, hi, sim = knn_edges(vectors, args.knn, args.seed, args.jobs)
            mean_sim = float(sim.mean())
            print(f"  {space:11s} edges={len(lo)} mean_sim={mean_sim:.3f}")

            floors: list[tuple[str, float, float]] = []
            for value in args.abs_floors:
                floors.append(("abs", value, value))
            for pct in args.pct_floors:
                cut = float(np.percentile(sim, pct)) if pct > 0 else 0.0
                floors.append(("pct", pct, cut))

            for mode, setting, cut in floors:
                keep = sim >= cut
                kept_lo, kept_hi, kept_sim = lo[keep], hi[keep], sim[keep]
                isolated = n - len(np.unique(np.concatenate([kept_lo, kept_hi]))) \
                    if kept_lo.size else n

                for resolution in args.resolutions:
                    if kept_lo.size == 0:
                        labels = np.full(n, -1)
                    else:
                        labels = leiden(n, kept_lo, kept_hi, kept_sim, resolution, args.seed)
                        # Nodes with no surviving edge are singletons -> noise.
                        connected = np.zeros(n, dtype=bool)
                        connected[kept_lo] = True
                        connected[kept_hi] = True
                        labels = np.where(connected, labels, -1)

                    for min_size in args.min_sizes:
                        final = apply_min_size(labels, min_size)
                        row = {
                            "model": model_key,
                            "space": space,
                            "floor_mode": mode,
                            "floor_setting": setting,
                            "floor_cosine": round(cut, 4),
                            "mean_edge_sim": round(mean_sim, 4),
                            "edges_kept_pct": round(100.0 * keep.sum() / len(sim), 1),
                            "isolated_nodes": int(isolated),
                            "resolution": resolution,
                            "min_size": min_size,
                            "n_keywords": n,
                        }
                        row.update(summarise(final, n))
                        rows.append(row)
                        writer.write(row)
                    writer.write(row)

    writer.close()
    print(f"\nwrote {args.out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
