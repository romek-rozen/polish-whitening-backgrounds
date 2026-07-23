"""Sweep the keyword_cluster method (threshold + union-find) for comparison.

Reimplements the semantic tier of
https://github.com/romek-rozen/bdos-ai-extensions/tree/main/keyword_cluster
so both clustering families can be measured on the same keywords, same models
and the same backgrounds, and land in one pivot table.

Their method, from ``api._semantic_cluster``::

    embed -> apply ZCA background -> connect every pair with cosine >= threshold
          -> connected components (union-find) -> clusters < min_cluster_size = noise

Defaults there: ``threshold=0.8`` on whitened embeddings, ``min_cluster_size=2``.

**This is a different family from kNN + Leiden**, not a parameter variation:

- union-find takes the *transitive closure*. A~B and B~C puts A and C in one
  group even when A and C are far apart — chains can snowball.
- Leiden optimises modularity on a kNN graph and will cut such a chain.
- a threshold is absolute, so it needs a calibrated space: raw anisotropic
  cosines cluster around 0.7-0.8 and a 0.8 cut behaves wildly differently there
  than on whitened vectors. That is precisely why their default is documented
  as "calibrated on the ZCA background".

The upstream implementation is a pure-Python double loop, O(n²) — 196 M pairs
at 19.8 k keywords, hours of runtime. The maths here is identical; the pair
scan is chunked matrix multiplication and the grouping is
``scipy.sparse.csgraph.connected_components``.

Usage::

    python sweep_threshold.py --csv work/keywords_url_2026-07-23.csv \\
        --out work/threshold_sweep.csv
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
from sweep_leiden import BACKGROUNDS, summarise

ROOT = Path(__file__).resolve().parent


def threshold_components(
    vectors: np.ndarray, threshold: float, chunk: int = 2048
) -> np.ndarray:
    """Connected components of the graph {(i,j) : cosine(i,j) >= threshold}.

    Same result as the upstream double-loop union-find, computed as chunked
    matrix products so 19.8 k keywords finish in seconds instead of hours.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    values = vectors.astype(np.float32)
    n = values.shape[0]
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = values[start:stop] @ values.T
        # Upper triangle only: pair (i,j) once, never i==i.
        local_i, local_j = np.nonzero(block >= threshold)
        global_i = local_i + start
        keep = global_i < local_j
        rows.append(global_i[keep])
        cols.append(local_j[keep])

    row = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    col = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    graph = coo_matrix(
        (np.ones(len(row), dtype=np.int8), (row, col)), shape=(n, n)
    )
    _, labels = connected_components(graph, directed=False)
    return labels, int(len(row))


def apply_min_cluster_size(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Clusters below min_size become noise (-1) — upstream behaviour."""
    counts = np.bincount(labels)
    small = np.flatnonzero(counts < min_size)
    out = labels.copy()
    out[np.isin(labels, small)] = -1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backgrounds", type=Path, default=ROOT.parent / "backgrounds")
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument(
        "--thresholds", type=float, nargs="*",
        default=[0.70, 0.75, 0.80, 0.85, 0.90],
        help="Cosine cutoffs. Upstream default is 0.80 on whitened vectors.",
    )
    ap.add_argument("--min-sizes", type=int, nargs="*", default=[2, 3, 10])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--refresh-cache", action="store_true",
                    help="Ignore cached vectors and re-embed.")
    args = ap.parse_args()

    keywords = load_keywords(args.csv)
    n = len(keywords)
    print(f"{n} unique keywords from {args.csv.name}\n")

    writer = IncrementalCsv(args.out, ['method', 'model', 'space', 'floor_mode', 'floor_setting', 'floor_cosine', 'mean_edge_sim', 'edges_kept_pct', 'isolated_nodes', 'resolution', 'min_size', 'n_keywords', 'n_edges', 'components_before_min_size', 'n_clusters', 'n_noise', 'noise_pct', 'largest', 'median_size', 'clusters_lt_10'])
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

        for space, vectors in spaces.items():
            for threshold in args.thresholds:
                started = time.perf_counter()
                labels, n_edges = threshold_components(vectors, threshold)
                elapsed = time.perf_counter() - started
                raw_groups = int(labels.max()) + 1
                print(f"  {space:11s} thr={threshold:.2f}  edges={n_edges:>9d}  "
                      f"components={raw_groups:>6d}  ({elapsed:.1f}s)")

                for min_size in args.min_sizes:
                    final = apply_min_cluster_size(labels, min_size)
                    row = {
                        "method": "threshold_unionfind",
                        "model": model_key,
                        "space": space,
                        "floor_mode": "abs",
                        "floor_setting": threshold,
                        "floor_cosine": threshold,
                        "mean_edge_sim": "",
                        "edges_kept_pct": "",
                        "isolated_nodes": "",
                        "resolution": "",
                        "min_size": min_size,
                        "n_keywords": n,
                        "n_edges": n_edges,
                        "components_before_min_size": raw_groups,
                    }
                    row.update(summarise(final, n))
                    rows.append(row)
                    writer.write(row)

    writer.close()
    print(f"\nwrote {args.out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
