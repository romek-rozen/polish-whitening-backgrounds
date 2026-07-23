"""Sweep UMAP + HDBSCAN — the third clustering family, for the same comparison.

The other two families label (almost) everything and have to be *forced* to
produce noise. HDBSCAN is density-based: it declares a point noise when it sits
in no dense region, with no threshold to hand-tune. That makes it the natural
answer to "how much of this keyword list genuinely belongs nowhere?" — and the
honest counterweight to the other methods' noise, which is an artefact of knobs
we chose.

HDBSCAN is run on a UMAP projection, not on the raw 768-1024 dimensions:
density estimation degrades badly in high dimensions, where every point is
roughly equidistant from every other, and HDBSCAN collapses into one blurry
mega-cluster. This mirrors the sitefocus bake-off, which used UMAP-30D.

Note the trade-off this family makes: it produces cleaner surviving clusters
precisely *because* it is allowed to discard the hard tail. A high noise share
is not a failure here — comparing its cluster count against Leiden's without
also reading `noise_pct` compares two different things.

Usage::

    python sweep_umap_hdbscan.py --csv work/keywords.csv --out work/umap_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "pl-keyword-embedding-cpu-bench"))

from bench import MODELS, apply_background, encode, l2_normalize  # noqa: E402
from cluster_keywords import load_keywords
from embed_cache import encode_cached
from incremental_csv import IncrementalCsv  # noqa: E402
from sweep_leiden import BACKGROUNDS, summarise  # noqa: E402


def umap_reduce(vectors: np.ndarray, n_components: int, n_neighbors: int,
                seed: int) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_components=n_components, n_neighbors=n_neighbors, min_dist=0.0,
        metric="cosine", random_state=seed, verbose=False,
    )
    return reducer.fit_transform(vectors.astype(np.float32))


def hdbscan_labels(points: np.ndarray, min_cluster_size: int) -> np.ndarray:
    from sklearn.cluster import HDBSCAN

    model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    return model.fit_predict(points)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--backgrounds", type=Path,
        default=ROOT.parent / "backgrounds",
    )
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--umap-dims", type=int, nargs="*", default=[10, 30])
    ap.add_argument("--umap-neighbors", type=int, default=15)
    ap.add_argument("--min-sizes", type=int, nargs="*", default=[3, 10, 25])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--refresh-cache", action="store_true",
                    help="Ignore cached vectors and re-embed.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    keywords = load_keywords(args.csv)
    n = len(keywords)
    print(f"{n} unique keywords from {args.csv.name}\n")

    writer = IncrementalCsv(args.out, ['method', 'model', 'space', 'floor_mode', 'floor_setting', 'floor_cosine', 'resolution', 'min_size', 'umap_dims', 'umap_neighbors', 'n_keywords', 'n_clusters', 'n_noise', 'noise_pct', 'largest', 'median_size', 'clusters_lt_10'])
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
            for dims in args.umap_dims:
                started = time.perf_counter()
                points = umap_reduce(vectors, dims, args.umap_neighbors, args.seed)
                reduce_s = time.perf_counter() - started

                for min_size in args.min_sizes:
                    started = time.perf_counter()
                    labels = hdbscan_labels(points, min_size)
                    stats = summarise(labels, n)
                    print(f"  {space:11s} umap{dims:<3d} min={min_size:<3d} "
                          f"clusters={stats['n_clusters']:>5d} "
                          f"noise={stats['noise_pct']:>6.2f}%  "
                          f"({reduce_s:.0f}s umap + {time.perf_counter()-started:.0f}s hdbscan)")
                    row = {
                        "method": "umap_hdbscan",
                        "model": model_key,
                        "space": space,
                        "floor_mode": "",
                        "floor_setting": "",
                        "floor_cosine": "",
                        "resolution": "",
                        "min_size": min_size,
                        "umap_dims": dims,
                        "umap_neighbors": args.umap_neighbors,
                        "n_keywords": n,
                    }
                    row.update(stats)
                    rows.append(row)
                    writer.write(row)

    writer.close()
    print(f"\nwrote {args.out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
