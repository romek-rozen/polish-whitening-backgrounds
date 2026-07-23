"""Can a CPU-only VPS actually cluster 100 000 Polish keywords?

The quality benchmark (`bench.py`) runs on 150 keywords — enough to compare
models, useless for answering "will this survive a real keyword list". At 100 k
the brute-force cosine kNN used there is 10^10 pair comparisons and a 40 GB
similarity matrix, so it cannot be the production path.

This measures the real pipeline on CPU, stage by stage, on embeddings that
already exist (embedding cost is measured separately by `bench.py` —
here the question is the *clustering*):

    load fp16 chunks -> apply background -> approximate kNN -> Leiden

Reports wall-clock and peak RSS per stage, so the VPS sizing question has a
number instead of a guess.

Usage::

    python scale_test_cpu.py --chunks work/chunks_kwmix_bgem3 \\
        --background ../backgrounds/bgem3_pl_kwmix900k_mrl1024 --n 100000
"""

from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import numpy as np

from bench import l2_normalize


def peak_rss_gb() -> float:
    # ru_maxrss is KiB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


class Stage:
    """Time a stage and report wall-clock plus peak RSS after it."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self.started
        print(f"  {self.name:28s} {elapsed:8.1f}s   peak RSS {peak_rss_gb():5.2f} GB")
        self.elapsed = elapsed
        return False


def load_chunks(chunks: Path, limit: int) -> np.ndarray:
    files = sorted(chunks.glob("chunk_*.npy"))
    if not files:
        raise SystemExit(f"no chunk_*.npy in {chunks}")
    out, total = [], 0
    for path in files:
        block = np.load(path)
        out.append(block)
        total += block.shape[0]
        if total >= limit:
            break
    return np.vstack(out)[:limit]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--background", type=Path, default=None)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--resolution", type=float, default=4.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import os

    # Keep BLAS to the simulated VPS core count too, not just torch.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(args.threads)

    print(f"n={args.n}  knn={args.knn}  r={args.resolution}  threads={args.threads}")
    print(f"chunks={args.chunks}")
    print(f"background={args.background}\n")

    total_started = time.perf_counter()

    with Stage("load fp16 chunks"):
        vectors = load_chunks(args.chunks, args.n)
        n, dim = vectors.shape

    with Stage("to float32 + L2"):
        vectors = l2_normalize(vectors.astype(np.float32))

    if args.background:
        with Stage("apply background"):
            mu = np.load(args.background / "mu_A.npy").astype(np.float32)
            whitening = np.load(args.background / "W_A.npy").astype(np.float32)
            vectors = l2_normalize((vectors - mu) @ whitening)

    with Stage("approximate kNN (pynndescent)"):
        from pynndescent import NNDescent

        index = NNDescent(
            vectors, metric="cosine", n_neighbors=args.knn + 1,
            random_state=args.seed, n_jobs=args.threads,
        )
        neighbours, distances = index.neighbor_graph

    with Stage("build graph"):
        import igraph as ig

        edges, weights = [], []
        for i in range(n):
            for j, dist in zip(neighbours[i], distances[i]):
                if int(j) == i:
                    continue
                a, b = (i, int(j)) if i < int(j) else (int(j), i)
                edges.append((a, b))
                weights.append(max(1.0 - float(dist), 1e-6))
        graph = ig.Graph(n=n, edges=edges)
        graph.es["weight"] = weights
        graph.simplify(combine_edges="max")

    with Stage("Leiden"):
        import leidenalg

        partition = leidenalg.find_partition(
            graph, leidenalg.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=args.resolution, seed=args.seed,
        )
        labels = np.asarray(partition.membership)

    total = time.perf_counter() - total_started
    sizes = np.bincount(labels)
    print(
        f"\n  TOTAL {total:.1f}s ({total/60:.1f} min) for {n} keywords, "
        f"peak RSS {peak_rss_gb():.2f} GB"
    )
    print(
        f"  clusters: {len(sizes)}   "
        f"largest {sizes.max()}   median {int(np.median(sizes))}   "
        f"singletons {(sizes == 1).sum()}"
    )
    print(f"  dim={dim}  edges={graph.ecount()}  mean degree={2*graph.ecount()/n:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
