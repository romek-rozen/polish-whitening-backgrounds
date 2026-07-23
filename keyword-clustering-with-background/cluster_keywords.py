"""Cluster a real keyword export and dump the groups for eyeballing.

No ground truth here on purpose. A keyword→URL export reflects what a site
ranked for at some point in the past; using those URLs as intent labels would
score models against stale, noisy annotations. What is left — and what actually
matters — is whether the produced groups look like usable ad groups.

So this reports what can be measured without labels:

- **coherence** — mean cosine of each keyword to its cluster centroid. Note it
  is *not* comparable between raw and whitened space: whitening shrinks all
  cosines by construction, so a lower number there means nothing on its own.
- **size distribution** — a good grouping has few singletons and no single
  bucket swallowing everything.
- **the clusters themselves** — printed, because that is the only honest test.

Usage::

    python cluster_real_keywords.py --csv work/keywords_url_2026-07-23.csv \\
        --background ../backgrounds/bgem3_pl_kwmix900k_mrl1024 \\
        --resolution 4.0 --show 25 > work/clusters_real.txt
"""

from __future__ import annotations

import argparse
import csv
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pl-keyword-embedding-cpu-bench"))

from embed_cache import encode_cached
from bench import MODELS, apply_background, encode, l2_normalize

ROOT = Path(__file__).resolve().parent


def clean(keyword: str) -> str:
    """Lowercase, collapse whitespace, drop invisible formatting characters.

    Real exports carry the odd zero-width joiner or RTL mark (category Cf/Cc).
    They survive `.strip()`, tokenise as junk, and — worse — can end up as a
    cluster's display label, where they render as an unexplained blank.
    """
    keyword = "".join(
        ch for ch in keyword if unicodedata.category(ch) not in ("Cf", "Cc")
    )
    return " ".join(keyword.split()).lower()


def load_keywords(path: Path) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        field = "KEYWORD" if "KEYWORD" in (reader.fieldnames or []) else None
        for row in reader:
            keyword = clean((row[field] if field else next(iter(row.values()), "")) or "")
            if keyword and keyword not in seen:
                seen.add(keyword)
                out.append(keyword)
    return out


def cluster(vectors: np.ndarray, knn: int, resolution: float, seed: int) -> np.ndarray:
    import igraph as ig
    import leidenalg
    from pynndescent import NNDescent

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
    partition = leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution, seed=seed,
    )
    return np.asarray(partition.membership)


def cluster_labels(keywords: list[str], vectors: np.ndarray,
                   labels: np.ndarray) -> dict[int, str]:
    """Name each cluster with its most central keyword.

    The keyword closest to the centroid is the least idiosyncratic member, so it
    reads as the group's theme — far more usable than a bare cluster number when
    the output lands in a spreadsheet.
    """
    members: dict[int, list[int]] = defaultdict(list)
    for row, cluster_id in enumerate(labels):
        members[int(cluster_id)].append(row)

    names: dict[int, str] = {}
    for cluster_id, rows in members.items():
        block = vectors[rows]
        centroid = block.mean(axis=0)
        norm = np.linalg.norm(centroid)
        best = rows[0] if norm == 0 else rows[int(np.argmax(block @ (centroid / norm)))]
        names[cluster_id] = keywords[best]
    return names


def write_csv(path: Path, keywords: list[str], spaces: dict[str, np.ndarray],
              labellings: dict[str, np.ndarray]) -> None:
    """One row per keyword, one column pair per space, biggest groups first."""
    primary = list(spaces)[-1]  # the whitened space when a background was given
    names = {s: cluster_labels(keywords, spaces[s], labellings[s]) for s in spaces}
    sizes = {s: Counter(labellings[s].tolist()) for s in spaces}

    fields = ["keyword"]
    for space in spaces:
        tag = "bg" if space.startswith("bg:") else "raw"
        fields += [f"{tag}_cluster", f"{tag}_label", f"{tag}_size"]

    rows = []
    for i, keyword in enumerate(keywords):
        row = {"keyword": keyword}
        for space in spaces:
            tag = "bg" if space.startswith("bg:") else "raw"
            cluster_id = int(labellings[space][i])
            row[f"{tag}_cluster"] = cluster_id
            row[f"{tag}_label"] = names[space][cluster_id]
            row[f"{tag}_size"] = sizes[space][cluster_id]
        rows.append(row)

    ptag = "bg" if primary.startswith("bg:") else "raw"
    rows.sort(key=lambda r: (-r[f"{ptag}_size"], r[f"{ptag}_cluster"], r["keyword"]))

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}  ({len(rows)} rows, {len(fields)} columns)")


def report(keywords: list[str], vectors: np.ndarray, labels: np.ndarray,
           title: str, show: int, members_shown: int) -> None:
    members: dict[int, list[int]] = defaultdict(list)
    for row, cluster_id in enumerate(labels):
        members[int(cluster_id)].append(row)

    sizes = np.array([len(v) for v in members.values()])
    coherences = []
    for rows in members.values():
        block = vectors[rows]
        centroid = block.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            coherences.append(float((block @ (centroid / norm)).mean()) * len(rows))
    coherence = sum(coherences) / len(keywords)

    print(f"\n{'='*78}\n{title}\n{'='*78}")
    print(f"clusters: {len(members)}   singletons: {int((sizes == 1).sum())}   "
          f"largest: {int(sizes.max())}   median: {int(np.median(sizes))}")
    print(f"size-weighted coherence: {coherence:.3f} "
          f"(not comparable across spaces — whitening shrinks all cosines)\n")

    order = sorted(members, key=lambda c: -len(members[c]))[:show]
    for rank, cluster_id in enumerate(order, start=1):
        rows = members[cluster_id]
        print(f"[{rank}] n={len(rows)}")
        for row in rows[:members_shown]:
            print(f"      {keywords[row]}")
        if len(rows) > members_shown:
            print(f"      … +{len(rows)-members_shown} more")
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--model", default="bge_m3", choices=list(MODELS))
    ap.add_argument("--background", type=Path, default=None)
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--resolution", type=float, default=4.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--device", default="cuda",
        help="Defaults to cuda: this script inspects cluster quality, it does "
             "not measure VPS throughput, and the vectors are identical either "
             "way. bench.py is the CPU-only one — keep it that way.",
    )
    ap.add_argument("--show", type=int, default=20, help="clusters to print")
    ap.add_argument("--members", type=int, default=12, help="keywords per cluster")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Write one row per keyword with its cluster in each space.")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="Ignore cached vectors and re-embed.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    keywords = load_keywords(args.csv)
    print(f"{len(keywords)} unique keywords from {args.csv.name}")

    started = time.perf_counter()
    encoded = encode_cached(args.model, keywords, batch_size=args.batch_size,
                            device=args.device, refresh=args.refresh_cache)
    print(f"encoded on {args.device} in {time.perf_counter()-started:.0f}s "
          f"({encoded.throughput:.0f} kw/s, dim={encoded.dim})")

    spaces: dict[str, np.ndarray] = {"raw+L2": l2_normalize(encoded.vectors)}
    if args.background:
        spaces[f"bg:{args.background.name}"] = apply_background(
            encoded.vectors, args.background)

    labellings: dict[str, np.ndarray] = {}
    for space, vectors in spaces.items():
        labels = cluster(vectors, args.knn, args.resolution, args.seed)
        labellings[space] = labels
        title = (f"RAW + L2   (model={args.model}, r={args.resolution})"
                 if space == "raw+L2"
                 else f"BACKGROUND {args.background.name}   (r={args.resolution})")
        report(keywords, vectors, labels, title, args.show, args.members)

    if args.out_csv:
        write_csv(args.out_csv, keywords, spaces, labellings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
