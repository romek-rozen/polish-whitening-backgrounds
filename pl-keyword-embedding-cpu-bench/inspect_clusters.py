"""Print the actual clusters a model produced, so the metrics can be eyeballed.

AMI tells you *how much* a model got wrong, never *what*. This dumps every
cluster with its keywords and their true groups, and flags the two failure
modes separately:

- **mixed cluster** — one cluster holds keywords from several true groups
  (lowers homogeneity: unrelated intents merged into one ad group),
- **split group** — one true group is spread over several clusters
  (lowers completeness: one intent fragmented into several ad groups).

Usage::

    python inspect_clusters.py --model bge_m3 --resolution 2.0
    python inspect_clusters.py --model embeddinggemma --resolution 4.0
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bench import MODELS, cluster_leiden, encode, load_dataset

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="bge_m3", choices=list(MODELS))
    parser.add_argument("--dataset", type=Path, default=ROOT / "test_dataset" / "keywords_pl.jsonl")
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--use-prompts", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)

    keywords, truth, group_names = load_dataset(args.dataset)
    spec = MODELS[args.model]
    encoded = encode(spec, keywords, args.batch_size, use_prompts=args.use_prompts)
    pred = cluster_leiden(encoded.vectors, args.knn, args.resolution, args.seed)

    members: dict[int, list[int]] = defaultdict(list)
    for row, cluster in enumerate(pred):
        members[int(cluster)].append(row)

    # Order clusters by size so the interesting big ones come first.
    order = sorted(members, key=lambda c: -len(members[c]))

    print(f"\nmodel={args.model}  r={args.resolution}  knn={args.knn}  "
          f"prompt={encoded.prompt_used}  -> {len(members)} clusters "
          f"(truth: {len(group_names)} groups)\n")

    mixed = 0
    for rank, cluster in enumerate(order, start=1):
        rows = members[cluster]
        counts = Counter(group_names[truth[r]] for r in rows)
        dominant, dominant_n = counts.most_common(1)[0]
        purity = dominant_n / len(rows)
        flag = ""
        if len(counts) > 1:
            mixed += 1
            flag = "  <== MIXED: " + ", ".join(
                f"{name}×{n}" for name, n in counts.most_common()
            )
        print(f"[{rank}] n={len(rows):>2}  purity={purity:.2f}  ~{dominant}{flag}")
        for row in rows:
            true_group = group_names[truth[row]]
            marker = "   " if true_group == dominant else " ! "
            print(f"   {marker}{keywords[row]}"
                  + (f"   (true: {true_group})" if true_group != dominant else ""))
        print()

    # Which true groups got fragmented across clusters?
    spread: dict[str, set[int]] = defaultdict(set)
    for row, cluster in enumerate(pred):
        spread[group_names[truth[row]]].add(int(cluster))
    split = {name: cl for name, cl in spread.items() if len(cl) > 1}

    print("-" * 70)
    print(f"mixed clusters (>1 true group): {mixed}/{len(members)}")
    if split:
        print(f"split groups (spread over >1 cluster): {len(split)}/{len(group_names)}")
        for name, clusters in sorted(split.items()):
            print(f"   {name}: {len(clusters)} clusters")
    else:
        print(f"split groups: 0/{len(group_names)} — every group stayed in one cluster")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
