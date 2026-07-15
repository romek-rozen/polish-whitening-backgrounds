"""Dump the medical corpus to plain .txt files for easy human review.

The corpus lives in Parquet (columnar/binary — not openable in a text
editor).  This writes N documents out as individual UTF-8 .txt files
you can open in anything.

Usage::

    # 30 whole ChPL documents -> data/med_pl/preview_doc/000.txt ...
    python scripts/dump_med_txt.py

    # a different granularity / count / location
    python scripts/dump_med_txt.py --source paragraphs --n 50
    python scripts/dump_med_txt.py --source chunks --n 40 --out /tmp/chunks
    python scripts/dump_med_txt.py --all            # every document (careful: 13k files)
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

SOURCES = {
    "doc": DATA / "corpus_med_pl.parquet",
    "paragraphs": DATA / "corpus_med_pl_paragraphs.parquet",
    "chunks": DATA / "corpus_med_pl_chunks_300_50.parquet",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=SOURCES, default="doc",
                    help="Which granularity to dump (default: doc = whole ChPL).")
    ap.add_argument("--n", type=int, default=30, help="How many to dump (random sample).")
    ap.add_argument("--all", action="store_true", help="Dump every row (ignores --n).")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir (default: data/med_pl/preview_<source>/).")
    args = ap.parse_args()

    path = SOURCES[args.source]
    if not path.is_file():
        raise SystemExit(f"not found: {path} — build the medical corpus first")

    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    n = len(texts)
    if args.all:
        idx = range(n)
    else:
        k = min(args.n, n)
        idx = sorted(random.Random(args.seed).sample(range(n), k))

    out = args.out or (DATA / "med_pl" / f"preview_{args.source}")
    out.mkdir(parents=True, exist_ok=True)
    for j, i in enumerate(idx):
        (out / f"{j:04d}_row{i}.txt").write_text(texts[i], encoding="utf-8")

    print(f"wrote {len(list(idx))} {args.source} file(s) to {out}/")
    print(f"open e.g.:  less {out}/0000_*.txt   (or any text editor / file browser)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
