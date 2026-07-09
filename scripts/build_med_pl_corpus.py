"""Assemble the Polish medical corpus from its per-source parquets.

Merges whatever ``build_med_pl_corpus_*.py`` have produced under
``data/med_pl/`` into a single doc-level corpus
``data/corpus_med_pl.parquet`` with the SAME schema as the general
corpus (``build_corpus.py``): ``{source, text, sha, n_chars}``.  That
lets the medical corpus flow through the existing derived-corpus and
fit pipeline unchanged — ``build_corpus_{chunks,segments,paragraphs,
keywords}.py`` and the ``run_*.sh`` orchestrators — just point them at
this parquet and use the corpus tag ``med_pl`` in the background name.

Sources (each a ``data/med_pl/<name>.parquet`` written by its builder):
  - ``chpl``  → ``chpl_rpl``  long-form drug characteristics (pypdf)
  - ``pes``   → ``pes_cem``   short exam questions (HF)
  - ``chpl_ocr`` → ``chpl_rpl_ocr``  scanned ChPLs recovered via OCR
    (optional; only if that builder has been run)

De-dup is global by ``sha`` (a ChPL and its OCR twin, or a question
repeated across editions, collapse to one).  Row order is shuffled
with a fixed seed so no single source dominates any contiguous span
(the fit is order-agnostic, but downstream chunk parquets inherit row
order).

Usage::

    python scripts/build_med_pl_corpus.py                  # all sources found
    python scripts/build_med_pl_corpus.py --sources chpl pes
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MED_DIR = DATA_DIR / "med_pl"

SEED = 42
DEFAULT_SOURCES = ("chpl", "pes", "chpl_ocr")

logger = logging.getLogger("build_med_pl_corpus")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def build(sources: list[str], out_path: Path, min_chars: int) -> dict:
    rows_text: list[str] = []
    rows_source: list[str] = []
    seen: set[str] = set()
    per_source: dict[str, int] = {}
    missing: list[str] = []

    for name in sources:
        p = MED_DIR / f"{name}.parquet"
        if not p.is_file():
            missing.append(name)
            continue
        t = pq.read_table(p, columns=["source", "text"])
        srcs = t.column("source").to_pylist()
        txts = t.column("text").to_pylist()
        kept = 0
        for src, txt in zip(srcs, txts):
            if not txt or len(txt) < min_chars:
                continue
            sha = _sha(txt)
            if sha in seen:
                continue
            seen.add(sha)
            rows_text.append(txt)
            rows_source.append(src)
            kept += 1
        per_source[name] = kept
        logger.info("source %-9s (%s): kept %d", name, p.name, kept)

    if not rows_text:
        raise SystemExit(
            f"no rows assembled (looked in {MED_DIR} for {sources}). "
            "Run the build_med_pl_corpus_*.py builders first."
        )

    order = list(range(len(rows_text)))
    random.Random(SEED).shuffle(order)
    rows_text = [rows_text[i] for i in order]
    rows_source = [rows_source[i] for i in order]

    table = pa.table({
        "source": rows_source,
        "text": rows_text,
        "sha": [_sha(t) for t in rows_text],
        "n_chars": pa.array([len(t) for t in rows_text], type=pa.int64()),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)

    chars = sorted(len(t) for t in rows_text)
    return {
        "total_docs": len(rows_text),
        "per_source": per_source,
        "missing_sources": missing,
        "min_chars": chars[0],
        "median_chars": chars[len(chars)//2],
        "max_chars": chars[-1],
        "out": str(out_path),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", nargs="*", default=list(DEFAULT_SOURCES),
                    help=f"Per-source names under data/med_pl/ "
                         f"(default: {' '.join(DEFAULT_SOURCES)}).")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "corpus_med_pl.parquet")
    ap.add_argument("--min-chars", type=int, default=200,
                    help="Global floor; drop rows shorter than this.")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    stats = build(args.sources, args.out, args.min_chars)
    logger.info("DONE: %s", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
