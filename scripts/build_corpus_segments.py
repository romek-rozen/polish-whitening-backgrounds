"""Segment the document corpus into section-level pieces for `_segments_` fits.

Reads ``data/corpus.parquet`` (produced by ``build_corpus.py``) and
writes ``data/corpus_segments_<size>.parquet`` where every row is one
section-level segment produced by :mod:`scripts.lib.segmenter` —
the unit used for internal-linking retrieval (segment→segment, with
target docs represented by their own segments).

Output schema::

    text        : str    # the segment
    doc_sha     : str    # sha of the parent document (from corpus.parquet)
    segment_idx : int32  # 0-based index of this segment within its parent doc
    source      : str    # wikipedia | fineweb | oasst (inherited)
    sha         : str    # sha of the segment text itself — embed_via_openrouter
                         # reads this column name unconditionally, same
                         # contract as build_corpus_chunks.py.

Same rationale as build_corpus_chunks.py for shipping a parquet rather
than segmenting on-the-fly in the embed step: one segmented parquet
feeds both Qwen3 models (shared tokenizer), re-runs are no-ops, and
the corpus we actually whitened against stays on disk and auditable.

Usage::

    python scripts/build_corpus_segments.py --model qwen/qwen3-embedding-4b
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from lib.segmenter import SEGMENT_SIZE_TOKENS, make_segmenter, segment_text

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

logger = logging.getLogger("build_corpus_segments")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def segment_corpus(
    corpus_path: Path, out_path: Path, model: str, segment_size: int,
) -> dict:
    """Read *corpus_path*, segment each doc, write *out_path*.

    Returns a stats dict (n_docs, n_segments, mean_segments_per_doc,
    min/max segment chars) for the caller to log.
    """
    if out_path.exists():
        logger.info("output already exists: %s — skipping", out_path)
        t = pq.read_table(out_path)
        return {
            "n_docs": len(set(t.column("doc_sha").to_pylist())),
            "n_segments": t.num_rows,
            "skipped": True,
        }

    logger.info("reading %s", corpus_path)
    table = pq.read_table(corpus_path, columns=["text", "sha", "source"])
    texts = table.column("text").to_pylist()
    shas = table.column("sha").to_pylist()
    sources = table.column("source").to_pylist()
    n_docs = len(texts)
    del table

    logger.info("building segmenter for %s (size=%d tokens, no overlap)",
                model, segment_size)
    splitter = make_segmenter(model, segment_size)

    out_text: list[str] = []
    out_doc_sha: list[str] = []
    out_segment_idx: list[int] = []
    out_source: list[str] = []
    out_sha: list[str] = []

    for text, doc_sha, source in tqdm(
        zip(texts, shas, sources), total=n_docs, desc="segment",
    ):
        for idx, seg in enumerate(segment_text(splitter, text)):
            out_text.append(seg)
            out_doc_sha.append(doc_sha)
            out_segment_idx.append(idx)
            out_source.append(source)
            out_sha.append(_sha(seg))

    n_segments = len(out_text)
    logger.info("writing %s (%d segments from %d docs)",
                out_path, n_segments, n_docs)
    out_table = pa.table({
        "text": out_text,
        "doc_sha": out_doc_sha,
        "segment_idx": pa.array(out_segment_idx, type=pa.int32()),
        "source": out_source,
        "sha": out_sha,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, out_path)

    chars = [len(t) for t in out_text]
    return {
        "n_docs": n_docs,
        "n_segments": n_segments,
        "mean_segments_per_doc": round(n_segments / max(1, n_docs), 2),
        "min_segment_chars": min(chars) if chars else 0,
        "max_segment_chars": max(chars) if chars else 0,
        "skipped": False,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--corpus", type=Path, default=DATA_DIR / "corpus.parquet",
        help="Input doc-level corpus (default: data/corpus.parquet).",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Output parquet path (default: "
             "data/corpus_segments_<size>.parquet).",
    )
    ap.add_argument(
        "--model", default="qwen/qwen3-embedding-4b",
        help="OpenRouter model id (picks the tokenizer). Both Qwen3 "
             "Embedding sizes ship the same tokenizer.json — either "
             "is fine.",
    )
    ap.add_argument("--segment-size", type=int, default=SEGMENT_SIZE_TOKENS)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.corpus.is_file():
        logger.error("corpus not found: %s — run build_corpus.py first",
                     args.corpus)
        return 2

    out = args.out or (DATA_DIR / f"corpus_segments_{args.segment_size}.parquet")

    stats = segment_corpus(args.corpus, out, args.model, args.segment_size)
    logger.info("stats: %s", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
