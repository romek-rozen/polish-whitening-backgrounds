"""Chunk the v2 document corpus into paragraph-level pieces for v3.

Reads ``data/corpus.parquet`` (produced by ``build_corpus.py``) and
writes ``data/corpus_chunks_<size>_<overlap>.parquet`` where every
row is one sentence-aware chunk produced by
:func:`scripts.lib.chunker.make_splitter`.

Output schema::

    text       : str    # the chunk
    doc_sha    : str    # sha of the parent document (from corpus.parquet)
    chunk_idx  : int32  # 0-based index of this chunk within its parent doc
    source     : str    # wikipedia | fineweb | oasst (inherited)
    sha        : str    # sha of the chunk text itself — embed_via_openrouter
                        # uses this column name unconditionally; we keep it
                        # so the downstream pipeline doesn't need to know
                        # whether it's working off docs or chunks.

Why ship chunks as a parquet rather than chunk on-the-fly inside the
embed step:

- The chunker depends on the model tokenizer, but the embed-time
  loop is otherwise model-agnostic — keeping the split out of the
  embed step means we can re-use the same chunked parquet to embed
  with multiple models (Qwen3 4B and 8B share the same tokenizer
  byte-for-byte, so one chunked parquet feeds both).
- Idempotent: re-running build_corpus_chunks.py is a no-op if the
  output exists.  Skipping the chunking phase on a re-run saves
  ~5-10 minutes on a 50k-doc corpus.
- Auditable: the corpus we actually whitened against is on disk and
  fingerprintable, not regenerated implicitly inside the embed loop.

Usage::

    python scripts/build_corpus_chunks.py \\
        --model qwen/qwen3-embedding-4b \\
        --chunk-size 512 --chunk-overlap 64
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

from lib.chunker import make_splitter

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

logger = logging.getLogger("build_corpus_chunks")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_corpus(
    corpus_path: Path, out_path: Path, model: str,
    chunk_size: int, chunk_overlap: int,
) -> dict:
    """Read *corpus_path*, chunk each doc, write *out_path*.

    Returns a stats dict (n_docs, n_chunks, mean_chunks_per_doc,
    min/max chunk_chars) for the caller to log.
    """
    if out_path.exists():
        logger.info("output already exists: %s — skipping", out_path)
        # Still load it to report stats consistently.
        t = pq.read_table(out_path)
        return {
            "n_docs": len(set(t.column("doc_sha").to_pylist())),
            "n_chunks": t.num_rows,
            "skipped": True,
        }

    logger.info("reading %s", corpus_path)
    table = pq.read_table(corpus_path, columns=["text", "sha", "source"])
    texts = table.column("text").to_pylist()
    shas = table.column("sha").to_pylist()
    sources = table.column("source").to_pylist()
    n_docs = len(texts)
    del table

    logger.info("building splitter for %s (size=%d overlap=%d tokens)",
                model, chunk_size, chunk_overlap)
    splitter = make_splitter(model, chunk_size, chunk_overlap)

    out_text: list[str] = []
    out_doc_sha: list[str] = []
    out_chunk_idx: list[int] = []
    out_source: list[str] = []
    out_sha: list[str] = []

    for text, doc_sha, source in tqdm(
        zip(texts, shas, sources), total=n_docs, desc="chunk",
    ):
        for idx, chunk in enumerate(splitter.split_text(text)):
            out_text.append(chunk)
            out_doc_sha.append(doc_sha)
            out_chunk_idx.append(idx)
            out_source.append(source)
            out_sha.append(_sha(chunk))

    n_chunks = len(out_text)
    logger.info("writing %s (%d chunks from %d docs)", out_path, n_chunks, n_docs)
    out_table = pa.table({
        "text": out_text,
        "doc_sha": out_doc_sha,
        "chunk_idx": pa.array(out_chunk_idx, type=pa.int32()),
        "source": out_source,
        "sha": out_sha,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, out_path)

    chars = [len(t) for t in out_text]
    return {
        "n_docs": n_docs,
        "n_chunks": n_chunks,
        "mean_chunks_per_doc": round(n_chunks / max(1, n_docs), 2),
        "min_chunk_chars": min(chars) if chars else 0,
        "max_chunk_chars": max(chars) if chars else 0,
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
             "data/corpus_chunks_<size>_<overlap>.parquet).",
    )
    ap.add_argument(
        "--model", default="qwen/qwen3-embedding-4b",
        help="OpenRouter model id (picks the tokenizer). Both Qwen3 "
             "Embedding sizes ship the same tokenizer.json — either "
             "is fine.",
    )
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--chunk-overlap", type=int, default=64)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.corpus.is_file():
        logger.error("corpus not found: %s — run build_corpus.py first",
                     args.corpus)
        return 2

    out = args.out or (
        DATA_DIR / f"corpus_chunks_{args.chunk_size}_{args.chunk_overlap}.parquet"
    )

    stats = chunk_corpus(
        args.corpus, out, args.model,
        args.chunk_size, args.chunk_overlap,
    )
    logger.info("stats: %s", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
