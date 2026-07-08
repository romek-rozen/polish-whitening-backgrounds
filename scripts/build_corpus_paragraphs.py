"""Split the document corpus into paragraph-level pieces for `_paragraphs_` fits.

Reads ``data/corpus.parquet`` (produced by ``build_corpus.py``) and
writes ``data/corpus_paragraphs.parquet`` where every row is one
blank-line paragraph produced by :mod:`scripts.lib.paragrapher` — the
finest structural unit above a sentence, sitting below `_chunks_` and
`_segments_` in the length hierarchy (see GOTCHAS.md §1).

Output schema::

    text          : str    # the paragraph
    doc_sha       : str    # sha of the parent document (from corpus.parquet)
    paragraph_idx : int32  # 0-based index of this paragraph within its parent doc
    source        : str    # wikipedia | fineweb | oasst (inherited)
    sha           : str    # sha of the paragraph text itself — embed_via_openrouter
                           # reads this column name unconditionally, same
                           # contract as build_corpus_chunks.py / _segments.py.

Same rationale as build_corpus_segments.py for shipping a parquet
rather than splitting on-the-fly in the embed step: one split parquet
feeds both Qwen3 models (shared tokenizer) *and* the two OpenAI
models, re-runs are no-ops, and the corpus we actually whitened
against stays on disk and auditable.

Usage::

    python scripts/build_corpus_paragraphs.py --model qwen/qwen3-embedding-4b
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

from lib.paragrapher import PARA_MAX_TOKENS, make_paragrapher, paragraph_text

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

logger = logging.getLogger("build_corpus_paragraphs")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def paragraph_corpus(
    corpus_path: Path, out_path: Path, model: str, para_max: int,
) -> dict:
    """Read *corpus_path*, split each doc into paragraphs, write *out_path*.

    Returns a stats dict (n_docs, n_paragraphs, mean_paragraphs_per_doc,
    min/max paragraph chars) for the caller to log.
    """
    if out_path.exists():
        logger.info("output already exists: %s — skipping", out_path)
        t = pq.read_table(out_path)
        return {
            "n_docs": len(set(t.column("doc_sha").to_pylist())),
            "n_paragraphs": t.num_rows,
            "skipped": True,
        }

    logger.info("reading %s", corpus_path)
    table = pq.read_table(corpus_path, columns=["text", "sha", "source"])
    texts = table.column("text").to_pylist()
    shas = table.column("sha").to_pylist()
    sources = table.column("source").to_pylist()
    n_docs = len(texts)
    del table

    logger.info("building paragrapher for %s (max=%d tokens, no overlap)",
                model, para_max)
    splitter = make_paragrapher(model, para_max)

    out_text: list[str] = []
    out_doc_sha: list[str] = []
    out_paragraph_idx: list[int] = []
    out_source: list[str] = []
    out_sha: list[str] = []

    for text, doc_sha, source in tqdm(
        zip(texts, shas, sources), total=n_docs, desc="paragraph",
    ):
        for idx, para in enumerate(paragraph_text(splitter, text)):
            out_text.append(para)
            out_doc_sha.append(doc_sha)
            out_paragraph_idx.append(idx)
            out_source.append(source)
            out_sha.append(_sha(para))

    n_paragraphs = len(out_text)
    logger.info("writing %s (%d paragraphs from %d docs)",
                out_path, n_paragraphs, n_docs)
    out_table = pa.table({
        "text": out_text,
        "doc_sha": out_doc_sha,
        "paragraph_idx": pa.array(out_paragraph_idx, type=pa.int32()),
        "source": out_source,
        "sha": out_sha,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, out_path)

    chars = [len(t) for t in out_text]
    return {
        "n_docs": n_docs,
        "n_paragraphs": n_paragraphs,
        "mean_paragraphs_per_doc": round(n_paragraphs / max(1, n_docs), 2),
        "min_paragraph_chars": min(chars) if chars else 0,
        "max_paragraph_chars": max(chars) if chars else 0,
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
        help="Output parquet path (default: data/corpus_paragraphs.parquet).",
    )
    ap.add_argument(
        "--model", default="qwen/qwen3-embedding-4b",
        help="OpenRouter model id (picks the tokenizer for the "
             "oversize-paragraph descent). Both Qwen3 Embedding sizes "
             "ship the same tokenizer.json — either is fine, and the "
             "resulting parquet feeds the OpenAI models too.",
    )
    ap.add_argument("--para-max", type=int, default=PARA_MAX_TOKENS)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.corpus.is_file():
        logger.error("corpus not found: %s — run build_corpus.py first",
                     args.corpus)
        return 2

    out = args.out or (DATA_DIR / "corpus_paragraphs.parquet")

    stats = paragraph_corpus(args.corpus, out, args.model, args.para_max)
    logger.info("stats: %s", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
