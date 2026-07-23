"""Embed a corpus parquet locally with sentence-transformers, save fp16 chunks.

Sibling of ``embed_via_openrouter.py`` (paid API) and
``embed_local_sentences.py`` (local vLLM HTTP endpoint). This one loads the
model in-process, so it needs no server at all — which is what makes it the
path for models nobody serves as an embedding API, such as ``BAAI/bge-m3``.

Output matches what ``fit_zca.py`` expects: ``chunk_XXXX.npy`` files of
row-L2-normalised fp16 vectors, in corpus row order.

Resumable: completed chunks are skipped on re-run, so an interrupted job
continues where it stopped.

Usage::

    python scripts/embed_local_st.py \\
        --corpus data/corpus_keywords_580k.parquet \\
        --model ~/models/bge-m3 \\
        --out data/kw580k_corpus/chunks_baai_bge-m3 \\
        --device cuda --batch-size 256
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("embed_local_st")


def l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("model returned a zero vector")
    return values / norms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True, help="Local model dir.")
    ap.add_argument("--out", type=Path, required=True, help="Dir for chunk_*.npy.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--chunk-size", type=int, default=20_000,
        help="Rows per chunk_*.npy file (resume granularity).",
    )
    ap.add_argument(
        "--prompt-name", default=None,
        help="sentence-transformers prompt to apply. Default none: bge-m3 and "
             "Qwen3-Embedding need no instruction, and a task prompt measurably "
             "hurt short phrases in pl-keyword-embedding-cpu-bench.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    texts = pq.read_table(args.corpus, columns=["text"]).column("text").to_pylist()
    args.out.mkdir(parents=True, exist_ok=True)
    total_chunks = (len(texts) + args.chunk_size - 1) // args.chunk_size
    logger.info(
        "corpus=%s rows=%d -> %d chunks of %d",
        args.corpus, len(texts), total_chunks, args.chunk_size,
    )

    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    model = SentenceTransformer(str(args.model), device=args.device)
    logger.info("loaded %s on %s in %.1fs", args.model.name, args.device, time.perf_counter() - started)

    kwargs: dict = {"batch_size": args.batch_size, "show_progress_bar": False}
    if args.prompt_name:
        available = set(getattr(model, "prompts", {}) or {})
        if args.prompt_name not in available:
            raise SystemExit(
                f"prompt {args.prompt_name!r} not defined by this model; "
                f"available: {sorted(available) or 'none'}"
            )
        kwargs["prompt_name"] = args.prompt_name

    embed_started = time.perf_counter()
    done_rows = 0
    dim = None
    for index in range(total_chunks):
        target = args.out / f"chunk_{index:04d}.npy"
        lo = index * args.chunk_size
        hi = min(lo + args.chunk_size, len(texts))
        if target.exists():
            existing = np.load(target, mmap_mode="r")
            if existing.shape[0] == hi - lo:
                dim = dim or int(existing.shape[1])
                done_rows += hi - lo
                continue
            logger.warning("%s has %d rows, expected %d — redoing",
                           target.name, existing.shape[0], hi - lo)

        chunk_started = time.perf_counter()
        vectors = np.asarray(model.encode(texts[lo:hi], **kwargs), dtype=np.float32)
        vectors = l2_normalize(vectors).astype(np.float16)
        # Write to a temp name first so an interrupted write cannot leave a
        # truncated chunk that a later resume would trust. The temp name must
        # itself end in .npy — np.save silently appends .npy otherwise, and the
        # rename would then look for a file that was never created.
        tmp = target.with_name(f"{target.stem}.tmp.npy")
        np.save(tmp, vectors)
        tmp.replace(target)

        dim = int(vectors.shape[1])
        done_rows += hi - lo
        elapsed = time.perf_counter() - embed_started
        rate = done_rows / elapsed if elapsed else 0.0
        remaining = (len(texts) - done_rows) / rate if rate else float("nan")
        logger.info(
            "chunk %d/%d rows=%d dim=%d  %.1fs  (%.0f rows/s, ~%.1f min left)",
            index + 1, total_chunks, hi - lo, dim,
            time.perf_counter() - chunk_started, rate, remaining / 60,
        )

    total_s = time.perf_counter() - embed_started
    manifest = {
        "corpus": str(args.corpus),
        "model": str(args.model),
        "device": args.device,
        "prompt_name": args.prompt_name,
        "n_rows": len(texts),
        "dim": dim,
        "chunk_size": args.chunk_size,
        "n_chunks": total_chunks,
        "embed_seconds": round(total_s, 1),
        "rows_per_second": round(len(texts) / total_s, 1) if total_s else None,
    }
    (args.out.parent / f"manifest_{args.out.name}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info("DONE %d rows, dim=%d, %.1f min", len(texts), dim or -1, total_s / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
