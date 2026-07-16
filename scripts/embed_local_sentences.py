"""Embed a sentence parquet with local vLLM, MRL-truncate, renormalize, and save fp16 chunks."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("embed_local_sentences")


def _l2(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding server returned a zero vector")
    return values / norms


def _embed(endpoint: str, model: str, texts: list[str], timeout: float) -> np.ndarray:
    response = requests.post(
        endpoint, json={"model": model, "input": texts}, timeout=timeout
    )
    response.raise_for_status()
    data = response.json().get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        raise RuntimeError(f"asked for {len(texts)} embeddings, got {len(data or [])}")
    ordered = sorted(data, key=lambda item: item["index"])
    return np.asarray([item["embedding"] for item in ordered], dtype=np.float64)


def embed_corpus(
    corpus: Path,
    out: Path,
    endpoint: str,
    model: str,
    mrl_dim: int,
    batch_size: int,
    chunk_size: int,
    timeout: float,
) -> tuple[int, float]:
    texts = pq.read_table(corpus, columns=["text"]).column("text").to_pylist()
    out.mkdir(parents=True, exist_ok=True)
    completed = sorted(out.glob("chunk_*.npy"))
    start_row = len(completed) * chunk_size
    started = time.perf_counter()
    for chunk_start in range(start_row, len(texts), chunk_size):
        chunk_texts = texts[chunk_start : chunk_start + chunk_size]
        rows: list[np.ndarray] = []
        for batch_start in range(0, len(chunk_texts), batch_size):
            native = _embed(
                endpoint,
                model,
                chunk_texts[batch_start : batch_start + batch_size],
                timeout,
            )
            if native.shape[1] < mrl_dim:
                raise ValueError(f"native dim {native.shape[1]} is below MRL dim {mrl_dim}")
            rows.append(_l2(native[:, :mrl_dim]))
        array = np.vstack(rows).astype(np.float16)
        chunk_index = chunk_start // chunk_size
        target = out / f"chunk_{chunk_index:04d}.npy"
        temporary = target.with_suffix(".tmp.npy")
        np.save(temporary, array)
        temporary.replace(target)
        logger.info("saved %s rows=%d total=%d/%d", target.name, len(array), chunk_start + len(array), len(texts))
    return len(texts), time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/corpus_sentences.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "data/chunks_qwen_qwen3-embedding-4b_sentences")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8002/v1/embeddings")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--mrl-dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    count, seconds = embed_corpus(
        args.corpus, args.out, args.endpoint, args.model, args.mrl_dim,
        args.batch_size, args.chunk_size, args.timeout,
    )
    logger.info("DONE rows=%d seconds=%.1f", count, seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
