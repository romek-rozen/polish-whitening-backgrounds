"""Validate raw and whitened embeddings on real website sentences."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ENTROPY_SRC = Path("/home/spark001/Workspaces/entropy_001/src")
sys.path.insert(0, str(ENTROPY_SRC))

from entropy_lab.features.chunk.sentences import split_sentences  # noqa: E402
from embed_local_sentences import _embed, _l2  # noqa: E402


def load_sentences(database: Path, domain: str) -> list[str]:
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            """
            SELECT extraction.main_content
            FROM extraction JOIN page USING (url_hash)
            WHERE page.domain = ? AND extraction.main_content IS NOT NULL
            ORDER BY page.url
            """,
            (domain,),
        ).fetchall()
    return [sentence for (text,) in rows for sentence in split_sentences(text)]


def embed(texts: list[str], endpoint: str, model: str, batch_size: int) -> np.ndarray:
    batches = []
    for start in range(0, len(texts), batch_size):
        native = _embed(endpoint, model, texts[start : start + batch_size], 120.0)
        batches.append(_l2(native[:, :1024]))
    return np.vstack(batches)


def transform(vectors: np.ndarray, background: Path) -> np.ndarray:
    mean = np.load(background / "mu_A.npy").astype(np.float64)
    matrix = np.load(background / "W_A.npy").astype(np.float64)
    return (vectors.astype(np.float64) - mean) @ matrix


def load_chunks(directory: Path) -> np.ndarray:
    paths = sorted(directory.glob("chunk_*.npy"))
    return _l2(np.concatenate([np.load(path).astype(np.float64) for path in paths]))


def top_eigenvalue_ratio(vectors: np.ndarray) -> float:
    covariance = np.cov(vectors.astype(np.float64), rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return float(eigenvalues[-1] / eigenvalues.mean())


def mean_absolute_cosine(vectors: np.ndarray, pairs: int = 100_000) -> float:
    rng = np.random.default_rng(42)
    left = rng.integers(0, len(vectors), size=pairs)
    right = rng.integers(0, len(vectors) - 1, size=pairs)
    right += right >= left
    return float(np.abs(np.sum(vectors[left] * vectors[right], axis=1)).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("/home/spark001/Workspaces/entropy_001/data/cache/pages.sqlite"),
    )
    parser.add_argument("--domain", default="rozenberger.com")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8002/v1/embeddings")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--background",
        type=Path,
        default=ROOT / "backgrounds/qwen3_4b_pl_mixed50k_sentences_mrl1024",
    )
    parser.add_argument(
        "--chunks-background",
        type=Path,
        default=ROOT / "backgrounds/qwen3_4b_pl_mixed50k_chunks_mrl1024",
    )
    parser.add_argument(
        "--heldout-vectors",
        type=Path,
        default=ROOT / "data/chunks_qwen_qwen3-embedding-4b_sentences_heldout",
    )
    args = parser.parse_args()

    sentences = load_sentences(args.database, args.domain)
    raw = embed(sentences, args.endpoint, args.model, args.batch_size)
    linear = {
        "raw": raw,
        "chunks_background": transform(raw, args.chunks_background),
        "sentences_background": transform(raw, args.background),
    }
    heldout = load_chunks(args.heldout_vectors)
    heldout_linear = {
        "raw": heldout,
        "chunks_background": transform(heldout, args.chunks_background),
        "sentences_background": transform(heldout, args.background),
    }
    result = {
        "domain": args.domain,
        "sentence_count": len(sentences),
        "pair_count": 100_000,
        "heldout_sentence_count": len(heldout),
        "website_metrics": {
            name: {
                "top_eigenvalue_over_mean": top_eigenvalue_ratio(vectors),
                "mean_absolute_cosine": mean_absolute_cosine(_l2(vectors)),
            }
            for name, vectors in linear.items()
        },
        "heldout_diagnostics": {
            name: {
                "top_eigenvalue_over_mean_linear": top_eigenvalue_ratio(vectors),
                "top_eigenvalue_over_mean_post_l2": top_eigenvalue_ratio(_l2(vectors)),
            }
            for name, vectors in heldout_linear.items()
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
