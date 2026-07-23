"""Cache embeddings on disk so a keyword list is embedded once, not per script.

Every sweep here re-embedded the same keywords from scratch. On GPU that is ~25 s
per model and merely wasteful; on a CPU-only box it is ~7 minutes per model, so
a four-script comparison costs over an hour of recomputing identical vectors.

The cache key is the model plus a hash of the exact keyword list — content and
order, since every downstream array is indexed by row. Change one keyword and
you get a fresh entry rather than a silently mismatched one.

Vectors are stored **raw** (L2-normalised, no background applied). Whitening is
cheap (0.4 s for 100 k rows) and background choice varies per run, so caching the
un-whitened vectors keeps one entry usable for every background.

Cached runs report ``from_cache=True`` and ``throughput=None``: a timing taken
from a disk read is not an embedding benchmark, and must not be presented as one.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "pl-keyword-embedding-cpu-bench"))

from bench import MODELS, encode  # noqa: E402

logger = logging.getLogger("embed_cache")
DEFAULT_CACHE = ROOT / "work" / "embeddings"


@dataclass
class CachedEmbedding:
    vectors: np.ndarray          # L2-normalised, no background applied
    dim: int
    from_cache: bool
    seconds: float
    throughput: float | None     # None when served from cache


def corpus_fingerprint(keywords: list[str]) -> str:
    digest = hashlib.sha256()
    for keyword in keywords:
        digest.update(keyword.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def cache_path(cache_dir: Path, model_key: str, fingerprint: str) -> Path:
    return cache_dir / f"{model_key}__{fingerprint}.npy"


def encode_cached(
    model_key: str,
    keywords: list[str],
    *,
    batch_size: int = 64,
    device: str = "cuda",
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> CachedEmbedding:
    """Embed `keywords` with `model_key`, reusing a cached array when possible."""
    fingerprint = corpus_fingerprint(keywords)
    path = cache_path(cache_dir, model_key, fingerprint)

    if path.exists() and not refresh:
        started = time.perf_counter()
        vectors = np.load(path).astype(np.float64)
        if vectors.shape[0] != len(keywords):
            logger.warning(
                "%s: cached rows %d != %d — re-embedding",
                path.name, vectors.shape[0], len(keywords),
            )
        else:
            elapsed = time.perf_counter() - started
            logger.info("%s: %d vectors from cache in %.1fs",
                        model_key, len(keywords), elapsed)
            return CachedEmbedding(vectors, int(vectors.shape[1]), True, elapsed, None)

    started = time.perf_counter()
    encoded = encode(MODELS[model_key], keywords, batch_size, device=device)
    elapsed = time.perf_counter() - started

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Write via a temp name so an interrupted run cannot leave a short array
    # that a later run would trust. (.npy suffix required: np.save appends it.)
    tmp = path.with_name(f"{path.stem}.tmp.npy")
    np.save(tmp, encoded.vectors.astype(np.float32))
    tmp.replace(path)

    logger.info("%s: embedded %d in %.0fs (%.0f/s) on %s -> %s",
                model_key, len(keywords), elapsed, encoded.throughput, device, path.name)
    return CachedEmbedding(
        encoded.vectors, encoded.dim, False, elapsed, encoded.throughput
    )
