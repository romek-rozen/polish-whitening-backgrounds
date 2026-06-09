"""ZCA whitening fit over chunked embeddings, with optional MRL slice.

Two streaming passes:

  1. μ  = E[x]                          — one pass, accumulate sums.
  2. Σ  = E[(x - μ)(x - μ)ᵀ]            — second pass, accumulate
     centred outer products into a (dim × dim) matrix.

Then ``W = U · diag(1 / √(S + ε)) · Uᵀ`` from ``SVD(Σ)`` with the
canonical ``ε = 1e-6``.  No GPU, no torch — just numpy + an outer
product per chunk.

Vectors are L2-renormalised row-wise on each pass: Qwen3 already
emits unit-norm vectors, but the chunks are stored as fp16, and the
renorm guards against the slight drift the round-trip introduces.

MRL refits: pass ``truncate_to=N`` to slice each chunk to the first
``N`` columns and renormalise *after* the slice.  This matches the
inference-time geometry of a Matryoshka-truncated embedding, so the
resulting (μ, Σ) reflect what the index actually stores.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def _l2_renorm(x: np.ndarray) -> np.ndarray:
    """Unit-L2 per row; zero rows stay zero (no NaN injection)."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n > 0, n, 1.0)
    return x / n


def fit(
    chunks_dir: Path, eps: float = 1e-6,
    truncate_to: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Two-pass μ / Σ + SVD over ``chunks_dir/chunk_*.npy``.

    Returns ``(W, mu, eigvals, diag)`` where:

    - ``W`` is the ZCA matrix, ``float32 (dim, dim)``
    - ``mu`` is the mean vector, ``float32 (dim,)``
    - ``eigvals`` are the post-SVD singular values, ``float32 (dim,)``
    - ``diag`` is a small dict of fit-time diagnostics suitable for
      stashing in a ``*.meta.json``.

    ``truncate_to=N`` enables an MRL slice; the returned ``dim`` is
    ``N`` and ``diag["mrl_truncated"]`` is set to ``True``.
    """
    chunk_paths = sorted(chunks_dir.glob("chunk_*.npy"))
    if not chunk_paths:
        raise SystemExit(f"no chunks in {chunks_dir}")

    head = np.load(chunk_paths[0], mmap_mode="r")
    native_dim = head.shape[1]
    if truncate_to and truncate_to > 0:
        if truncate_to > native_dim:
            raise SystemExit(
                f"--truncate-to {truncate_to} > native dim {native_dim}"
            )
        dim = truncate_to
        logger.info(
            "source: %d chunks, native_dim=%d, MRL slice → %d",
            len(chunk_paths), native_dim, dim,
        )
    else:
        dim = native_dim
        logger.info("source: %d chunks, dim=%d", len(chunk_paths), dim)

    def _load_sliced(cp: Path) -> np.ndarray:
        x = np.load(cp).astype(np.float32, copy=False)
        if dim < native_dim:
            x = x[:, :dim]
        return _l2_renorm(x)

    # Pass 1: mean.
    t0 = time.perf_counter()
    n_total = 0
    sum_vec = np.zeros(dim, dtype=np.float64)
    for cp in chunk_paths:
        x = _load_sliced(cp)
        sum_vec += x.sum(axis=0, dtype=np.float64)
        n_total += x.shape[0]
    mu = (sum_vec / n_total).astype(np.float32)
    logger.info("μ from N=%d in %.1fs", n_total, time.perf_counter() - t0)

    # Pass 2: covariance.
    t1 = time.perf_counter()
    cov = np.zeros((dim, dim), dtype=np.float64)
    for cp in chunk_paths:
        x = _load_sliced(cp)
        xc = (x - mu).astype(np.float64, copy=False)
        cov += xc.T @ xc
    cov /= (n_total - 1)
    logger.info("Σ %dx%d in %.1fs", dim, dim, time.perf_counter() - t1)

    # SVD + ZCA assembly.
    t2 = time.perf_counter()
    U, S, _ = np.linalg.svd(cov.astype(np.float64))
    inv = 1.0 / np.sqrt(S + eps)
    W = ((U * inv) @ U.T).astype(np.float32)
    eigvals = S.astype(np.float32)
    logger.info("SVD in %.1fs", time.perf_counter() - t2)

    rank_def = int(np.sum(eigvals < 1e-6))
    diag = {
        "n_total": int(n_total),
        "eps": eps,
        "dim": int(dim),
        "native_dim": int(native_dim),
        "mrl_truncated": bool(dim < native_dim),
        "rank_deficient_eigvals": rank_def,
        "rank_full_eigvals": int(dim - rank_def),
        "top_ev_ratio_pre": float(eigvals.max() / max(eigvals.mean(), 1e-12)),
        "eigvals_min": float(eigvals.min()),
        "eigvals_median": float(np.median(eigvals)),
        "eigvals_max": float(eigvals.max()),
        "eigvals_top5": [float(v) for v in np.sort(eigvals)[::-1][:5]],
        "fit_s": round(time.perf_counter() - t0, 3),
    }
    logger.info("ZCA: %s", diag)
    return W, mu, eigvals, diag


def write_meta(
    out_dir: Path, name: str, model: str, diag: dict,
    corpus_path: Path | None = None,
    cost_report_path: Path | None = None,
) -> None:
    """Persist ``<name>.meta.json`` next to the W / μ / eigvals files.

    Best-effort: corpus fingerprinting and cost-report attachment are
    wrapped in try/except so a missing or malformed sidecar never
    blocks the meta write.
    """
    meta: dict = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            f"ZCA whitening background for {model} on Polish text "
            "(no per-document character cap). Drop into siteFocus / any "
            "retrieval pipeline that L2-normalises Qwen3 embeddings."
        ),
        "language": "pl",
        "embedding_model": model,
        "embedding_dim": diag["dim"],
        "max_chars_per_doc": None,
        "embedding_endpoint": "https://openrouter.ai/api/v1/embeddings",
        "provider": "openrouter",
        "build_script": "scripts/fit_zca.py",
        "diagnostics": diag,
        "notes": [
            "Apply: x_white = (x - mu) @ W where x is unit-L2 in `embedding_dim` space.",
            "Embeddings were L2-renormalised row-wise before mean and covariance.",
            "ZCA = U · diag(1 / sqrt(S + eps)) · U^T where Σ = U S U^T.",
        ],
    }

    if corpus_path and corpus_path.is_file():
        # Fingerprint = sha256 over the sorted sha column.  Stable
        # across runs of build_corpus.py on the same seed/mix.
        try:
            shas = (
                pq.read_table(corpus_path, columns=["sha"])
                .column("sha").to_pylist()
            )
            meta["corpus_fingerprint_sha256"] = hashlib.sha256(
                "\n".join(sorted(shas)).encode()
            ).hexdigest()
            sources = (
                pq.read_table(corpus_path, columns=["source"])
                .column("source").to_pylist()
            )
            counts: dict[str, int] = {}
            for s in sources:
                counts[s] = counts.get(s, 0) + 1
            meta["sample_size_actual"] = counts
        except Exception as e:
            logger.warning("could not fingerprint corpus: %s", e)

    if cost_report_path and cost_report_path.is_file():
        try:
            cost = json.loads(cost_report_path.read_text())
            meta["embedding_cost"] = {
                "prompt_tokens": cost.get("prompt_tokens"),
                "total_tokens": cost.get("total_tokens"),
                "cost_usd": cost.get("cost_usd"),
                "n_calls": cost.get("n_calls"),
                "n_429": cost.get("n_429"),
            }
        except Exception:
            pass

    (out_dir / f"{name}.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )
