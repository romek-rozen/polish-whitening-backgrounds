"""Fit a ZCA whitening transform from chunked embeddings.

Reads ``data/chunks_<model_slug>/chunk_*.npy``, fits ZCA via two streaming
passes (mean, covariance), then SVD, and writes the canonical artefacts
to ``backgrounds/<name>/``:

    W_A.npy        # (dim, dim) float32
    mu_A.npy       # (dim,)    float32
    eigvals_A.npy  # (dim,)    float32
    <name>.meta.json

Embeddings are L2-renormalised row-wise before mean/covariance — this
matches what the publisher pipeline does at apply time (Qwen3 already
emits unit vectors, but renorm guards against float-drift after fp16
storage).

Usage::

    python scripts/fit_zca.py \\
        --chunks data/chunks_qwen_qwen3-embedding-8b \\
        --name polish_mixed_50k_v1_qwen3-8b-nocap \\
        --model qwen/qwen3-embedding-8b \\
        --corpus data/corpus.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
BG_ROOT = REPO_ROOT / "backgrounds"
DATA_DIR = REPO_ROOT / "data"

logger = logging.getLogger("fit_zca")


def _l2_renorm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n > 0, n, 1.0)
    return x / n


def fit(chunks_dir: Path, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    chunk_paths = sorted(chunks_dir.glob("chunk_*.npy"))
    if not chunk_paths:
        raise SystemExit(f"no chunks in {chunks_dir}")

    head = np.load(chunk_paths[0], mmap_mode="r")
    dim = head.shape[1]
    logger.info("source: %d chunks, dim=%d", len(chunk_paths), dim)

    t0 = time.perf_counter()
    n_total = 0
    sum_vec = np.zeros(dim, dtype=np.float64)
    for cp in chunk_paths:
        x = np.load(cp).astype(np.float32, copy=False)
        x = _l2_renorm(x)
        sum_vec += x.sum(axis=0, dtype=np.float64)
        n_total += x.shape[0]
    mu = (sum_vec / n_total).astype(np.float32)
    logger.info("μ from N=%d in %.1fs", n_total, time.perf_counter() - t0)

    t1 = time.perf_counter()
    cov = np.zeros((dim, dim), dtype=np.float64)
    for cp in chunk_paths:
        x = np.load(cp).astype(np.float32, copy=False)
        x = _l2_renorm(x)
        xc = (x - mu).astype(np.float64, copy=False)
        cov += xc.T @ xc
    cov /= (n_total - 1)
    logger.info("Σ %dx%d in %.1fs", dim, dim, time.perf_counter() - t1)

    t2 = time.perf_counter()
    U, S, _ = np.linalg.svd(cov.astype(np.float64))
    inv = 1.0 / np.sqrt(S + eps)
    W = (U * inv) @ U.T
    W = W.astype(np.float32)
    eigvals = S.astype(np.float32)
    logger.info("SVD in %.1fs", time.perf_counter() - t2)

    rank_def = int(np.sum(eigvals < 1e-6))
    diag = {
        "n_total": int(n_total),
        "eps": eps,
        "dim": int(dim),
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


def write_meta(out_dir: Path, name: str, model: str, diag: dict,
               corpus_path: Path | None, cost_report_path: Path | None) -> None:
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
        # Corpus fingerprint — sha of sorted sha column. Cross-comparable
        # across backgrounds fitted on the same parquet.
        try:
            shas = pq.read_table(corpus_path, columns=["sha"])\
                .column("sha").to_pylist()
            meta["corpus_fingerprint_sha256"] = hashlib.sha256(
                "\n".join(sorted(shas)).encode()).hexdigest()
            sources = pq.read_table(corpus_path, columns=["source"])\
                .column("source").to_pylist()
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chunks", type=Path, required=True,
                    help="Dir with chunk_*.npy files (output of embed_via_openrouter.py).")
    ap.add_argument("--name", required=True,
                    help="Background name; output goes to backgrounds/<name>/.")
    ap.add_argument("--model", required=True,
                    help="OpenRouter model id used to produce the chunks (for meta).")
    ap.add_argument("--corpus", type=Path, default=DATA_DIR / "corpus.parquet",
                    help="Corpus parquet (for fingerprint + sample counts).")
    ap.add_argument("--cost-report", type=Path, default=None,
                    help="cost_report_<slug>.json from embed step.")
    ap.add_argument("--out", type=Path, default=BG_ROOT,
                    help="Root for backgrounds/ (default: ./backgrounds).")
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    out_dir = args.out / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    W, mu, eigvals, diag = fit(args.chunks, eps=args.eps)
    np.save(out_dir / "W_A.npy", W)
    np.save(out_dir / "mu_A.npy", mu)
    np.save(out_dir / "eigvals_A.npy", eigvals)

    # Auto-resolve cost report if not given.
    cost_report = args.cost_report
    if cost_report is None:
        slug = args.model.replace("/", "_").replace(":", "_")
        candidate = DATA_DIR / f"cost_report_{slug}.json"
        if candidate.is_file():
            cost_report = candidate

    write_meta(
        out_dir, name=args.name, model=args.model, diag=diag,
        corpus_path=args.corpus, cost_report_path=cost_report,
    )
    logger.info("DONE %s → %s", args.name, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
