"""Fit a ZCA whitening transform from chunked embeddings.

Thin CLI over :mod:`scripts.lib.zca`.

Reads ``data/chunks_<model_slug>/chunk_*.npy``, fits ZCA via two
streaming passes (mean, covariance) then SVD, and writes the
canonical artefacts to ``backgrounds/<name>/``:

    W_A.npy        # (dim, dim) float32
    mu_A.npy       # (dim,)    float32
    eigvals_A.npy  # (dim,)    float32
    <name>.meta.json

Embeddings are L2-renormalised row-wise before mean/covariance; this
matches what the apply step does at inference time.  See
``scripts/lib/zca.py`` for the algorithm itself.

Usage::

    python scripts/fit_zca.py \\
        --chunks data/chunks_qwen_qwen3-embedding-8b \\
        --name polish_mixed_50k_v2_qwen3-8b_mrl4096 \\
        --model qwen/qwen3-embedding-8b \\
        --truncate-to 4096
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from lib.tokenizer import model_slug
from lib.zca import fit, write_meta

REPO_ROOT = Path(__file__).resolve().parent.parent
BG_ROOT = REPO_ROOT / "backgrounds"
DATA_DIR = REPO_ROOT / "data"

logger = logging.getLogger("fit_zca")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--chunks", type=Path, required=True,
        help="Dir with chunk_*.npy files (output of embed_via_openrouter.py).",
    )
    ap.add_argument(
        "--name", required=True,
        help="Background name; output goes to backgrounds/<name>/.",
    )
    ap.add_argument(
        "--model", required=True,
        help="OpenRouter model id used to produce the chunks (for meta).",
    )
    ap.add_argument(
        "--corpus", type=Path, default=DATA_DIR / "corpus.parquet",
        help="Corpus parquet (for fingerprint + sample counts).",
    )
    ap.add_argument(
        "--cost-report", type=Path, default=None,
        help="cost_report_<slug>.json from embed step.",
    )
    ap.add_argument(
        "--out", type=Path, default=BG_ROOT,
        help="Root for backgrounds/ (default: ./backgrounds).",
    )
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument(
        "--truncate-to", type=int, default=None,
        help="MRL refit: slice each chunk to the first N columns and "
             "L2-renormalise row-wise before fitting ZCA. Default: use "
             "native dim from chunk_0000.npy.",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    out_dir = args.out / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    W, mu, eigvals, diag = fit(
        args.chunks, eps=args.eps, truncate_to=args.truncate_to,
    )
    np.save(out_dir / "W_A.npy", W)
    np.save(out_dir / "mu_A.npy", mu)
    np.save(out_dir / "eigvals_A.npy", eigvals)

    # Auto-resolve cost report from the model slug if the user didn't
    # point at one explicitly.
    cost_report = args.cost_report
    if cost_report is None:
        candidate = DATA_DIR / f"cost_report_{model_slug(args.model)}.json"
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
