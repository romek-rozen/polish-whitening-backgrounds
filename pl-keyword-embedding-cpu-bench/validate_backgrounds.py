"""Validate shipped backgrounds on held-out phrases.

A background has one job: remove the shared dominant direction that makes every
embedding look similar to every other. This checks that it actually does that
on text it was **not** fitted on, rather than trusting the fit-time diagnostics
(which are computed on the fit corpus itself and are therefore self-confirming).

What is measured, raw vs. whitened:

- ``top_ev_ratio`` — largest eigenvalue over the mean eigenvalue of the
  held-out covariance. This *is* the anisotropy. A working background drives it
  toward ~1.
- ``mean_cosine`` — average pairwise cosine of unrelated phrases. Anisotropy
  inflates it (the "everything is 0.8 similar" baseline); it should drop toward 0.
- ``finite`` / ``unit_norm`` — the artefacts must not produce NaN/inf or break
  L2 normalisation.

Usage::

    python validate_backgrounds.py --backgrounds ../backgrounds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench import MODELS, encode, l2_normalize, load_dataset

ROOT = Path(__file__).resolve().parent

# background dir name -> bench model key
PAIRS = {
    "bgem3_pl_kwmix900k_mrl1024": "bge_m3",
    "qwen3_06b_pl_kwmix900k_mrl1024": "qwen3_06b",
    "embgemma_pl_kwmix900k_mrl768": "embeddinggemma",
}


def anisotropy(vectors: np.ndarray) -> float:
    """Largest eigenvalue / mean eigenvalue of the covariance."""
    cov = np.cov(vectors - vectors.mean(axis=0, keepdims=True), rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    return float(eigvals.max() / eigvals.mean())


def mean_pairwise_cosine(vectors: np.ndarray) -> float:
    similarity = vectors @ vectors.T
    n = similarity.shape[0]
    return float((similarity.sum() - np.trace(similarity)) / (n * (n - 1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backgrounds", type=Path, default=ROOT.parent / "backgrounds")
    ap.add_argument("--dataset", type=Path, default=ROOT / "test_dataset" / "keywords_pl.jsonl")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    if args.threads:
        import torch

        torch.set_num_threads(args.threads)

    keywords, _, _ = load_dataset(args.dataset)
    print(f"held-out phrases: {len(keywords)} (hand-written, not in any fit corpus)\n")

    rows = []
    failures = []
    for bg_name, model_key in PAIRS.items():
        bg_dir = args.backgrounds / bg_name
        if not bg_dir.exists():
            failures.append(f"{bg_name}: directory missing")
            continue

        meta_path = bg_dir / f"{bg_name}.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mu = np.load(bg_dir / "mu_A.npy").astype(np.float64)
        whitening = np.load(bg_dir / "W_A.npy").astype(np.float64)

        encoded = encode(MODELS[model_key], keywords, batch_size=32)
        raw = encoded.vectors

        if mu.shape[0] != raw.shape[1]:
            failures.append(
                f"{bg_name}: dim {mu.shape[0]} != model dim {raw.shape[1]}"
            )
            continue

        white = l2_normalize((raw - mu) @ whitening)

        finite = bool(np.all(np.isfinite(white)))
        unit = bool(np.allclose(np.linalg.norm(white, axis=1), 1.0, atol=1e-6))
        if not finite:
            failures.append(f"{bg_name}: produced non-finite values")
        if not unit:
            failures.append(f"{bg_name}: output is not unit-L2")

        raw_aniso, white_aniso = anisotropy(raw), anisotropy(white)
        row = {
            "background": bg_name,
            "model": meta["embedding_model"],
            "dim": int(meta["diagnostics"]["dim"]),
            "fit_n": int(meta["diagnostics"]["n_total"]),
            "aniso_raw": raw_aniso,
            "aniso_white": white_aniso,
            "cos_raw": mean_pairwise_cosine(raw),
            "cos_white": mean_pairwise_cosine(white),
            "finite": finite,
            "unit_norm": unit,
        }
        rows.append(row)

        # A background that does not reduce anisotropy on held-out text is
        # broken, whatever its fit-time diagnostics claim.
        if white_aniso >= raw_aniso:
            failures.append(
                f"{bg_name}: anisotropy NOT reduced "
                f"({raw_aniso:.1f} -> {white_aniso:.1f})"
            )

    print(f"{'background':34s} {'dim':>5} {'aniso raw':>10} {'aniso white':>12} "
          f"{'cos raw':>9} {'cos white':>10}  ok")
    print("-" * 90)
    for row in rows:
        ok = "yes" if row["finite"] and row["unit_norm"] else "NO"
        print(
            f"{row['background']:34s} {row['dim']:5d} {row['aniso_raw']:10.1f} "
            f"{row['aniso_white']:12.1f} {row['cos_raw']:9.3f} "
            f"{row['cos_white']:10.3f}  {ok}"
        )

    print()
    if failures:
        print("FAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("all backgrounds validated: anisotropy reduced, output finite and unit-L2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
