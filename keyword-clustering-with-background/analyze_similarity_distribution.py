"""Where does a similarity threshold actually cut? The distribution decides.

A fixed cutoff like 0.8 does not separate "meaningful" from "junk" — it
separates "has a very close twin" from "does not". If a keyword's nearest
neighbour sits at 0.65, a 0.8 threshold makes it noise no matter how sensible
the keyword is. So a 60% noise share may mean the list is fragmented, not
dirty.

This reports, per model and per space:
- the CDF of each keyword's TOP-1 similarity (its closest neighbour)
- what share of keywords can survive each threshold at all
- the CDF of all pairwise similarities, sampled

Read the top-1 curve: at threshold t, the keywords below t are unclusterable by
any threshold method, by construction.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "pl-keyword-embedding-cpu-bench"))
from bench import MODELS, apply_background, l2_normalize  # noqa: E402
from cluster_keywords import load_keywords  # noqa: E402
from embed_cache import encode_cached  # noqa: E402
from sweep_leiden import BACKGROUNDS  # noqa: E402

CSV = ROOT / "work" / "keywords_url_2026-07-23.csv"
THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]

def top1_and_sample(V, chunk=2048, rng=None):
    V = V.astype(np.float32); n = len(V)
    top1 = np.empty(n, np.float32); sample = []
    for s in range(0, n, chunk):
        e = min(s+chunk, n)
        B = V[s:e] @ V.T
        np.fill_diagonal(B[:, s:e], -np.inf)
        top1[s:e] = B.max(axis=1)
        sample.append(rng.choice(B.ravel(), size=20000, replace=False))
    return top1, np.concatenate(sample)

def main():
    kws = load_keywords(CSV); n = len(kws)
    rng = np.random.default_rng(42)
    print(f"{n} fraz\n")
    print(f"{'model':15s}{'space':6s}{'top1 med':>9}{'top1 p10':>9}" +
          "".join(f"{'>='+str(t):>8}" for t in THRESHOLDS))
    for key in MODELS:
        enc = encode_cached(key, kws, device="cpu")
        for space in ("raw", "bg"):
            V = (l2_normalize(enc.vectors) if space == "raw"
                 else apply_background(enc.vectors, ROOT.parent / "backgrounds" / BACKGROUNDS[key]))
            top1, samp = top1_and_sample(V, rng=rng)
            shares = [f"{100*(top1>=t).mean():7.1f}%" for t in THRESHOLDS]
            print(f"{key:15s}{space:6s}{np.median(top1):9.3f}{np.percentile(top1,10):9.3f}" + "".join(shares))
            if space == "bg":
                print(f"{'':21s}wszystkie pary: mediana {np.median(samp):.3f}  p99 {np.percentile(samp,99):.3f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
