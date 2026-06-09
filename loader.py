"""Load a whitening background from this repo.

No dependencies beyond numpy.

>>> from loader import load_background
>>> W, mu, meta = load_background("polish_mixed_50k_v1_mrl1024")
>>> import numpy as np
>>> x = np.random.randn(5, 1024).astype("float32")  # MRL-truncated, L2-normalised
>>> x_white = (x - mu) @ W
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "backgrounds"


@dataclass
class Background:
    """Wraps the three arrays + meta. ``W`` and ``mu`` are float32; meta is a dict."""
    name: str
    W: np.ndarray
    mu: np.ndarray
    eigvals: np.ndarray
    meta: dict

    @property
    def dim(self) -> int:
        return int(self.W.shape[0])

    def apply(self, x: np.ndarray) -> np.ndarray:
        """``(x - mu) @ W``. ``x`` must be (N, dim), L2-normalised, fp32."""
        return (x.astype(np.float32, copy=False) - self.mu) @ self.W


def list_backgrounds() -> list[str]:
    return sorted(p.name for p in ROOT.iterdir() if (p / "W_A.npy").is_file())


def load_background(name: str, root: Path | None = None) -> Background:
    base = (root or ROOT) / name
    if not (base / "W_A.npy").is_file():
        raise FileNotFoundError(
            f"background {name!r} not found under {base}. "
            f"Available: {list_backgrounds()}"
        )
    W = np.load(base / "W_A.npy")
    mu = np.load(base / "mu_A.npy")
    eig = np.load(base / "eigvals_A.npy")
    meta_path = next(base.glob("*.meta.json"))
    meta = json.loads(meta_path.read_text())
    return Background(name=name, W=W, mu=mu, eigvals=eig, meta=meta)
