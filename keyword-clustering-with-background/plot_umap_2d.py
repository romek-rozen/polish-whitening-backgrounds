"""Plot the clusterings as 2D UMAP scatters — one PNG per model x space x method.

**UMAP is used here for visualisation only, never for clustering.** Every label
plotted was produced in the full embedding space (768 or 1024 dims) by
``export_clusters_xlsx.run_methods``; the 2D projection only decides *where a
dot lands*, not which group it belongs to. The one exception is inherent to the
method itself: ``umap_hdbscan`` clusters on its own 30-dim projection, which is
why this project keeps it as a reference rather than a candidate. Even there,
the 2D layout drawn here is a *separate* projection and is not what HDBSCAN saw.

Consequences to keep in mind when reading the pictures:

- Two dots far apart on the page can be neighbours in 1024 dims. A cluster that
  looks torn in 2D is not evidence that the clustering is wrong.
- UMAP is stochastic. The seed is fixed, so the plots are reproducible, but a
  different seed rearranges the page.
- Colours are cluster *identity* only. There are up to a few thousand clusters,
  far past what any categorical palette can separate, so hue is deliberately
  not readable as a label — it is there to show whether groups occupy compact
  regions. Noise (label -1) is the one colour that means something fixed: grey,
  low alpha, drawn underneath everything else.

Vectors come from the embedding cache; nothing is embedded here unless the cache
is cold. The 2D projections and the labels are themselves cached under
``<out>/cache`` so re-plotting with different cosmetics costs seconds.

Usage::

    python plot_umap_2d.py --csv work/keywords_url_2026-07-23.csv --out work/plots
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "pl-keyword-embedding-cpu-bench"))

from bench import MODELS, apply_background, l2_normalize  # noqa: E402
from cluster_keywords import load_keywords  # noqa: E402
from embed_cache import encode_cached  # noqa: E402
from export_clusters_xlsx import run_methods  # noqa: E402
from sweep_leiden import BACKGROUNDS  # noqa: E402
from sweep_umap_hdbscan import umap_reduce  # noqa: E402

logger = logging.getLogger("plot_umap_2d")


def cached_npy(path: Path, build) -> np.ndarray:
    """Return `build()` , reusing `path` when it already holds the array."""
    if path.exists():
        logger.info("  reuse %s", path.name)
        return np.load(path)
    array = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp.npy")
    np.save(tmp, array)
    tmp.replace(path)
    return array


def scatter(points: np.ndarray, labels: np.ndarray, title: str, out: Path,
            noise_color: str = "#9AA0A6") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    is_noise = labels < 0
    real = labels[~is_noise]

    figure, axes = plt.subplots(figsize=(9, 9), dpi=130)
    axes.set_facecolor("#FFFFFF")

    # Noise first so every labelled point is drawn on top of it.
    if is_noise.any():
        axes.scatter(points[is_noise, 0], points[is_noise, 1], s=2.0,
                     c=noise_color, alpha=0.18, linewidths=0, zorder=1,
                     rasterized=True)

    if real.size:
        # Hue = identity only. With hundreds to thousands of clusters no
        # palette can keep them apart, so the colour is scrambled per cluster
        # id purely to stop neighbouring ids sharing a shade.
        unique = np.unique(real)
        rng = np.random.default_rng(0)
        shuffled = rng.permutation(len(unique))
        slot = {int(c): int(s) for c, s in zip(unique, shuffled)}
        colors = np.array([slot[int(c)] for c in real]) / max(len(unique) - 1, 1)
        axes.scatter(points[~is_noise, 0], points[~is_noise, 1], s=2.6,
                     c=colors, cmap="turbo", alpha=0.75, linewidths=0,
                     zorder=2, rasterized=True)

    axes.set_title(title, fontsize=11, color="#202124", pad=12)
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_color("#DADCE0")
    axes.text(0.5, -0.02,
              "UMAP 2D is visualisation only — clusters were computed in the "
              "full embedding space",
              transform=axes.transAxes, ha="center", va="top",
              fontsize=8, color="#5F6368")
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    logger.info("  wrote %s", out.name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "work" / "plots")
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--spaces", nargs="*", default=["raw", "background"],
                    choices=["raw", "background"])
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Subset of method names; default is all five.")
    ap.add_argument("--backgrounds", type=Path, default=ROOT.parent / "backgrounds")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--neighbors", type=int, default=15,
                    help="UMAP n_neighbors for the 2D layout.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    keywords = load_keywords(args.csv)
    logger.info("%d unique keywords", len(keywords))
    cache = args.out / "cache"

    for model_key in args.models:
        encoded = encode_cached(model_key, keywords, batch_size=args.batch_size,
                                device=args.device)
        for space in args.spaces:
            if space == "raw":
                vectors = l2_normalize(encoded.vectors)
                space_name = "raw"
            else:
                bg = args.backgrounds / BACKGROUNDS[model_key]
                if not bg.exists():
                    logger.warning("%s: %s missing — skipping background space",
                                   model_key, bg.name)
                    continue
                vectors = apply_background(encoded.vectors, bg)
                space_name = "background"
            logger.info("%s / %s", model_key, space_name)

            stem = f"{model_key}__{space_name}"
            points = cached_npy(
                cache / f"{stem}__umap2d.npy",
                lambda: umap_reduce(vectors, 2, args.neighbors, args.seed),
            )

            label_path = cache / f"{stem}__labels.npz"
            if label_path.exists():
                logger.info("  reuse %s", label_path.name)
                stored = np.load(label_path)
                labels_by_method = {k: stored[k] for k in stored.files}
            else:
                labels_by_method = run_methods(vectors, args.seed)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(label_path, **labels_by_method)

            for method, labels in labels_by_method.items():
                if args.methods and method not in args.methods:
                    continue
                is_noise = labels < 0
                title = (f"{model_key} / {space_name} / {method}\n"
                         f"{int(np.unique(labels[~is_noise]).size)} clusters · "
                         f"{100.0 * is_noise.sum() / len(labels):.1f}% noise")
                scatter(points, labels, title,
                        args.out / f"{stem}__{method}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
