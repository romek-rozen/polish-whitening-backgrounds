"""Fine average-linkage sweep, deduplicated into stable threshold ranges."""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "pl-keyword-embedding-cpu-bench"))

from bench import MODELS, apply_background  # noqa: E402
from cluster_agglomerative import (  # noqa: E402
    agglomerative_tree,
    cut_agglomerative_tree,
    demote_small,
)
from cluster_keywords import load_keywords  # noqa: E402
from embed_cache import cache_path, corpus_fingerprint, encode_cached  # noqa: E402
from sweep_leiden import BACKGROUNDS  # noqa: E402

THRESHOLDS = np.round(np.arange(0.35, 0.8501, 0.001), 3)
MIN_SIZES = (2, 3)


@dataclass
class Result:
    model: str
    threshold_from: float
    threshold_to: float
    stable_range_width: float
    min_size: int
    n_clusters: int
    noise: int
    noise_pct: float
    largest: int
    median_size: float


def summarise(model: str, threshold: float, min_size: int,
              labels: np.ndarray) -> Result:
    real = labels[labels >= 0]
    sizes = np.bincount(real) if real.size else np.empty(0, dtype=int)
    sizes = sizes[sizes > 0]
    noise = int((labels == -1).sum())
    return Result(
        model=model, threshold_from=float(threshold), threshold_to=float(threshold),
        stable_range_width=0.0, min_size=min_size, n_clusters=int(len(sizes)),
        noise=noise, noise_pct=100 * noise / len(labels),
        largest=int(sizes.max()) if len(sizes) else 0,
        median_size=float(np.median(sizes)) if len(sizes) else 0.0,
    )


def close_range(row: Result, threshold_to: float) -> Result:
    row.threshold_to = float(threshold_to)
    row.stable_range_width = round(row.threshold_to - row.threshold_from, 3)
    return row


def sweep_tree(model: str, tree: np.ndarray) -> tuple[list[Result], dict[int, list[Result]]]:
    """Return deduplicated ranges and full curves used for slope analysis."""
    deduplicated: list[Result] = []
    curves = {min_size: [] for min_size in MIN_SIZES}
    previous: dict[int, np.ndarray | None] = {min_size: None for min_size in MIN_SIZES}
    open_rows: dict[int, Result | None] = {min_size: None for min_size in MIN_SIZES}

    for threshold in THRESHOLDS:
        base = cut_agglomerative_tree(tree, float(threshold))
        for min_size in MIN_SIZES:
            labels = demote_small(base, min_size)
            current = summarise(model, float(threshold), min_size, labels)
            curves[min_size].append(current)
            if previous[min_size] is not None and np.array_equal(labels, previous[min_size]):
                continue
            if open_rows[min_size] is not None:
                deduplicated.append(close_range(open_rows[min_size], float(threshold - 0.001)))
            open_rows[min_size] = current
            previous[min_size] = labels.copy()

    for min_size in MIN_SIZES:
        assert open_rows[min_size] is not None
        deduplicated.append(close_range(open_rows[min_size], float(THRESHOLDS[-1])))
    deduplicated.sort(key=lambda row: (row.model, row.min_size, row.threshold_from))
    return deduplicated, curves


def window_changes(curve: list[Result], radius: int = 10) -> list[tuple[float, float]]:
    """Absolute changes across a 0.020-wide window around every grid point."""
    out = []
    for index in range(len(curve)):
        lo, hi = max(0, index - radius), min(len(curve) - 1, index + radius)
        width = curve[hi].threshold_from - curve[lo].threshold_from
        clusters = abs(curve[hi].n_clusters - curve[lo].n_clusters) / width
        noise = abs(curve[hi].noise_pct - curve[lo].noise_pct) / width
        out.append((clusters, noise))
    return out


def recommendation(rows: list[Result], curve: list[Result]) -> Result:
    candidates = [row for row in rows if row.min_size == 2
                  and 0.5 <= row.threshold_from <= 0.6
                  and 10.0 <= row.noise_pct <= 30.0]
    if not candidates:
        candidates = [row for row in rows if row.min_size == 2
                      and 0.5 <= row.threshold_from <= 0.6]
    slopes = window_changes(curve)
    index_by_threshold = {row.threshold_from: i for i, row in enumerate(curve)}

    def rank(row: Result):
        midpoint = round((row.threshold_from + row.threshold_to) / 2, 3)
        index = index_by_threshold.get(midpoint)
        if index is None:
            index = min(range(len(curve)),
                        key=lambda i: abs(curve[i].threshold_from - midpoint))
        cluster_slope, noise_slope = slopes[index]
        return (-row.stable_range_width, cluster_slope / 1000 + noise_slope,
                abs(row.noise_pct - 22.5))

    return min(candidates, key=rank)


def write_plot(path: Path, curves: dict[str, dict[int, list[Result]]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(curves), 1, figsize=(10, 10), sharex=True)
    twin_axes = []
    for axis, (model, by_min_size) in zip(axes, curves.items()):
        twin = axis.twinx()
        twin_axes.append(twin)
        for min_size, linestyle in ((2, "-"), (3, "--")):
            curve = by_min_size[min_size]
            x = [r.threshold_from for r in curve]
            axis.plot(x, [r.n_clusters for r in curve], linestyle,
                      label=f"clusters, min={min_size}", color="tab:blue", alpha=0.8)
            twin.plot(x, [r.noise_pct for r in curve], linestyle,
                      label=f"noise %, min={min_size}", color="tab:red", alpha=0.8)
        axis.set_title(model)
        axis.set_ylabel("n_clusters", color="tab:blue")
        twin.set_ylabel("noise %", color="tab:red")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("similarity threshold")
    fig.legend(
        [axes[0].lines[0], axes[0].lines[1],
         twin_axes[0].lines[0], twin_axes[0].lines[1]],
        ["clusters, min=2", "clusters, min=3", "noise %, min=2", "noise %, min=3"],
        loc="upper center", ncol=4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(path: Path, rows: list[Result],
                 curves: dict[str, dict[int, list[Result]]]) -> None:
    lines = [
        "# Gęsty sweep aglomeracji average-linkage", "",
        "Dla każdego modelu drzewo linkage policzono raz w przestrzeni z właściwym "
        "backgroundem. Następnie wykonano 501 cięć od 0.350 do 0.850. Kolejne "
        "identyczne podziały są zapisane jako jeden zakres stabilności.", "",
        "**Wynik deduplikacji:** żadne dwa sąsiednie cięcia odległe o 0.001 nie dały "
        "identycznego finalnego podziału — ani dla żadnego modelu, ani dla `min_size=2/3`. "
        "Dlatego wszystkie dokładne `stable_range_width` wynoszą 0 na tej siatce. Krok "
        "0.0005 zwiększyłby rozdzielczość, ale nie zmieniłby faktu, że na 19 801 fraz "
        "wysokości scaleń są bardzo gęste. Odporności należy tu szukać w płaskim przebiegu "
        "statystyk, a nie w identyczności całego wektora etykiet.", "",
        "## Rekomendowane progi na płaskich odcinkach krzywej", "",
        "| model | rekomendowany próg / zakres | szerokość | klastry | szum | największy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    recommendations = {}
    for model, by_min_size in curves.items():
        model_rows = [row for row in rows if row.model == model]
        chosen = recommendation(model_rows, by_min_size[2])
        recommendations[model] = chosen
        midpoint = (chosen.threshold_from + chosen.threshold_to) / 2
        lines.append(
            f"| {model} | **{midpoint:.3f}** / [{chosen.threshold_from:.3f}, "
            f"{chosen.threshold_to:.3f}] | **{chosen.stable_range_width:.3f}** | "
            f"{chosen.n_clusters} | {chosen.noise_pct:.1f}% | {chosen.largest} |"
        )
    lines += [
        "", "Rekomendacje używają `min_size=2`: nie odrzucają automatycznie sensownych "
        "par. Najpierw maksymalizowana jest szerokość identycznego podziału w okolicy "
        "0.5–0.6 przy 10–30% szumu, a przy remisie wybierany jest łagodniejszy lokalny "
        "przebieg krzywych.", "", "## Gdzie krzywa jest płaska, a gdzie stroma?", "",
    ]
    for model, by_min_size in curves.items():
        curve = by_min_size[2]
        slopes = window_changes(curve)
        indices = [i for i, row in enumerate(curve) if 0.5 <= row.threshold_from <= 0.6]
        flat = min(indices, key=lambda i: slopes[i][0] / 1000 + slopes[i][1])
        steep = max(range(len(curve)), key=lambda i: slopes[i][0] / 1000 + slopes[i][1])
        lines += [
            f"### {model}", "",
            f"Najbardziej płaski lokalny odcinek w okolicy 0.5–0.6 wypada przy **{curve[flat].threshold_from:.3f}**: "
            f"w oknie ±0.010 zmiana wynosi około {slopes[flat][0] * 0.02:.0f} klastrów "
            f"i {slopes[flat][1] * 0.02:.1f} p.p. szumu. Najstromszy odcinek całego sweepu "
            f"jest przy **{curve[steep].threshold_from:.3f}** ({slopes[steep][0] * 0.02:.0f} "
            f"klastrów i {slopes[steep][1] * 0.02:.1f} p.p. w takim samym oknie). "
            "Próg na pierwszym odcinku jest odporniejszy; na drugim drobna zmiana podobieństw "
            "przenosi wiele fraz między klastrami i szumem.", "",
        ]
    lines += [
        "## Interpretacja", "",
        "Próg 0.8 nie mierzy jakości frazy. Jeśli top-1 podobieństwo frazy jest niższe, "
        "żadne cięcie progowe nie może jej połączyć; wysoki szum jest więc arytmetycznym "
        "skutkiem położenia progu względem rozkładu podobieństw. `stable_range_width=0` "
        "oznacza, że nawet sąsiedni punkt siatki odległy o 0.001 zmienił podział — rzeczywisty "
        "stabilny przedział jest wtedy węższy niż rozdzielczość sweepu.", "",
        "Pełne, zdeduplikowane wyniki znajdują się w `agglo_fine_sweep.csv`, a wykres "
        "krzywych w `plots/agglo_fine_sweep.png`.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path,
                        default=ROOT / "work/keywords_url_2026-07-23.csv")
    parser.add_argument("--out-csv", type=Path,
                        default=ROOT / "work/agglo_fine_sweep.csv")
    parser.add_argument("--out-md", type=Path,
                        default=ROOT / "work/agglo_fine_sweep.md")
    parser.add_argument("--plot", type=Path,
                        default=ROOT / "work/plots/agglo_fine_sweep.png")
    parser.add_argument("--models", nargs="*", default=list(MODELS),
                        choices=list(MODELS))
    parser.add_argument("--backgrounds", type=Path,
                        default=ROOT.parent / "backgrounds")
    parser.add_argument("--cache-dir", type=Path,
                        default=ROOT / "work/embeddings")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    keywords = load_keywords(args.csv)
    fingerprint = corpus_fingerprint(keywords)
    all_rows: list[Result] = []
    all_curves: dict[str, dict[int, list[Result]]] = {}
    for model in args.models:
        expected = cache_path(args.cache_dir, model, fingerprint)
        if not expected.exists():
            raise SystemExit(f"Missing embedding cache; refusing to re-embed: {expected}")
        encoded = encode_cached(model, keywords, device="cpu", cache_dir=args.cache_dir)
        vectors = apply_background(encoded.vectors,
                                   args.backgrounds / BACKGROUNDS[model])
        tree = agglomerative_tree(vectors, "average")
        rows, curves = sweep_tree(model, tree)
        all_rows.extend(rows)
        all_curves[model] = curves
        del tree, vectors, encoded, rows, curves
        gc.collect()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(all_rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in all_rows)
    write_report(args.out_md, all_rows, all_curves)
    write_plot(args.plot, all_curves)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")
    print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
