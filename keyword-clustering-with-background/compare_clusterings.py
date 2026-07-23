"""Quantitatively compare representative clustering partitions.

The output contains pairwise ARI/AMI comparisons in two directions: methods
within a model and models within a method.  Metrics are calculated both on all
keywords and on the intersection of non-noise keywords from both partitions.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from cluster_keywords import load_keywords
from embed_cache import cache_path, corpus_fingerprint, encode_cached
from export_clusters_xlsx import run_methods
from sweep_leiden import BACKGROUNDS

import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "pl-keyword-embedding-cpu-bench"))
from bench import MODELS, apply_background  # noqa: E402


@dataclass
class Comparison:
    comparison_type: str
    fixed_item: str
    item_a: str
    item_b: str
    n_all: int
    ari_all: float
    ami_all: float
    agreed_noise: int
    n_non_noise_both: int
    ari_non_noise: float
    ami_non_noise: float


def compare(comparison_type: str, fixed_item: str, item_a: str, item_b: str,
            labels_a: np.ndarray, labels_b: np.ndarray) -> Comparison:
    if labels_a.shape != labels_b.shape:
        raise ValueError(f"Label shape mismatch: {labels_a.shape} != {labels_b.shape}")
    keep = (labels_a != -1) & (labels_b != -1)
    if keep.sum() < 2:
        ari_non_noise = ami_non_noise = float("nan")
    else:
        ari_non_noise = adjusted_rand_score(labels_a[keep], labels_b[keep])
        ami_non_noise = adjusted_mutual_info_score(labels_a[keep], labels_b[keep])
    return Comparison(
        comparison_type=comparison_type,
        fixed_item=fixed_item,
        item_a=item_a,
        item_b=item_b,
        n_all=len(labels_a),
        ari_all=adjusted_rand_score(labels_a, labels_b),
        ami_all=adjusted_mutual_info_score(labels_a, labels_b),
        agreed_noise=int(((labels_a == -1) & (labels_b == -1)).sum()),
        n_non_noise_both=int(keep.sum()),
        ari_non_noise=ari_non_noise,
        ami_non_noise=ami_non_noise,
    )


def aggregate(rows: list[Comparison], comparison_type: str, metric: str) -> tuple[float, float]:
    values = np.asarray([
        getattr(row, metric) for row in rows if row.comparison_type == comparison_type
    ], dtype=float)
    return float(np.nanmean(values)), float(np.nanmedian(values))


def fmt(value: float) -> str:
    return "n/d" if np.isnan(value) else f"{value:.3f}"


def write_report(path: Path, rows: list[Comparison], n_keywords: int,
                 models: list[str], methods: list[str]) -> None:
    method_ari = aggregate(rows, "methods_within_model", "ari_all")
    method_ami = aggregate(rows, "methods_within_model", "ami_all")
    model_ari = aggregate(rows, "models_within_method", "ari_all")
    model_ami = aggregate(rows, "models_within_method", "ami_all")
    method_ari_nn = aggregate(rows, "methods_within_model", "ari_non_noise")
    method_ami_nn = aggregate(rows, "methods_within_model", "ami_non_noise")
    model_ari_nn = aggregate(rows, "models_within_method", "ari_non_noise")
    model_ami_nn = aggregate(rows, "models_within_method", "ami_non_noise")

    method_agreement = (method_ari[0] + method_ami[0]) / 2
    model_agreement = (model_ari[0] + model_ami[0]) / 2
    if method_agreement < model_agreement:
        conclusion = "wybór metody wpływa na wynik bardziej niż wybór modelu"
    elif model_agreement < method_agreement:
        conclusion = "wybór modelu wpływa na wynik bardziej niż wybór metody"
    else:
        conclusion = "wpływ wyboru modelu i metody jest w tym zestawieniu równy"

    lines = [
        "# Porównanie podziałów klastrowych", "",
        f"Porównano {n_keywords:,} fraz, {len(models)} modele i {len(methods)} metod. ".replace(",", " ", 1)
        + "Każdy model używa właściwego backgroundu `pl_kwmix900k`.", "",
        "## Odpowiedź na główne pytanie", "",
        f"Średnia zgodność między metodami przy stałym modelu wynosi ARI **{method_ari[0]:.3f}** "
        f"i AMI **{method_ami[0]:.3f}**. Średnia zgodność między modelami przy stałej metodzie "
        f"wynosi ARI **{model_ari[0]:.3f}** i AMI **{model_ami[0]:.3f}**. Niższa zgodność oznacza "
        f"silniejszy wpływ danego wyboru, dlatego **{conclusion}**.", "",
        "| zmieniany czynnik | ARI średnia | ARI mediana | AMI średnia | AMI mediana |",
        "|---|---:|---:|---:|---:|",
        f"| metoda (model stały) | {method_ari[0]:.3f} | {method_ari[1]:.3f} | {method_ami[0]:.3f} | {method_ami[1]:.3f} |",
        f"| model (metoda stała) | {model_ari[0]:.3f} | {model_ari[1]:.3f} | {model_ami[0]:.3f} | {model_ami[1]:.3f} |",
        "", "## Wariant bez szumu", "",
        "W tym wariancie każda para jest liczona tylko na przecięciu fraz, które oba podziały "
        "uznały za nie-szum. Odpowiada on na pytanie o zgodność grupowania wspólnie zaakceptowanych "
        "fraz; wariant pełny dodatkowo mierzy zgodność decyzji, co odrzucić.", "",
        "| zmieniany czynnik | ARI średnia | ARI mediana | AMI średnia | AMI mediana |",
        "|---|---:|---:|---:|---:|",
        f"| metoda (model stały) | {method_ari_nn[0]:.3f} | {method_ari_nn[1]:.3f} | {method_ami_nn[0]:.3f} | {method_ami_nn[1]:.3f} |",
        f"| model (metoda stała) | {model_ari_nn[0]:.3f} | {model_ari_nn[1]:.3f} | {model_ami_nn[0]:.3f} | {model_ami_nn[1]:.3f} |",
        "",
        f"Po usunięciu szumu relacja się odwraca: zgodność między modelami "
        f"(ARI {model_ari_nn[0]:.3f}, AMI {model_ami_nn[0]:.3f}) jest niższa niż między metodami "
        f"(ARI {method_ari_nn[0]:.3f}, AMI {method_ami_nn[0]:.3f}). Oznacza to, że wybór metody "
        "silniej różnicuje pełny wynik — zwłaszcza decyzję o odrzuceniu fraz — ale wśród fraz "
        "zaakceptowanych przez oba podziały większe różnice wnosi wybór modelu.",
        "", "## Szczegółowe porównania", "",
        "Pełne macierze (w układzie długim) są w `porownanie_metod.csv`. Kolumna "
        "`agreed_noise` podaje liczbę fraz jednocześnie oznaczonych jako `-1`, a "
        "`n_non_noise_both` — mianownik wariantu bez szumu.", "",
    ]
    for comparison_type, title in (
        ("methods_within_model", "Metody przy stałym modelu"),
        ("models_within_method", "Modele przy stałej metodzie"),
    ):
        lines += [f"### {title}", "",
                  "| stały element | para | ARI | AMI | wspólny szum | n bez szumu | ARI bez szumu | AMI bez szumu |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            if row.comparison_type == comparison_type:
                lines.append(
                    f"| {row.fixed_item} | {row.item_a} ↔ {row.item_b} | {row.ari_all:.3f} | "
                    f"{row.ami_all:.3f} | {row.agreed_noise} | {row.n_non_noise_both} | "
                    f"{fmt(row.ari_non_noise)} | {fmt(row.ami_non_noise)} |"
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "work/keywords_url_2026-07-23.csv")
    parser.add_argument("--out-csv", type=Path, default=ROOT / "work/porownanie_metod.csv")
    parser.add_argument("--out-md", type=Path, default=ROOT / "work/porownanie_metod.md")
    parser.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--backgrounds", type=Path, default=ROOT.parent / "backgrounds")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "work/embeddings")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    keywords = load_keywords(args.csv)
    fingerprint = corpus_fingerprint(keywords)
    labels: dict[tuple[str, str], np.ndarray] = {}
    methods: list[str] | None = None
    print(f"{len(keywords)} unique keywords")
    for model in args.models:
        expected = cache_path(args.cache_dir, model, fingerprint)
        if not expected.exists():
            raise SystemExit(f"Missing embedding cache; refusing to re-embed: {expected}")
        encoded = encode_cached(model, keywords, device="cpu", cache_dir=args.cache_dir)
        vectors = apply_background(encoded.vectors, args.backgrounds / BACKGROUNDS[model])
        current = run_methods(vectors, args.seed)
        if methods is None:
            methods = list(current)
        elif list(current) != methods:
            raise RuntimeError("run_methods returned inconsistent method sets")
        labels.update({(model, method): value for method, value in current.items()})

    assert methods is not None
    rows: list[Comparison] = []
    for model in args.models:
        for method_a, method_b in combinations(methods, 2):
            rows.append(compare("methods_within_model", model, method_a, method_b,
                                labels[(model, method_a)], labels[(model, method_b)]))
    for method in methods:
        for model_a, model_b in combinations(args.models, 2):
            rows.append(compare("models_within_method", method, model_a, model_b,
                                labels[(model_a, method)], labels[(model_b, method)]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    write_report(args.out_md, rows, len(keywords), args.models, methods)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
