"""CPU-only benchmark of embedding models for Polish keyword clustering.

Answers one question: which locally-servable embedding model groups Polish
search keywords best on a CPU-only VPS, and how fast?

Every model is loaded through ``sentence-transformers`` on ``device="cpu"`` --
the same path a FastAPI/TEI wrapper would use on a VPS. No vLLM, no GPU.

Two axes are measured:

- **quality**: cluster the keywords, compare against the hand-labelled groups
  in ``test_dataset/keywords_pl.jsonl`` (homogeneity / completeness / AMI / ARI).
- **cost**: wall-clock encode throughput (keywords/s) and per-keyword latency
  on CPU -- the numbers that decide whether a VPS can serve it.

Clustering mirrors production (`analysis/clustering_Louvain_Leiden_UmapHdbscan`
in the sitefocus repo): cosine kNN graph + Leiden, *never* a similarity
threshold -- a fixed cosine floor empties the graph in high-dim anisotropic
space and every node becomes its own "cluster".

Usage::

    python bench.py                        # all models, default sweep
    python bench.py --models embeddinggemma qwen3_06b
    python bench.py --whiten               # add ZCA-whitened variants
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS_DIR = Path.home() / "models"
logger = logging.getLogger("bench")


@dataclass(frozen=True)
class ModelSpec:
    """A locally-downloaded embedding model and how to call it."""

    key: str
    path: Path
    # sentence-transformers prompt to use for a symmetric clustering task.
    # None = feed the raw text (bge-m3 and Qwen3-Embedding need no prefix).
    prompt_name: str | None = None
    note: str = ""


MODELS: dict[str, ModelSpec] = {
    "embeddinggemma": ModelSpec(
        key="embeddinggemma",
        path=MODELS_DIR / "embeddinggemma-300m",
        prompt_name="Clustering",
        note="308M, 768d, true Matryoshka, dedicated Clustering prompt",
    ),
    "qwen3_06b": ModelSpec(
        key="qwen3_06b",
        path=MODELS_DIR / "qwen3-embedding-0.6b",
        prompt_name=None,
        note="0.6B, 1024d, PL-MTEB clustering leader among small models",
    ),
    "bge_m3": ModelSpec(
        key="bge_m3",
        path=MODELS_DIR / "bge-m3",
        prompt_name=None,
        note="568M, 1024d, multilingual generalist, no task prompts",
    ),
}


@dataclass
class EncodeResult:
    vectors: np.ndarray
    encode_s: float
    per_item_ms: float
    throughput: float
    dim: int
    prompt_used: str | None
    load_s: float = 0.0


@dataclass
class ClusterResult:
    label: str
    resolution: float
    n_clusters: int
    homogeneity: float
    completeness: float
    v_measure: float
    ami: float
    ari: float
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def load_dataset(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    """Return (keywords, integer truth labels, group names in label order)."""
    keywords: list[str] = []
    groups: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            keywords.append(row["keyword"])
            groups.append(row["group"])
    names = sorted(set(groups))
    index = {name: i for i, name in enumerate(names)}
    return keywords, np.asarray([index[g] for g in groups]), names


# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #


def l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("model returned a zero vector")
    return values / norms


def encode(
    spec: ModelSpec, texts: list[str], batch_size: int, use_prompts: bool = False
) -> EncodeResult:
    """Load the model on CPU and embed every keyword, timing both steps.

    ``use_prompts`` is **off by default**. Only embeddinggemma defines task
    prompts; bge-m3 and Qwen3-Embedding take raw text. Enabling the prompt for
    one model and not the others compares different things, so the default is
    the common denominator: plain embeddings, no instruction.

    Note the trade-off when reading a no-prompt embeddinggemma number: that
    model was *trained* with its prompts, so raw text is off-distribution for
    it and the score is a floor, not its ceiling. Run both ways to see the gap.
    """
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    model = SentenceTransformer(str(spec.path), device="cpu")
    load_s = time.perf_counter() - started

    kwargs: dict = {"batch_size": batch_size, "show_progress_bar": False}
    prompt_used = spec.prompt_name if use_prompts else None
    if prompt_used:
        available = set(getattr(model, "prompts", {}) or {})
        if prompt_used in available:
            kwargs["prompt_name"] = prompt_used
        else:
            logger.warning(
                "%s: prompt %r not in model config (%s) -- encoding raw",
                spec.key,
                prompt_used,
                sorted(available) or "none",
            )
            prompt_used = None

    # Warm-up: first forward pass pays lazy-init costs that would otherwise
    # be charged to the measured run and inflate CPU latency.
    model.encode(texts[:8], **kwargs)

    started = time.perf_counter()
    vectors = np.asarray(model.encode(texts, **kwargs), dtype=np.float64)
    encode_s = time.perf_counter() - started

    del model
    return EncodeResult(
        vectors=l2_normalize(vectors),
        encode_s=encode_s,
        per_item_ms=1000.0 * encode_s / len(texts),
        throughput=len(texts) / encode_s,
        dim=vectors.shape[1],
        prompt_used=prompt_used,
        load_s=load_s,
    )


def apply_background(vectors: np.ndarray, background: Path) -> np.ndarray:
    """Apply a *fitted* ZCA background: x_white = (x - mu) @ W, then re-L2.

    This is the real thing, unlike :func:`zca_whiten` — mu and W were fitted on
    a large keyword corpus, so the covariance is well-conditioned instead of
    being guessed from the 150 benchmark rows.
    """
    mu = np.load(background / "mu_A.npy").astype(np.float64)
    whitening = np.load(background / "W_A.npy").astype(np.float64)
    if mu.shape[0] != vectors.shape[1]:
        raise SystemExit(
            f"background dim {mu.shape[0]} != embedding dim {vectors.shape[1]} "
            f"({background})"
        )
    return l2_normalize((vectors - mu) @ whitening)


def zca_whiten(vectors: np.ndarray, shrinkage: float = 0.10, eps: float = 1e-6) -> np.ndarray:
    """Batch ZCA whitening, shrinkage-stabilised (n << dim here).

    This is the *batch* fallback, not a corpus-fitted background: with 150
    keywords and 768-1024 dims the covariance is massively rank-deficient, so
    it is shrunk toward the identity before inversion. Treat the whitened
    numbers as an indication, not as the quality of a real fitted background.
    """
    mu = vectors.mean(axis=0, keepdims=True)
    centered = vectors - mu
    cov = np.cov(centered, rowvar=False)
    cov = (1.0 - shrinkage) * cov + shrinkage * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + eps)) @ eigvecs.T
    return l2_normalize(centered @ whitening)


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #


def knn_graph(vectors: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    """Symmetric cosine kNN edge list. Vectors must already be unit-L2."""
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -np.inf)
    neighbours = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    edges: dict[tuple[int, int], float] = {}
    for i, row in enumerate(neighbours):
        for j in row:
            weight = float(similarity[i, j])
            # cosine can be negative; a negative weight is meaningless as a
            # modularity edge weight, so clamp to a small positive floor.
            key = (i, int(j)) if i < j else (int(j), i)
            edges[key] = max(edges.get(key, 0.0), max(weight, 1e-6))
    return [(i, j, w) for (i, j), w in edges.items()]


def cluster_leiden(vectors: np.ndarray, k: int, resolution: float, seed: int) -> np.ndarray:
    import igraph as ig
    import leidenalg

    edges = knn_graph(vectors, k)
    graph = ig.Graph(n=len(vectors), edges=[(i, j) for i, j, _ in edges])
    graph.es["weight"] = [w for _, _, w in edges]
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )
    return np.asarray(partition.membership)


def cluster_agglomerative(vectors: np.ndarray, distance_threshold: float) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    return model.fit_predict(vectors)


def score(label: str, resolution: float, truth: np.ndarray, pred: np.ndarray) -> ClusterResult:
    from sklearn import metrics

    return ClusterResult(
        label=label,
        resolution=resolution,
        n_clusters=int(len(set(pred.tolist()))),
        homogeneity=float(metrics.homogeneity_score(truth, pred)),
        completeness=float(metrics.completeness_score(truth, pred)),
        v_measure=float(metrics.v_measure_score(truth, pred)),
        ami=float(metrics.adjusted_mutual_info_score(truth, pred)),
        ari=float(metrics.adjusted_rand_score(truth, pred)),
    )


# --------------------------------------------------------------------------- #
# baseline
# --------------------------------------------------------------------------- #


def lexical_baseline(texts: list[str], truth: np.ndarray, threshold: float) -> ClusterResult:
    """TF-IDF word overlap -- what you get WITHOUT semantic embeddings.

    Included so the embedding numbers have a floor to beat: the dataset is
    built with cross-group lexical traps ("warszawa", "cena", "ranking"
    appear in several unrelated groups).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    matrix = TfidfVectorizer(analyzer="word", ngram_range=(1, 2)).fit_transform(texts)
    vectors = l2_normalize(np.asarray(matrix.todense(), dtype=np.float64))
    pred = cluster_agglomerative(vectors, distance_threshold=threshold)
    return score("tfidf_lexical", float("nan"), truth, pred)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def markdown_table(rows: list[dict]) -> str:
    header = (
        "| model | space | dim | r | #clusters | homog. | compl. | AMI | ARI "
        "| kw/s (CPU) | ms/kw | load s |"
    )
    divider = "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"
    lines = [header, divider]
    for row in rows:
        lines.append(
            "| {model} | {space} | {dim} | {r} | {n_clusters} | {homogeneity:.3f} "
            "| {completeness:.3f} | {ami:.3f} | {ari:.3f} | {throughput} | {per_item_ms} "
            "| {load_s} |".format(**row)
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--dataset", type=Path, default=ROOT / "test_dataset" / "keywords_pl.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument(
        "--resolutions", type=float, nargs="*", default=[0.5, 1.0, 2.0, 4.0]
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=0, help="0 = torch default")
    parser.add_argument("--whiten", action="store_true", help="also score ZCA-whitened space")
    parser.add_argument(
        "--background", type=Path, default=None,
        help="Dir with a FITTED background (mu_A.npy + W_A.npy). Adds a "
        "'background' space. Its dim must match the model's.",
    )
    parser.add_argument(
        "--use-prompts",
        action="store_true",
        help="use each model's task prompt (only embeddinggemma has one); "
        "off by default so every model is compared on plain embeddings",
    )
    parser.add_argument("--baseline-threshold", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.threads:
        import torch

        torch.set_num_threads(args.threads)

    keywords, truth, group_names = load_dataset(args.dataset)
    logger.info("dataset: %d keywords in %d groups", len(keywords), len(group_names))

    rows: list[dict] = []
    payload: dict = {
        "dataset": {
            "path": str(args.dataset),
            "n_keywords": len(keywords),
            "n_groups": len(group_names),
            "groups": group_names,
        },
        "params": {
            "knn": args.knn,
            "resolutions": args.resolutions,
            "batch_size": args.batch_size,
            "threads": args.threads or "torch-default",
            "seed": args.seed,
        },
        "models": {},
    }

    baseline = lexical_baseline(keywords, truth, args.baseline_threshold)
    rows.append(
        {
            "model": "tfidf (baseline)",
            "space": "lexical",
            "dim": "-",
            "r": "-",
            "n_clusters": baseline.n_clusters,
            "homogeneity": baseline.homogeneity,
            "completeness": baseline.completeness,
            "ami": baseline.ami,
            "ari": baseline.ari,
            "throughput": "-",
            "per_item_ms": "-",
            "load_s": "-",
        }
    )
    payload["baseline"] = baseline.__dict__

    for key in args.models:
        spec = MODELS[key]
        if not spec.path.exists():
            logger.warning("%s: %s not found -- skipping", key, spec.path)
            continue

        logger.info("%s: encoding %d keywords on CPU...", key, len(keywords))
        encoded = encode(spec, keywords, args.batch_size, use_prompts=args.use_prompts)
        logger.info(
            "%s: dim=%d  %.1f kw/s  %.1f ms/kw  (load %.1fs, prompt=%s)",
            key,
            encoded.dim,
            encoded.throughput,
            encoded.per_item_ms,
            encoded.load_s,
            encoded.prompt_used,
        )

        spaces = {"raw+L2": encoded.vectors}
        if args.background:
            spaces[f"bg:{args.background.name}"] = apply_background(
                encoded.vectors, args.background
            )
        if args.whiten:
            spaces["ZCA(batch)"] = zca_whiten(encoded.vectors)

        model_payload: dict = {
            "path": str(spec.path),
            "note": spec.note,
            "dim": encoded.dim,
            "prompt_used": encoded.prompt_used,
            "load_s": round(encoded.load_s, 2),
            "encode_s": round(encoded.encode_s, 3),
            "throughput_kw_per_s": round(encoded.throughput, 1),
            "latency_ms_per_kw": round(encoded.per_item_ms, 2),
            "runs": [],
        }

        for space_name, vectors in spaces.items():
            for resolution in args.resolutions:
                try:
                    pred = cluster_leiden(vectors, args.knn, resolution, args.seed)
                except ImportError:
                    logger.warning("leidenalg/igraph missing -- agglomerative fallback")
                    pred = cluster_agglomerative(vectors, distance_threshold=0.35)
                result = score(f"{key}|{space_name}", resolution, truth, pred)
                model_payload["runs"].append(result.__dict__)
                rows.append(
                    {
                        "model": key,
                        "space": space_name,
                        "dim": encoded.dim,
                        "r": resolution,
                        "n_clusters": result.n_clusters,
                        "homogeneity": result.homogeneity,
                        "completeness": result.completeness,
                        "ami": result.ami,
                        "ari": result.ari,
                        "throughput": round(encoded.throughput, 1),
                        "per_item_ms": round(encoded.per_item_ms, 2),
                        "load_s": round(encoded.load_s, 1),
                    }
                )

        payload["models"][key] = model_payload

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    table = markdown_table(rows)
    (args.out / "results.md").write_text(table + "\n", encoding="utf-8")
    print()
    print(table)
    print(f"\nwrote {args.out / 'results.json'} and {args.out / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
