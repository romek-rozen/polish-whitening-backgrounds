"""Build a deterministic 130k sentence sample from the 50k-document corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ENTROPY_SRC = Path("/home/spark001/Workspaces/entropy_001/src")
sys.path.insert(0, str(ENTROPY_SRC))

from entropy_lab.features.chunk import split_sentences

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("build_corpus_sentences")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(corpus: Path, out: Path, sample_size: int, seed: int) -> dict[str, object]:
    """Reservoir-sample sentences while counting the complete generated population."""
    table = pq.read_table(corpus, columns=["text", "sha", "source"])
    rng = random.Random(seed)
    reservoir: list[dict[str, object]] = []
    total = 0
    per_source: dict[str, int] = {}
    for text, doc_sha, source in tqdm(
        zip(
            table.column("text").to_pylist(),
            table.column("sha").to_pylist(),
            table.column("source").to_pylist(),
        ),
        total=table.num_rows,
        desc="sentences",
    ):
        for index, sentence in enumerate(split_sentences(text)):
            row = {
                "text": sentence,
                "doc_sha": doc_sha,
                "chunk_idx": index,
                "source": source,
                "sha": _sha(sentence),
            }
            total += 1
            per_source[source] = per_source.get(source, 0) + 1
            if len(reservoir) < sample_size:
                reservoir.append(row)
            else:
                replacement = rng.randrange(total)
                if replacement < sample_size:
                    reservoir[replacement] = row
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "text": [row["text"] for row in reservoir],
                "doc_sha": [row["doc_sha"] for row in reservoir],
                "chunk_idx": pa.array(
                    [row["chunk_idx"] for row in reservoir], type=pa.int32()
                ),
                "source": [row["source"] for row in reservoir],
                "sha": [row["sha"] for row in reservoir],
            }
        ),
        out,
    )
    stats = {
        "n_documents": table.num_rows,
        "n_sentences_total": total,
        "n_sentences_sampled": len(reservoir),
        "seed": seed,
        "population_by_source": per_source,
    }
    out.with_suffix(".stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/corpus.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "data/corpus_sentences.parquet")
    parser.add_argument("--sample-size", type=int, default=130_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.out.exists():
        logger.error("refusing to overwrite existing corpus: %s", args.out)
        return 2
    logger.info("stats: %s", build(args.corpus, args.out, args.sample_size, args.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
