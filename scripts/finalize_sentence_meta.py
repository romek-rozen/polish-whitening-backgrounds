"""Finalize metadata generated for the local sentence-granularity fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name", default="qwen3_4b_pl_mixed50k_sentences_mrl1024"
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8002/v1/embeddings")
    args = parser.parse_args()
    path = ROOT / "backgrounds" / args.name / f"{args.name}.meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    stats_path = ROOT / "data/corpus_sentences.stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "purpose": (
                "ZCA whitening background for Polish sentence embeddings from the "
                "pl_mixed50k corpus; use only when each indexed unit is one sentence."
            ),
            "granularity": "sentences",
            "embedding_endpoint": args.endpoint,
            "provider": "local_vllm",
            "build_script": (
                "scripts/build_corpus_sentences.py + scripts/embed_local_sentences.py + "
                "scripts/fit_zca.py"
            ),
            "sentence_population_size": stats["n_sentences_total"],
            "sample_seed": stats["seed"],
        }
    )
    meta["diagnostics"]["mrl_truncated"] = True
    meta["diagnostics"]["native_dim"] = 2560
    meta["notes"] = [
        "Unit of fit: one Polish sentence from entropy_lab.features.chunk.split_sentences.",
        "Sentence sample was drawn uniformly by reservoir sampling with seed 42.",
        "Native 2560-d vectors were sliced to 1024 and L2-renormalized before storage and fit.",
        "Apply: x_white = (x - mu) @ W, then L2-normalize for cosine similarity.",
    ]
    validation_path = ROOT / "data/validation_qwen3_4b_pl_mixed50k_sentences_mrl1024.json"
    if validation_path.is_file():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        meta["validation"] = {
            "heldout_sentence_count": validation["heldout_sentence_count"],
            "heldout_seed": 43,
            "heldout_diagnostics": validation["heldout_diagnostics"],
            "website": validation["domain"],
            "website_sentence_count": validation["sentence_count"],
            "website_metrics": validation["website_metrics"],
        }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
