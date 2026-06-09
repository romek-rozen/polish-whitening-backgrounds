"""Build a Polish text corpus parquet for whitening-background fitting.

Samples a deterministic mix of public Polish corpora and writes them to
``data/corpus.parquet`` as ``{source, text, sha, n_chars}`` rows. By
default applies **no character cap** — full documents go through.

Mix (matches the original ``polish_mixed_50k_v1`` background so that
downstream comparisons are apples-to-apples):

    wikipedia: 20 000 docs (wikimedia/wikipedia config 20231101.pl)
    mc4:       20 000 docs (allenai/c4 config pl)
    klej:       5 000 docs (allegro/klej-nkjp-ner + dyk + cdsc-r)
    oasst:      5 000 docs (OpenAssistant/oasst1 filtered lang=='pl';
                            yields ~156 in practice on the public dump)

Idempotent: skipped if ``data/corpus.parquet`` already exists.

Usage::

    python scripts/build_corpus.py
    python scripts/build_corpus.py --max-chars 8000   # optional cap

The output schema matches what ``embed_via_openrouter.py`` consumes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

logger = logging.getLogger("build_corpus")

DEFAULT_MIX = {
    "wikipedia": 20000,
    "mc4":       20000,
    "klej":       5000,
    "oasst":      5000,
}


def _maybe_truncate(text: str, max_chars: int | None) -> str:
    if max_chars is None:
        return text
    return text[:max_chars]


def _wiki_iter(seed: int, max_chars: int | None) -> Iterator[tuple[str, str]]:
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", name="20231101.pl",
                      split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    for row in ds:
        text = (row.get("text") or "").strip()
        if len(text) < 200:
            continue
        yield "wikipedia", _maybe_truncate(text, max_chars)


def _mc4_iter(seed: int, max_chars: int | None) -> Iterator[tuple[str, str]]:
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", name="pl",
                      split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    for row in ds:
        text = (row.get("text") or "").strip()
        if len(text) < 200:
            continue
        yield "mc4", _maybe_truncate(text, max_chars)


def _klej_iter(seed: int, max_chars: int | None) -> Iterator[tuple[str, str]]:
    """KLEJ — NKJP-NER + DYK + CDSC-R."""
    from datasets import load_dataset
    pool: list[tuple[str, str]] = []
    sub_specs = [
        ("allegro/klej-nkjp-ner", "sentence"),
        ("allegro/klej-dyk", "question"),
        ("allegro/klej-cdsc-r", "sentence_A"),
        ("allegro/klej-cdsc-r", "sentence_B"),
    ]
    for repo, field in sub_specs:
        try:
            ds = load_dataset(repo, split="train")
            for row in ds:
                t = (row.get(field) or "").strip()
                # KLEJ items are naturally short — keep a floor so we drop
                # one-word artefacts, but skip the upper cap entirely when
                # max_chars is None.
                if len(t) < 50:
                    continue
                if max_chars is not None and len(t) > max_chars:
                    continue
                pool.append(("klej", _maybe_truncate(t, max_chars)))
        except Exception as e:
            logger.warning("klej sub %s failed: %s", repo, e)
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    for item in pool:
        yield item


def _oasst_iter(seed: int, max_chars: int | None) -> Iterator[tuple[str, str]]:
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1",
                      split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    for row in ds:
        if row.get("lang") != "pl":
            continue
        text = (row.get("text") or "").strip()
        if len(text) < 100:
            continue
        yield "oasst", _maybe_truncate(text, max_chars)


SOURCES = {
    "wikipedia": _wiki_iter,
    "mc4":       _mc4_iter,
    "klej":      _klej_iter,
    "oasst":     _oasst_iter,
}


def build_corpus(out_dir: Path, mix: dict, seed: int,
                 max_chars: int | None) -> Path:
    out_path = out_dir / "corpus.parquet"
    if out_path.exists():
        logger.info("[skip] corpus exists: %s", out_path)
        return out_path
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for source_name, target_n in mix.items():
        loader = SOURCES[source_name]
        logger.info("loading %s: target=%d docs (cap=%s)",
                    source_name, target_n,
                    f"{max_chars} chars" if max_chars else "none")
        it = loader(seed=seed, max_chars=max_chars)
        bar = tqdm(total=target_n, desc=source_name, unit="doc")
        taken = 0
        for src, text in it:
            sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
            rows.append({
                "source": src, "text": text, "sha": sha,
                "n_chars": len(text),
            })
            taken += 1
            bar.update(1)
            if taken >= target_n:
                break
        bar.close()
        if taken < target_n:
            logger.warning("source %s yielded only %d / %d",
                           source_name, taken, target_n)

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path, compression="zstd")

    # Tiny manifest with the sampling recipe (helps reproducibility).
    fp = hashlib.sha256(
        "\n".join(sorted(r["sha"] for r in rows)).encode()
    ).hexdigest()
    (out_dir / "corpus_manifest.json").write_text(json.dumps({
        "mix_target": mix,
        "mix_actual": {s: sum(1 for r in rows if r["source"] == s) for s in mix},
        "seed": seed,
        "max_chars_per_doc": max_chars,
        "n_total": len(rows),
        "n_chars_total": sum(r["n_chars"] for r in rows),
        "corpus_fingerprint_sha256": fp,
    }, indent=2))

    logger.info("wrote corpus parquet: %s (%d rows, %.1f MB chars, fp=%s)",
                out_path, len(rows),
                sum(r["n_chars"] for r in rows) / 1e6, fp[:16])
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DATA_DIR,
                    help="Output dir (default: ./data).")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for HF streaming shuffle (matches v1).")
    ap.add_argument("--max-chars", type=int, default=None,
                    help="Optional per-document char cap. Default: no cap.")
    ap.add_argument("--wiki", type=int, default=DEFAULT_MIX["wikipedia"])
    ap.add_argument("--mc4", type=int, default=DEFAULT_MIX["mc4"])
    ap.add_argument("--klej", type=int, default=DEFAULT_MIX["klej"])
    ap.add_argument("--oasst", type=int, default=DEFAULT_MIX["oasst"])
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    mix = {
        "wikipedia": args.wiki,
        "mc4": args.mc4,
        "klej": args.klej,
        "oasst": args.oasst,
    }
    build_corpus(args.out, mix=mix, seed=args.seed, max_chars=args.max_chars)
    return 0


if __name__ == "__main__":
    sys.exit(main())
