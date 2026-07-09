"""Build the PES source of the Polish medical corpus.

Downloads the Hugging Face dataset
``amu-cai/medical-exams-PES-PL-2007-2024`` — 180k questions from the
Polish Board Certification Examinations (Państwowy Egzamin
Specjalizacyjny), 84 medical specialties, 2007-2024, sourced from
cem.edu.pl — and reshapes each question into a corpus row.

This is the short-form medical-Polish source (question stems + A-E
options, ~100-600 chars each): the natural fit for a `kw` / sentence
scale medical background, complementing the long-form ChPL source
(``build_med_pl_corpus_chpl.py``).

We pull the Hub's auto-converted parquet (no heavy ``datasets``
dependency), keep only the text, and write the canonical corpus
schema ``{source, text, sha, n_chars}`` to ``data/med_pl/pes.parquet``
with ``source = "pes_cem"``.

Note on rights: the exam questions are published by CEM; amu-cai state
"We do not own the rights to this dataset; we processed and published
it here."  This corpus is for whitening-background fitting (statistical
covariance only — no redistribution of the questions).

Usage::

    python scripts/build_med_pl_corpus_pes.py
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MED_DIR = DATA_DIR / "med_pl"

SOURCE = "pes_cem"
PARQUET_URL = (
    "https://huggingface.co/api/datasets/"
    "amu-cai/medical-exams-PES-PL-2007-2024/parquet/default/train/0.parquet"
)
# The text column; falls back through these if the schema shifts.
TEXT_COLS = ("question_w_options", "question", "text")

logger = logging.getLogger("build_med_pl_corpus_pes")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _download(url: str, dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 0:
        logger.info("raw parquet cached: %s (%.1f MB)",
                    dest, dest.stat().st_size / 1e6)
        return dest
    logger.info("downloading %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    logger.info("saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def build(raw_path: Path, out_path: Path, min_chars: int) -> dict:
    table = pq.read_table(raw_path)
    cols = table.column_names
    text_col = next((c for c in TEXT_COLS if c in cols), None)
    if text_col is None:
        raise SystemExit(f"no text column among {TEXT_COLS} in {cols}")
    logger.info("raw rows=%d, using text column %r (schema: %s)",
                table.num_rows, text_col, cols)

    texts = table.column(text_col).to_pylist()
    seen: set[str] = set()
    out_text: list[str] = []
    kept = dropped = dupes = 0
    for t in texts:
        if not t:
            dropped += 1
            continue
        t = t.strip()
        if len(t) < min_chars:
            dropped += 1
            continue
        sha = _sha(t)
        if sha in seen:
            dupes += 1
            continue
        seen.add(sha)
        out_text.append(t)
        kept += 1

    out = pa.table({
        "source": [SOURCE] * len(out_text),
        "text": out_text,
        "sha": [_sha(t) for t in out_text],
        "n_chars": pa.array([len(t) for t in out_text], type=pa.int64()),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, out_path)
    chars = [len(t) for t in out_text]
    return {
        "kept": kept, "dropped": dropped, "dupes": dupes,
        "min_chars": min(chars) if chars else 0,
        "median_chars": int(sorted(chars)[len(chars)//2]) if chars else 0,
        "max_chars": max(chars) if chars else 0,
        "out": str(out_path),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=MED_DIR / "pes_raw.parquet")
    ap.add_argument("--out", type=Path, default=MED_DIR / "pes.parquet")
    ap.add_argument("--min-chars", type=int, default=60,
                    help="Drop questions shorter than this (default 60).")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    raw = _download(PARQUET_URL, args.raw)
    stats = build(raw, args.out, args.min_chars)
    logger.info("stats: %s", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
