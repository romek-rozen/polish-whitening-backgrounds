"""Build a mixed Polish short-phrase corpus for fitting keyword backgrounds.

**This is a NEW corpus, not the one behind the shipped `*_kw_*` backgrounds.**
Those were fitted on `data/corpus_keywords.parquet`: 50 000 phrases mined as
n-grams from Polish web text only. A background fitted here is therefore NOT
comparable to them and must carry a different name (`pl_kwmix*`, not
`pl_mixed50k_kw`).

Why a mix at all: a whitening background models the *distribution* of what you
will whiten at inference. Web n-grams alone are a poor stand-in for search
keywords — they are full of grammatical fragments (`operacja rozpoczęła`,
`europejskiego i rady z dnia`) that nobody ever types into a search box.
Combining sources of different shape covers the space better than any one of
them:

===================  ==========================================================
source               what it contributes
===================  ==========================================================
``ngrams``           natural web language, long tail, realistic noise
``wiki_titles``      native Polish noun phrases — the canonical keyword *shape*
                     (no finite verbs), which is exactly what POS filtering
                     would have produced, minus the tagger and its errors
``msmarco``          real search-query shape and intent (translated from
                     English, so the phrasing is translationese and the topics
                     skew Anglo — a known, accepted limitation)
===================  ==========================================================

Every source is reproducible from public data: a Wikipedia dump, a HuggingFace
dataset, and the repo's own miner. No account-gated or proprietary keyword data.

**Casing**: everything is lowercased. bge-m3, Qwen3-Embedding and
embeddinggemma are all case-sensitive (SentencePiece), so mixing cased
Wikipedia titles (`Bitwa pod Grunwaldem`) with lowercase n-grams would make the
background model a bimodal distribution matching neither. Google Ads keyword
lists are lowercase by convention, so lowercase is also what inference will
see. Consistency with the inference input matters more than the choice itself.

Usage::

    python build_mixed_corpus.py \\
        --ngrams work/corpus_keywords_580k.parquet \\
        --wiki-titles work/sources/plwiki-titles.gz \\
        --msmarco work/sources/msmarco_pl_queries.txt \\
        --target 900000 --out work/corpus_kwmix.parquet
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("build_mixed_corpus")

# Same length band and mix as the repo's keyword miner, so this corpus stays in
# the `kw` granularity contract (see parent GOTCHAS.md §1 on why granularity
# must match).
LENGTH_MIX = {1: 0.10, 2: 0.35, 3: 0.30, 4: 0.15, 5: 0.10}
MIN_CHARS, MAX_CHARS = 3, 60

_WS_RE = re.compile(r"\s+")
# Wikipedia disambiguation/qualifier suffix: "Zamek (Poznań)" -> "Zamek".
_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")
# A phrase must be mostly letters; drop pure punctuation/number soup.
_HAS_LETTER_RE = re.compile(r"[a-ząćęłńóśźż]")
_ALLOWED_RE = re.compile(r"^[a-ząćęłńóśźż0-9][a-ząćęłńóśźż0-9 \-'/&.]*$")


def normalize(phrase: str) -> str | None:
    """Lowercase, strip decoration, and reject anything not keyword-shaped."""
    text = unicodedata.normalize("NFC", phrase).replace("_", " ").strip().lower()
    text = _PAREN_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip(" -–—,.;:!?\"'")
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return None
    if not _HAS_LETTER_RE.search(text):
        return None
    if not _ALLOWED_RE.match(text):
        return None
    if not 1 <= len(text.split()) <= max(LENGTH_MIX):
        return None
    return text


def read_ngrams(path: Path) -> list[str]:
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    logger.info("ngrams: %d raw rows from %s", len(texts), path.name)
    return texts


def read_wiki_titles(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    out: list[str] = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            # Header row of the dump, plus namespace-ish leftovers.
            if not line or line == "page_title" or ":" in line:
                continue
            out.append(line)
    logger.info("wiki_titles: %d raw titles from %s", len(out), path.name)
    return out


def read_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        out = [line.strip() for line in handle if line.strip()]
    logger.info("msmarco: %d raw queries from %s", len(out), path.name)
    return out


def bucket_by_length(
    phrases: list[str], source: str, seen: set[str], buckets: dict[int, list[tuple[str, str]]]
) -> int:
    """Normalise, dedup globally, and file each phrase under its word count."""
    kept = 0
    for raw in phrases:
        text = normalize(raw)
        if text is None or text in seen:
            continue
        seen.add(text)
        buckets[len(text.split())].append((text, source))
        kept += 1
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ngrams", type=Path, default=None)
    ap.add_argument("--wiki-titles", type=Path, default=None)
    ap.add_argument("--msmarco", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", type=int, default=900_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    rng = random.Random(args.seed)

    seen: set[str] = set()
    buckets: dict[int, list[tuple[str, str]]] = defaultdict(list)

    # Source order is also dedup priority: the first source to claim a phrase
    # keeps it. n-grams first so the web-language tail is not crowded out by
    # the (much larger) title dump.
    for label, path, reader in (
        ("ngram_web", args.ngrams, read_ngrams),
        ("wiki_title", args.wiki_titles, read_wiki_titles),
        ("msmarco_query", args.msmarco, read_lines),
    ):
        if path is None:
            logger.warning("%s: not provided — skipping", label)
            continue
        if not path.exists():
            raise SystemExit(f"{label}: {path} does not exist")
        kept = bucket_by_length(reader(path), label, seen, buckets)
        logger.info("%s: %d phrases kept after normalise+dedup", label, kept)

    for n in sorted(LENGTH_MIX):
        logger.info("bucket %d-word: %d candidates", n, len(buckets[n]))

    # The smallest bucket relative to its share caps a mix-preserving sample.
    feasible = min(
        int(len(buckets[n]) / share) for n, share in LENGTH_MIX.items() if buckets[n]
    )
    if feasible < args.target:
        logger.warning(
            "target %d would distort the length mix; capping at %d "
            "(binding bucket decides)", args.target, feasible,
        )
    target = min(args.target, feasible)

    picked: list[tuple[str, str]] = []
    for n, share in LENGTH_MIX.items():
        want = int(round(target * share))
        pool = buckets[n]
        rng.shuffle(pool)
        picked.extend(pool[:want])
    rng.shuffle(picked)

    by_source = Counter(source for _, source in picked)
    by_length = Counter(len(text.split()) for text, _ in picked)
    logger.info("picked %d phrases", len(picked))
    logger.info("by source: %s", dict(by_source))
    logger.info("by length: %s", dict(sorted(by_length.items())))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "source": pa.array([source for _, source in picked], pa.string()),
            "text": pa.array([text for text, _ in picked], pa.string()),
            "sha": pa.array(
                [hashlib.sha256(text.encode()).hexdigest() for text, _ in picked], pa.string()
            ),
            "n_chars": pa.array([len(text) for text, _ in picked], pa.int32()),
        }
    )
    pq.write_table(table, args.out)

    stats = {
        "out": str(args.out),
        "n_phrases": len(picked),
        "by_source": dict(by_source),
        "by_length": dict(sorted(by_length.items())),
        "lowercased": True,
        "seed": args.seed,
        "note": "NEW corpus — not the 50k web-ngram corpus behind the shipped "
                "*_pl_mixed50k_kw_* backgrounds. Not comparable to them.",
    }
    args.out.with_suffix(".stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("DONE %d phrases → %s", len(picked), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
