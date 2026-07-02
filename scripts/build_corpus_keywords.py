"""Extract keyword-like phrases from the v2 document corpus.

Reads ``data/corpus.parquet`` (produced by ``build_corpus.py``) and
writes ``data/corpus_keywords.parquet`` where every row is one short
Polish phrase (1-5 words) mined from the document texts.  These
phrases approximate the embedding-space distribution of Google Ads /
search keywords, which is *very* different from whole documents or
512-token chunks — see GOTCHAS.md §1 on why granularity must match.

Output schema (same columns as corpus.parquet so the downstream
embed step works unchanged)::

    source     : str    # "ngram_pl_mixed50k"
    text       : str    # the phrase, lowercased
    sha        : str    # sha256 of the phrase text
    n_chars    : int32

Extraction method:

- lowercase, sentence-split on terminal punctuation, word-split on a
  Polish-aware token regex,
- candidate n-grams (n = 1..5) that do not start or end with a
  stopword, contain no bare numbers at the edges, and are 3-60 chars,
- document frequency counted with periodic pruning of singletons
  (memory bound, slightly lossy for rare phrases — irrelevant here
  because we only keep phrases with df >= --min-df anyway),
- final sample of --target phrases stratified by phrase length to
  mimic a realistic keyword-list mix (10% 1-word, 35% 2-word,
  30% 3-word, 15% 4-word, 10% 5-word), deterministic via --seed.

Usage::

    python scripts/build_corpus_keywords.py
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

logger = logging.getLogger("build_corpus_keywords")

# Stopwords reject n-grams that *start or end* with one ("do sklepu"
# bad, "buty do biegania" fine).  Primary source is the 328-word
# stopwords-iso/stopwords-pl list (data/stopwords-pl.txt, downloaded
# from https://github.com/stopwords-iso/stopwords-pl); the tiny
# fallback below only exists so the script still runs without it.
_FALLBACK_STOPWORDS = frozenset("""
a aby ale albo aż bardzo bez bo być by był była było były co coś czy
często dla do gdy gdyby gdzie go i ich im ile jak jaka jaki jakie jako
je jego jej jest jeszcze jeśli już ją kiedy kto która które który
których lub ma mają miał mnie mogą może można mu na nad nam nas nawet
nie niż no o od ona one oni ono oraz po pod ponad ponieważ przed przez
przy się są ta tak takie taki także tam te tego tej ten teraz też to
tu tych tylko tym u w we więc wszystko z za zawsze że żeby ów jednak
został została zostały zostało być będzie były
""".split())

# A few English function words — FineWeb-2 PL has stray EN fragments
# and stopwords-pl obviously doesn't cover them.
_EN_STOPWORDS = frozenset("the of and in a an to for on with is".split())


def load_stopwords(path: Path) -> frozenset:
    if path.is_file():
        words = frozenset(
            w.strip().lower() for w in path.read_text("utf-8").splitlines()
            if w.strip()
        )
        logger.info("loaded %d stopwords from %s", len(words), path)
        return words | _EN_STOPWORDS
    logger.warning("stopword file missing (%s) — using tiny fallback", path)
    return _FALLBACK_STOPWORDS | _EN_STOPWORDS


STOPWORDS = _FALLBACK_STOPWORDS | _EN_STOPWORDS  # replaced in main()

# Numbers ARE tokens — otherwise "wydany 12 lipca" silently collapses
# into the unnatural bigram "wydany lipca".  Bare-number tokens are
# rejected at the phrase edges (and as unigrams) in doc_phrases().
_WORD_RE = re.compile(r"[a-ząćęłńóśźż0-9][a-ząćęłńóśźż0-9\-]*", re.UNICODE)
_NUM_RE = re.compile(r"[0-9][0-9\-]*$")
_SENT_RE = re.compile(r"[.!?;:\n()\[\]{}«»\"„”]+")

NGRAM_RANGE = (1, 5)
# Target mix by phrase word-count, roughly matching real keyword lists.
LENGTH_MIX = {1: 0.10, 2: 0.35, 3: 0.30, 4: 0.15, 5: 0.10}


def doc_phrases(text: str) -> set[str]:
    """All acceptable candidate phrases in one document (deduped)."""
    out: set[str] = set()
    for sent in _SENT_RE.split(text.lower()):
        words = _WORD_RE.findall(sent)
        n_words = len(words)
        for n in range(NGRAM_RANGE[0], NGRAM_RANGE[1] + 1):
            for i in range(n_words - n + 1):
                gram = words[i:i + n]
                if gram[0] in STOPWORDS or gram[-1] in STOPWORDS:
                    continue
                if _NUM_RE.match(gram[0]) or _NUM_RE.match(gram[-1]):
                    continue
                # Unigrams: require a real word, not too short.
                if n == 1 and len(gram[0]) < 4:
                    continue
                if len(gram[0]) < 2 or len(gram[-1]) < 2:
                    continue
                phrase = " ".join(gram)
                if not 3 <= len(phrase) <= 60:
                    continue
                out.add(phrase)
    return out


def count_df(
    texts, n_docs: int, prune_every: int, prune_above: int,
) -> Counter:
    """Document-frequency counter with periodic singleton pruning."""
    df: Counter = Counter()
    for i, text in enumerate(tqdm(texts, total=n_docs, desc="mine", unit="doc")):
        df.update(doc_phrases(text))
        if (i + 1) % prune_every == 0 and len(df) > prune_above:
            before = len(df)
            df = Counter({k: v for k, v in df.items() if v > 1})
            logger.info(
                "pruned singletons at doc %d: %d -> %d entries",
                i + 1, before, len(df),
            )
    return df


def stratified_sample(
    df: Counter, target: int, min_df: int, seed: int,
) -> list[str]:
    """Sample *target* phrases with the LENGTH_MIX word-count mix.

    Within each length bucket we sample uniformly from all phrases
    with df >= min_df — NOT by frequency, which would over-select
    generic boilerplate ("strona główna", "polityka prywatności").
    Deficits in one bucket spill over into the others.
    """
    rng = random.Random(seed)
    buckets: dict[int, list[str]] = {n: [] for n in LENGTH_MIX}
    for phrase, count in df.items():
        if count < min_df:
            continue
        n = phrase.count(" ") + 1
        buckets[n].append(phrase)
    for n, b in sorted(buckets.items()):
        logger.info("bucket %d-word: %d candidates (df>=%d)", n, len(b), min_df)

    picked: list[str] = []
    deficit = 0
    # Largest buckets last so deficits can spill into them.
    order = sorted(LENGTH_MIX, key=lambda n: len(buckets[n]))
    for n in order:
        want = int(round(target * LENGTH_MIX[n])) + deficit
        b = buckets[n]
        take = min(want, len(b))
        deficit = want - take
        picked.extend(rng.sample(b, take))
    if deficit:
        logger.warning("corpus too small for target: short by %d", deficit)
    rng.shuffle(picked)
    return picked[:target]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--corpus", type=Path, default=DATA_DIR / "corpus.parquet",
        help="Input corpus parquet (default: ./data/corpus.parquet).",
    )
    ap.add_argument(
        "--out", type=Path, default=DATA_DIR / "corpus_keywords.parquet",
        help="Output parquet (default: ./data/corpus_keywords.parquet).",
    )
    ap.add_argument("--target", type=int, default=50_000)
    ap.add_argument(
        "--min-df", type=int, default=3,
        help="Minimum document frequency for a phrase to be eligible.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--stopwords", type=Path, default=DATA_DIR / "stopwords-pl.txt",
        help="One-word-per-line stopword list "
             "(default: ./data/stopwords-pl.txt from stopwords-iso).",
    )
    ap.add_argument("--prune-every", type=int, default=2_000)
    ap.add_argument(
        "--prune-above", type=int, default=5_000_000,
        help="Only prune singletons when the counter exceeds this size.",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.out.exists():
        logger.info("output already exists: %s — skipping", args.out)
        return 0

    global STOPWORDS
    STOPWORDS = load_stopwords(args.stopwords)

    table = pq.read_table(args.corpus, columns=["text"])
    texts = table.column("text").to_pylist()
    logger.info("mining phrases from %d docs", len(texts))

    df = count_df(texts, len(texts), args.prune_every, args.prune_above)
    logger.info("counter holds %d distinct phrases", len(df))

    phrases = stratified_sample(df, args.target, args.min_df, args.seed)
    logger.info("sampled %d phrases", len(phrases))

    out_table = pa.table({
        "source": pa.array(["ngram_pl_mixed50k"] * len(phrases)),
        "text": pa.array(phrases),
        "sha": pa.array([_sha(p) for p in phrases]),
        "n_chars": pa.array([len(p) for p in phrases], type=pa.int32()),
    })
    pq.write_table(out_table, args.out)
    logger.info("DONE %d phrases → %s", len(phrases), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
