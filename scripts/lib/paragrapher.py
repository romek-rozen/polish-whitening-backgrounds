"""Paragraph-level splitter for the `_paragraphs_` backgrounds.

A **paragraph** is the finest structural unit above a sentence: the
block of text between two blank lines (`\\n\\n`).  It sits below both
`lib.chunker` (fixed 512-token RAG windows) and `lib.segmenter`
(whole H2/H3-scale sections) in the length hierarchy — median ~430
chars on the fit corpus, roughly half a chunk — and it is a
measurably distinct embedding distribution: whitening paragraphs with
a `_chunks_` background leaves ~7x the residual anisotropy of a proper
`_paragraphs_` background (top_ev/mean ~13.5 vs ~2.0 on Qwen3-4B), so
the granularity contract in GOTCHAS.md §1 applies here too.

Differences from :mod:`lib.chunker` and :mod:`lib.segmenter`:

- The primary boundary is the **blank line** (`\\n\\n`) and each
  paragraph becomes its own row.  Unlike the segmenter — which packs
  consecutive paragraphs up to a 1024-token budget — this splitter
  never merges *across* paragraph boundaries, so short adjacent
  paragraphs stay separate (that separation is exactly the finer
  granularity we're fitting for).
- ``PARA_MAX_TOKENS = 512``: a paragraph longer than the cap (a wall
  of text with no blank lines) is descended into at sentence
  boundaries so no single row blows the embedder's context or drifts
  into chunk-scale length.  Normal paragraphs pass through untouched.
- ``merge_tiny`` runs with ``min_chars=120``: a bare heading or a
  one-word line is not a paragraph; forward-merging folds it into the
  paragraph body that follows (same trick the other two splitters use,
  smaller floor because real paragraphs legitimately run short).

Granularity contract (GOTCHAS.md §1) applies unchanged: whiten
paragraph embeddings only with a ``_paragraphs_`` background, and
split your production text with **this same splitter** (blank-line
paragraphs — which is what any Markdown/HTML renderer already treats
as `<p>` boundaries).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


PARA_MAX_TOKENS = 512
PARA_MIN_CHARS = 120

# Oversize-paragraph descent order: no "\n\n" here — we've already
# split on it, and we feed one paragraph at a time so the recursive
# splitter can only *subdivide* an over-long paragraph, never merge
# across the blank-line boundaries we just cut on.
PARA_SEPARATORS: list[str] = [
    "\n", ". ", "? ", "! ", "; ", ", ", " ", "",
]


def make_paragrapher(
    model: str,
    para_max: int = PARA_MAX_TOKENS,
    tokenizer_repo: str | None = None,
):
    """Build the oversize-paragraph splitter, sized in **Qwen3 tokens**.

    Thin wrapper over :func:`lib.chunker.make_splitter` (shared
    tokenizer resolution + token-length function); only the
    separators / size / zero overlap differ.  This splitter is applied
    *per paragraph* by :func:`paragraph_text`, so its only job is to
    subdivide the rare paragraph that exceeds ``para_max`` tokens.
    """
    from .chunker import make_splitter

    return make_splitter(
        model,
        chunk_size=para_max,
        chunk_overlap=0,
        separators=PARA_SEPARATORS,
        tokenizer_repo=tokenizer_repo,
    )


def paragraph_text(splitter, text: str) -> list[str]:
    """Split one document into cleaned, one-per-row paragraphs.

    Blank-line split first, then each paragraph is passed through
    *splitter* — which returns it unchanged when it fits under
    ``PARA_MAX_TOKENS`` and descends to sentence boundaries only when
    it doesn't.  Post-processing mirrors the chunk/segment pipeline:
    ``merge_tiny`` folds heading-only / one-liner splits forward into
    the paragraph they introduce, and ``strip_overlap_fragments``
    drops any leading ``". "``-style fragment a sentence-level descent
    can leave behind.
    """
    from .chunker import merge_tiny, strip_overlap_fragments

    pieces: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        # split_text returns [para] verbatim when it's under the token
        # cap; only oversize walls of text get subdivided.
        pieces.extend(splitter.split_text(para))

    return strip_overlap_fragments(merge_tiny(pieces, min_chars=PARA_MIN_CHARS))
