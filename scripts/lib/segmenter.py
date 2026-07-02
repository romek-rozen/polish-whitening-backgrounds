"""Section-level segmenter for the `_segments_` backgrounds.

A **segment** is the unit of internal-linking retrieval: one topically
coherent section of an article (the thing under one H2/H3 in a real
markdown post), as opposed to a fixed 512-token RAG window
(`lib.chunker`) or a whole document.  Internal linking works
segment→segment: the source article's sections are the queries, the
target documents are represented by *their* sections, and a target
doc's score is an aggregate (max / top-k mean) over its segments —
so both sides of the cosine live in the same whitened space.

Differences from :mod:`lib.chunker`:

- ``chunk_size=1024`` tokens (sections run longer than RAG windows),
  ``chunk_overlap=0`` — real article sections don't overlap, so the
  fit distribution shouldn't either.
- The separator list tries **markdown headings first**
  (``\\n## ``, ``\\n### ``) before paragraph boundaries.  On the fit
  corpus (Wikipedia PL / FineWeb-2 plain text — no markdown) these
  simply never match and the splitter falls through to ``\\n\\n``,
  which packs whole paragraphs up to the token budget.  On real
  markdown articles at inference time the same splitter breaks at
  section boundaries.  One splitter, both worlds — no heading
  *detection* heuristics (see GOTCHAS.md §6 rule 3 for why we don't
  try to be clever about structure).
- ``merge_tiny`` runs with ``min_chars=300`` (vs 100 for chunks): a
  bare heading or one-liner is not a section; forward-merging makes
  it the opening line of the section body that follows.

Granularity contract (GOTCHAS.md §1) applies unchanged: whiten
segment embeddings only with a ``_segments_`` background, and segment
your production articles with **this same splitter** (or your CMS's
real H2/H3 boundaries — which is what the heading-first separators
emulate).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


SEGMENT_SIZE_TOKENS = 1024
SEGMENT_MIN_CHARS = 300

# Heading separators first: with keep_separator=True LangChain prepends
# the matched separator to the *following* split, so "## Nagłówek"
# opens its own section instead of dangling at the tail of the previous
# one (verified empirically; the standalone-heading wart that remains
# is exactly what merge_tiny fixes).  H1 ("\n# ") is included for
# multi-article dumps; a title-only H1 at position 0 has no leading
# newline and never matches.  H4+ is rare enough that "\n\n" handles it.
SEGMENT_SEPARATORS: list[str] = [
    "\n# ", "\n## ", "\n### ",
    "\n\n", "\n", ". ", "? ", "! ", " ", "",
]


def make_segmenter(
    model: str,
    segment_size: int = SEGMENT_SIZE_TOKENS,
    tokenizer_repo: str | None = None,
):
    """Build the section-level splitter, sized in **Qwen3 tokens**.

    Thin wrapper over :func:`lib.chunker.make_splitter` so tokenizer
    resolution and the token-counting length function stay in one
    place; only the separators / size / overlap differ.
    """
    from .chunker import make_splitter

    return make_splitter(
        model,
        chunk_size=segment_size,
        chunk_overlap=0,
        separators=SEGMENT_SEPARATORS,
        tokenizer_repo=tokenizer_repo,
    )


def segment_text(splitter, text: str) -> list[str]:
    """Split one document into cleaned segments.

    Post-processing mirrors the chunk pipeline: ``merge_tiny`` folds
    heading-only / one-liner splits into the section they introduce,
    and ``strip_overlap_fragments`` drops the leading ``". "``-style
    fragments that ``keep_separator`` can leave at the start of a
    split when the splitter had to descend to sentence separators
    (only happens inside \\n\\n-free walls of text longer than the
    segment budget).
    """
    from .chunker import merge_tiny, strip_overlap_fragments

    raw = splitter.split_text(text)
    return strip_overlap_fragments(merge_tiny(raw, min_chars=SEGMENT_MIN_CHARS))
