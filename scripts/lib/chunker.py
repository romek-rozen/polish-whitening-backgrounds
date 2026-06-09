"""Sentence-aware document chunker for v3 paragraph-level backgrounds.

Wraps LangChain's :class:`RecursiveCharacterTextSplitter` with a
length function that counts **Qwen3 tokens** (via the same
``tokenizer.json`` we use for pre-flight truncation in
:mod:`scripts.lib.tokenizer`).  Net effect: ``chunk_size=512`` means
512 tokens — not 512 characters — so the chunks line up with what
the embedding model actually consumes.

Why this splitter rather than a hand-rolled regex chain:

- Recursive fallback over ``["\\n\\n", "\\n", ". ", "? ", "! ", " ",
  ""]`` means we try paragraph boundaries first and only descend to
  finer granularity when a paragraph is too big.  Single words never
  get cut in half on natural prose.
- ``keep_separator=True`` keeps the sentence-final punctuation at
  the **end** of the previous chunk, so chunks read as complete
  sentences instead of trailing off mid-phrase.
- Overlap is built in — adjacent chunks share their boundary
  sentences so a fact straddling a chunk boundary lives whole in
  at least one chunk.

The whole point of pulling this into ``lib/`` is that the
**same** splitter is used at fit time (to build the corpus) and at
inference time (to chunk new docs before embedding + whitening).
Different splitters → different Σ → wrong whitening — see
``GOTCHAS.md`` §1.
"""
from __future__ import annotations

import logging

from .tokenizer import OPENROUTER_TO_HF_TOKENIZER, load_tokenizer

logger = logging.getLogger(__name__)


# Order matters: recursive fallback tries these top-to-bottom.  We
# want to break at paragraph boundaries before we descend to single
# newlines, sentence ends, then whitespace, then characters (last
# resort, only triggered by a single token-dense word longer than the
# chunk size — vanishingly rare on FineWeb-2 PL).
DEFAULT_SEPARATORS: list[str] = [
    "\n\n", "\n", ". ", "? ", "! ", " ", "",
]


def make_splitter(
    model: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: list[str] | None = None,
    tokenizer_repo: str | None = None,
):
    """Build a sentence-aware splitter sized in **Qwen3 tokens**.

    Parameters
    ----------
    model
        OpenRouter model id (e.g. ``qwen/qwen3-embedding-4b``).
        Used to resolve the matching HF tokenizer.json.
    chunk_size, chunk_overlap
        In **tokens**, not characters.  Defaults match the typical
        RAG window (512/64) and are well below Qwen3's 32k context.
    separators
        Override the recursive-fallback list.  Defaults to
        :data:`DEFAULT_SEPARATORS`.
    tokenizer_repo
        Override the HF repo id for the tokenizer (when *model*
        isn't in :data:`OPENROUTER_TO_HF_TOKENIZER`).
    """
    # Local import keeps the dep optional: importing ``lib`` doesn't
    # pull langchain unless you actually build a splitter.
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    repo = tokenizer_repo or OPENROUTER_TO_HF_TOKENIZER.get(model.lower())
    if not repo:
        raise SystemExit(
            f"no tokenizer mapping for {model} — pass tokenizer_repo "
            f"or extend OPENROUTER_TO_HF_TOKENIZER"
        )
    logger.info("loading tokenizer %s for chunker", repo)
    tok = load_tokenizer(repo)

    def _len(s: str) -> int:
        # add_special_tokens=False because chunks get composed before
        # the embedder adds BOS/EOS itself — we count payload only.
        return len(tok.encode(s, add_special_tokens=False).ids)

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_len,
        separators=separators or DEFAULT_SEPARATORS,
        keep_separator=True,
    )
