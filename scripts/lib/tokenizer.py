"""Pre-flight per-doc token-precise truncation.

OpenRouter caps inputs at the model's context window (32k for Qwen3
Embedding).  Naively sending a 167k-char doc gets you a HTTP 200 with
an error body and forces the runtime into the "skip with zero-vector
placeholder" path.  We avoid that by tokenising each doc locally with
the model's own ``tokenizer.json`` (the same one the provider would
use server-side) and truncating ahead of time.

Why not pull in ``transformers``? The Qwen3 ``tokenizer.json`` works
straight through the Rust ``tokenizers`` crate, and we already need
``huggingface_hub`` to fetch it.  Skipping ``transformers`` keeps the
dep footprint small (no ``torch``, no ``safetensors``, …).
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


# OpenRouter id → HuggingFace repo id.  Both Qwen3-Embedding-4B and
# Qwen3-Embedding-8B ship a byte-identical ``tokenizer.json`` (sha256
# ``83cdf8c3a34f6886…``) so loading either is fine in practice, but we
# keep the mapping explicit so future models / sizes are obvious.
OPENROUTER_TO_HF_TOKENIZER: dict[str, str] = {
    "qwen/qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen/qwen3-embedding-4b":   "Qwen/Qwen3-Embedding-4B",
    "qwen/qwen3-embedding-8b":   "Qwen/Qwen3-Embedding-8B",
}


def model_slug(model: str) -> str:
    """Filesystem-safe slug for an OpenRouter model id.

    ``qwen/qwen3-embedding-4b`` → ``qwen_qwen3-embedding-4b``.
    Used to name chunk dirs, cost reports, and skip logs.
    """
    return model.replace("/", "_").replace(":", "_")


def load_tokenizer(hf_repo_id: str):
    """Pull ``tokenizer.json`` from HF and load it via the Rust crate.

    Cached under ``~/.cache/huggingface/hub/`` on the first call, so
    repeated runs are zero-network after the initial fetch.
    """
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    path = hf_hub_download(repo_id=hf_repo_id, filename="tokenizer.json")
    return Tokenizer.from_file(path)


def truncate_to_tokens(
    texts: list[str], tokenizer, max_tokens: int,
) -> tuple[int, int, int]:
    """Mutate *texts* in place so no entry exceeds *max_tokens*.

    Returns ``(n_capped, max_seen_tokens, total_tokens)``.  Useful for
    logging the safety-cap impact: on the 45k-doc Polish corpus only
    ~25 docs are typically affected by a 30 000-token cap, so the
    pre-flight pass is cheap and obvious.
    """
    n_capped = 0
    max_seen = 0
    total = 0
    encs = tokenizer.encode_batch(texts, add_special_tokens=False)
    for i, enc in enumerate(encs):
        ids = enc.ids
        n = len(ids)
        total += n
        if n > max_seen:
            max_seen = n
        if n > max_tokens:
            texts[i] = tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)
            n_capped += 1
    return n_capped, max_seen, total


def resolve_and_apply_token_cap(
    texts: list[str], model: str, max_tokens: int,
    tokenizer_repo: str | None = None,
) -> tuple[int, int, int]:
    """One-shot helper: pick the right HF repo, load the tokenizer,
    truncate in place, log a summary, return ``(n_capped, max_seen,
    total)``.

    Raises ``SystemExit`` if there's no mapping for *model* and the
    caller didn't override ``tokenizer_repo``.
    """
    repo = tokenizer_repo or OPENROUTER_TO_HF_TOKENIZER.get(model.lower())
    if not repo:
        raise SystemExit(
            f"no tokenizer mapping for {model} — pass --tokenizer-repo "
            f"or extend OPENROUTER_TO_HF_TOKENIZER"
        )
    logger.info("loading tokenizer %s for token-precise truncation", repo)
    t0 = time.monotonic()
    tok = load_tokenizer(repo)
    n_capped, max_seen, total_tok = truncate_to_tokens(texts, tok, max_tokens)
    dt = time.monotonic() - t0
    logger.info(
        "tokenized %d docs in %.1fs: total=%s tokens, max=%d, "
        "truncated=%d at %d-tok cap",
        len(texts), dt, f"{total_tok:,}", max_seen, n_capped, max_tokens,
    )
    return n_capped, max_seen, total_tok
