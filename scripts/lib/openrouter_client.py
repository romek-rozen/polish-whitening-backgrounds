"""Single embeddings POST against OpenRouter, plus the helpers needed
to decide what to do when it doesn't return clean ``{"data": [...]}``.

Why this is its own module:

- The retry / adaptive-batch loop lives in the calling script.  That
  loop just wants a function it can call and a clean signal of "this
  is transient" vs "this is a real 4xx".  Everything that mucks with
  HTTP semantics belongs here.
- OpenRouter sometimes returns HTTP 200 with ``{"error": {...}}``
  instead of ``{"data": [...]}`` — either a real upstream error
  (``code: 400`` / context overflow) or a temporary provider hiccup
  (``code: 429`` / "engine_overloaded").  We re-classify those into a
  proper ``requests.HTTPError`` with a synthetic status code so the
  caller's existing TRANSIENT_STATUSES path or 400-shrink path picks
  them up correctly.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import requests

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"

# Status codes the caller should treat as "wait and retry, don't
# shrink the batch", per OpenRouter docs (429 + the standard 5xx
# transient set).
TRANSIENT_STATUSES: set[int] = {408, 429, 500, 502, 503, 504, 529}


def _build_provider_block(
    provider_order: list[str] | None,
    ignore_providers: list[str] | None,
) -> dict[str, Any] | None:
    """Compose the optional ``provider`` payload sub-object.

    Returns ``None`` when neither pin nor exclusion is set — letting
    OpenRouter apply its default price-weighted load balancing.
    """
    block: dict[str, Any] = {}
    if provider_order:
        # Pin specific providers first; allow_fallbacks lets the next
        # cheapest healthy option pick up when the pinned ones throttle.
        block["order"] = provider_order
        block["allow_fallbacks"] = True
    if ignore_providers:
        # Hard-exclude providers we don't want any spend going to (e.g.
        # SiliconFlow, which is ~4× the price of nebius / deepinfra
        # for Qwen3-Embedding).
        block["ignore"] = ignore_providers
    return block or None


def post_embed_batch(
    session: requests.Session,
    api_key: str,
    model: str,
    texts: list[str],
    timeout: float,
    provider_order: list[str] | None = None,
    ignore_providers: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Send one batch to ``/embeddings``.

    Returns ``(arr, usage)`` where ``arr`` is ``(len(texts), dim)``
    float32 and ``usage`` is the raw ``body["usage"]`` dict (may be
    empty if the provider didn't include one).

    Raises ``requests.HTTPError`` on every non-2xx **and** on
    "HTTP 200 + error body" responses (see module docstring).  The
    synthetic 4xx / 5xx status on the latter is chosen so that
    ``status in TRANSIENT_STATUSES`` does the right thing in the
    caller's retry loop.
    """
    payload: dict[str, Any] = {
        "model": model,
        "input": texts,
        "encoding_format": "float",
    }
    block = _build_provider_block(provider_order, ignore_providers)
    if block is not None:
        payload["provider"] = block
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional analytics tag — harmless if dropped.
        "X-Title": "polish-whitening-backgrounds",
    }

    r = session.post(OPENROUTER_URL, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    body = r.json()

    # Happy path: providers return {"data": [{"embedding": [...]}, …]}.
    if "data" in body:
        arr = np.array([d["embedding"] for d in body["data"]], dtype=np.float32)
        usage = body.get("usage") or {}
        return arr, usage

    # 200-but-no-data: two failure modes hide here.
    #
    #   1. Doc exceeds the provider's max context — typically surfaces
    #      as a 40x in the inner body (or no inner code).  Caller
    #      should shrink, and eventually skip with a zero-vector
    #      placeholder.
    #   2. Provider is overloaded — surfaces as inner ``code: 429`` (or
    #      a 5xx) with a "Model busy, retry later" message.  Fully
    #      transient; shrinking + skipping would silently throw away
    #      perfectly good documents.
    #
    # Pull the inner code out and mirror it on the synthesised
    # Response.  The existing TRANSIENT_STATUSES path triggers for
    # case 2; case 1 falls through to the 400-shrink path.
    err = body.get("error") or body
    inner_code = err.get("code") if isinstance(err, dict) else None
    try:
        inner_code = int(inner_code) if inner_code is not None else None
    except (TypeError, ValueError):
        inner_code = None
    fake_status = (
        inner_code
        if isinstance(inner_code, int) and inner_code in TRANSIENT_STATUSES
        else 400
    )
    fake = requests.Response()
    fake.status_code = fake_status
    fake._content = json.dumps(body).encode()
    raise requests.HTTPError(
        f"OpenRouter 200-but-no-data (mapped to HTTP {fake_status}): {err}",
        response=fake,
    )
