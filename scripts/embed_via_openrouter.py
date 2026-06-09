"""Embed corpus.parquet via OpenRouter, persisting chunked vectors.

Thin CLI / orchestrator over :mod:`scripts.lib` modules.  The HTTP
POST itself, the tokenizer, the chunk layout, and the ``.env`` loader
all live under ``scripts/lib/``; this file just wires them together
and runs the adaptive-batch retry loop.

Behaviour:
  - Resumes from the highest existing ``chunk_NNNN.npy``.
  - Pre-flight token-precise truncation under the model's context.
  - Adaptive batch size: shrinks on 429 / 5xx, grows after success
    streaks.  Backoff is bounded.
  - 200-but-no-data responses are reclassified by the inner ``code``
    so transient overloads retry (don't get treated as overflow).
  - Skipped docs (single-row 400 after shrink) get a zero-vector
    placeholder, keeping chunk row N aligned with corpus row N.

Auth: reads ``OPENROUTER_API_KEY`` from the environment.  We never
accept it as a CLI argument — that would leak it into the process
list and logs.  A ``.env`` file in the repo root is loaded if present.

Usage::

    export OPENROUTER_API_KEY=sk-or-...
    python scripts/embed_via_openrouter.py --model qwen/qwen3-embedding-4b
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

from lib.chunk_store import (
    detect_resume_state, load_or_init_cost,
    log_skipped_doc, write_chunk,
)
from lib.dotenv import load_dotenv
from lib.openrouter_client import (
    TRANSIENT_STATUSES, post_embed_batch,
)
from lib.tokenizer import model_slug, resolve_and_apply_token_cap

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

logger = logging.getLogger("embed_via_openrouter")


def embed_corpus(
    out_dir: Path,
    corpus_path: Path,
    model: str,
    api_key: str,
    start_batch: int = 16,
    max_batch: int = 32,
    min_batch: int = 1,
    chunk_size: int = 1000,
    request_timeout: float = 120.0,
    grow_after_successes: int = 16,
    max_retries_per_batch: int = 15,
    provider_order: list[str] | None = None,
    ignore_providers: list[str] | None = None,
    max_tokens_per_doc: int = 30_000,
    tokenizer_repo: str | None = None,
) -> dict[str, Any]:
    slug = model_slug(model)
    state = detect_resume_state(out_dir, slug)

    # Load corpus into RAM (45k strings ≈ 100 MB of text, ~300 MB with
    # Python str overhead).  Drop the pyarrow Table immediately after
    # extracting columns — its internal Arrow buffer is another full
    # copy that we'd otherwise hold for the lifetime of embed_corpus.
    table = pq.read_table(corpus_path, columns=["text", "sha", "source"])
    texts_all = table.column("text").to_pylist()
    shas_all = table.column("sha").to_pylist()
    sources_all = table.column("source").to_pylist()
    n_total = len(texts_all)
    del table
    gc.collect()

    if max_tokens_per_doc and max_tokens_per_doc > 0:
        resolve_and_apply_token_cap(
            texts_all, model, max_tokens_per_doc, tokenizer_repo,
        )
    logger.info(
        "corpus: %d docs total, %d to do",
        n_total, n_total - state.rows_covered,
    )

    cost = load_or_init_cost(state.cost_report_path)

    session = requests.Session()
    chunk_id = state.chunks_written
    buf_vec: list[np.ndarray] = []
    buf_meta: list[dict] = []
    # Dim is learned from the first successful response so we can
    # write zero-vector placeholders for skipped docs without
    # hard-coding it.
    dim_known: int | None = state.dim

    batch = start_batch
    succ_streak = 0
    bar = tqdm(
        total=n_total - state.rows_covered,
        desc=f"embed {slug}", unit="doc",
    )
    i = state.rows_covered
    while i < n_total:
        end = min(i + batch, n_total)
        texts = texts_all[i:end]

        attempt = 0
        backoff = 1.5
        skip_this_doc = False
        while True:
            try:
                vecs, usage = post_embed_batch(
                    session, api_key, model, texts,
                    timeout=request_timeout,
                    provider_order=provider_order,
                    ignore_providers=ignore_providers,
                )
                break
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if status == 429:
                    cost["n_429"] += 1
                cost["n_retried"] += 1
                # Transient — halve batch (down to min) and back off.
                if status in TRANSIENT_STATUSES and len(texts) > min_batch:
                    new_batch = max(min_batch, len(texts) // 2)
                    logger.warning(
                        "HTTP %s — shrinking batch %d → %d, sleep %.1fs",
                        status, len(texts), new_batch, backoff,
                    )
                    texts = texts_all[i:i + new_batch]
                    end = i + new_batch
                    batch = new_batch
                    succ_streak = 0
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    attempt += 1
                    if attempt > max_retries_per_batch:
                        raise
                    continue
                # 400 from our 200-no-data remapping → context overflow
                # or a doc the provider just can't handle.  Shrink to
                # isolate; at min_batch, skip + zero-vector placeholder.
                if status == 400 and len(texts) > min_batch:
                    new_batch = max(min_batch, len(texts) // 2)
                    logger.warning(
                        "HTTP 400 (likely context overflow) — shrinking %d → %d: %s",
                        len(texts), new_batch,
                        (e.response.text[:200] if e.response is not None else ""),
                    )
                    texts = texts_all[i:i + new_batch]
                    end = i + new_batch
                    batch = new_batch
                    succ_streak = 0
                    time.sleep(0.5)
                    attempt += 1
                    if attempt > max_retries_per_batch:
                        raise
                    continue
                if status == 400 and len(texts) == 1:
                    if dim_known is None:
                        logger.error(
                            "HTTP 400 on first batch and no successful "
                            "embedding yet — cannot synthesise placeholder. "
                            "Body: %s",
                            (e.response.text[:300] if e.response is not None else ""),
                        )
                        raise
                    body_text = (
                        e.response.text[:300] if e.response is not None else ""
                    )
                    logger.warning(
                        "SKIP doc i=%d sha=%s source=%s (len=%d) — "
                        "zero-vector placeholder. %s",
                        i, shas_all[i], sources_all[i],
                        len(texts[0]), body_text,
                    )
                    log_skipped_doc(
                        state.skipped_log_path,
                        i=i, sha=shas_all[i], source=sources_all[i],
                        n_chars=len(texts[0]), reason=body_text,
                    )
                    skip_this_doc = True
                    break
                # Other 4xx (auth, payment) — bubble up.
                if status not in TRANSIENT_STATUSES:
                    body_text = ""
                    try:
                        body_text = e.response.text[:300]
                    except Exception:
                        pass
                    logger.error("HTTP %s non-retryable: %s", status, body_text)
                    raise
                # Transient at min_batch — backoff only.
                logger.warning(
                    "HTTP %s at min_batch=%d — sleep %.1fs",
                    status, len(texts), backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                attempt += 1
                if attempt > max_retries_per_batch:
                    raise
            except (requests.RequestException, KeyError, ValueError) as e:
                cost["n_retried"] += 1
                logger.warning(
                    "network/parse error: %s — sleep %.1fs", e, backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                attempt += 1
                if attempt > max_retries_per_batch:
                    raise

        if skip_this_doc:
            # Zero-vector placeholder so chunk row N corresponds to
            # corpus row N.  ZCA fit is robust to it (L2-renorm guard
            # + zero rows don't pull μ).
            zero = np.zeros((1, dim_known), dtype=np.float32)
            buf_vec.append(zero)
            buf_meta.append({
                "sha": shas_all[i], "source": sources_all[i],
                "skipped": True,
            })
            i += 1
            bar.update(1)
            succ_streak = 0
            rows_in_buf = sum(v.shape[0] for v in buf_vec)
            if rows_in_buf >= chunk_size or i == n_total:
                write_chunk(
                    state.chunks_dir, chunk_id, buf_vec, buf_meta,
                    state.manifest_path, state.manifest_lines,
                    cost, state.cost_report_path,
                )
                buf_vec.clear()
                buf_meta.clear()
                chunk_id += 1
            continue

        if dim_known is None:
            dim_known = int(vecs.shape[1])
        buf_vec.append(vecs)
        for j in range(i, end):
            buf_meta.append({"sha": shas_all[j], "source": sources_all[j]})
        cost["n_calls"] += 1
        cost["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        cost["total_tokens"] += int(usage.get("total_tokens", 0))
        cost["cost_usd"] += float(usage.get("cost", 0.0))
        i = end
        bar.update(end - (i - vecs.shape[0]))

        succ_streak += 1
        if succ_streak >= grow_after_successes and batch < max_batch:
            old, batch = batch, min(max_batch, batch * 2)
            logger.info(
                "growing batch %d → %d after %d clean successes",
                old, batch, succ_streak,
            )
            succ_streak = 0

        rows_in_buf = sum(v.shape[0] for v in buf_vec)
        if rows_in_buf >= chunk_size or i == n_total:
            write_chunk(
                state.chunks_dir, chunk_id, buf_vec, buf_meta,
                state.manifest_path, state.manifest_lines,
                cost, state.cost_report_path,
            )
            buf_vec.clear()
            buf_meta.clear()
            chunk_id += 1
    bar.close()

    state.cost_report_path.write_text(json.dumps(cost, indent=2))
    logger.info(
        "DONE %s: %d docs in %d chunks  ·  tokens=%s  ·  cost=$%.4f",
        model, n_total, chunk_id,
        f"{cost['prompt_tokens']:,}", cost["cost_usd"],
    )
    return cost


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True,
                    help="OpenRouter model id, e.g. qwen/qwen3-embedding-8b")
    ap.add_argument("--corpus", type=Path,
                    default=DATA_DIR / "corpus.parquet",
                    help="Path to corpus.parquet (default: ./data/corpus.parquet).")
    ap.add_argument("--out", type=Path, default=DATA_DIR,
                    help="Output dir (default: ./data).")
    ap.add_argument("--start-batch", type=int, default=16)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--min-batch", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=1000,
                    help="Rows per chunk_NNNN.npy.")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument(
        "--provider-order", default="",
        help="CSV of preferred OpenRouter provider slugs (e.g. 'nebius,deepinfra'). "
             "Empty = let OpenRouter pick. allow_fallbacks=True is always on.",
    )
    ap.add_argument(
        "--ignore-providers", default="siliconflow",
        help="CSV of OpenRouter provider slugs to hard-exclude (default: "
             "'siliconflow', which is ~4× the price of nebius/deepinfra "
             "for Qwen3-Embedding). Empty string disables the exclusion.",
    )
    ap.add_argument(
        "--max-tokens-per-doc", type=int, default=30_000,
        help="Token-precise per-doc cap, enforced with the model's own "
             "tokenizer pulled from HF.  Default 30 000 leaves ~2k margin "
             "below the 32k context window.  Set 0 to disable (large docs "
             "may then trigger HTTP 200-but-no-data and be skipped).",
    )
    ap.add_argument(
        "--tokenizer-repo", default="",
        help="Override HF repo for tokenizer.json (default: derived from "
             "--model via OPENROUTER_TO_HF_TOKENIZER).",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set — see .env.example")
        return 2

    if not args.corpus.is_file():
        logger.error(
            "corpus not found: %s — run scripts/build_corpus.py first",
            args.corpus,
        )
        return 2

    provider_order = (
        [p.strip() for p in args.provider_order.split(",") if p.strip()]
        if args.provider_order else None
    )
    ignore_providers = (
        [p.strip() for p in args.ignore_providers.split(",") if p.strip()]
        if args.ignore_providers else None
    )
    embed_corpus(
        out_dir=args.out,
        corpus_path=args.corpus,
        model=args.model,
        api_key=api_key,
        start_batch=args.start_batch,
        max_batch=args.max_batch,
        min_batch=args.min_batch,
        chunk_size=args.chunk_size,
        request_timeout=args.timeout,
        provider_order=provider_order,
        ignore_providers=ignore_providers,
        max_tokens_per_doc=args.max_tokens_per_doc,
        tokenizer_repo=(args.tokenizer_repo or None),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
