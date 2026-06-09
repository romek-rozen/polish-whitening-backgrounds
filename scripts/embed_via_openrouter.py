"""Embed a corpus.parquet via OpenRouter and persist chunked vectors.

Reads ``data/corpus.parquet`` (produced by ``build_corpus.py``), POSTs
each batch to ``https://openrouter.ai/api/v1/embeddings``, and writes
fp16 chunks to ``data/chunks_<model_slug>/chunk_NNNN.npy`` plus a
sidecar ``chunk_NNNN.index.jsonl`` and a corpus-wide ``manifest.jsonl``.

Features
--------
- **Adaptive batch size**: starts at ``--start-batch`` (default 16),
  halves on 429 / 5xx-503 / 529 ("provider overloaded"), grows back
  gradually after a streak of successes. Bounded by ``--max-batch``.
- **Retry with exponential backoff** on network errors.
- **Idempotent**: resumes from the highest chunk already on disk.
- **Cost & token logging**: aggregates ``usage.prompt_tokens`` /
  ``usage.cost`` returned by OpenRouter and writes a final
  ``cost_report.json``.

Auth
----
Reads ``OPENROUTER_API_KEY`` from the environment. Never accept it as
a CLI argument — that would leak it into the process list and any logs.
A ``.env`` file in the repo root will be loaded if present.

Usage
-----
::

    export OPENROUTER_API_KEY=sk-or-...
    python scripts/embed_via_openrouter.py --model qwen/qwen3-embedding-4b
    python scripts/embed_via_openrouter.py --model qwen/qwen3-embedding-8b
"""
from __future__ import annotations

import argparse
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

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

logger = logging.getLogger("embed_via_openrouter")

OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"
TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504, 529}

# OpenRouter id → HuggingFace repo id (used to fetch tokenizer.json for
# precise pre-flight truncation under the 32k token context window).
OPENROUTER_TO_HF_TOKENIZER = {
    "qwen/qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen/qwen3-embedding-4b":   "Qwen/Qwen3-Embedding-4B",
    "qwen/qwen3-embedding-8b":   "Qwen/Qwen3-Embedding-8B",
}


def _load_tokenizer(hf_repo_id: str):
    """Pull just tokenizer.json from HF and load it via the Rust `tokenizers`
    crate. Cached under ~/.cache/huggingface/hub/ on first call."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    path = hf_hub_download(repo_id=hf_repo_id, filename="tokenizer.json")
    return Tokenizer.from_file(path)


def _truncate_to_tokens(
    texts: list[str], tokenizer, max_tokens: int
) -> tuple[int, int, int]:
    """In-place truncate any doc whose token count exceeds ``max_tokens``.

    Returns ``(n_capped, max_seen, total_tokens)`` for logging.
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


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader (only KEY=VALUE lines, no quoting tricks)."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _model_slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def _post_batch(
    session: requests.Session,
    api_key: str,
    model: str,
    texts: list[str],
    timeout: float,
    provider_order: list[str] | None = None,
    ignore_providers: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Single POST. Raises on non-2xx; returns (arr, usage_dict)."""
    payload: dict[str, Any] = {
        "model": model, "input": texts, "encoding_format": "float",
    }
    provider_block: dict[str, Any] = {}
    if provider_order:
        # Pin specific providers first; allow_fallbacks lets the next
        # cheapest option pick up when the pinned ones are throttled.
        provider_block["order"] = provider_order
        provider_block["allow_fallbacks"] = True
    if ignore_providers:
        # Hard-exclude providers we don't want any spend going to (e.g.
        # SiliconFlow, which is ~4× the price of nebius / deepinfra).
        provider_block["ignore"] = ignore_providers
    if provider_block:
        payload["provider"] = provider_block
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Site-name headers are optional; OpenRouter uses them for analytics
        # on the leaderboard. Harmless to set.
        "X-Title": "polish-whitening-backgrounds",
    }
    r = session.post(OPENROUTER_URL, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    # OpenRouter occasionally returns HTTP 200 with `{"error": {...}}` instead
    # of `{"data": [...]}`.  Two distinct failure modes hide here:
    #
    #   1. Doc exceeds the provider's max context — typically surfaces as a
    #      40x in the inner body (or no inner code).  Caller should shrink
    #      and eventually skip.
    #   2. Provider is overloaded — surfaces as inner `code: 429` (or 5xx)
    #      with a message like "Model busy, retry later".  This is fully
    #      transient; shrinking + skipping would silently throw away
    #      perfectly good documents.
    #
    # Inspect the inner `code` and mirror it on the synthesised Response so
    # the caller's existing TRANSIENT_STATUSES path (backoff + retry,
    # WITHOUT skipping) triggers for case 2.
    if "data" not in body:
        err = body.get("error") or body
        inner_code = err.get("code") if isinstance(err, dict) else None
        # Accept either an int or a numeric string.
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
    data = body["data"]
    arr = np.array([d["embedding"] for d in data], dtype=np.float32)
    usage = body.get("usage") or {}
    return arr, usage


def _write_chunk(chunks_dir: Path, chunk_id: int,
                 buf_vec: list[np.ndarray], buf_meta: list[dict],
                 manifest_path: Path, manifest_lines: list[str],
                 cost: dict, cost_report_path: Path) -> None:
    """Flush buffered embeddings to disk as a single chunk + sidecar.

    Called both on regular size triggers and after a skipped doc — kept
    in one place so chunk layout stays consistent.
    """
    arr = np.concatenate(buf_vec, axis=0).astype(np.float16)
    cpath = chunks_dir / f"chunk_{chunk_id:04d}.npy"
    ipath = chunks_dir / f"chunk_{chunk_id:04d}.index.jsonl"
    np.save(cpath, arr)
    with ipath.open("w") as f:
        for m in buf_meta:
            f.write(json.dumps(m) + "\n")
    for m in buf_meta:
        manifest_lines.append(json.dumps(m))
    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    logger.info("wrote %s (%d rows, dim=%d) — running cost=$%.4f",
                cpath.name, arr.shape[0], arr.shape[1],
                cost["cost_usd"])
    cost_report_path.write_text(json.dumps(cost, indent=2))


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
    slug = _model_slug(model)
    chunks_dir = out_dir / f"chunks_{slug}"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"manifest_{slug}.jsonl"
    cost_report_path = out_dir / f"cost_report_{slug}.json"

    # Resume.
    written_chunks = sorted(chunks_dir.glob("chunk_*.npy"))
    covered = (
        sum(int(np.load(c, mmap_mode="r").shape[0]) for c in written_chunks)
        if written_chunks else 0
    )
    if covered:
        logger.info("[resume] %d chunks, %d rows already embedded for %s",
                    len(written_chunks), covered, model)

    # Load corpus into RAM (45k strings = OK).
    table = pq.read_table(corpus_path, columns=["text", "sha", "source"])
    texts_all = table.column("text").to_pylist()
    shas_all = table.column("sha").to_pylist()
    sources_all = table.column("source").to_pylist()
    n_total = len(texts_all)
    # Pre-flight token-level truncation under the model's 32k context.
    # Uses the model's own tokenizer (pulled from HF), so the count matches
    # exactly what the provider will compute server-side.  Default 30 000
    # tokens leaves a ~2k margin for safety + any special tokens added by
    # the embed endpoint.  Without this, the rare ultra-long wiki article
    # would trigger an HTTP 200 + error-body ("context length exceeded")
    # that the runtime can only handle by skipping the doc.
    if max_tokens_per_doc and max_tokens_per_doc > 0:
        repo = tokenizer_repo or OPENROUTER_TO_HF_TOKENIZER.get(model.lower())
        if not repo:
            logger.error(
                "no tokenizer mapping for %s — pass --tokenizer-repo or extend "
                "OPENROUTER_TO_HF_TOKENIZER",
                model,
            )
            raise SystemExit(2)
        logger.info("loading tokenizer %s for token-precise truncation", repo)
        t0 = time.monotonic()
        tok = _load_tokenizer(repo)
        n_capped, max_seen, total_tok = _truncate_to_tokens(
            texts_all, tok, max_tokens_per_doc
        )
        dt = time.monotonic() - t0
        logger.info(
            "tokenized %d docs in %.1fs: total=%s tokens, max=%d, "
            "truncated=%d at %d-tok cap",
            n_total, dt, f"{total_tok:,}", max_seen, n_capped,
            max_tokens_per_doc,
        )
    logger.info("corpus: %d docs total, %d to do", n_total, n_total - covered)

    # Load any prior cost report so resumed totals are correct.
    cost = {
        "prompt_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "n_calls": 0,
        "n_retried": 0,
        "n_429": 0,
    }
    if cost_report_path.is_file():
        try:
            prev = json.loads(cost_report_path.read_text())
            for k in cost:
                cost[k] = prev.get(k, cost[k])
        except Exception:
            pass

    session = requests.Session()
    chunk_id = len(written_chunks)
    buf_vec: list[np.ndarray] = []
    buf_meta: list[dict] = []
    # Dim is learned from the first successful response so we can write
    # zero-vector placeholders for skipped docs without hard-coding it.
    dim_known: int | None = None
    if written_chunks:
        dim_known = int(np.load(written_chunks[0], mmap_mode="r").shape[1])
    skipped_log = out_dir / f"skipped_{_model_slug(model)}.jsonl"
    manifest_lines: list[str] = (
        manifest_path.read_text().splitlines() if manifest_path.is_file() else []
    )

    batch = start_batch
    succ_streak = 0
    bar = tqdm(total=n_total - covered, desc=f"embed {slug}", unit="doc")
    i = covered
    while i < n_total:
        # Take up to `batch` docs.
        end = min(i + batch, n_total)
        texts = texts_all[i:end]

        # Try this batch with adaptive shrink + retry/backoff.
        attempt = 0
        backoff = 1.5
        skip_this_doc = False
        while True:
            try:
                vecs, usage = _post_batch(
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
                # On retryable status, halve batch (down to min) and back off.
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
                # 400 from our own "200-no-data" remapping → that batch has
                # something the provider can't handle (typically: doc still
                # too long even after the char cap, or one specific doc with
                # bad encoding).  Shrink to isolate; at min_batch, skip the
                # bad doc and log.
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
                    # Single doc still rejected — even after char-cap
                    # truncation.  Log + write a zero-vector placeholder so
                    # row alignment with corpus.parquet is preserved.
                    if dim_known is None:
                        # Can't placeholder yet; bubble up so the run fails
                        # loudly rather than producing misaligned chunks.
                        logger.error(
                            "HTTP 400 on first batch and no successful "
                            "embedding yet — cannot synthesise placeholder. "
                            "Body: %s",
                            (e.response.text[:300] if e.response is not None else ""),
                        )
                        raise
                    body_text = (e.response.text[:300] if e.response is not None else "")
                    logger.warning(
                        "SKIP doc i=%d sha=%s source=%s (len=%d) — zero-vector placeholder. %s",
                        i, shas_all[i], sources_all[i],
                        len(texts[0]), body_text,
                    )
                    with skipped_log.open("a") as f:
                        f.write(json.dumps({
                            "i": i, "sha": shas_all[i], "source": sources_all[i],
                            "n_chars": len(texts[0]),
                            "reason": body_text,
                        }) + "\n")
                    skip_this_doc = True
                    break  # exit retry loop
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
                logger.warning("HTTP %s at min_batch=%d — sleep %.1fs",
                               status, len(texts), backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                attempt += 1
                if attempt > max_retries_per_batch:
                    raise
            except (requests.RequestException, KeyError, ValueError) as e:
                cost["n_retried"] += 1
                logger.warning("network/parse error: %s — sleep %.1fs", e, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                attempt += 1
                if attempt > max_retries_per_batch:
                    raise

        if skip_this_doc:
            # Append a zero-vector placeholder so chunk row N corresponds to
            # corpus row N.  ZCA fit is robust to it: after L2-renorm guard
            # zero stays zero, and a zero row contributes nothing to μ or Σ.
            zero = np.zeros((1, dim_known), dtype=np.float32)
            buf_vec.append(zero)
            buf_meta.append({"sha": shas_all[i], "source": sources_all[i],
                             "skipped": True})
            i += 1
            bar.update(1)
            succ_streak = 0
            # Flush chunk if needed.
            rows_in_buf = sum(v.shape[0] for v in buf_vec)
            if rows_in_buf >= chunk_size or i == n_total:
                _write_chunk(chunks_dir, chunk_id, buf_vec, buf_meta,
                             manifest_path, manifest_lines, cost,
                             cost_report_path)
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
            logger.info("growing batch %d → %d after %d clean successes",
                        old, batch, succ_streak)
            succ_streak = 0

        # Flush chunk.
        rows_in_buf = sum(v.shape[0] for v in buf_vec)
        if rows_in_buf >= chunk_size or i == n_total:
            _write_chunk(chunks_dir, chunk_id, buf_vec, buf_meta,
                         manifest_path, manifest_lines, cost,
                         cost_report_path)
            buf_vec.clear()
            buf_meta.clear()
            chunk_id += 1
    bar.close()

    cost_report_path.write_text(json.dumps(cost, indent=2))
    logger.info("DONE %s: %d docs in %d chunks  ·  tokens=%s  ·  cost=$%.4f",
                model, n_total, chunk_id,
                f"{cost['prompt_tokens']:,}", cost["cost_usd"])
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

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    _load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set — see .env.example")
        return 2

    if not args.corpus.is_file():
        logger.error("corpus not found: %s — run scripts/build_corpus.py first",
                     args.corpus)
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
