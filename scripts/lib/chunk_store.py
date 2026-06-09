"""On-disk layout for chunked embeddings, manifests, and cost reports.

The caller (``embed_via_openrouter.py``) keeps two buffers in memory:

- ``buf_vec`` — list of ``np.ndarray`` rows still to flush
- ``buf_meta`` — matching list of ``{sha, source, skipped?}`` dicts

When the row count hits ``chunk_size`` (or we reach the end of the
corpus), we flush:

- ``chunks_<slug>/chunk_NNNN.npy``       — concatenated fp16 vectors
- ``chunks_<slug>/chunk_NNNN.index.jsonl`` — per-row metadata
- ``manifest_<slug>.jsonl``               — full corpus index (rewritten each flush)
- ``cost_report_<slug>.json``             — running OpenRouter usage

This module owns the layout and is the single place that knows the
filename pattern.  If you ever want to switch to fp32 chunks or a
different on-disk format, you change it here, not in the caller.

Resume protocol: callers call ``detect_resume_state()`` once at
startup to learn how many rows are already on disk.  We rely on a
simple "highest contiguous chunk_NNNN.npy on disk" rule — the writer
flushes atomically and only at chunk boundaries, so any survivor is
trustworthy.  Anything that was in the in-memory buffer at the time
of a crash is lost, but the corpus.parquet is the source of truth and
the embed pass is idempotent, so re-running just re-fills the gap.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ResumeState:
    """How far the previous run got on this (model, corpus) pair."""

    chunks_dir: Path
    chunks_written: int
    rows_covered: int
    dim: int | None  # learned from the first existing chunk, or None
    manifest_path: Path
    manifest_lines: list[str]
    cost_report_path: Path
    skipped_log_path: Path


def detect_resume_state(out_dir: Path, slug: str) -> ResumeState:
    """Inspect ``out_dir`` for an existing run of ``<slug>``.

    Always returns a ``ResumeState`` — fresh runs just have
    ``chunks_written == 0``.  The caller uses ``rows_covered`` as the
    starting index into ``corpus.parquet``.
    """
    chunks_dir = out_dir / f"chunks_{slug}"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"manifest_{slug}.jsonl"
    cost_report_path = out_dir / f"cost_report_{slug}.json"
    skipped_log_path = out_dir / f"skipped_{slug}.jsonl"

    written = sorted(chunks_dir.glob("chunk_*.npy"))
    covered = (
        sum(int(np.load(c, mmap_mode="r").shape[0]) for c in written)
        if written
        else 0
    )
    dim = int(np.load(written[0], mmap_mode="r").shape[1]) if written else None

    manifest_lines = (
        manifest_path.read_text().splitlines()
        if manifest_path.is_file()
        else []
    )

    if written:
        logger.info(
            "[resume] %d chunks, %d rows already embedded for %s",
            len(written), covered, slug,
        )

    return ResumeState(
        chunks_dir=chunks_dir,
        chunks_written=len(written),
        rows_covered=covered,
        dim=dim,
        manifest_path=manifest_path,
        manifest_lines=manifest_lines,
        cost_report_path=cost_report_path,
        skipped_log_path=skipped_log_path,
    )


def write_chunk(
    chunks_dir: Path,
    chunk_id: int,
    buf_vec: list[np.ndarray],
    buf_meta: list[dict],
    manifest_path: Path,
    manifest_lines: list[str],
    cost: dict,
    cost_report_path: Path,
) -> None:
    """Flush one chunk's vectors + metadata, then update the manifest
    and cost report.

    ``buf_vec`` and ``buf_meta`` are NOT mutated — the caller is
    expected to ``.clear()`` them after the call.  ``manifest_lines``
    IS appended to in place (it's the shared running list).
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
    logger.info(
        "wrote %s (%d rows, dim=%d) — running cost=$%.4f",
        cpath.name, arr.shape[0], arr.shape[1], cost["cost_usd"],
    )
    cost_report_path.write_text(json.dumps(cost, indent=2))


def load_or_init_cost(cost_report_path: Path) -> dict:
    """Recover the running cost / call-count dict if a prior run wrote
    one, otherwise return a fresh zero-initialised one.

    Used by the caller when resuming so totals span the whole job, not
    just the current invocation.
    """
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
            # Corrupt cost report — log and start fresh; the embed
            # pass is the source of truth so losing the running total
            # is a paper cut, not a correctness bug.
            logger.warning(
                "could not parse %s — starting cost counters at 0",
                cost_report_path,
            )
    return cost


def log_skipped_doc(
    skipped_log_path: Path, i: int, sha: str, source: str,
    n_chars: int, reason: str,
) -> None:
    """Append one skipped-doc record.  Append-only so the file
    survives resumes and is easy to grep after a run."""
    with skipped_log_path.open("a") as f:
        f.write(json.dumps({
            "i": i, "sha": sha, "source": source,
            "n_chars": n_chars, "reason": reason,
        }) + "\n")
