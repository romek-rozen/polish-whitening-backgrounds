"""Build the ChPL source of the Polish medical corpus.

Downloads *Charakterystyka Produktu Leczniczego* (Summary of Product
Characteristics) documents from the official Polish drug registry —
Rejestr Produktów Leczniczych (RPL) — and extracts their text.  These
are long-form, professional Polish medical documents (indications,
dosing, contraindications, pharmacology, adverse effects), the richest
freely-available raw medical-Polish text there is.

Source API (verified 2026-07-08)::

    GET https://rejestrymedyczne.ezdrowie.gov.pl/api/rpl
        /medicinal-products/<id>/characteristic
    -> 200 application/pdf   for a valid product id
    -> 400                   for a gap in the id space

The registry's *list* endpoint is access-denied and the site is a
SPA, so there is no bulk manifest — we enumerate the integer id space
directly (valid ids are dense-ish up to a few tens of thousands) and
keep every 200.  Politeness: bounded concurrency + retry/backoff on
429/5xx.

Resumable: every attempted id is journalled to
``data/med_pl/chpl.jsonl`` (one line per id: id, http, n_chars, and
the extracted text on success).  Re-running skips journalled ids and
appends only new ones.  ``--finalize`` (implied at the end of a full
pass) rebuilds ``data/med_pl/chpl.parquet`` from the journal with the
canonical corpus schema ``{source, text, sha, n_chars}``.

Usage::

    python scripts/build_med_pl_corpus_chpl.py --id-max 60000 --workers 8
    python scripts/build_med_pl_corpus_chpl.py --finalize   # journal -> parquet only
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MED_DIR = DATA_DIR / "med_pl"

SOURCE = "chpl_rpl"
API = ("https://rejestrymedyczne.ezdrowie.gov.pl/api/rpl"
       "/medicinal-products/{id}/characteristic")
MIN_DOC_CHARS = 500

logger = logging.getLogger("build_med_pl_corpus_chpl")

_ws = re.compile(r"[ \t]+")
_nl = re.compile(r"\n{3,}")
# pypdf can emit lone UTF-16 surrogates from malformed PDFs; these are
# not valid UTF-8 and blow up json.dumps / parquet writes.  Strip them.
_surrogates = re.compile(r"[\ud800-\udfff]")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _clean_pdf_text(raw: str) -> str:
    """Normalise pypdf output: collapse intra-line whitespace, keep
    paragraph breaks (``\\n\\n``) so the downstream chunk / segment /
    paragraph splitters have real boundaries to cut on."""
    raw = _surrogates.sub("", raw)
    lines = [_ws.sub(" ", ln).strip() for ln in raw.splitlines()]
    text = "\n".join(lines)
    text = _nl.sub("\n\n", text)
    return text.strip()


def _extract_pdf(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return _clean_pdf_text("\n".join(parts))


def _load_done(journal: Path) -> set[int]:
    done: set[int] = set()
    if journal.is_file():
        for line in journal.read_text().splitlines():
            try:
                done.add(int(json.loads(line)["id"]))
            except Exception:
                continue
    return done


def _fetch_one(
    session: requests.Session, pid: int, timeout: float, max_retries: int,
) -> dict:
    """Fetch + extract one product id.  Returns a journal record."""
    url = API.format(id=pid)
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=timeout)
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                return {"id": pid, "http": 0, "err": str(e)[:120]}
            time.sleep(backoff); backoff = min(backoff * 2, 30)
            continue

        if r.status_code == 200 and "pdf" in r.headers.get("content-type", ""):
            try:
                text = _extract_pdf(r.content)
            except Exception as e:
                return {"id": pid, "http": 200, "err": f"extract:{e}"[:120]}
            # ~20% of ChPLs are image-only scans with no text layer;
            # pypdf yields near-nothing.  Flag them (scan=True, no text
            # stored) so a later OCR pass can target exactly these ids
            # without re-probing the whole space.  See
            # build_med_pl_corpus_chpl_ocr.py.
            if len(text) < 200:
                return {"id": pid, "http": 200, "scan": True,
                        "n_chars": len(text)}
            return {"id": pid, "http": 200, "n_chars": len(text), "text": text}
        if r.status_code in (400, 404):
            return {"id": pid, "http": r.status_code}
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff); backoff = min(backoff * 2, 30)
            continue
        return {"id": pid, "http": r.status_code}
    return {"id": pid, "http": 429, "err": "retries_exhausted"}


def finalize(journal: Path, out_path: Path) -> dict:
    """Rebuild the parquet from the journal (canonical schema)."""
    rows_text: list[str] = []
    seen_sha: set[str] = set()
    ok = short = 0
    if not journal.is_file():
        raise SystemExit(f"no journal at {journal}")
    for line in journal.read_text().splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("http") != 200 or "text" not in rec:
            continue
        text = rec["text"]
        if len(text) < MIN_DOC_CHARS:
            short += 1
            continue
        sha = _sha(text)
        if sha in seen_sha:            # identical ChPL across pack sizes
            continue
        seen_sha.add(sha)
        rows_text.append(text)
        ok += 1

    table = pa.table({
        "source": [SOURCE] * len(rows_text),
        "text": rows_text,
        "sha": [_sha(t) for t in rows_text],
        "n_chars": pa.array([len(t) for t in rows_text], type=pa.int64()),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
    chars = [len(t) for t in rows_text]
    return {
        "docs": ok, "deduped_or_short": short,
        "min_chars": min(chars) if chars else 0,
        "median_chars": int(sorted(chars)[len(chars)//2]) if chars else 0,
        "max_chars": max(chars) if chars else 0,
        "out": str(out_path),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--id-min", type=int, default=1)
    ap.add_argument("--id-max", type=int, default=60_000,
                    help="Upper bound of the id space to probe (inclusive).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--journal", type=Path, default=MED_DIR / "chpl.jsonl")
    ap.add_argument("--out", type=Path, default=MED_DIR / "chpl.parquet")
    ap.add_argument("--finalize", action="store_true",
                    help="Skip fetching; just rebuild parquet from journal.")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args.journal.parent.mkdir(parents=True, exist_ok=True)

    if args.finalize:
        logger.info("finalize: %s", finalize(args.journal, args.out))
        return 0

    done = _load_done(args.journal)
    todo = [i for i in range(args.id_min, args.id_max + 1) if i not in done]
    logger.info("id space %d..%d — %d already journalled, %d to fetch",
                args.id_min, args.id_max, len(done), len(todo))

    session = requests.Session()
    session.headers.update({
        "Accept": "application/pdf",
        "User-Agent": "polish-whitening-backgrounds/med-corpus (research)",
    })
    lock = threading.Lock()
    jf = args.journal.open("a", encoding="utf-8", errors="replace")
    n_ok = n_pdf_chars = 0

    def _record(rec: dict) -> None:
        nonlocal n_ok, n_pdf_chars
        with lock:
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()
            if rec.get("http") == 200 and "text" in rec:
                n_ok += 1
                n_pdf_chars += rec.get("n_chars", 0)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(_fetch_one, session, pid, args.timeout,
                          args.max_retries): pid
                for pid in todo
            }
            bar = tqdm(total=len(futs), desc="chpl", unit="id")
            for fut in as_completed(futs):
                _record(fut.result())
                bar.update(1)
                bar.set_postfix(ok=n_ok)
            bar.close()
    finally:
        jf.close()

    logger.info("fetched: %d valid ChPL, ~%.1f M chars",
                n_ok, n_pdf_chars / 1e6)
    logger.info("finalize: %s", finalize(args.journal, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
