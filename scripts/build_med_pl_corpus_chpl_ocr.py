"""OCR the scanned ChPL documents that pypdf could not read.

``build_med_pl_corpus_chpl.py`` journals every image-only ChPL (no
text layer) as ``{"id": N, "http": 200, "scan": true}`` in
``data/med_pl/chpl.jsonl``.  This builder targets exactly those ids:
re-fetches each PDF, rasterises its pages with ``pdftoppm`` (poppler),
and transcribes them with a local vision model (Qwen3-VL via Ollama by
default) — recovering the ~1/5 of the registry that would otherwise be
lost.  Output is the canonical corpus schema written to
``data/med_pl/chpl_ocr.parquet`` with ``source = "chpl_rpl_ocr"``.

Why a VLM and not tesseract: ChPLs are dense, multi-column, table-heavy
Polish medical documents; a document-OCR-tuned VLM handles the layout
and diacritics far better than classic OCR.  Qwen3-VL-30B-A3B is a MoE
(3B active) so it is fast per page and fits the GB10's unified memory.

Resumable: every attempted id is journalled to
``data/med_pl/chpl_ocr.jsonl`` (id, n_chars, n_pages, and text on
success).  Re-running skips journalled ids.  ``--finalize`` rebuilds
the parquet from the journal.

Serving (Ollama, OpenAI-compatible images via /api/chat)::

    ollama pull qwen3-vl:30b-a3b-instruct
    python scripts/build_med_pl_corpus_chpl_ocr.py --workers 3

Endpoint / model are overridable for a vLLM deployment via
``--ollama-url`` / ``--model``.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
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

SOURCE = "chpl_rpl_ocr"
API = ("https://rejestrymedyczne.ezdrowie.gov.pl/api/rpl"
       "/medicinal-products/{id}/characteristic")
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3-vl:30b-a3b-instruct"
MIN_DOC_CHARS = 400
DPI = 150

OCR_PROMPT = (
    "Przepisz dokładnie cały tekst widoczny na tym obrazie strony "
    "dokumentu medycznego (Charakterystyka Produktu Leczniczego). "
    "Zachowaj kolejność, akapity i pozycje list. Nie dodawaj żadnych "
    "komentarzy, nagłówków ani wyjaśnień od siebie — zwróć wyłącznie "
    "tekst przepisany z obrazu."
)

logger = logging.getLogger("build_med_pl_corpus_chpl_ocr")

_ws = re.compile(r"[ \t]+")
_nl = re.compile(r"\n{3,}")
_surrogates = re.compile(r"[\ud800-\udfff]")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _clean(raw: str) -> str:
    raw = _surrogates.sub("", raw)
    lines = [_ws.sub(" ", ln).strip() for ln in raw.splitlines()]
    return _nl.sub("\n\n", "\n".join(lines)).strip()


def _scan_ids(journal: Path) -> list[int]:
    ids: list[int] = []
    if not journal.is_file():
        raise SystemExit(f"no ChPL journal at {journal} — run the scraper first")
    for line in journal.read_text().splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("scan") is True:
            ids.append(int(rec["id"]))
    return sorted(set(ids))


def _load_done(journal: Path) -> set[int]:
    done: set[int] = set()
    if journal.is_file():
        for line in journal.read_text().splitlines():
            try:
                done.add(int(json.loads(line)["id"]))
            except Exception:
                continue
    return done


def _render_pages(pdf_bytes: bytes, dpi: int) -> list[bytes]:
    """PDF bytes -> list of PNG bytes, one per page, via pdftoppm."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "in.pdf").write_bytes(pdf_bytes)
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "in.pdf", "page"],
            cwd=td, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        pages = sorted(tdp.glob("page*.png"))
        return [p.read_bytes() for p in pages]


def _ocr_page(session: requests.Session, url: str, model: str,
              png: bytes, timeout: float, api: str, api_key: str | None) -> str:
    b64 = base64.b64encode(png).decode()
    if api == "ollama":
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": OCR_PROMPT, "images": [b64]}],
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 8192},
        }
        r = session.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")
    # OpenAI-compatible (OpenRouter / api.openai.com) vision chat.
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": OCR_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = session.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _process_id(pid: int, http_session: requests.Session,
                ocr_session: requests.Session, args) -> dict:
    try:
        r = http_session.get(API.format(id=pid), timeout=args.timeout)
        if r.status_code != 200 or "pdf" not in r.headers.get("content-type", ""):
            return {"id": pid, "ok": False, "err": f"http {r.status_code}"}
        pages = _render_pages(r.content, args.dpi)
        if not pages:
            return {"id": pid, "ok": False, "err": "no pages"}
        parts = []
        for png in pages:
            for attempt in range(args.max_retries):
                try:
                    parts.append(_ocr_page(ocr_session, args.url, args.model,
                                           png, args.ocr_timeout, args.api,
                                           args._api_key))
                    break
                except requests.RequestException as e:
                    if attempt == args.max_retries - 1:
                        parts.append("")
                    else:
                        time.sleep(2 * (attempt + 1))
        text = _clean("\n".join(parts))
        return {"id": pid, "ok": True, "n_pages": len(pages),
                "n_chars": len(text), "text": text}
    except Exception as e:
        return {"id": pid, "ok": False, "err": str(e)[:150]}


def finalize(journal: Path, out_path: Path) -> dict:
    rows: list[str] = []
    seen: set[str] = set()
    ok = short = 0
    for line in journal.read_text().splitlines() if journal.is_file() else []:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not rec.get("ok") or "text" not in rec:
            continue
        text = rec["text"]
        if len(text) < MIN_DOC_CHARS:
            short += 1
            continue
        sha = _sha(text)
        if sha in seen:
            continue
        seen.add(sha)
        rows.append(text)
        ok += 1
    table = pa.table({
        "source": [SOURCE] * len(rows),
        "text": rows,
        "sha": [_sha(t) for t in rows],
        "n_chars": pa.array([len(t) for t in rows], type=pa.int64()),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
    chars = sorted(len(t) for t in rows)
    return {
        "docs": ok, "short_or_dup": short,
        "median_chars": chars[len(chars)//2] if chars else 0,
        "out": str(out_path),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src-journal", type=Path, default=MED_DIR / "chpl.jsonl",
                    help="Scraper journal that flags scans.")
    ap.add_argument("--journal", type=Path, default=MED_DIR / "chpl_ocr.jsonl")
    ap.add_argument("--out", type=Path, default=MED_DIR / "chpl_ocr.parquet")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--api", choices=["ollama", "openai"], default="ollama",
                    help="ollama = local /api/chat (default); openai = "
                         "OpenAI-compatible /v1/chat/completions "
                         "(OpenRouter / api.openai.com vision).")
    ap.add_argument("--url", default=OLLAMA_URL,
                    help="OCR endpoint. For --api openai pass e.g. "
                         "https://openrouter.ai/api/v1/chat/completions")
    ap.add_argument("--api-key-env", default=None,
                    help="Env var holding the bearer key for --api openai "
                         "(e.g. OPENROUTER_API_KEY / OPENAI_API_KEY).")
    ap.add_argument("--workers", type=int, default=3,
                    help="Concurrent docs in flight (Ollama serialises GPU; "
                         "this overlaps fetch+render with OCR).")
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--ocr-timeout", type=float, default=300.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N scan ids (for validation).")
    ap.add_argument("--ids", type=int, nargs="*", default=None,
                    help="Explicit ids to OCR (overrides journal scan set; "
                         "for quality tests).")
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args(argv)
    args._api_key = os.environ.get(args.api_key_env) if args.api_key_env else None

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args.journal.parent.mkdir(parents=True, exist_ok=True)

    if args.finalize:
        logger.info("finalize: %s", finalize(args.journal, args.out))
        return 0

    if args.ids:
        todo = list(args.ids)
    else:
        scans = _scan_ids(args.src_journal)
        done = _load_done(args.journal)
        todo = [i for i in scans if i not in done]
        if args.limit:
            todo = todo[: args.limit]
        logger.info("scans=%d, already done=%d, to OCR=%d",
                    len(scans), len(done), len(todo))
    if not todo:
        logger.info("nothing to do; finalizing")
        logger.info("finalize: %s", finalize(args.journal, args.out))
        return 0

    http_session = requests.Session()
    http_session.headers.update({"Accept": "application/pdf",
                                 "User-Agent": "med-corpus-ocr/research"})
    ocr_session = requests.Session()
    lock = threading.Lock()
    jf = args.journal.open("a", encoding="utf-8", errors="replace")
    n_ok = n_chars = 0

    def _record(rec: dict) -> None:
        nonlocal n_ok, n_chars
        with lock:
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()
            if rec.get("ok"):
                n_ok += 1
                n_chars += rec.get("n_chars", 0)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_process_id, pid, http_session, ocr_session, args): pid
                    for pid in todo}
            bar = tqdm(total=len(futs), desc="ocr", unit="doc")
            for fut in as_completed(futs):
                _record(fut.result())
                bar.update(1); bar.set_postfix(ok=n_ok)
            bar.close()
    finally:
        jf.close()

    logger.info("OCR done: %d docs, ~%.2f M chars", n_ok, n_chars / 1e6)
    logger.info("finalize: %s", finalize(args.journal, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
