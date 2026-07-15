"""Download the original ChPL PDFs to disk (the scraper kept only text).

Reads the ids the scraper journalled in ``data/med_pl/chpl.jsonl`` (every
id with a valid 200 response — text-layer OR scan) and downloads each
PDF to ``data/med_pl/pdf/<id>.pdf``.  Resumable (skips files already on
disk), bounded concurrency.

Usage::

    python scripts/download_med_pdf.py                 # all valid ids
    python scripts/download_med_pdf.py --scans-only     # just the OCR scans
    python scripts/download_med_pdf.py --workers 12
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
MED = REPO / "data" / "med_pl"
API = ("https://rejestrymedyczne.ezdrowie.gov.pl/api/rpl"
       "/medicinal-products/{id}/characteristic")


def ids_from_journal(journal: Path, scans_only: bool) -> list[int]:
    out = []
    for line in journal.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if scans_only:
            if r.get("scan") is True:
                out.append(int(r["id"]))
        elif r.get("http") == 200 and ("text" in r or r.get("scan")):
            out.append(int(r["id"]))
    return sorted(set(out))


def fetch(session: requests.Session, pid: int, out_dir: Path, timeout: float) -> int:
    dest = out_dir / f"{pid}.pdf"
    if dest.exists() and dest.stat().st_size > 0:
        return 0
    for attempt in range(4):
        try:
            r = session.get(API.format(id=pid), timeout=timeout)
            if r.status_code == 200 and "pdf" in r.headers.get("content-type", ""):
                dest.write_bytes(r.content)
                return len(r.content)
            return 0
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--journal", type=Path, default=MED / "chpl.jsonl")
    ap.add_argument("--out", type=Path, default=MED / "pdf")
    ap.add_argument("--scans-only", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    ids = ids_from_journal(args.journal, args.scans_only)
    args.out.mkdir(parents=True, exist_ok=True)
    have = {int(p.stem) for p in args.out.glob("*.pdf")}
    todo = [i for i in ids if i not in have]
    print(f"{len(ids)} ids, {len(have)} already on disk, {len(todo)} to download → {args.out}/")

    session = requests.Session()
    session.headers.update({"Accept": "application/pdf",
                            "User-Agent": "med-corpus-pdf/research"})
    total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, session, pid, args.out, args.timeout): pid for pid in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="pdf", unit="file"):
            total += fut.result()
    print(f"done. downloaded ~{total/1e9:.2f} GB this run; "
          f"{len(list(args.out.glob('*.pdf')))} PDFs on disk total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
