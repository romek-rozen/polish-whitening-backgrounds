"""Controlled OCR benchmark: same documents, measure wall time + tokens/s.

Runs a fixed set of scanned ChPL documents through an OpenAI-compatible
vision endpoint (gemma-26B or diffusiongemma, both on :8001), page by
page, and records:
  - per-document wall time
  - total wall time
  - total completion tokens (from usage) and tokens/s
  - characters produced

The document set is DETERMINISTIC (first N scan ids, sorted) so two
models are compared on identical inputs. Results are written to JSON.

Usage::

    python scripts/ocr_bench.py --model gemma4-26b-it-fp8 --out bench_gemma.json
    python scripts/ocr_bench.py --model diffusiongemma --out bench_diff.json
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_med_pl_corpus_chpl_ocr import _render_pages, OCR_PROMPT  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
API = ("https://rejestrymedyczne.ezdrowie.gov.pl/api/rpl"
       "/medicinal-products/{id}/characteristic")


def fixed_scan_ids(journal: Path, n: int) -> list[int]:
    """First N scan ids (sorted) — deterministic across models."""
    ids = []
    for line in journal.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("scan") is True:
            ids.append(int(r["id"]))
    return sorted(set(ids))[:n]


def ocr_page(sess, url, model, b64, timeout, prompt, use_temp=True):
    payload = {
        "model": model, "max_tokens": 4096,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
    }
    # Diffusion models (block-diffusion sampler) reject `temperature`; the
    # autoregressive models are benchmarked greedily (temperature=0).
    if use_temp:
        payload["temperature"] = 0.0
    r = sess.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    return (d["choices"][0]["message"]["content"],
            d.get("usage", {}).get("completion_tokens", 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001/v1/chat/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--journal", type=Path,
                    default=REPO / "data/med_pl/chpl.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--prompt", default=None,
                    help="Override the OCR prompt (GLM-OCR wants 'Text Recognition:').")
    ap.add_argument("--no-temp", action="store_true",
                    help="Drop `temperature` from requests (diffusion models reject it).")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    prompt = args.prompt or OCR_PROMPT
    use_temp = not args.no_temp

    ids = fixed_scan_ids(args.journal, args.n)
    print(f"[bench] model={args.model}  docs={ids}", flush=True)

    http = requests.Session(); http.headers["Accept"] = "application/pdf"
    ocr = requests.Session()

    # Pre-render all pages once (network/CPU, not counted in model time).
    docs = []
    for pid in ids:
        r = http.get(API.format(id=pid), timeout=60)
        pages = _render_pages(r.content, 150)
        docs.append((pid, [base64.b64encode(p).decode() for p in pages]))
    total_pages = sum(len(p) for _, p in docs)
    print(f"[bench] {len(docs)} docs, {total_pages} pages rendered", flush=True)

    per_doc = []
    t_all = time.time()
    tok_all = chars_all = 0
    for pid, pages in docs:
        t0 = time.time(); tok = chars = 0
        for b64 in pages:
            txt, ct = ocr_page(ocr, args.url, args.model, b64, args.timeout, prompt, use_temp)
            tok += ct; chars += len(txt)
        dt = time.time() - t0
        per_doc.append({"id": pid, "pages": len(pages), "sec": round(dt, 1),
                        "tokens": tok, "chars": chars})
        tok_all += tok; chars_all += chars
        print(f"  id={pid} pages={len(pages)} {dt:.1f}s tok={tok} "
              f"({tok/dt:.1f} tok/s)", flush=True)
    wall = time.time() - t_all

    result = {
        "model": args.model, "n_docs": len(docs), "total_pages": total_pages,
        "wall_sec": round(wall, 1),
        "sec_per_doc": round(wall / len(docs), 1),
        "sec_per_page": round(wall / total_pages, 1),
        "total_tokens": tok_all, "tokens_per_sec": round(tok_all / wall, 1),
        "total_chars": chars_all, "per_doc": per_doc,
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\n[bench] {args.model}: wall={wall:.1f}s  "
          f"{result['sec_per_page']}s/page  {result['tokens_per_sec']} tok/s"
          f"  → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
