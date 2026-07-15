#!/usr/bin/env bash
# Finish med_pl DOC-level backgrounds with OCR scans included.
# Reuses the 13,514 existing embeddings (stable prefix) and embeds ONLY
# the 878 new OCR docs, refits doc-level Σ (both Qwen), indexes, commits, pushes.
set -uo pipefail
cd /home/spark001/Spark-testy/polish-whitening-backgrounds
export PY=.venv/bin/python
LOG=data/med_pl/finish_doc.log; mkdir -p data/med_pl
{
echo "[finish $(date -u +%H:%M:%S)] START doc-only reuse (+878 OCR)"
GRANS="doc" MODELS_FILTER="qwen3_4b qwen3_8b" PY=.venv/bin/python bash scripts/run_med.sh
rc=$?
echo "[finish $(date -u +%H:%M:%S)] run_med rc=$rc"
if [ "$rc" = 0 ]; then
  git add backgrounds/ data/corpus_med_pl.parquet 2>/dev/null || git add backgrounds/
  git commit -m "backgrounds: med_pl doc refit incl. OCR scans (13514+878=14392 docs) — Qwen 4b/8b" || echo "nic do commita"
  git push origin main && echo "[finish] PUSHED"
fi
echo "FINISH_DOC_DONE rc=$rc"
} >> "$LOG" 2>&1
