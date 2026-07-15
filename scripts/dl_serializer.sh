#!/usr/bin/env bash
# Serialize the big downloads one at a time (they choke the link in
# parallel), in the order Roman asked:
#   1. diffusiongemma  2. gemma-31B  3. cu130-nightly image  4. resume OCR
# Runs detached in its own tmux; survives session moves.
set -uo pipefail
REPO=/home/spark001/Spark-testy/polish-whitening-backgrounds
DOCKER=/home/spark001/Spark-testy/llm-extraction-embedding-dgx-final-setup/scripts/docker
MODELS=/home/spark001/Spark-testy/llm-extraction-embedding-dgx-final-setup/models
PY=$REPO/.venv/bin/python
HF=$REPO/.venv/bin/hf
cd "$REPO"
LOG=data/med_pl/serializer.log
exec >> "$LOG" 2>&1

echo "[serial] $(date -u) STEP 1: wait diffusiongemma download"
until grep -q "DL_EXIT" data/med_pl/dl_diff.log 2>/dev/null; do sleep 30; done
echo "[serial] $(date -u) diffusiongemma done ($(grep -aoE 'DL_EXIT=[0-9]+' data/med_pl/dl_diff.log|tail -1))"

echo "[serial] $(date -u) STEP 2: resume+finish gemma-31B"
$HF download RedHatAI/gemma-4-31B-it-FP8-block --local-dir "$MODELS/gemma-4-31B-it-FP8-block" >> data/med_pl/dl31b.log 2>&1
echo "DL_EXIT=$?" >> data/med_pl/dl31b.log
echo "[serial] $(date -u) 31B done"

echo "[serial] $(date -u) STEP 3: ensure cu130-nightly image present"
until docker images | grep -q "vllm/vllm-openai.*cu130-nightly" || grep -q "PULL_EXIT" data/med_pl/pull_vllm.log 2>/dev/null; do sleep 30; done
echo "[serial] $(date -u) image ready"

echo "[serial] $(date -u) STEP 4: restart gemma-26B SOLO + resume OCR"
( cd "$DOCKER" && GPU_MEM_LLM=0.75 docker compose --profile extract up -d )
until docker ps --format '{{.Names}} {{.Status}}' | grep -qE 'vllm-gemma4.*healthy'; do sleep 10; done
echo "[serial] $(date -u) gemma healthy — launching OCR"
: > data/med_pl/ocr.log
setsid bash -c "cd '$REPO' && $PY scripts/build_med_pl_corpus_chpl_ocr.py --api openai --url http://localhost:8001/v1/chat/completions --model gemma4-26b-it-fp8 --workers 8 >> data/med_pl/ocr.log 2>&1; echo OCR_EXIT=\$? >> data/med_pl/ocr.log" >/dev/null 2>&1 &
echo "[serial] $(date -u) DONE — all downloads finished, OCR resumed"
