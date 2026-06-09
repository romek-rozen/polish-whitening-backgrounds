#!/usr/bin/env bash
# End-to-end: corpus → embed (each model) → fit (each model) → index.
#
# Idempotent at every step:
#   - data/corpus.parquet is reused if present
#   - chunks_<slug>/*.npy resume from highest existing chunk
#   - fit overwrites W_A.npy / mu_A.npy / eigvals_A.npy in the target dir
#
# Required env:
#   OPENROUTER_API_KEY   — see .env.example
#
# Optional env:
#   MODELS               — space-separated OpenRouter model ids
#                          (default: both Qwen3 embedders)
#   NAME_PREFIX          — backgrounds/<NAME_PREFIX>_<model_short>_nocap
#                          (default: polish_mixed_50k_v1)
#   MAX_CHARS            — pass --max-chars N to build_corpus.py
#                          (default: unset = no cap)
#   START_BATCH          — initial batch size to OpenRouter (default: 16)
#   MAX_BATCH            — upper bound after success streaks (default: 32)
#   PROVIDER_ORDER       — CSV of preferred providers, cheapest first
#                          (default: "nebius,deepinfra" — both at ~$0.01-0.02/M)
#
# Usage:
#   cp .env.example .env   # fill OPENROUTER_API_KEY
#   pip install -r requirements.txt
#   bash scripts/run_full.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python}"

# Load .env if present (so the script also works under `bash` without `set -a`).
if [ -f .env ]; then
    set -a; . ./.env; set +a
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set. See .env.example." >&2
    exit 2
fi

MODELS_DEFAULT="qwen/qwen3-embedding-4b qwen/qwen3-embedding-8b"
MODELS="${MODELS:-$MODELS_DEFAULT}"
NAME_PREFIX="${NAME_PREFIX:-polish_mixed_50k_v1}"
START_BATCH="${START_BATCH:-16}"
MAX_BATCH="${MAX_BATCH:-32}"
PROVIDER_ORDER="${PROVIDER_ORDER:-nebius,deepinfra}"

CORPUS_ARGS=()
if [ -n "${MAX_CHARS:-}" ]; then
    CORPUS_ARGS+=(--max-chars "$MAX_CHARS")
fi

echo "==> Phase 1: build corpus"
$PY scripts/build_corpus.py "${CORPUS_ARGS[@]+"${CORPUS_ARGS[@]}"}"

short() {
    # qwen/qwen3-embedding-8b → qwen3-8b
    echo "$1" | sed -E 's|^qwen/qwen3-embedding-([0-9a-z]+)$|qwen3-\1|; s|/|-|g'
}

for MODEL in $MODELS; do
    SHORT="$(short "$MODEL")"
    SLUG="$(echo "$MODEL" | tr '/' '_' | tr ':' '_')"
    NAME="${NAME_PREFIX}_${SHORT}_nocap"
    echo "==> Phase 2: embed $MODEL  →  data/chunks_${SLUG}/"
    $PY scripts/embed_via_openrouter.py \
        --model "$MODEL" \
        --start-batch "$START_BATCH" \
        --max-batch "$MAX_BATCH" \
        --provider-order "$PROVIDER_ORDER"

    echo "==> Phase 3: fit ZCA $MODEL  →  backgrounds/${NAME}/"
    $PY scripts/fit_zca.py \
        --chunks "data/chunks_${SLUG}" \
        --name "$NAME" \
        --model "$MODEL"
done

echo "==> Phase 4: regenerate registry"
$PY scripts/index_backgrounds.py

echo "==> DONE.  Next:  cd $REPO_ROOT  &&  git add backgrounds REGISTRY.md registry.json scripts/ requirements.txt README.md README.pl.md .env.example .gitignore  &&  git commit -m 'add ...'  &&  git push"
