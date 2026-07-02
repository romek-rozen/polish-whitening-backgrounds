#!/usr/bin/env bash
# Fit all keyword-granularity (kw) backgrounds from the embeddings
# under data/kw_corpus/.  Assumes embed_via_openrouter.py already
# finished for the relevant models (Qwen via OpenRouter, OpenAI via
# api.openai.com — see README "kw granularity").
#
# Idempotent: fit_zca overwrites the target dir; index at the end.
#
# Usage:
#   bash scripts/run_kw_fits.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python}"
CORPUS="data/corpus_keywords.parquet"
KW_DIR="data/kw_corpus"

# <short-name>|<model-id>|<chunks-slug>|<dims>
FITS=(
    "qwen3_4b|qwen/qwen3-embedding-4b|qwen_qwen3-embedding-4b|2560 1536 1024 768 512"
    "qwen3_8b|qwen/qwen3-embedding-8b|qwen_qwen3-embedding-8b|4096 3072 2048 1024 768 512"
    "te3small|text-embedding-3-small|text-embedding-3-small|1536 1024 768 512 256"
    "te3large|text-embedding-3-large|text-embedding-3-large|3072 2048 1536 1024 768 512 256"
)

for SPEC in "${FITS[@]}"; do
    IFS='|' read -r SHORT MODEL SLUG DIMS <<< "$SPEC"
    CHUNKS="$KW_DIR/chunks_$SLUG"
    if [ ! -d "$CHUNKS" ]; then
        echo "==> SKIP $SHORT — no embeddings at $CHUNKS" >&2
        continue
    fi
    # Refuse partial embeddings — an in-flight embed run leaves a
    # valid-looking chunk dir that would silently fit on a subset.
    MANIFEST="$KW_DIR/manifest_$SLUG.jsonl"
    ROWS=$(wc -l < "$MANIFEST" 2>/dev/null || echo 0)
    WANT=$($PY -c "import pyarrow.parquet as pq, sys; print(pq.read_table(sys.argv[1], columns=['sha']).num_rows)" "$CORPUS")
    if [ "$ROWS" -lt "$WANT" ]; then
        echo "==> SKIP $SHORT — manifest has $ROWS/$WANT rows (embed still running?)" >&2
        continue
    fi
    for DIM in $DIMS; do
        NAME="${SHORT}_pl_mixed50k_kw_mrl${DIM}"
        echo "==> fit $NAME"
        $PY scripts/fit_zca.py \
            --chunks "$CHUNKS" \
            --name "$NAME" \
            --model "$MODEL" \
            --corpus "$CORPUS" \
            --cost-report "$KW_DIR/cost_report_$SLUG.json" \
            --truncate-to "$DIM"
    done
done

echo "==> regenerate registry"
$PY scripts/index_backgrounds.py
echo "==> DONE"
