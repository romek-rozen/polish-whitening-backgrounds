#!/usr/bin/env bash
# Fit doc- and chunk-granularity backgrounds for the OpenAI
# text-embedding-3-* models, mirroring the Qwen families shipped by
# run_full.sh.  Keyword-granularity fits live in run_kw_fits.sh.
#
# Assumes embed_via_openrouter.py already finished against
# api.openai.com for the relevant (model, corpus) pairs:
#   doc    → data/chunks_<model>/           (corpus.parquet)
#   chunks → data/chunks_corpus/chunks_<model>/  (corpus_chunks_512_64.parquet)
#
# Idempotent: fit_zca overwrites the target dir; index at the end.
#
# Usage:
#   bash scripts/run_oai_fits.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python}"

DIMS_SMALL="1536 1024 768 512 256"
DIMS_LARGE="3072 2048 1536 1024 768 512 256"

# <short>|<model-id>|<granularity>|<data-dir>|<corpus>|<dims>
FITS=(
    "te3small|text-embedding-3-small|doc|data|data/corpus.parquet|$DIMS_SMALL"
    "te3small|text-embedding-3-small|chunks|data/chunks_corpus|data/corpus_chunks_512_64.parquet|$DIMS_SMALL"
    "te3large|text-embedding-3-large|doc|data|data/corpus.parquet|$DIMS_LARGE"
    "te3large|text-embedding-3-large|chunks|data/chunks_corpus|data/corpus_chunks_512_64.parquet|$DIMS_LARGE"
)

for SPEC in "${FITS[@]}"; do
    IFS='|' read -r SHORT MODEL GRAN DIR CORPUS DIMS <<< "$SPEC"
    CHUNKS="$DIR/chunks_$MODEL"
    if [ ! -d "$CHUNKS" ]; then
        echo "==> SKIP $SHORT $GRAN — no embeddings at $CHUNKS" >&2
        continue
    fi
    # Refuse partial embeddings — an in-flight embed run leaves a
    # valid-looking chunk dir that would silently fit on a subset.
    MANIFEST="$DIR/manifest_$MODEL.jsonl"
    ROWS=$(wc -l < "$MANIFEST" 2>/dev/null || echo 0)
    WANT=$($PY -c "import pyarrow.parquet as pq, sys; print(pq.read_table(sys.argv[1], columns=['sha']).num_rows)" "$CORPUS")
    if [ "$ROWS" -lt "$WANT" ]; then
        echo "==> SKIP $SHORT $GRAN — manifest has $ROWS/$WANT rows (embed still running?)" >&2
        continue
    fi
    for DIM in $DIMS; do
        NAME="${SHORT}_pl_mixed50k_${GRAN}_mrl${DIM}"
        echo "==> fit $NAME"
        $PY scripts/fit_zca.py \
            --chunks "$CHUNKS" \
            --name "$NAME" \
            --model "$MODEL" \
            --corpus "$CORPUS" \
            --cost-report "$DIR/cost_report_$MODEL.json" \
            --truncate-to "$DIM"
    done
done

echo "==> regenerate registry"
$PY scripts/index_backgrounds.py
echo "==> DONE"
