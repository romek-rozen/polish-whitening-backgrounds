#!/usr/bin/env bash
# Embed the mixed keyword corpus with all three CPU-servable models and fit one
# ZCA background per model.
#
# GPU is used here on purpose: fitting is a one-off offline job. The *benchmark*
# stays CPU-only (see AGENTS.md) — these backgrounds are then applied to
# CPU-produced embeddings, which is valid because the same model produces the
# same vectors on either device.
#
# Names carry `pl_kwmix900k`, NOT `pl_mixed50k_kw`: this is a different corpus
# (Wikipedia titles + web n-grams + msmarco-pl queries, all lowercased) and its
# backgrounds are not comparable to the shipped keyword backgrounds.
#
# Usage:  bash run_kwmix_backgrounds.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="$REPO_ROOT/pl-keyword-embedding-cpu-bench"
WORK="$BENCH/work"
CORPUS="$WORK/corpus_kwmix.parquet"
PY_GPU="$REPO_ROOT/.venv-cuda/bin/python"
PY="$REPO_ROOT/.venv/bin/python"

cd "$REPO_ROOT"

# <key>|<model dir>|<dim>|<canonical model id>
#
# The last field is what lands in the background's meta.json. It must be the
# real HuggingFace id, never our internal short key: once a background dir is
# copied somewhere else, that string is the only record of which model it was
# fitted for.
MODELS=(
    "bgem3|$HOME/models/bge-m3|1024|BAAI/bge-m3"
    "qwen3_06b|$HOME/models/qwen3-embedding-0.6b|1024|Qwen/Qwen3-Embedding-0.6B"
    "embgemma|$HOME/models/embeddinggemma-300m|768|google/embeddinggemma-300m"
)

for SPEC in "${MODELS[@]}"; do
    IFS='|' read -r KEY MODEL_DIR DIM MODEL_ID <<< "$SPEC"
    CHUNKS="$WORK/chunks_kwmix_$KEY"
    NAME="${KEY}_pl_kwmix900k_mrl${DIM}"

    echo "=============================================================="
    echo "==> $KEY  (dim $DIM)  ->  $NAME"
    echo "=============================================================="

    # Prompts stay off: only embeddinggemma has them, and its Clustering prompt
    # measurably hurt short phrases. Using it for one model would also make the
    # three backgrounds inconsistent with each other.
    "$PY_GPU" scripts/embed_local_st.py \
        --corpus "$CORPUS" \
        --model "$MODEL_DIR" \
        --out "$CHUNKS" \
        --device cuda \
        --batch-size 256

    "$PY" scripts/fit_zca.py \
        --chunks "$CHUNKS" \
        --name "$NAME" \
        --model "$MODEL_ID" \
        --corpus "$CORPUS" \
        --out "$WORK/backgrounds"
done

echo "=============================================================="
echo "ALL BACKGROUNDS DONE"
ls -1 "$WORK/backgrounds"
