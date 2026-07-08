#!/usr/bin/env bash
# End-to-end paragraph-granularity backgrounds for ALL four models.
#
#   corpus.parquet ─▶ corpus_paragraphs.parquet ─▶ embed (per model)
#                   ─▶ fit ZCA (full MRL grid) ─▶ index
#
# One launch covers Qwen3-4B + Qwen3-8B (OpenRouter) and
# text-embedding-3-small + -large (api.openai.com), writing
#   backgrounds/<model>_pl_mixed50k_paragraphs_mrl<dim>/
# for every model × dim in the grid — mirroring the doc/chunks/
# segments/kw families already shipped.
#
# All paragraph embeddings land under their OWN out-root
# (data/paragraphs_corpus/) so resume state and cost reports never
# collide with the doc / chunks / segments / kw runs (GOTCHAS + the
# parallel-session race note: a shared out-root would let
# detect_resume_state append paragraph rows after doc embeddings and
# silently poison Σ).
#
# Idempotent at every step:
#   - corpus_paragraphs.parquet is reused if present
#   - chunks_<slug>/*.npy resume from the highest existing chunk
#   - fit overwrites W_A.npy / mu_A.npy / eigvals_A.npy in the target dir
#
# Required env (from .env):
#   OPENROUTER_API_KEY   — Qwen3 embedders
#   OPENAI_API_KEY       — text-embedding-3-* embedders
#
# Optional env:
#   MODELS_FILTER  — space-separated <short> names to run a subset,
#                    e.g. MODELS_FILTER="qwen3_4b te3small"
#   START_BATCH / MAX_BATCH — OpenRouter adaptive batch bounds
#   PROVIDER_ORDER — CSV of preferred OpenRouter providers (Qwen only)
#
# Usage:
#   bash scripts/run_paragraphs.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python}"

if [ -f .env ]; then
    set -a; . ./.env; set +a
fi

OUT_ROOT="data/paragraphs_corpus"
CORPUS="data/corpus_paragraphs.parquet"
START_BATCH="${START_BATCH:-16}"
MAX_BATCH="${MAX_BATCH:-32}"
PROVIDER_ORDER="${PROVIDER_ORDER:-}"
OPENROUTER_URL="https://openrouter.ai/api/v1/embeddings"
OPENAI_URL="https://api.openai.com/v1/embeddings"

# <short>|<model-id>|<base-url>|<api-key-env>|<dims>
SPECS=(
    "qwen3_4b|qwen/qwen3-embedding-4b|$OPENROUTER_URL|OPENROUTER_API_KEY|2560 1536 1024 768 512"
    "qwen3_8b|qwen/qwen3-embedding-8b|$OPENROUTER_URL|OPENROUTER_API_KEY|4096 3072 2048 1024 768 512"
    "te3small|text-embedding-3-small|$OPENAI_URL|OPENAI_API_KEY|1536 1024 768 512 256"
    "te3large|text-embedding-3-large|$OPENAI_URL|OPENAI_API_KEY|3072 2048 1536 1024 768 512 256"
)

# ---- preflight: the keys we actually need for the selected models ----
need_openrouter=0; need_openai=0
for SPEC in "${SPECS[@]}"; do
    IFS='|' read -r SHORT MODEL BASE_URL KEY_ENV DIMS <<< "$SPEC"
    if [ -n "${MODELS_FILTER:-}" ] && [[ " $MODELS_FILTER " != *" $SHORT "* ]]; then
        continue
    fi
    [ "$KEY_ENV" = "OPENROUTER_API_KEY" ] && need_openrouter=1
    [ "$KEY_ENV" = "OPENAI_API_KEY" ] && need_openai=1
done
if [ "$need_openrouter" = 1 ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set (needed for Qwen). See .env.example." >&2
    exit 2
fi
if [ "$need_openai" = 1 ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY not set (needed for text-embedding-3-*). See .env.example." >&2
    exit 2
fi

# ---- Phase 1: build the paragraph corpus (once, shared by all models) ----
echo "==> Phase 1: build paragraph corpus → $CORPUS"
$PY scripts/build_corpus_paragraphs.py --corpus data/corpus.parquet --out "$CORPUS"

WANT=$($PY -c "import pyarrow.parquet as pq, sys; print(pq.read_table(sys.argv[1], columns=['sha']).num_rows)" "$CORPUS")
echo "==> paragraph corpus rows: $WANT"

# ---- Phase 2+3: per model, embed then fit the full MRL grid ----
for SPEC in "${SPECS[@]}"; do
    IFS='|' read -r SHORT MODEL BASE_URL KEY_ENV DIMS <<< "$SPEC"
    if [ -n "${MODELS_FILTER:-}" ] && [[ " $MODELS_FILTER " != *" $SHORT "* ]]; then
        echo "==> skip $SHORT (not in MODELS_FILTER)"
        continue
    fi

    SLUG="$(echo "$MODEL" | tr '/' '_' | tr ':' '_')"
    CHUNKS="$OUT_ROOT/chunks_$SLUG"

    echo "==> Phase 2: embed $MODEL  →  $CHUNKS/"
    EMBED_ARGS=(
        --model "$MODEL"
        --corpus "$CORPUS"
        --out "$OUT_ROOT"
        --start-batch "$START_BATCH"
        --max-batch "$MAX_BATCH"
        --base-url "$BASE_URL"
        --api-key-env "$KEY_ENV"
    )
    if [ "$KEY_ENV" = "OPENROUTER_API_KEY" ]; then
        EMBED_ARGS+=(--provider-order "$PROVIDER_ORDER")
    fi
    $PY scripts/embed_via_openrouter.py "${EMBED_ARGS[@]}"

    # Refuse to fit on a partial embed (parallel-session race guard):
    # the manifest must have one row per corpus paragraph.
    MANIFEST="$OUT_ROOT/manifest_$SLUG.jsonl"
    ROWS=$(wc -l < "$MANIFEST" 2>/dev/null || echo 0)
    if [ "$ROWS" -lt "$WANT" ]; then
        echo "==> SKIP fit $SHORT — manifest has $ROWS/$WANT rows (embed incomplete?)" >&2
        continue
    fi

    for DIM in $DIMS; do
        NAME="${SHORT}_pl_mixed50k_paragraphs_mrl${DIM}"
        echo "==> Phase 3: fit ZCA $MODEL  dim=$DIM  →  backgrounds/$NAME/"
        $PY scripts/fit_zca.py \
            --chunks "$CHUNKS" \
            --name "$NAME" \
            --model "$MODEL" \
            --corpus "$CORPUS" \
            --cost-report "$OUT_ROOT/cost_report_$SLUG.json" \
            --truncate-to "$DIM"
    done
done

# ---- Phase 4: regenerate the registry ----
echo "==> Phase 4: regenerate registry"
$PY scripts/index_backgrounds.py

echo "==> DONE.  Review, then:  git add backgrounds REGISTRY.md registry.json scripts/ README*.md AGENTS.md GOTCHAS.md"
