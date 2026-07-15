#!/usr/bin/env bash
# End-to-end medical (med_pl) backgrounds for ALL four models.
#
#   corpus_med_pl.parquet ─▶ [derive <gran>] ─▶ embed (per model)
#                          ─▶ fit ZCA (full MRL grid) ─▶ index
#
# Mirrors run_paragraphs.sh (same proven embed/fit hydraulics) but on
# the Polish MEDICAL corpus assembled by build_med_pl_corpus.py
# (ChPL drug characteristics + OCR of scanned ChPLs + PES exam
# questions).  Background names carry the corpus tag `med_pl`, e.g.
#   backgrounds/qwen3_4b_med_pl_paragraphs_mrl1024/
#
# ⚠️ COSTS CLOUD MONEY (OpenRouter + OpenAI) — this is the paid step,
# intentionally separate from the free corpus build.  Estimate for
# Qwen (both sizes) ≈ $2-4; OpenAI te3large is the expensive one.
# Trim MODELS_FILTER / GRANS to control spend.
#
# Required env (from .env): OPENROUTER_API_KEY, OPENAI_API_KEY
#
# Optional env:
#   GRANS          — granularities to build (default: "doc paragraphs";
#                    full set: "doc paragraphs segments chunks kw")
#   MODELS_FILTER  — subset of <short> names (qwen3_4b qwen3_8b te3small te3large)
#   START_BATCH / MAX_BATCH / PROVIDER_ORDER — OpenRouter tuning
#
# Usage:
#   bash scripts/run_med.sh                        # doc + paragraphs, all models
#   MODELS_FILTER="qwen3_4b qwen3_8b" bash scripts/run_med.sh   # Qwen only (cheap)
#   GRANS="doc" MODELS_FILTER=qwen3_4b bash scripts/run_med.sh  # minimal

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python}"

if [ -f .env ]; then
    set -a; . ./.env; set +a
fi

MED_CORPUS="${MED_CORPUS:-data/corpus_med_pl.parquet}"
OUT_BASE="${OUT_BASE:-data/med_corpus}"
START_BATCH="${START_BATCH:-16}"
MAX_BATCH="${MAX_BATCH:-32}"
PROVIDER_ORDER="${PROVIDER_ORDER:-}"
GRANS="${GRANS:-doc paragraphs}"
# Fine granularities on the long ChPL docs explode (paragraphs → ~1.1M
# rows).  A whitening Σ needs only a representative sample, so cap the
# fit set — same principle as the 50k-doc general corpus.  Set empty to
# use every row.
PARA_SAMPLE="${PARA_SAMPLE:-150000}"
# Medical `chunks`: tighter than the general 512/64 — default 300-token
# windows, 50-token overlap. Also sampled (chunks explode to ~800k rows).
CHUNK_SIZE="${CHUNK_SIZE:-300}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-50}"
CHUNK_SAMPLE="${CHUNK_SAMPLE:-150000}"
OPENROUTER_URL="https://openrouter.ai/api/v1/embeddings"
OPENAI_URL="https://api.openai.com/v1/embeddings"

# <short>|<model-id>|<base-url>|<api-key-env>|<dims>
SPECS=(
    "qwen3_4b|qwen/qwen3-embedding-4b|$OPENROUTER_URL|OPENROUTER_API_KEY|2560 1536 1024 768 512"
    "qwen3_8b|qwen/qwen3-embedding-8b|$OPENROUTER_URL|OPENROUTER_API_KEY|4096 3072 2048 1024 768 512"
    "te3small|text-embedding-3-small|$OPENAI_URL|OPENAI_API_KEY|1536 1024 768 512 256"
    "te3large|text-embedding-3-large|$OPENAI_URL|OPENAI_API_KEY|3072 2048 1536 1024 768 512 256"
)

selected() {  # $1 = short name
    [ -z "${MODELS_FILTER:-}" ] && return 0
    [[ " $MODELS_FILTER " == *" $1 "* ]]
}

# ---- preflight ----
if [ ! -f "$MED_CORPUS" ]; then
    echo "==> assembling $MED_CORPUS"
    $PY scripts/build_med_pl_corpus.py
fi
need_or=0; need_oa=0
for SPEC in "${SPECS[@]}"; do
    IFS='|' read -r SHORT _ _ KEY_ENV _ <<< "$SPEC"
    selected "$SHORT" || continue
    [ "$KEY_ENV" = "OPENROUTER_API_KEY" ] && need_or=1
    [ "$KEY_ENV" = "OPENAI_API_KEY" ] && need_oa=1
done
[ "$need_or" = 1 ] && [ -z "${OPENROUTER_API_KEY:-}" ] && { echo "ERROR: OPENROUTER_API_KEY unset" >&2; exit 2; }
[ "$need_oa" = 1 ] && [ -z "${OPENAI_API_KEY:-}" ] && { echo "ERROR: OPENAI_API_KEY unset" >&2; exit 2; }

# derive the corpus parquet for a granularity; echoes its path
derive_corpus() {  # $1 = gran
    local gran="$1"
    case "$gran" in
        doc)        echo "$MED_CORPUS"; return 0 ;;
        paragraphs) full="data/corpus_med_pl_paragraphs.parquet"
                    [ -f "$full" ] || $PY scripts/build_corpus_paragraphs.py --corpus "$MED_CORPUS" --out "$full" >&2
                    if [ -n "$PARA_SAMPLE" ]; then
                        out="data/corpus_med_pl_paragraphs_${PARA_SAMPLE}.parquet"
                        [ -f "$out" ] || $PY -c "import sys,random,pyarrow.parquet as pq; t=pq.read_table('$full'); n=t.num_rows; k=min($PARA_SAMPLE,n); idx=sorted(random.Random(42).sample(range(n),k)); pq.write_table(t.take(idx),'$out'); print(f'sampled {k} of {n} paragraphs',file=sys.stderr)" >&2
                    else
                        out="$full"
                    fi ;;
        segments)   out="data/corpus_med_pl_segments.parquet"
                    [ -f "$out" ] || $PY scripts/build_corpus_segments.py --corpus "$MED_CORPUS" --out "$out" >&2 ;;
        chunks)     full="data/corpus_med_pl_chunks_${CHUNK_SIZE}_${CHUNK_OVERLAP}.parquet"
                    [ -f "$full" ] || $PY scripts/build_corpus_chunks.py --corpus "$MED_CORPUS" --out "$full" --chunk-size "$CHUNK_SIZE" --chunk-overlap "$CHUNK_OVERLAP" >&2
                    if [ -n "$CHUNK_SAMPLE" ]; then
                        out="data/corpus_med_pl_chunks_${CHUNK_SIZE}_${CHUNK_OVERLAP}_${CHUNK_SAMPLE}.parquet"
                        [ -f "$out" ] || $PY -c "import sys,random,pyarrow.parquet as pq; t=pq.read_table('$full'); n=t.num_rows; k=min($CHUNK_SAMPLE,n); idx=sorted(random.Random(42).sample(range(n),k)); pq.write_table(t.take(idx),'$out'); print(f'sampled {k} of {n} chunks',file=sys.stderr)" >&2
                    else
                        out="$full"
                    fi ;;
        kw)         out="data/corpus_med_pl_kw.parquet"
                    [ -f "$out" ] || $PY scripts/build_corpus_keywords.py --corpus "$MED_CORPUS" --out "$out" >&2 ;;
        *) echo "unknown gran $gran" >&2; return 2 ;;
    esac
    echo "$out"
}

for GRAN in $GRANS; do
    CORPUS="$(derive_corpus "$GRAN")"
    OUT_ROOT="$OUT_BASE/$GRAN"
    echo "==> granularity=$GRAN  corpus=$CORPUS  out=$OUT_ROOT"
    WANT=$($PY -c "import pyarrow.parquet as pq,sys; print(pq.read_table(sys.argv[1],columns=['sha']).num_rows)" "$CORPUS")

    for SPEC in "${SPECS[@]}"; do
        IFS='|' read -r SHORT MODEL BASE_URL KEY_ENV DIMS <<< "$SPEC"
        selected "$SHORT" || { echo "==> skip $SHORT (filtered)"; continue; }

        SLUG="$(echo "$MODEL" | tr '/' '_' | tr ':' '_')"
        CHUNKS="$OUT_ROOT/chunks_$SLUG"
        echo "==> embed $MODEL ($GRAN) → $CHUNKS/"
        EMBED_ARGS=(--model "$MODEL" --corpus "$CORPUS" --out "$OUT_ROOT"
                    --start-batch "$START_BATCH" --max-batch "$MAX_BATCH"
                    --base-url "$BASE_URL" --api-key-env "$KEY_ENV")
        [ "$KEY_ENV" = "OPENROUTER_API_KEY" ] && EMBED_ARGS+=(--provider-order "$PROVIDER_ORDER")
        $PY scripts/embed_via_openrouter.py "${EMBED_ARGS[@]}"

        ROWS=$(wc -l < "$OUT_ROOT/manifest_$SLUG.jsonl" 2>/dev/null || echo 0)
        if [ "$ROWS" -lt "$WANT" ]; then
            echo "==> SKIP fit $SHORT/$GRAN — manifest $ROWS/$WANT (embed incomplete?)" >&2
            continue
        fi
        for DIM in $DIMS; do
            NAME="${SHORT}_med_pl_${GRAN}_mrl${DIM}"
            echo "==> fit $NAME"
            $PY scripts/fit_zca.py --chunks "$CHUNKS" --name "$NAME" --model "$MODEL" \
                --corpus "$CORPUS" --cost-report "$OUT_ROOT/cost_report_$SLUG.json" \
                --truncate-to "$DIM"
        done
    done
done

echo "==> regenerate registry"
$PY scripts/index_backgrounds.py
echo "==> DONE med_pl backgrounds"
