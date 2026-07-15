#!/usr/bin/env bash
# 3-way OCR benchmark on 10 fixed scanned ChPL docs:
#   gemma-4-26B (MoE)  vs  gemma-4-31B (dense)  vs  diffusiongemma (block diffusion)
# Waits for all artifacts, serves one model at a time on :8001, measures
# wall time + tokens/s with scripts/ocr_bench.py. Detached in tmux.
set -uo pipefail
REPO=/home/spark001/Spark-testy/polish-whitening-backgrounds
DOCKER=/home/spark001/Spark-testy/llm-extraction-embedding-dgx-final-setup/scripts/docker
MODELS=/home/spark001/Spark-testy/llm-extraction-embedding-dgx-final-setup/models
PY=$REPO/.venv/bin/python
cd "$REPO"
OUT=data/med_pl/bench3; mkdir -p "$OUT"
exec >> "$OUT/bench.log" 2>&1
say(){ echo "[3way $(date -u +%H:%M:%S)] $*"; }

# ---- helper: wait until a served-model-name answers on :8001, or fail ----
wait_load(){  # $1 container  $2 served-model-name  $3 max_tries(10s each)
    local c="$1" m="$2" n="${3:-90}"
    for i in $(seq 1 "$n"); do
        if curl -sf -m5 http://localhost:8001/v1/models 2>/dev/null | grep -q "\"$m\""; then return 0; fi
        local st; st=$(docker inspect -f '{{.State.Status}}' "$c" 2>/dev/null || echo missing)
        if [ "$st" = "exited" ] || docker logs "$c" 2>&1 | grep -qiE "not recognize|no module|does not support|ValueError|Traceback|CUDA error|out of memory"; then
            say "$c FAILED to load:"; docker logs --tail 20 "$c" 2>&1 | grep -iE "error|not recognize|support|arch|diffusion|valueerror|cuda" | tail -8 | sed 's/^/    /'
            return 1
        fi
        sleep 10
    done
    say "$c timeout waiting for /v1/models"; return 1
}

# ================= STEP 0: wait for the essentials (NOT 31B) =================
# 31B is decoupled — it downloads slowly (HF throttle) and is the least
# interesting (dense, ~7 tok/s). Benchmark runs on 26B + diffusiongemma +
# GLM-OCR as soon as the vLLM-0.24 image is ready; 31B is included only if
# its shards happen to be complete by STEP 2.
say "STEP 0: waiting for OCR done + vLLM-0.24 image (31B optional)"
until grep -q "OCR_EXIT" data/med_pl/ocr.log 2>/dev/null; do sleep 30; done
until docker images | grep -q "vllm/vllm-openai.*nightly-aarch64"; do sleep 30; done
say "essentials present (image + OCR)."

# ---- resolve the diffusion image (nightly, + transformers-from-source if needed) ----
DIFF_IMAGE="vllm/vllm-openai:nightly-aarch64"
if docker run --rm --entrypoint python3 "$DIFF_IMAGE" -c "from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion" 2>/dev/null; then
    say "nightly-aarch64 transformers already knows diffusion_gemma"
else
    say "nightly-aarch64 transformers lacks diffusion_gemma — building image with tf from source"
    td=$(mktemp -d)
    printf 'FROM %s\nRUN pip install --no-cache-dir --upgrade "git+https://github.com/huggingface/transformers.git"\n' "$DIFF_IMAGE" > "$td/Dockerfile"
    docker build -t vllm-diffusion:local "$td" && DIFF_IMAGE="vllm-diffusion:local"
fi
say "diffusion image = $DIFF_IMAGE"

# ================= STEP 1: gemma-4-26B =================
say "STEP 1: gemma-4-26B"
( cd "$DOCKER" && GPU_MEM_LLM=0.75 docker compose --profile extract up -d )
if wait_load vllm-gemma4 gemma4-26b-it-fp8; then
    $PY scripts/ocr_bench.py --model gemma4-26b-it-fp8 --n 10 --out "$OUT/gemma26b.json"
fi

# ================= STEP 2: gemma-4-31B (only if fully downloaded) =================
G31="$MODELS/gemma-4-31B-it-FP8-block"
g31_ready=$($PY -c "
import json,os
M='$G31'
try:
    idx=json.load(open(M+'/model.safetensors.index.json'))
    files=set(idx.get('weight_map',{}).values())
    ok=all(os.path.exists(M+'/'+f) and not os.path.exists(M+'/'+f+'.incomplete') for f in files)
    print('yes' if ok else 'no')
except Exception: print('no')
")
if [ "$g31_ready" = "yes" ]; then
    say "STEP 2: gemma-4-31B (dense) — model complete, benchmarking"
    docker stop vllm-gemma4
    ( cd "$DOCKER" && docker compose --profile gemma31 up -d )
    if wait_load vllm-gemma31b gemma31 120; then
        $PY scripts/ocr_bench.py --model gemma31 --n 10 --out "$OUT/gemma31.json"
    else
        say "gemma-31B nie wystartował — pomijam"
    fi
    docker stop vllm-gemma31b 2>/dev/null || true; docker rm -f vllm-gemma31b 2>/dev/null || true
else
    say "STEP 2: gemma-31B jeszcze się ściąga (shard niekompletny) — POMIJAM (bonus później)"
fi

# ================= STEP 3: diffusiongemma =================
say "STEP 3: diffusiongemma (block diffusion, V2 runner)"
docker stop vllm-gemma4 2>/dev/null || true   # free :8001 (gemma may still be up if 31B was skipped)
docker rm -f vllm-diffusiongemma 2>/dev/null || true
docker run -d --name vllm-diffusiongemma --gpus all --ipc host -p 8001:8000 \
    -v "$MODELS/diffusiongemma-26B-A4B-it-FP8-dynamic":/model:ro \
    -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_SERVER_DEV_MODE=1 \
    --entrypoint vllm "$DIFF_IMAGE" \
    serve /model --served-model-name diffusiongemma --trust-remote-code \
    --max-num-seqs 4 --gpu-memory-utilization 0.75 --max-model-len 8192 \
    --hf-overrides '{"diffusion_sampler": "entropy_bound", "diffusion_entropy_bound": 0.1}' \
    --default-chat-template-kwargs '{"enable_thinking": true}'
if wait_load vllm-diffusiongemma diffusiongemma 120; then
    $PY scripts/ocr_bench.py --model diffusiongemma --n 10 --out "$OUT/diffusiongemma.json"
else
    say "diffusiongemma nie wystartował — pomijam (patrz błąd wyżej)"
fi
docker rm -f vllm-diffusiongemma 2>/dev/null || true

# ================= STEP 3b: GLM-OCR =================
say "STEP 3b: GLM-OCR (dedicated OCR, no MTP per D51)"
if [ -d "$MODELS/GLM-OCR" ]; then
    docker rm -f vllm-glmocr 2>/dev/null || true
    docker run -d --name vllm-glmocr --gpus all --ipc host -p 8001:8000 \
        -v "$MODELS/GLM-OCR":/model:ro -e VLLM_SERVER_DEV_MODE=1 \
        --entrypoint vllm "$DIFF_IMAGE" \
        serve /model --served-model-name glm-ocr --trust-remote-code \
        --max-num-seqs 4 --gpu-memory-utilization 0.5 --max-model-len 8192
    if wait_load vllm-glmocr glm-ocr 120; then
        $PY scripts/ocr_bench.py --model glm-ocr --prompt "Text Recognition:" --n 10 --out "$OUT/glmocr.json"
    else
        say "GLM-OCR nie wystartował — pomijam"
    fi
    docker rm -f vllm-glmocr 2>/dev/null || true
else
    say "GLM-OCR model brak — pomijam"
fi

# ================= STEP 4: restore gemma-26B =================
say "STEP 4: restore gemma-26B"
( cd "$DOCKER" && GPU_MEM_LLM=0.75 docker compose --profile extract up -d )

# ================= STEP 5: comparison =================
say "STEP 5: WYNIK"
$PY - <<'PYEOF'
import json, os
d="data/med_pl/bench3"
def load(f):
    p=os.path.join(d,f)
    return json.load(open(p)) if os.path.exists(p) else None
rows=[("gemma-4-26B (MoE)","gemma26b.json"),
      ("gemma-4-31B (dense)","gemma31.json"),
      ("diffusiongemma (diff)","diffusiongemma.json"),
      ("GLM-OCR (0.9B)","glmocr.json")]
print("\n"+"="*74)
print("  OCR BENCHMARK — te same 10 skanów ChPL")
print("="*74)
h=f"{'model':<22}{'wall_s':>9}{'s/doc':>8}{'s/page':>8}{'tok/s':>9}{'chars':>9}"
print(h); print("-"*len(h))
base=None
for name,f in rows:
    r=load(f)
    if r:
        print(f"{name:<22}{r['wall_sec']:>9}{r['sec_per_doc']:>8}{r['sec_per_page']:>8}{r['tokens_per_sec']:>9}{r['total_chars']:>9}")
        if f=="gemma26b.json": base=r
    else:
        print(f"{name:<22}{'— nie wystartował —':>43}")
if base:
    print("\n  Względem gemma-26B (wall time):")
    for name,f in rows:
        r=load(f)
        if r and f!="gemma26b.json":
            sp=base['wall_sec']/r['wall_sec']
            print(f"    {name}: {sp:.2f}x {'SZYBSZY' if sp>1 else 'WOLNIEJSZY'}")
print("="*74)
PYEOF
say "DONE"
echo "BENCH3_DONE"
