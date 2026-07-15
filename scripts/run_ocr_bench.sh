#!/usr/bin/env bash
# Controlled A/B OCR benchmark, run AFTER the OCR deliverable finishes.
#   1. wait OCR done   2. bench gemma-26B (already up)
#   3. swap to diffusiongemma (cu130-nightly)   4. bench it (or note it
#      failed to load)   5. restore gemma-26B   6. print comparison
# Detached in tmux; survives disconnects.
set -uo pipefail
REPO=/home/spark001/Spark-testy/polish-whitening-backgrounds
DOCKER=/home/spark001/Spark-testy/llm-extraction-embedding-dgx-final-setup/scripts/docker
PY=$REPO/.venv/bin/python
cd "$REPO"
OUT=data/med_pl/bench
mkdir -p "$OUT"
LOG="$OUT/bench.log"
exec >> "$LOG" 2>&1
say(){ echo "[bench $(date -u +%H:%M:%S)] $*"; }

say "STEP 1: wait for OCR deliverable to finish"
until grep -q "OCR_EXIT" data/med_pl/ocr.log 2>/dev/null; do sleep 60; done
say "OCR finished. unique=$($PY -c "import json;print(len({json.loads(l)['id'] for l in open('data/med_pl/chpl_ocr.jsonl') if json.loads(l).get('ok')}))")/879"

say "STEP 2: benchmark gemma-26B (10 fixed scans)"
$PY scripts/ocr_bench.py --model gemma4-26b-it-fp8 --n 10 --out "$OUT/gemma26b.json"

say "STEP 3: swap :8001 → diffusiongemma (cu130-nightly)"
docker stop vllm-gemma4
( cd "$DOCKER" && docker compose --profile diffocr up -d )

say "STEP 4: wait for diffusiongemma to load (or fail)"
loaded=0
for i in $(seq 1 60); do   # up to ~10 min
    if curl -sf -m 5 http://localhost:8001/v1/models 2>/dev/null | grep -q diffusiongemma; then loaded=1; break; fi
    if [ "$(docker inspect -f '{{.State.Status}}' vllm-diffusiongemma 2>/dev/null)" = "exited" ] \
       || docker logs vllm-diffusiongemma 2>&1 | grep -qiE "not supported|no module|ValueError|does not recognize|Traceback|out of memory"; then
        say "diffusiongemma FAILED to load — capturing error"
        docker logs --tail 25 vllm-diffusiongemma 2>&1 | grep -iE "error|not support|recognize|valueerror|arch|diffusion" | tail -8 | sed 's/^/    /'
        break
    fi
    sleep 10
done

if [ "$loaded" = 1 ]; then
    say "diffusiongemma LOADED — benchmarking"
    $PY scripts/ocr_bench.py --model diffusiongemma --n 10 --out "$OUT/diffusiongemma.json"
else
    say "SKIP diffusiongemma benchmark (did not serve on this vLLM)"
fi

say "STEP 5: restore gemma-26B on :8001"
docker stop vllm-diffusiongemma 2>/dev/null || true
( cd "$DOCKER" && GPU_MEM_LLM=0.75 docker compose --profile extract up -d )

say "STEP 6: COMPARISON"
$PY - <<'PY'
import json, os
d="data/med_pl/bench"
def load(f):
    p=os.path.join(d,f)
    return json.load(open(p)) if os.path.exists(p) else None
g=load("gemma26b.json"); x=load("diffusiongemma.json")
print("\n==================== WYNIK BENCHMARKU (10 tych samych skanów) ====================")
hdr=f"{'model':<18}{'wall_s':>9}{'s/doc':>8}{'s/page':>8}{'tok/s':>8}{'chars':>9}"
print(hdr); print("-"*len(hdr))
for r in (g,x):
    if r: print(f"{r['model']:<18}{r['wall_sec']:>9}{r['sec_per_doc']:>8}{r['sec_per_page']:>8}{r['tokens_per_sec']:>8}{r['total_chars']:>9}")
if g and x:
    sp=g['wall_sec']/x['wall_sec']
    print(f"\n  diffusiongemma jest {sp:.2f}x {'SZYBSZY' if sp>1 else 'WOLNIEJSZY'} od gemma-26B (wall time)")
elif g and not x:
    print("\n  diffusiongemma nie wystartował — porównanie niemożliwe (patrz błąd wyżej)")
PY
say "DONE"
echo "BENCH_DONE"
