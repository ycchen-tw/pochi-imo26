#!/usr/bin/env bash
# Start FM Pochi on 8x B200. Precision and backend are recorded per run.
set -Eeuo pipefail
umask 0002
CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# env.sh is load-bearing, not convenience: it sets
# SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER=0.125, which keeps the per-request SWA
# footprint at ~4736 tokens. At the upstream default of 1.0 it is ~8320 and the
# KV sizing below no longer fits.
source "$CODE_DIR/env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export POCHI_PORT="${POCHI_PORT:-30000}"
export POCHI_DFLASH="${POCHI_DFLASH:-0}"
export POCHI_QUANTIZATION="${POCHI_QUANTIZATION:-fp8}"
export POCHI_KV_DTYPE="${POCHI_KV_DTYPE:-fp8_e4m3}"
export POCHI_ATTENTION_BACKEND="${POCHI_ATTENTION_BACKEND:-trtllm_mha}"
export POCHI_CONTEXT_LENGTH="${POCHI_CONTEXT_LENGTH:-262144}"
export POCHI_MAX_RUNNING_REQUESTS="${POCHI_MAX_RUNNING_REQUESTS:-64}"
MODEL="$FM_POCHI_ROOT/models/opd-32b-bf16-step-225"
QUANT_ARGS=()
case "$POCHI_QUANTIZATION" in
  fp8) QUANT_ARGS=(--quantization fp8) ;;
  bf16) ;;
  *) echo 'POCHI_QUANTIZATION must be fp8 or bf16.' >&2; exit 1 ;;
esac
if [[ "$POCHI_KV_DTYPE" == fp8* && "$POCHI_ATTENTION_BACKEND" == fa4 ]]; then
  echo 'This pinned FA4 backend does not support the required FP8 KV path; use trtllm_mha.' >&2
  exit 1
fi

# Speculative decoding is opt-in. Measured +31% at 128 concurrent requests, but
# NOT measured at the 384 this config targets, where compute is already
# saturated. Benchmark before trusting it.
SPEC_ARGS=()
case "$POCHI_DFLASH" in
  0) ;;
  1)
    if [[ "$POCHI_QUANTIZATION" != bf16 || "$POCHI_KV_DTYPE" != auto || "$POCHI_ATTENTION_BACKEND" != fa4 ]]; then
      echo 'DFlash has only been checked with BF16 model/KV and FA4; select that profile explicitly.' >&2
      exit 1
    fi
    DRAFT="$FM_POCHI_ROOT/models/dflash-32b-draft-v2test-phaseL"
    test -f "$DRAFT/config.json"
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1   # draft says 65536, target is 139264
    export SGLANG_DFLASH_DRAFT_RING=1
    export SGLANG_DFLASH_DRAFT_RING_QUOTA=4
    SPEC_ARGS=(--speculative-algorithm DFLASH --speculative-draft-model-path "$DRAFT"
      --speculative-dflash-block-size 8 --speculative-num-draft-tokens 8
      --speculative-draft-window-size 512 --speculative-draft-attention-backend fa4)
    ;;
  *) echo 'POCHI_DFLASH must be 0 or 1.' >&2; exit 1 ;;
esac

test -f "$FM_POCHI_ROOT/runtime/READY.json" || { echo 'Run setup first: runtime/READY.json missing.' >&2; exit 1; }
test -f "$MODEL/config.json"

"$VENV/bin/python" - <<'PY'
import os, socket, subprocess
devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
if len(devices) != 8 or set(devices) != set(map(str, range(8))):
    raise SystemExit('This TP4 x DP2 launcher requires physical GPUs 0 through 7 exactly once.')
busy = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).strip()
if busy:
    raise SystemExit('GPU processes already exist; refusing to start:\n' + busy)
with socket.socket() as s:
    s.bind(('127.0.0.1', int(os.environ['POCHI_PORT'])))
PY

RUNS="${ARC_RUNS:-/data/home/ycc/work/runs}"
RUN_DIR="${RUN_DIR:-$RUNS/fm-pochi-$(id -un)-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_DIR"
export RUN_DIR
nvidia-smi > "$RUN_DIR/gpu-before.txt"
cd "$RUN_DIR"

nohup setsid "$VENV/bin/python" -u -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name fm-pochi \
  --tp 4 --dp 2 --load-balance-method round_robin \
  --host 127.0.0.1 --port "$POCHI_PORT" \
  --dtype bfloat16 "${QUANT_ARGS[@]}" --kv-cache-dtype "$POCHI_KV_DTYPE" \
  --attention-backend "$POCHI_ATTENTION_BACKEND" --page-size 128 \
  --context-length "$POCHI_CONTEXT_LENGTH" --chunked-prefill-size 4096 \
  --max-running-requests "$POCHI_MAX_RUNNING_REQUESTS" --max-total-tokens 5502848 \
  --swa-full-tokens-ratio 0.2144 --mem-fraction-static 0.85 \
  --cuda-graph-max-bs-decode "$POCHI_MAX_RUNNING_REQUESTS" \
  --reasoning-parser deepseek-r1 --random-seed 0 \
  "${SPEC_ARGS[@]}" \
  > "$RUN_DIR/server.log" 2>&1 < /dev/null &
export POCHI_SERVER_PID=$!

"$VENV/bin/python" - <<'PY'
import json, os
from pathlib import Path
pid = int(os.environ['POCHI_SERVER_PID'])
state = {'pid':pid, 'start_ticks':Path(f'/proc/{pid}/stat').read_text().split()[21],
         'run_dir':os.environ['RUN_DIR'], 'port':int(os.environ['POCHI_PORT']),
         'gpus':os.environ['CUDA_VISIBLE_DEVICES'], 'tp':4, 'dp':2,
         'dflash':os.environ['POCHI_DFLASH']=='1',
         'quantization':os.environ['POCHI_QUANTIZATION'],
         'kv_cache_dtype':os.environ['POCHI_KV_DTYPE'],
         'attention_backend':os.environ['POCHI_ATTENTION_BACKEND'],
         'context_length':int(os.environ['POCHI_CONTEXT_LENGTH']),
         'max_running_requests':int(os.environ['POCHI_MAX_RUNNING_REQUESTS']),
         'client_sampling':{'temperature':1.0,'top_p':0.95,'max_tokens':131072}}
Path(os.environ['RUN_DIR'],'service.json').write_text(json.dumps(state,indent=2)+'\n')
Path(os.environ['FM_POCHI_ROOT'],'service.json').write_text(json.dumps(state,indent=2)+'\n')
print(f"Starting PID {pid}; log: {state['run_dir']}/server.log")
print(f"API: http://127.0.0.1:{state['port']}/v1   (model name: fm-pochi)")
PY
