#!/usr/bin/env bash
set -Eeuo pipefail
umask 0002
CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$CODE_DIR/.." && pwd)"
export FM_POCHI_RUNTIME="${FM_POCHI_RUNTIME:-$CODE_DIR/runtime}"
if [[ -f "$FM_POCHI_RUNTIME/READY.json" ]]; then
  echo "Runtime already prepared: $FM_POCHI_RUNTIME"
  exit 0
fi
RUN_DIR="${RUN_DIR:-${ARC_RUNS:-/data/home/ycc/work/runs}/fm-pochi-setup-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_DIR"
python3 "$CODE_DIR/bootstrap_runtime.py" "$FM_POCHI_RUNTIME" "$RUN_DIR"
source "$CODE_DIR/env.sh"
"$FM_POCHI_RUNTIME/uv" pip install --python "$VENV/bin/python" -r "$REPO_DIR/evaluation/requirements.txt"
"$FM_POCHI_RUNTIME/uv" pip install --python "$VENV/bin/python" --no-deps --reinstall nvidia-cutlass-dsl-libs-cu13==4.5.2
bash "$REPO_DIR/sglang_patches/apply_patches.sh" "$VENV" \
  "$FM_POCHI_RUNTIME/proof-pilot/deploy/w4a8/humming_w4a8.py"
"$VENV/bin/python" "$CODE_DIR/patch_fa4_decode.py" "$VENV"
"$VENV/bin/python" "$REPO_DIR/docker/validate_cutlass_install.py"
"$VENV/bin/python" - <<'PY'
import hashlib, importlib.metadata, json, os, sys
from pathlib import Path
import sglang, torch, flash_attn
runtime = Path(os.environ['FM_POCHI_RUNTIME'])
venv = Path(os.environ['VENV'])
assert Path(sys.base_prefix).resolve() == (runtime/'pybase').resolve()
model = venv/'lib/python3.12/site-packages/sglang/srt/models/olmo2.py'
assert 'class Olmo3SinkForCausalLM' in model.read_text()
packages = ['sglang','torch','transformers','flash-attn-4','flashinfer-python',
            'nvidia-cutlass-dsl','nvidia-cutlass-dsl-libs-cu13']
versions = {name:importlib.metadata.version(name) for name in packages}
report = {'upstream_commit':'5d23e406e150088c4634afe83db8468c483f2fa1',
          'runtime_layer_sha256':'27c911493f490231f95909cb831ce7d958cd5f2604968dedde7930744708c130',
          'packages':versions, 'olmo3_sink_patch_sha256':hashlib.sha256(model.read_bytes()).hexdigest()}
backend = venv/'lib/python3.12/site-packages/sglang/srt/layers/attention/flashattention_backend.py'
assert 'FM_POCHI_FA4_DECODE_SPLITKV_V1' in backend.read_text()
report['fa4_decode_splitkv'] = {'marker':'FM_POCHI_FA4_DECODE_SPLITKV_V1',
    'backend_sha256':hashlib.sha256(backend.read_bytes()).hexdigest(),
    'splits_by_graph_batch':{'1':32,'2':16,'4':8,'8':4},
    'scope':'Pochi TP2 BF16 full-attention decode only'}
(runtime/'READY.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
PY
