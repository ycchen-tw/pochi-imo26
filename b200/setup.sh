#!/usr/bin/env bash
set -Eeuo pipefail
umask 0002
CODE_DIR="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")" && pwd)"
REPO_DIR="$(cd -- "$CODE_DIR/.." && pwd)"
export FM_POCHI_RUNTIME="${FM_POCHI_RUNTIME:-$CODE_DIR/runtime}"
if [[ -f "$FM_POCHI_RUNTIME/READY.json" ]]; then
  python3 "$CODE_DIR/relocate_runtime.py" "$FM_POCHI_RUNTIME"
  echo "Runtime already prepared: $FM_POCHI_RUNTIME"
  exit 0
fi
RUN_DIR="${RUN_DIR:-${ARC_RUNS:-/data/home/ycc/work/runs}/fm-pochi-setup-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_DIR"
python3 "$CODE_DIR/bootstrap_runtime.py" "$FM_POCHI_RUNTIME" "$RUN_DIR"
python3 "$CODE_DIR/relocate_runtime.py" "$FM_POCHI_RUNTIME"
source "$CODE_DIR/env.sh"
"$FM_POCHI_RUNTIME/uv" pip install --python "$VENV/bin/python" -r "$REPO_DIR/evaluation/requirements.txt"
"$FM_POCHI_RUNTIME/uv" pip install --python "$VENV/bin/python" --no-deps --reinstall nvidia-cutlass-dsl-libs-cu13==4.5.2
bash "$REPO_DIR/sglang_patches/apply_patches.sh" "$VENV" \
  "$FM_POCHI_RUNTIME/proof-pilot/deploy/w4a8/humming_w4a8.py"
"$VENV/bin/python" "$CODE_DIR/patch_fa4_decode.py" "$VENV"
"$VENV/bin/python" "$REPO_DIR/docker/validate_cutlass_install.py"
# READY.json is written by verify_runtime.py, which the container build calls
# too -- one definition of "this runtime is correct", asserted identically
# bare-metal and in the image.
# --correctness-cases carries forward the 40 dynamic cases the split-KV patch
# was checked against on 2026-09-11. It is recorded, not re-measured; the
# backend_sha256 in the same report is what ties the claim to this patch.
"$VENV/bin/python" "$CODE_DIR/verify_runtime.py" \
  --runtime "$FM_POCHI_RUNTIME" --venv "$VENV" --correctness-cases 40
