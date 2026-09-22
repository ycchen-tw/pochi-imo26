#!/usr/bin/env bash
# Container entrypoint for the FM Pochi B200 FP8 profile.
#
# Why this exists rather than calling b200/start.sh directly: start.sh is a
# bare-metal launcher. It backgrounds the server with `nohup setsid ... &` and
# returns, which is correct on a login node and fatal in a container -- PID 1
# exits, the container stops, and the server goes with it. Everything here runs
# the server in the FOREGROUND of its own process group, forwards signals, and
# exits with a status that means something to an orchestrator.
#
# Modes:
#   harness [args]  AIMO integer-answer generate-verify-refine run: bring up the
#                   server, wait for it, run run_math_harness.py, tear down.
#                   Exits with the HARNESS's status (2 = invalid answers).
#   direct  [args]  same lifecycle, but run_direct.py (direct sampling pass@k).
#   prepare [args]  run_math_harness.py --prepare-only. No server, no GPU.
#   serve           server only, foreground. For a separate harness container.
#   verify          runtime self-check. No GPU.
#   <other>         exec'd verbatim (e.g. `bash`).
set -Eeuo pipefail

# Image layout, overridable so this script can be exercised outside the image
# (against a bare-metal checkout) instead of only being testable by building
# 19 GB and running it.
CODE_DIR="${FM_POCHI_CODE_DIR:-/opt/fm-pochi/b200}"
REPO_DIR="${FM_POCHI_REPO_DIR:-/opt/fm-pochi}"

log() { printf '\033[1;36m[entrypoint]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[entrypoint] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# env.sh is load-bearing, not convenience -- see the comment in start.sh about
# SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER and the KV sizing that depends on it.
# Every path it resolves is ${VAR:-default}, and the image sets those defaults
# to container mount points, so sourcing it here needs no container-specific
# copy of the file.
# shellcheck disable=SC1091
source "$CODE_DIR/env.sh"

MODEL="$FM_POCHI_MODEL_ROOT/opd-32b-bf16-step-225"
export POCHI_PORT="${POCHI_PORT:-30000}"
# Loopback inside a container is the container's own, so start.sh's 127.0.0.1
# default would make a published port reach nothing. Bind all interfaces unless
# told otherwise; under Apptainer (host network) either works.
export POCHI_HOST="${POCHI_HOST:-0.0.0.0}"
# Loading 61 GB and quantizing to FP8 at load is not fast. start.sh's documented
# readiness loop has no timeout at all; a container needs one, or a server that
# dies on startup leaves the orchestrator waiting forever.
READY_TIMEOUT="${POCHI_READY_TIMEOUT_SECONDS:-2700}"

server_pid=""

# --------------------------------------------------------------- preflight
require_runtime() {
    [[ -f "$FM_POCHI_RUNTIME/READY.json" ]] \
        || die "runtime/READY.json missing -- this image was not built by b200/container/Dockerfile"
}

require_weights() {
    [[ -f "$MODEL/config.json" ]] || die "no model at $MODEL
Mount the weights read-only at \$FM_POCHI_MODEL_ROOT (default /models), e.g.
  docker:     -v /host/models:/models:ro
  apptainer:  --bind /host/models:/models:ro
The directory must contain opd-32b-bf16-step-225/ (and dflash-32b-draft-v2test-phaseL/
if POCHI_DFLASH=1)."
}

require_writable() {
    # Under a read-only rootfs (the Apptainer/SIF default, and `docker run
    # --read-only`) these must be mounts or tmpfs. Failing here with the path
    # named beats failing twenty minutes later inside a JIT compile with
    # "Read-only file system".
    local var path
    for var in FM_POCHI_STATE_ROOT FM_POCHI_CACHE TMPDIR; do
        path="${!var}"
        mkdir -p "$path" 2>/dev/null || true
        [[ -w "$path" ]] || die "\$$var ($path) is not writable.
Mount a writable volume there, or run with a writable overlay
  (docker: drop --read-only;  apptainer: --writable-tmpfs, or --bind a host dir)."
    done
}

seed_caches() {
    # Cold JIT on first launch costs minutes; the image carries a warm copy.
    local seed="$FM_POCHI_RUNTIME/cache-seed"
    if [[ -d "$seed" && -z "$(ls -A "$FM_POCHI_CACHE" 2>/dev/null)" ]]; then
        log "seeding JIT caches from $seed"
        cp -a "$seed/." "$FM_POCHI_CACHE/" 2>/dev/null || log "  seed copy incomplete (continuing)"
    fi
}

check_gpus() {
    # start.sh asserts CUDA_VISIBLE_DEVICES is exactly 0..7 because TP4 x DP2
    # needs eight devices. A container runtime renumbers the injected GPUs to
    # 0..N-1, so `--gpus all` on an 8-GPU host satisfies it; a subset does not,
    # and that has to be a hard error rather than a silently wrong topology.
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    local visible
    visible=$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | grep -c .)
    (( visible == 8 )) || die "this TP4 x DP2 profile needs exactly 8 GPUs, got $visible ($CUDA_VISIBLE_DEVICES).
Pass all eight: docker --gpus all  /  apptainer --nv with all devices visible."

    # start.sh's occupancy gate shells out to `nvidia-smi --query-compute-apps`.
    # That query reports PIDs, and a container cannot see PIDs in other
    # namespaces, so inside a container it comes back EMPTY even when the GPUs
    # are fully occupied -- the guard is still there but no longer guards
    # anything. memory.used IS visible across namespaces.
    [[ "${POCHI_SKIP_GPU_OCCUPANCY_CHECK:-0}" == 1 ]] && return 0
    local busy="" idx used
    while read -r idx used; do
        (( used > 2048 )) && busy+="  GPU ${idx%,}: ${used} MiB"$'\n'
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    [[ -z "$busy" ]] || die "GPUs already occupied; refusing to start:
$busy
This host shares GPUs. Set POCHI_SKIP_GPU_OCCUPANCY_CHECK=1 only if you know the
other memory is yours."
}

# --------------------------------------------------------------- server
start_server() {
    local run_dir="$1"
    local plan
    plan="$("$VENV/bin/python" "$CODE_DIR/server_argv.py" --model "$MODEL")" \
        || die "server_argv.py rejected this profile (see message above)"
    printf '%s\n' "$plan" > "$run_dir/server-plan.json"

    local argv=() key value
    mapfile -t argv < <(jq -r '.argv[]' <<<"$plan")
    (( ${#argv[@]} > 0 )) || die "server_argv.py produced no argv"
    # DFlash needs a few SGLANG_* variables exported alongside its flags.
    while IFS=$'\t' read -r key value; do
        [[ -n "$key" ]] || continue
        log "env $key=$value"
        export "$key=$value"
    done < <(jq -r '.env | to_entries[] | "\(.key)\t\(.value)"' <<<"$plan")

    log "profile: $(jq -c '.profile' <<<"$plan")"
    nvidia-smi > "$run_dir/gpu-before.txt" 2>&1 || true

    "$VENV/bin/python" -u -m sglang.launch_server "${argv[@]}" \
        > "$run_dir/server.log" 2>&1 &
    server_pid=$!
    log "server pid $server_pid; log $run_dir/server.log"
}

wait_ready() {
    # Readiness by MODEL NAME, not by the port answering: a port that answers is
    # not the same as this model being loaded. Also watch the server process --
    # if it dies during load, say so with the log instead of timing out blind.
    local deadline=$(( SECONDS + READY_TIMEOUT )) run_dir="$1"
    log "waiting up to ${READY_TIMEOUT}s for fm-pochi on port $POCHI_PORT"
    while (( SECONDS < deadline )); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            wait "$server_pid" 2>/dev/null || true
            log "---- last 40 lines of server.log ----"
            tail -40 "$run_dir/server.log" >&2 || true
            die "server exited during startup (see $run_dir/server.log)"
        fi
        if curl -sf "http://127.0.0.1:$POCHI_PORT/v1/models" 2>/dev/null | grep -q fm-pochi; then
            log "server ready after $(( SECONDS ))s"
            return 0
        fi
        sleep 10
    done
    log "---- last 40 lines of server.log ----"
    tail -40 "$run_dir/server.log" >&2 || true
    die "server did not report model fm-pochi within ${READY_TIMEOUT}s"
}

stop_server() {
    [[ -n "$server_pid" ]] || return 0
    kill -0 "$server_pid" 2>/dev/null || return 0
    log "stopping server $server_pid"
    kill -TERM "$server_pid" 2>/dev/null || true
    local deadline=$(( SECONDS + 120 ))
    while (( SECONDS < deadline )) && kill -0 "$server_pid" 2>/dev/null; do sleep 2; done
    kill -KILL "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
}

new_run_dir() {
    local name="$1" dir
    dir="${RUN_DIR:-$FM_POCHI_STATE_ROOT/$name-$(date -u +%Y%m%dT%H%M%SZ)}"
    mkdir -p "$dir"
    printf '%s' "$dir"
}

# --------------------------------------------------------------- dispatch
mode="${1:-serve}"
shift || true

case "$mode" in
verify)
    exec "$VENV/bin/python" "$CODE_DIR/verify_runtime.py" \
        --runtime "$FM_POCHI_RUNTIME" --venv "$VENV" "$@"
    ;;

prepare)
    # Pins inputs and prints the resolved search settings without contacting a
    # model. No GPU, no server -- the cheapest possible check that an input CSV
    # and a budget are actually accepted before booking eight B200s.
    require_runtime
    args=("$@")
    [[ " ${args[*]} " == *" --input "* ]] \
        || args=(--input "$FM_POCHI_DATA_ROOT/aimo3-reference/problems.csv" "${args[@]}")
    [[ " ${args[*]} " == *" --run-dir "* ]] \
        || args=("${args[@]}" --run-dir "$(new_run_dir prepare)")
    exec "$VENV/bin/python" "$CODE_DIR/run_math_harness.py" --prepare-only "${args[@]}"
    ;;

serve)
    require_runtime; require_weights; require_writable; seed_caches; check_gpus
    run_dir="$(new_run_dir serve)"
    log "run dir: $run_dir"
    start_server "$run_dir"
    trap 'stop_server' TERM INT
    log "API on http://${POCHI_HOST}:${POCHI_PORT}/v1 (model name: fm-pochi)"
    # Mirror the log so `docker logs` shows it, without losing the file.
    tail -f "$run_dir/server.log" &
    wait "$server_pid"
    status=$?
    log "server exited with status $status"
    exit "$status"
    ;;

harness|direct)
    require_runtime; require_weights; require_writable; seed_caches; check_gpus
    run_dir="$(new_run_dir "$mode")"
    log "run dir: $run_dir"
    start_server "$run_dir"
    trap 'stop_server; exit 143' TERM INT
    wait_ready "$run_dir"

    args=("$@")
    if [[ "$mode" == harness ]]; then
        runner="$CODE_DIR/run_math_harness.py"
        # run_math_harness.py's --url includes /v1; run_direct.py's does not.
        default_url="http://127.0.0.1:$POCHI_PORT/v1"
        [[ " ${args[*]} " == *" --input "* ]] \
            || args=(--input "$FM_POCHI_DATA_ROOT/aimo3-reference/problems.csv" "${args[@]}")
    else
        runner="$CODE_DIR/run_direct.py"
        default_url="http://127.0.0.1:$POCHI_PORT"
    fi
    [[ " ${args[*]} " == *" --url "* ]]     || args=("${args[@]}" --url "$default_url")
    [[ " ${args[*]} " == *" --run-dir "* ]] || args=("${args[@]}" --run-dir "$run_dir")

    log "running $(basename "$runner") ${args[*]}"
    set +e
    "$VENV/bin/python" "$runner" "${args[@]}"
    status=$?
    set -e
    # 2 means completed_with_invalid_answers -- a real, distinct outcome the
    # caller has to be able to see. Do not collapse it into 0 or 1.
    log "$(basename "$runner") exited with status $status"
    stop_server
    exit "$status"
    ;;

*)
    exec "$mode" "$@"
    ;;
esac
