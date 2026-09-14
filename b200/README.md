# FM Pochi — B200 FP8 deployment and short-answer evaluation

This fork-only directory contains the tested B200 SM100 profile. The upstream
H200 deployment remains unchanged at the repository root.

| | Upstream H200 | This B200 profile |
|---|---|---|
| topology | TP2 x DP4 | TP4 x DP2 |
| weights / KV | BF16 / BF16 | online FP8 / FP8 E4M3 |
| attention | FA3, page 1 | `trtllm_mha`, page 128 |
| DFlash | enabled | disabled |

For the original generate-verify-refine harness adapted to integer short answers,
see [harness/README.md](harness/README.md).

Everything needed to sample this model at scale on this host's 8x B200.
The environment is already built. Only model weights live on NFS; source,
runtime, caches, state, datasets, logs and results are local under `/data`.

## Run

```bash
cd /data/home/ycc/work/fm-pochi/upstream/b200
source env.sh
RUN_DIR="${ARC_RUNS:-/data/home/ycc/work/runs}/fm-pochi-direct-$(date -u +%Y%m%dT%H%M%SZ)"

./start.sh
until curl -sf http://127.0.0.1:30000/v1/models | grep -q fm-pochi; do sleep 10; done

python run_direct.py --problems problems.csv --reference answers.csv \
  --k 10 --run-dir "$RUN_DIR"

./stop.sh
```

`problems.csv` needs two columns, `id` and `problem`:

```csv
id,problem
p001,"Find the remainder when 7^2025 is divided by 1000."
p002,"..."
```

`answers.csv` needs `id,answer` with the same IDs. Gold answers are used only
for scoring. Omitting both CSV arguments uses the Git-tracked local AIMO3 public
10 set in `data/aimo3-reference/`.

Each sample saves its request, raw SSE stream, full response (including reasoning)
and result under `samples/`. `status.json` reports progress. Repeat the same
command and output directory to resume. After the artifact audit passes,
`summary.json` reports sample accuracy and per-question pass@k; `results.jsonl`
contains all final sample records. Length-truncated samples count as failures.
The direct runner uses real async HTTP with 100 concurrent requests by default.

## The prompt

`prompt.txt` holds the system prompt. Edit it directly; `run_direct.py` reads it at
startup. `--prompt <file>` selects a different one.

Two things to know if you change it:

- The model's chat template prepends the system message as **raw text with no
  separator**, then `<｜User｜>` and the problem. Keep it self-contained.
- Generation always starts inside `<think>`, and the server splits reasoning from
  the answer with the `deepseek-r1` reasoning parser. **Only the text after
  `</think>` is parsed**, so the instruction about the *final response* is the one
  that decides what the parser sees.

The parser accepts a single integer inside `\boxed{}` and nothing else. Problems
with non-integer answers need a different extractor in `run_direct.py`.

## Sampling settings

Temperature 1.0, top-p 0.95, max output 131,072 tokens including reasoning.
The checkpoint's `generation_config.json` suggests temperature 0.6; 1.0 is what
this setup was configured and exercised with.

## Server settings

TP4 x DP2 over GPUs 0-7, FP8 linear weights and FP8 E4M3 KV, `trtllm_mha`, CUDA
graphs. Context 262,144; 64 running requests per replica, 128 in total.
API at `http://127.0.0.1:30000/v1`,
model name `fm-pochi`.

FP8 weights are quantized at load time from the original BF16 checkpoint using
`--quantization fp8`; no separate FP8 checkpoint is created. Norms, embeddings
and other unquantized operations retain BF16. KV scales use the runtime default
1.0 (no calibrated KV-scale file). The old BF16/FA4 throughput measurements do
not describe this FP8 profile.

Precision, backend, context and per-replica capacity can be set with
`POCHI_QUANTIZATION`, `POCHI_KV_DTYPE`, `POCHI_ATTENTION_BACKEND`,
`POCHI_CONTEXT_LENGTH` and `POCHI_MAX_RUNNING_REQUESTS`. To select the previous
BF16 batch profile:

```bash
POCHI_QUANTIZATION=bf16 POCHI_KV_DTYPE=auto POCHI_ATTENTION_BACKEND=fa4 \
POCHI_CONTEXT_LENGTH=139264 POCHI_MAX_RUNNING_REQUESTS=192 ./start.sh
```

Storage paths are independent:

| Variable | Default | Storage |
|---|---|---|
| `FM_POCHI_MODEL_ROOT` | `/nfs/aimo/shared/fm-pochi/models` | NFS, read-only weights |
| `FM_POCHI_RUNTIME` | `b200/runtime` | local `/data` runtime |
| `FM_POCHI_DATA_ROOT` | `b200/data` | Git-tracked local data |
| `FM_POCHI_STATE_ROOT` | `$ARC_RUNS/_state/fm-pochi-$USER` | local mutable state |

No start, stop or evaluation path writes to NFS. The only runtime NFS access is
through `FM_POCHI_MODEL_ROOT` for model and tokenizer files.

Watch the KV pool while a real job is running:

```bash
grep -oE 'full token usage: [0-9.]+.*swa token usage: [0-9.]+' server.log | tail -20
grep -ci retract server.log
```

If `swa token usage` sits near 1.0 or the retraction count climbs, lower
`--max-running-requests` in `start.sh` (and `--cuda-graph-max-bs-decode` with it —
batches above that value fall out of CUDA graphs). Retraction is not a failure:
requests are recomputed, results stay correct, throughput drops.

## Things that will bite you

- **First start after a reboot is slow** — several minutes, because the 66 GB of
  weights come from NFS with a cold page cache. Watch `server.log` in the run
  directory `start.sh` prints; it has not hung.
- **`start.sh` refuses to start if any GPU is busy.** This is a shared machine.
- **`start.sh` must source `env.sh`.** It sets
  `SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER=0.125`, which controls how often
  out-of-window SWA KV is released. At the upstream default of 1.0 the per-request
  SWA footprint is far larger and the KV sizing in `start.sh` no longer fits.
- **DFlash is off.** `POCHI_DFLASH=1` is accepted only with an explicitly selected
  BF16 model/KV and FA4 profile; FP8+DFlash has not been validated.

## Files

| | |
| --- | --- |
| `start.sh` / `stop.sh` | launch and stop the server |
| `run_direct.py` | direct pass@k with SSE archive, resume and final audit |
| `run_passk.py` | legacy batch client (default thread pool limits effective concurrency) |
| `prompt.txt` | the system prompt |
| `env.sh` | environment; sourced by `start.sh`, not optional |
| `runtime/` | the pinned Python runtime, 11 GB |
| `setup.sh`, `bootstrap_runtime.py`, `relocate_runtime.py`, `patch_fa4_decode.py`, parent repo | rebuild or relocate the runtime |

`stop.sh` checks PID, start_ticks, cmdline and process-group leadership before
signalling, so it cannot kill a server it did not start.

## Rebuilding the runtime

`runtime/` came from a SHA256-pinned container layer on ghcr.io. To recreate it
here or on another host:

```bash
FM_POCHI_RUNTIME_STORAGE=/path/on/local/disk ./setup.sh
```

Needs network access to ghcr.io. Keep it on local disk, not NFS — it is ~88k small
files and import latency dominates.

The former NFS handoff metadata, code and result bundle was archived locally
before NFS cleanup; see [NFS-MIGRATION.md](NFS-MIGRATION.md).

## AIMO3 evaluation

<!-- AIMO3_RESULTS_START -->

State: **paused by user** at 2026-09-14 04:52 UTC. The supervisor, current
attempt and SGLang service stopped cleanly; GPUs 0-7 and port 30000 were released.

| Mode | k | Completed attempts | Correct attempts | pass@k |
|---|---:|---:|---:|---:|
| direct | 10 | 100/100 | 60 | 90.0% |
| medium | 4 | 10/40 | 8 | pending |
| high | 2 | 0/20 | 0 | pending |
| xhigh | 1 | 0/10 | 0 | pending |

Medium replicate 0 completed all 10 problems and selected 8 correct answers.
This is one sample per problem, so it is not a final pass@4 result. Medium
replicate 1 stopped during `26de63` round 1 after 32 generations and 509/512
verifier calls (541 successful calls, 4,504,846 completion tokens, no failed
call records). Its artifacts are checkpointed and resumable. High and xhigh
were not started.

[Per-problem results and validation details](harness/RESULTS.md).

<!-- AIMO3_RESULTS_END -->

[Direct per-problem results and audit](harness/DIRECT-RESULTS.md).
[Deployment and evaluation bug fixes](harness/BUGFIXES.md).
