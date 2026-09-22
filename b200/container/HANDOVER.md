# FM Pochi B200 — how to run this

Everything an operator needs, on one page. The container is self-contained
apart from the model weights, which are downloaded separately because they are
61 GB.

## What it does

Solves mathematical short-answer problems whose answer is a single integer. For
each problem it runs a generate-verify-refine search — candidate solutions, peer
verification, refinement over several rounds, then an LLM selector picks one —
and writes the selected solution's boxed integer to `submission.csv`.

Model: FM-Pochi-32B-IMO26 (step-225), served with online FP8 weights and an
FP8 E4M3 KV cache on 8× B200.

## What you need

| | |
|---|---|
| GPUs | **8× NVIDIA B200** (SM100). The topology is TP4 × DP2 and the container refuses to start with any other device count. |
| Driver | one supporting CUDA 13 (validated against 590.48.01) |
| Disk | ~19 GB for the image, ~61 GB for the weights, plus room for run outputs |
| Runtime | Docker, or Apptainer/Singularity |
| Network | none at run time. Pulling the image and the weights needs it; neither needs a login. |

## 1. Get the image

Pinned by digest, because tags can move and this one should not:

```bash
docker pull ghcr.io/ycchen-tw/pochi-imo26-b200@sha256:4efab850aec148c52b56165f31ae68f04d3c564bc956a6c3275662f50ebda278
```

Apptainer:

```bash
apptainer build pochi-b200.sif \
  docker://ghcr.io/ycchen-tw/pochi-imo26-b200@sha256:4efab850aec148c52b56165f31ae68f04d3c564bc956a6c3275662f50ebda278
```

The tag `b200-v0.1.1` points at the same image. Public; no GitHub login needed.

Check what you got, without a GPU:

```bash
docker run --rm ghcr.io/ycchen-tw/pochi-imo26-b200:b200-v0.1.1 verify
```

It prints the runtime's package versions and the SHA-256 of the two patches the
model's numerics depend on. It exits non-zero if anything drifted.

## 2. Get the weights

Public HuggingFace repo, pinned revision, **no token required**. 61 GB.

```bash
pip install "huggingface-hub>=1.0"
hf download fieldsmodelorg/FM-Pochi-32B-IMO26 \
  --revision f14030d3c65e1ed59e4e70477297053fc9a75151 \
  --include 'opd-32b-bf16-step-225/*' \
  --local-dir /path/to/models
```

Result: `/path/to/models/opd-32b-bf16-step-225/` with 14 safetensors shards and
the tokenizer. That is the only checkpoint the default profile loads.

## 3. Run

```bash
docker run --rm --gpus all --ipc=host --shm-size=32g \
  -v /path/to/models:/models:ro \
  -v /path/to/state:/state \
  ghcr.io/ycchen-tw/pochi-imo26-b200:b200-v0.1.1 \
  harness --budget medium
```

```bash
apptainer run --nv \
  --bind /path/to/models:/models:ro --bind /path/to/state:/state \
  pochi-b200.sif harness --budget medium
```

One command does the whole job: starts the server, waits for the model to load
(several minutes — it quantizes to FP8 while loading), runs the search, shuts
the server down.

**Your own problems** — a CSV with exactly the columns `id,problem`:

```bash
... -v /path/to/mine.csv:/in/problems.csv:ro <image> harness --input /in/problems.csv --budget medium
```

Without `--input` it uses the AIMO3 public 10-problem set baked into the image.

`--budget medium | high | xhigh` trades compute for accuracy: medium is 32
candidates per round and up to 4 rounds; xhigh is 128 and up to 8. Medium on ten
problems is a multi-hour job on 8 B200s.

## Results

Everything lands under `/state/harness-<timestamp>/`:

| file | contents |
|---|---|
| `submission.csv` | `id,answer` — the deliverable |
| `status.json` | progress; also how you resume |
| `problems/row-NNNN/` | every model call, verification, refinement and selector ballot |
| `server.log` | the inference server's log |
| `manifest.json` | exactly which code, inputs and settings produced this run |

**Exit codes carry meaning. Do not treat non-zero as simply "failed":**

| code | meaning |
|---|---|
| 0 | finished; every selected solution had a boxed integer |
| 2 | finished, but some answers are blank — the model did not reach an answer it would commit to. `submission.csv` is still valid, with empty cells. |
| 1 | it did not run: bad arguments, missing weights, wrong GPU count, or the server failed to start |

To **resume** an interrupted run, point at the same directory:

```bash
... <image> harness --run-dir /state/harness-<timestamp>
```

Finished problems are skipped and a half-finished one continues from its last
completed round. Changing the input, settings or budget requires a new directory
— the container enforces this rather than silently mixing two runs.

## If it does not start

The container fails loudly and names the cause. The common ones:

- **`no model at /models/opd-32b-bf16-step-225`** — the weights are not mounted
  where it expects. Check the `-v` / `--bind` path.
- **`needs exactly 8 GPUs, got N`** — the topology is TP4 × DP2. Fewer GPUs is
  not a smaller run, it is a different server; it refuses rather than guess.
- **`GPUs already occupied`** — something else is on the cards. Set
  `POCHI_SKIP_GPU_OCCUPANCY_CHECK=1` only if that memory is yours.
- **`$FM_POCHI_CACHE is not writable`** — under a read-only rootfs, mount
  writable volumes at `/state` and `/cache`.
- **server never becomes ready** — look at `/state/<run>/server.log`. Loading 61
  GB takes minutes; the container waits up to 45 minutes before giving up.

`status.json` only updates at round boundaries, so it can sit unchanged for many
minutes during a round. That is not a hang. `server.log` is where you see live
activity.

## What has been verified, and what has not

Stated plainly, because someone else is going to run this.

**Verified.** The image builds reproducibly and its runtime self-check passes:
the required attention-sink patch and the B200 decode patch are present, byte
for byte, matching the development machine. The harness accepts its inputs and
resolves its search configuration. The failure paths above exit non-zero with
actionable messages. The server configuration the container launches is
mechanically identical to the one the profile was developed against.

**Not verified.** *This image has never been run on GPUs.* It was assembled on a
machine with no container runtime, so the checks above are all GPU-free. The
underlying runtime, server and inference were validated on 8× B200 outside a
container — same code, same runtime, same launch arguments — but the container
layer itself (device injection, the read-only filesystem, the mount contract)
has not been exercised against hardware.

Separately: a full generate-verify-refine run to `submission.csv` has not been
measured end to end on GPUs, in a container or otherwise.

**Before relying on this, do one short run** — `harness --budget medium --limit 1`
on a single problem — and confirm `submission.csv` appears. That exercises every
untested layer in about an hour.

## Source

Built from [`ycchen-tw/pochi-imo26`](https://github.com/ycchen-tw/pochi-imo26)
at commit `fe35d51`, tag `b200-v0.1.1`. The build is reproducible from that
commit; see [README.md](README.md) for how the image is assembled and what it
contains.
