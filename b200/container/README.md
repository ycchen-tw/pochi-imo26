# FM Pochi B200 — container handover

The B200 SM100 FP8 profile packaged as an OCI image, for a recipient who runs
Docker or Apptainer/Singularity and should not have to reproduce any of the
runtime assembly.

| | value |
|---|---|
| topology | TP4 × DP2 over 8× B200 |
| weights / KV | online FP8 (quantized at load from BF16) / FP8 E4M3 |
| attention | `trtllm_mha`, page 128 |
| context | 262,144 |
| API | OpenAI-compatible, port 30000, model name `fm-pochi` |

## What this image has that the root `Dockerfile` does not

The repository-root image ships the upstream H200 runtime and applies the
SGLang patches at **boot**, from `docker/entrypoint.sh`. It never runs
`b200/patch_fa4_decode.py`, so an image built from it has no
`FM_POCHI_FA4_DECODE_SPLITKV_V1` and no `READY.json`, and `b200/setup.sh`
cannot fix that after the fact — it early-returns the moment `READY.json`
exists, which in a fresh container it does not, but by then the server has
already been launched from an unpatched venv.

This image instead bakes the whole B200 patch set at **build** time and writes
`READY.json` into the layer, so the artifact can be checked before it is run:

```bash
docker run --rm ghcr.io/ycchen-tw/pochi-imo26-b200:latest verify
```

`verify` needs no GPU. It imports sglang/torch/flash_attn, asserts the venv
relocated onto the bundled CPython, and asserts both the required Olmo3Sink
patch and the FA4 split-KV marker are present, then prints the same JSON
report `b200/setup.sh` writes bare-metal. Same script, `b200/verify_runtime.py`,
in both places — so the image's report and a node's are directly comparable.

## Build

**In CI (no local Docker needed).** Actions → *Publish B200 image* → Run
workflow. It runs the argv drift tests, builds, runs `verify` and the
no-weights failure path against the built image, and only then pushes to
`ghcr.io/<owner>/pochi-imo26-b200`.

**Locally**, from the repository root (the build context is the root, not
`b200/`):

```bash
docker build -f b200/container/Dockerfile -t fm-pochi-b200 \
  --build-arg VCS_REF=$(git rev-parse HEAD) .
```

`VCS_REF` is not optional in practice. `run_math_harness.py` pins the source
commit into every run manifest, so a changed run cannot silently reuse an old
run directory. Bare-metal it shells out to `git`; the image has neither `.git`
(`.dockerignore` drops it) nor a `git` binary, so the commit is baked in at
build time. Omit the build-arg and it stays `unknown`, which the harness
rejects at startup rather than writing as provenance — a manifest that looks
pinned and is not is worse than a refusal.

The build needs no GPU. `/opt/pp` is inherited from a published base image that
already carries it (`RUNTIME_BASE_IMAGE`, default
`ghcr.io/fieldsmodelorg/aimo-proof-pilot:sha-463682b`); nothing is compiled.

## Run

Weights are never baked in — 61 GB, and they are fetched separately by
`download_models.sh`. Mount them read-only.

### AIMO integer-answer inference, one command

This is the whole job in a single container: it brings up the server, waits
for it by model name, runs the generate-verify-refine short-answer harness
(`run_math_harness.py`), tears the server down, and exits with the *harness's*
status.

```bash
docker run --rm --gpus all --ipc=host --shm-size=32g \
  -v /path/to/models:/models:ro \
  -v /path/to/state:/state \
  ghcr.io/ycchen-tw/pochi-imo26-b200:latest \
  harness --budget medium
```

```bash
apptainer build pochi-b200.sif docker://ghcr.io/ycchen-tw/pochi-imo26-b200:latest
apptainer run --nv \
  --bind /path/to/models:/models:ro --bind /path/to/state:/state \
  pochi-b200.sif harness --budget medium
```

Everything after `harness` goes to `run_math_harness.py` verbatim, so
`--budget high`, `--limit 3`, `--seed 1`, `--config`, `--problems` all work as
documented in [../harness/README.md](../harness/README.md). Two defaults are
filled in if you omit them: `--input` becomes the AIMO3 public 10 set baked at
`/opt/fm-pochi/b200/data/aimo3-reference/problems.csv`, and `--run-dir` becomes
a timestamped directory under `/state`.

Output lands in that run directory: `submission.csv` (`id,answer`), plus
`problems/row-NNNN/` with every call, review, refinement and selector ballot,
and `status.json`.

**Exit codes are the harness's and are meaningful** — the entrypoint passes
them through rather than collapsing them:

| code | meaning |
|---|---|
| 0 | completed, every selected solution had a boxed integer |
| 2 | `completed_with_invalid_answers` — finished, but some answers are blank |
| 1 | setup/argument error, or the server never came up |

Resume by pointing at the same directory: `harness --run-dir /state/<dir>`.

Use `direct` in place of `harness` for `run_direct.py` (direct sampling pass@k)
with the same lifecycle.

### Server only

For a two-container split, or to drive the API yourself:

```bash
docker run --rm --gpus all --ipc=host --shm-size=32g \
  -v /path/to/models:/models:ro -p 30000:30000 \
  ghcr.io/ycchen-tw/pochi-imo26-b200:latest serve
```

Readiness — by model name, not by the port answering:

```bash
until curl -sf http://127.0.0.1:30000/v1/models | grep -q fm-pochi; do sleep 10; done
```

### Dry runs, no GPU

```bash
docker run --rm <image> verify                      # runtime self-check
docker run --rm -v /path/to/models:/models:ro \
  <image> prepare --budget medium                   # pin inputs, resolve search
                                                    # config, contact nothing
```

`prepare` needs only the tokenizer files from the checkpoint directory
(~10 MB), not the weights. Both run in CI on every build.

### Mount contract

| path | mode | purpose |
|---|---|---|
| `/models` | read-only | `opd-32b-bf16-step-225/`, plus `dflash-32b-draft-v2test-phaseL/` if `POCHI_DFLASH=1` |
| `/state` | writable | run directories, `server.log`, `server-plan.json` |
| `/cache` | writable | JIT caches (Triton, flashinfer, inductor). Seeded from the image on first start |

`/state` and `/cache` are `1777` in the image, so Apptainer — which runs as the
invoking user against a read-only squashfs — works without binds, writing into
the container's tmpfs. Bind them to real paths to keep logs and to avoid paying
JIT cost on every start.

### Knobs

`POCHI_QUANTIZATION`, `POCHI_KV_DTYPE`, `POCHI_ATTENTION_BACKEND`,
`POCHI_CONTEXT_LENGTH`, `POCHI_MAX_RUNNING_REQUESTS`, `POCHI_DFLASH`,
`POCHI_HOST`, `POCHI_PORT` — identical to `b200/start.sh`, because both build
their command line from `b200/server_argv.py`. The previous BF16 batch profile:

```bash
docker run ... -e POCHI_QUANTIZATION=bf16 -e POCHI_KV_DTYPE=auto \
  -e POCHI_ATTENTION_BACKEND=fa4 -e POCHI_CONTEXT_LENGTH=139264 \
  -e POCHI_MAX_RUNNING_REQUESTS=192 ...
```

Invalid combinations are rejected before launch, not at load: FP8 KV with `fa4`,
DFlash outside the BF16/`auto`/`fa4` profile, an unknown quantization.

## Where the container differs from `start.sh`, and why

These are the bare-metal assumptions that do not survive containerization. Each
is handled in `container/entrypoint.sh`; they are listed because a reviewer
should know the launcher is not byte-identical to the node's.

**Foreground, not `nohup setsid … &`.** `start.sh` backgrounds the server and
returns, which is right on a login node and fatal as PID 1 — the container would
exit immediately and take the server with it. The entrypoint runs it in the
foreground under `tini` and forwards SIGTERM, so `docker stop` reaches sglang.

**Binds `0.0.0.0`, not `127.0.0.1`.** Loopback inside a container is the
container's own; a published port would reach nothing. Override with
`POCHI_HOST`. Under Apptainer (host network) either value works.

**Occupancy gate reads `memory.used`, not `--query-compute-apps`.** That query
reports PIDs, and a container cannot see PIDs in other namespaces, so inside a
container it returns empty even on fully occupied GPUs — the guard would still
be there and no longer guard anything. `memory.used` is visible across
namespaces. `POCHI_SKIP_GPU_OCCUPANCY_CHECK=1` disables it.

**Writable paths are checked up front.** Under a read-only rootfs the first
symptom of an unwritable cache is otherwise `Read-only file system` from inside
a JIT compile, twenty minutes in.

## What has NOT been verified

The image has not been run end-to-end on GPUs, because the machine it was
assembled on has no container runtime — no Docker daemon access, and
unprivileged user namespaces are blocked by AppArmor, which rules out rootless
Podman and Apptainer alike.

Verified: the build is deterministic and GPU-free; `verify` passes against the
bare-metal runtime the image reproduces; `prepare` resolves the harness search
config and pins the AIMO3 inputs; the argv the container launches is
mechanically identical to `start.sh`'s (`harness/test_server_argv.py`); the
failure paths exit non-zero with actionable messages.

Not verified: the image actually serving on 8× B200, and the
nvidia-container-toolkit `--gpus all` / `--nv` device injection. Before handing
this over, run it once on a host with a container runtime and confirm the
readiness check above returns `fm-pochi`.

Separately, and independent of containerization: `../harness/README.md` records
that **GPU end-to-end operation of the full generate-verify-refine short-answer
harness has not yet been measured** — direct-mode results do not validate that
flow. Packaging it does not change that. The first real `harness` run, in a
container or not, is still the first real run.
