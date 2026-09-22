# syntax=docker/dockerfile:1.7
#
# FM Pochi -- B200 SM100 FP8 profile, packaged for handover.
#
# Differs from the repository-root Dockerfile in exactly one way that matters:
# that image ships the upstream H200 runtime and applies the SGLang patches at
# *boot* via docker/entrypoint.sh. This one bakes the full B200 patch set --
# including b200/patch_fa4_decode.py, which the root Dockerfile never applies --
# and writes READY.json at BUILD time, so the shipped artifact is verifiable
# before it is ever run. `docker run ... verify` re-checks it without a GPU.
#
# Build (no GPU, no local Docker needed -- see container/README.md for the
# GitHub Actions path):
#   docker build -f b200/container/Dockerfile -t fm-pochi-b200 .
#
# The build context is the REPOSITORY ROOT, not b200/.

ARG RUNTIME_BASE_IMAGE=ghcr.io/fieldsmodelorg/aimo-proof-pilot:sha-463682b
ARG CUDA_BASE_IMAGE=nvidia/cuda:13.0.3-devel-ubuntu24.04

# ---------------------------------------------------------------------------
# Stage: take the pinned /opt/pp out of the published base and apply the B200
# delta on top. Nothing here needs a GPU: patch_fa4_decode.py is a source
# rewrite guarded by ast.parse, and validate_cutlass_install.py only hashes
# wheel files.
# ---------------------------------------------------------------------------
FROM ${RUNTIME_BASE_IMAGE} AS runtime
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY evaluation/requirements.txt /tmp/requirements.txt
COPY docker/validate_cutlass_install.py /tmp/validate_cutlass_install.py
COPY sglang_patches /tmp/sglang_patches
COPY b200/patch_fa4_decode.py b200/verify_runtime.py /tmp/

RUN set -Eeuo pipefail; \
    export VENV=/opt/pp/venv; \
    test -x "$VENV/bin/python"; \
    test -x /opt/pp/pybase/bin/python3; \
    # requirements.txt stays authoritative even though the base already has it.
    UV_LINK_MODE=copy /opt/pp/uv pip install --python "$VENV/bin/python" \
        -r /tmp/requirements.txt; \
    # The base and CUDA-13 CUTLASS wheels own overlapping paths and cu13 must be
    # the LAST writer, or FA4 on B200 loads a mixed install. Re-assert it after
    # every other dependency has settled, then prove it with the repo's own
    # checker rather than trusting install order.
    UV_LINK_MODE=copy /opt/pp/uv pip install --python "$VENV/bin/python" \
        --no-deps --reinstall nvidia-cutlass-dsl-libs-cu13==4.5.2; \
    LD_LIBRARY_PATH=/opt/pp/pybase/lib "$VENV/bin/python" /tmp/validate_cutlass_install.py; \
    # The REQUIRED Olmo3Sink patch plus the DFlash/sampling patch set.
    bash /tmp/sglang_patches/apply_patches.sh "$VENV" \
        /opt/pp/proof-pilot/deploy/w4a8/humming_w4a8.py; \
    # The B200-only piece the root Dockerfile does not have.
    LD_LIBRARY_PATH=/opt/pp/pybase/lib "$VENV/bin/python" /tmp/patch_fa4_decode.py "$VENV"; \
    # READY.json, written by the same script b200/setup.sh uses bare-metal --
    # same asserts, same fields, so the image's report and the node's are
    # directly comparable. --correctness-cases carries forward the 40 dynamic
    # cases recorded for the split-KV patch; backend_sha256 in the same report
    # is what ties that claim to the patch actually applied here.
    LD_LIBRARY_PATH=/opt/pp/pybase/lib "$VENV/bin/python" /tmp/verify_runtime.py \
        --runtime /opt/pp --venv "$VENV" --correctness-cases 40; \
    # Park the warm JIT caches where the entrypoint can seed a writable cache
    # from them. Cold flashinfer JIT on first launch costs minutes.
    mkdir -p /opt/pp/cache-seed; \
    cp -a /opt/pp/flashinfer_cache /opt/pp/cache-seed/flashinfer 2>/dev/null || true; \
    rm -rf /tmp/requirements.txt /tmp/validate_cutlass_install.py /tmp/sglang_patches \
           /tmp/patch_fa4_decode.py /tmp/verify_runtime.py

# ---------------------------------------------------------------------------
# Final image
# ---------------------------------------------------------------------------
FROM ${CUDA_BASE_IMAGE}
ARG DEBIAN_FRONTEND=noninteractive
ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="FM Pochi B200 FP8"
LABEL org.opencontainers.image.description="OPD-32B on 8x B200 (SM100): TP4xDP2, online FP8 weights + FP8 E4M3 KV, trtllm_mha"
LABEL org.opencontainers.image.revision="$VCS_REF"

# Runtime libraries only. No CUDA toolkit: the venv ships its own nvcc/ptxas
# under nvidia/cu13/bin and torch carries its own CUDA 13 libraries, so the
# host's toolchain is neither used nor required.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash ca-certificates curl jq \
        libaio1t64 libgl1 libglib2.0-0t64 libgomp1 libibverbs1 libnuma1 \
        numactl pciutils procps tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=runtime /opt/pp /opt/pp
COPY . /opt/fm-pochi
COPY b200/container/entrypoint.sh /usr/local/bin/fm-pochi-entrypoint

# b200/runtime is a symlink to the bare-metal runtime outside the repo; inside
# the image it would dangle and FM_POCHI_RUNTIME below supersedes it anyway.
RUN rm -f /opt/fm-pochi/b200/runtime \
    && chmod 0755 /usr/local/bin/fm-pochi-entrypoint /opt/fm-pochi/b200/*.sh

# Every one of these is a ${VAR:-default} in b200/env.sh, so pointing them at
# container paths needs no container-specific copy of that file.
ENV FM_POCHI_RUNTIME=/opt/pp \
    FM_POCHI_MODEL_ROOT=/models \
    FM_POCHI_STATE_ROOT=/state \
    FM_POCHI_CACHE=/cache \
    VENV=/opt/pp/venv \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Apptainer runs the container as the INVOKING USER, not root, and the SIF
# rootfs is read-only squashfs. These have to be world-writable or every run
# without an explicit bind dies inside a JIT compile. /models stays read-only:
# weights are mounted, never baked (61 GB).
RUN mkdir -p /models /state /cache \
    && chmod 1777 /state /cache

WORKDIR /opt/fm-pochi
VOLUME ["/state", "/cache"]
EXPOSE 30000
STOPSIGNAL SIGTERM

# Readiness by MODEL NAME, not by the port being open: a port that answers is
# not the same as this model being loaded, and on a shared host it may not even
# be our server.
HEALTHCHECK --start-period=45m --interval=30s --timeout=10s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${POCHI_PORT:-30000}/v1/models" | grep -q fm-pochi || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/fm-pochi-entrypoint"]
CMD ["serve"]
