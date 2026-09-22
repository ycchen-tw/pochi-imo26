#!/usr/bin/env python3
"""Build the sglang.launch_server argv (and the env it needs) for this profile.

Single source of truth for the server command line. start.sh and the container
entrypoint both call this, so the bare-metal launcher and the shipped container
cannot drift into two different servers -- which is the failure mode that makes
a container "work on my node" and not in the handover.

Emits JSON on stdout:  {"argv": [...], "env": {...}, "profile": {...}}

Every knob is read from the same POCHI_* environment variables start.sh already
documented; the defaults here are that file's defaults.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# TP4 x DP2 over 8 GPUs is the tested B200 topology. --max-total-tokens and
# --swa-full-tokens-ratio are sized against it together with
# SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER=0.125 from env.sh; changing one
# without re-deriving the others overflows the KV pool.
TENSOR_PARALLEL = 4
DATA_PARALLEL = 2
MAX_TOTAL_TOKENS = 5502848
SWA_FULL_TOKENS_RATIO = 0.2144
MEM_FRACTION_STATIC = 0.85
PAGE_SIZE = 128
CHUNKED_PREFILL = 4096


def environment(name: str, default: str) -> str:
    return os.environ.get(name, default)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to the target checkpoint")
    parser.add_argument("--draft", default=None, help="draft checkpoint (DFlash only)")
    args = parser.parse_args()

    quantization = environment("POCHI_QUANTIZATION", "fp8")
    kv_dtype = environment("POCHI_KV_DTYPE", "fp8_e4m3")
    backend = environment("POCHI_ATTENTION_BACKEND", "trtllm_mha")
    context_length = environment("POCHI_CONTEXT_LENGTH", "262144")
    max_running = environment("POCHI_MAX_RUNNING_REQUESTS", "64")
    dflash = environment("POCHI_DFLASH", "0")
    port = environment("POCHI_PORT", "30000")
    # start.sh binds loopback, which is right on a login node. In a container
    # loopback is the container's own, so a published port reaches nothing --
    # the entrypoint overrides this to 0.0.0.0. Left as a variable rather than
    # branching on "am I in a container", which is never reliably detectable.
    host = environment("POCHI_HOST", "127.0.0.1")

    if quantization not in ("fp8", "bf16"):
        sys.exit("POCHI_QUANTIZATION must be fp8 or bf16.")
    if kv_dtype.startswith("fp8") and backend == "fa4":
        sys.exit("This pinned FA4 backend does not support the required FP8 KV path; use trtllm_mha.")

    env: dict[str, str] = {}
    argv = [
        "--model-path", args.model, "--served-model-name", "fm-pochi",
        "--tp", str(TENSOR_PARALLEL), "--dp", str(DATA_PARALLEL),
        "--load-balance-method", "round_robin",
        "--host", host, "--port", port,
        "--dtype", "bfloat16",
    ]
    if quantization == "fp8":
        argv += ["--quantization", "fp8"]
    argv += [
        "--kv-cache-dtype", kv_dtype,
        "--attention-backend", backend, "--page-size", str(PAGE_SIZE),
        "--context-length", context_length,
        "--chunked-prefill-size", str(CHUNKED_PREFILL),
        "--max-running-requests", max_running,
        "--max-total-tokens", str(MAX_TOTAL_TOKENS),
        "--swa-full-tokens-ratio", str(SWA_FULL_TOKENS_RATIO),
        "--mem-fraction-static", str(MEM_FRACTION_STATIC),
        "--cuda-graph-max-bs-decode", max_running,
        "--reasoning-parser", "deepseek-r1", "--random-seed", "0",
    ]

    # Speculative decoding is opt-in. Measured +31% at 128 concurrent requests,
    # but NOT measured at the 384 this config targets, where compute is already
    # saturated. Benchmark before trusting it.
    if dflash == "1":
        if quantization != "bf16" or kv_dtype != "auto" or backend != "fa4":
            sys.exit("DFlash has only been checked with BF16 model/KV and FA4; "
                     "select that profile explicitly.")
        draft = args.draft
        if draft is None:
            model_root = os.environ.get("FM_POCHI_MODEL_ROOT")
            if not model_root:
                sys.exit("DFlash needs --draft or $FM_POCHI_MODEL_ROOT.")
            draft = str(Path(model_root, "dflash-32b-draft-v2test-phaseL"))
        if not Path(draft, "config.json").is_file():
            sys.exit(f"no draft checkpoint at {draft}")
        # The draft declares 65536; the target runs longer.
        env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        env["SGLANG_DFLASH_DRAFT_RING"] = "1"
        env["SGLANG_DFLASH_DRAFT_RING_QUOTA"] = "4"
        argv += [
            "--speculative-algorithm", "DFLASH",
            "--speculative-draft-model-path", draft,
            "--speculative-dflash-block-size", "8",
            "--speculative-num-draft-tokens", "8",
            "--speculative-draft-window-size", "512",
            "--speculative-draft-attention-backend", "fa4",
        ]
    elif dflash != "0":
        sys.exit("POCHI_DFLASH must be 0 or 1.")

    profile = {
        "tp": TENSOR_PARALLEL, "dp": DATA_PARALLEL,
        "quantization": quantization, "kv_cache_dtype": kv_dtype,
        "attention_backend": backend, "context_length": int(context_length),
        "max_running_requests": int(max_running), "dflash": dflash == "1",
        "host": host, "port": int(port),
        "client_sampling": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 131072},
    }
    json.dump({"argv": argv, "env": env, "profile": profile}, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
