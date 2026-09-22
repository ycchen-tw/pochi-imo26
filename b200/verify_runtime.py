#!/usr/bin/env python3
"""Verify a prepared Pochi runtime and write READY.json.

Extracted verbatim from the heredoc that used to live at the end of setup.sh so
the bare-metal installer and the container build assert the *same* invariants
against the same file. Two callers, one definition -- if this drifts, both drift
together and the mismatch is visible in READY.json rather than hidden.

Usage: verify_runtime.py [--runtime DIR] [--venv DIR] [--layer-sha256 HEX]
                         [--upstream-commit SHA]

Defaults come from $FM_POCHI_RUNTIME and $VENV, which is how setup.sh calls it.
The container build passes them explicitly because /opt/pp does not match the
bare-metal layout.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

# The runtime layer these checks were written against, and the upstream commit
# the patch set came from. Overridable so a rebased runtime records its own.
DEFAULT_UPSTREAM_COMMIT = "5d23e406e150088c4634afe83db8468c483f2fa1"
DEFAULT_LAYER_SHA256 = "27c911493f490231f95909cb831ce7d958cd5f2604968dedde7930744708c130"

PACKAGES = [
    "sglang", "torch", "transformers", "flash-attn-4", "flashinfer-python",
    "nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-cu13",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", default=os.environ.get("FM_POCHI_RUNTIME"))
    parser.add_argument("--venv", default=os.environ.get("VENV"))
    parser.add_argument("--layer-sha256", default=DEFAULT_LAYER_SHA256)
    parser.add_argument("--upstream-commit", default=DEFAULT_UPSTREAM_COMMIT)
    parser.add_argument("--output", default=None, type=Path,
                        help="where to write READY.json (default: <runtime>/READY.json). "
                             "Point it elsewhere to check a runtime without "
                             "overwriting the report it already carries.")
    # The 2026-09-11 runtime/READY.json carries dynamic_correctness_cases_passed=40
    # for the split-KV patch, but nothing in the repo produced it -- it was added
    # by hand and re-running setup.sh silently dropped it. Recording it is now an
    # explicit choice by the caller rather than an accident of which script ran.
    parser.add_argument("--correctness-cases", type=int, default=None,
                        help="number of dynamic correctness cases the FA4 split-KV "
                             "patch was checked against; recorded verbatim")
    args = parser.parse_args()
    if not args.runtime or not args.venv:
        parser.error("set --runtime/--venv or FM_POCHI_RUNTIME/VENV")

    runtime = Path(args.runtime)
    venv = Path(args.venv)

    # Importing here, not at module scope: this file is also read by the
    # container build before the venv is on sys.path.
    import sglang, torch, flash_attn  # noqa: F401  (imported for the side effect of proving they load)

    # The venv ships site-packages only; if relocation did not take, the
    # interpreter silently uses the wrong stdlib. Catch that here, loudly.
    assert Path(sys.base_prefix).resolve() == (runtime / "pybase").resolve(), (
        f"base_prefix {sys.base_prefix} is not {runtime / 'pybase'} -- relocation did not take"
    )

    # The REQUIRED Olmo3Sink patch. Without it the model runs without attention
    # sinks and silently produces wrong numerics.
    model = venv / "lib/python3.12/site-packages/sglang/srt/models/olmo2.py"
    assert "class Olmo3SinkForCausalLM" in model.read_text(), (
        f"Olmo3Sink patch missing from {model}"
    )

    report = {
        "upstream_commit": args.upstream_commit,
        "runtime_layer_sha256": args.layer_sha256,
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "olmo3_sink_patch_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
    }

    backend = venv / "lib/python3.12/site-packages/sglang/srt/layers/attention/flashattention_backend.py"
    assert "FM_POCHI_FA4_DECODE_SPLITKV_V1" in backend.read_text(), (
        f"FA4 decode split-KV patch missing from {backend}"
    )
    report["fa4_decode_splitkv"] = {
        "marker": "FM_POCHI_FA4_DECODE_SPLITKV_V1",
        "backend_sha256": hashlib.sha256(backend.read_bytes()).hexdigest(),
        "splits_by_graph_batch": {"1": 32, "2": 16, "4": 8, "8": 4},
        "scope": "Pochi TP2 BF16 full-attention decode only",
    }
    if args.correctness_cases is not None:
        report["fa4_decode_splitkv"]["dynamic_correctness_cases_passed"] = args.correctness_cases

    destination = args.output or (runtime / "READY.json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
