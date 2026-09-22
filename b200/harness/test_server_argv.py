"""server_argv.py must stay identical to what start.sh launches.

The container entrypoint and start.sh both build the server command line from
server_argv.py. That is only worth anything if the command line it produces is
the one this profile was actually measured with -- otherwise the container
quietly serves a different model configuration than the node it was validated
on, which is the exact failure a handover artifact must not have.

Two kinds of check here:

  * golden argv -- the literal command line, per profile.
  * drift -- every `--flag` still written out in start.sh must appear in some
    profile's argv. This is the one that fires when someone tunes start.sh and
    forgets the container.
"""
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

B200 = Path(__file__).resolve().parent.parent
SERVER_ARGV = B200 / "server_argv.py"
START_SH = B200 / "start.sh"
MODEL = "/models/opd-32b-bf16-step-225"

# Flags start.sh writes that are deliberately NOT server argv: shell plumbing
# and env exports rather than launch_server options.
NOT_SERVER_FLAGS = {"--format", "--query-compute-apps", "--query-gpu"}


def render(**environment):
    """Run server_argv.py with a clean POCHI_* environment."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("POCHI_")}
    env["FM_POCHI_MODEL_ROOT"] = "/models"
    env.update(environment)
    result = subprocess.run(
        [sys.executable, str(SERVER_ARGV), "--model", MODEL],
        capture_output=True, text=True, env=env,
    )
    return result


def argv_of(**environment):
    result = render(**environment)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


class ServerArgvTest(unittest.TestCase):
    def test_default_profile_is_fp8_trtllm_mha(self):
        plan = argv_of()
        self.assertEqual(plan["argv"], [
            "--model-path", MODEL, "--served-model-name", "fm-pochi",
            "--tp", "4", "--dp", "2", "--load-balance-method", "round_robin",
            "--host", "127.0.0.1", "--port", "30000",
            "--dtype", "bfloat16", "--quantization", "fp8",
            "--kv-cache-dtype", "fp8_e4m3",
            "--attention-backend", "trtllm_mha", "--page-size", "128",
            "--context-length", "262144", "--chunked-prefill-size", "4096",
            "--max-running-requests", "64", "--max-total-tokens", "5502848",
            "--swa-full-tokens-ratio", "0.2144", "--mem-fraction-static", "0.85",
            "--cuda-graph-max-bs-decode", "64",
            "--reasoning-parser", "deepseek-r1", "--random-seed", "0",
        ])
        self.assertEqual(plan["env"], {})

    def test_bf16_fa4_profile_drops_quantization_flag(self):
        plan = argv_of(POCHI_QUANTIZATION="bf16", POCHI_KV_DTYPE="auto",
                       POCHI_ATTENTION_BACKEND="fa4",
                       POCHI_CONTEXT_LENGTH="139264",
                       POCHI_MAX_RUNNING_REQUESTS="192")
        self.assertNotIn("--quantization", plan["argv"])
        self.assertEqual(plan["argv"][plan["argv"].index("--kv-cache-dtype") + 1], "auto")
        # cuda-graph-max-bs-decode tracks max-running-requests, not a constant.
        self.assertEqual(
            plan["argv"][plan["argv"].index("--cuda-graph-max-bs-decode") + 1], "192")

    def test_cuda_graph_batch_tracks_max_running_requests(self):
        plan = argv_of(POCHI_MAX_RUNNING_REQUESTS="128")
        index = plan["argv"].index("--cuda-graph-max-bs-decode")
        self.assertEqual(plan["argv"][index + 1], "128")

    def test_host_is_overridable_for_containers(self):
        # Loopback is unreachable from outside a container; the entrypoint sets
        # 0.0.0.0. If this stops being a knob the published port serves nothing.
        plan = argv_of(POCHI_HOST="0.0.0.0")
        self.assertEqual(plan["argv"][plan["argv"].index("--host") + 1], "0.0.0.0")

    def test_fp8_kv_with_fa4_is_rejected(self):
        result = render(POCHI_ATTENTION_BACKEND="fa4")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("trtllm_mha", result.stderr)

    def test_dflash_requires_the_bf16_fa4_profile(self):
        result = render(POCHI_DFLASH="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("BF16", result.stderr)

    def test_unknown_quantization_is_rejected(self):
        result = render(POCHI_QUANTIZATION="int4")
        self.assertNotEqual(result.returncode, 0)

    def test_unknown_dflash_value_is_rejected(self):
        result = render(POCHI_DFLASH="yes")
        self.assertNotEqual(result.returncode, 0)

    def test_no_launch_flag_in_start_sh_is_missing_from_server_argv(self):
        """The anti-drift check: tune start.sh, this fires."""
        source = START_SH.read_text()
        launch = source[source.index("sglang.launch_server"):]
        launch = launch[:launch.index("\n\n")]
        in_start = {flag for flag in re.findall(r"--[a-z0-9][a-z0-9-]+", launch)}
        in_start -= NOT_SERVER_FLAGS

        covered = set()
        for environment in (
            {},
            {"POCHI_QUANTIZATION": "bf16", "POCHI_KV_DTYPE": "auto",
             "POCHI_ATTENTION_BACKEND": "fa4"},
        ):
            covered.update(flag for flag in argv_of(**environment)["argv"]
                           if flag.startswith("--"))
        # DFlash flags are only reachable with a real draft checkpoint on disk,
        # so assert them against the source rather than by rendering.
        covered.update(re.findall(r'"(--[a-z0-9][a-z0-9-]+)"',
                                  SERVER_ARGV.read_text()))

        missing = sorted(in_start - covered)
        self.assertEqual(missing, [], f"start.sh launches with flags server_argv.py "
                                      f"never emits: {missing}")


if __name__ == "__main__":
    unittest.main()
