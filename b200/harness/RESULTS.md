# AIMO3 FP8 direct and original-harness evaluation

Updated: 2026-09-14T04:52:34.375055+00:00

Artifacts: `/data/home/ycc/work/runs/fm-pochi-fp8-harness-efforts-20260913-pOLs4R`
Campaign state: **paused by user** at 2026-09-14 04:52 UTC. All 10 public
reference problems are used. The supervisor, current attempt and model service
were stopped; GPUs 0-7 and port 30000 were verified released.

FP8 linear weights (online quantization), FP8 E4M3 KV, TP4 x DP2 on 8 B200.
The harness uses the original medium/high/xhigh search presets; only replicate seeds and the integer-answer task contract are adapted.

| Mode | k | Completed attempts | Correct attempts | pass@k |
|---|---:|---:|---:|---:|
| direct | 10 | 100/100 | 60 | 90.0% |
| medium | 4 | 10/40 | 8 | pending |
| high | 2 | 0/20 | 0 | pending |
| xhigh | 1 | 0/10 | 0 | pending |

pass@k means at least one correct selected answer among k independent complete attempts. A partial arm has no final pass@k. Different k values are not a controlled comparison of effort alone.

| Problem | medium correct/4 | high correct/2 | xhigh correct/1 |
|---|---:|---:|---:|
| 0e644e | pending (1/4) | pending (0/2) | pending (0/1) |
| 26de63 | pending (1/4) | pending (0/2) | pending (0/1) |
| 424e18 | pending (1/4) | pending (0/2) | pending (0/1) |
| 42d360 | pending (1/4) | pending (0/2) | pending (0/1) |
| 641659 | pending (1/4) | pending (0/2) | pending (0/1) |
| 86e8e5 | pending (1/4) | pending (0/2) | pending (0/1) |
| 92ba6a | pending (1/4) | pending (0/2) | pending (0/1) |
| 9c1c5f | pending (1/4) | pending (0/2) | pending (0/1) |
| a295e9 | pending (1/4) | pending (0/2) | pending (0/1) |
| dd7f5e | pending (1/4) | pending (0/2) | pending (0/1) |

## Completed medium replicate 0

| Problem | Reference | Selected answer | Correct | Rounds | Logical calls |
|---|---:|---:|---:|---:|---:|
| 0e644e | 336 | 336 | yes | 1 | 560 |
| 26de63 | 32951 | 32951 | yes | 1 | 576 |
| 424e18 | 21818 | 21818 | yes | 2 | 1,104 |
| 42d360 | 32193 | 32193 | yes | 2 | 1,088 |
| 641659 | 57447 | 57447 | yes | 4 | 2,160 |
| 86e8e5 | 8687 | 25 | no | 4 | 1,664 |
| 92ba6a | 50 | unparsed | no | 3 | 1,632 |
| 9c1c5f | 580 | 580 | yes | 1 | 576 |
| a295e9 | 520 | 520 | yes | 4 | 2,144 |
| dd7f5e | 160 | 160 | yes | 1 | 528 |

These ten completed attempts used 129,649,084 completion tokens and 12,201
physical requests. Every completed attempt passed the selected-proof, prompt,
review and artifact-linkage audit. There were 85 calls whose token usage was
estimated after real-time loop detection aborted and salvaged the stream; totals
therefore include estimates.

## Paused checkpoint

Medium replicate 1 stopped during problem `26de63`, round 1. It retains 32
completed generations and 509 completed verifier calls: 541 successful logical
calls, 4,504,846 completion tokens, zero failed call records. No round winner or
answer had been selected, so these partial calls are excluded from the result
table above. A resume will reuse them and request only missing work.

Resume only after checking GPU and port ownership:

```bash
cd /data/home/ycc/work/fm-pochi
nvidia-smi
runtime/venv/bin/python -u run_harness_campaign.py \
  --run-dir /data/home/ycc/work/runs/fm-pochi-fp8-harness-efforts-20260913-pOLs4R \
  --direct-run /data/home/ycc/work/runs/fm-pochi-fp8-direct-pass10-20260913-piYoKj \
  --interval 300
```

The planned medium pass@4, high pass@2 and xhigh pass@1 remain incomplete.
No pass@k is reported for a partial arm.

Full selected solutions, reasoning, reviews, refinements, selector ballots and per-attempt audits are under `tasks/`. `status.json` identifies the current job and any failure. Re-run the supervisor with the same run directory to resume.

KV scales use the uncalibrated runtime default 1.0. No matched BF16 comparison was run. Direct uses 131,072 output tokens; the harness preserves its original 128,000 per-call budget and continuation/selector budgets.

Fixed issues: FP8 KV uses trtllm_mha instead of the incompatible pinned FA4 path; direct uses async HTTP instead of the legacy 32-thread bottleneck; harness request IDs are namespaced by run/problem to avoid cross-run collisions.
