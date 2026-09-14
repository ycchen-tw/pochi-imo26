# FP8 direct: AIMO3 public 10, 2026-09-13

**pass@10: 9/10 problems (90%). Sample accuracy: 60/100 (60%).**

All 100 independent samples completed and passed request/response/SSE linkage,
gzip integrity and SHA256 audits. There were no failed HTTP attempts or retries.
Nine samples reached the output cap and count as failures. Eleven answers were
unparsed, including those nine truncated samples and two normally stopped replies
without a boxed integer.

| Problem | Reference answer | Correct / 10 | Truncated |
|---|---:|---:|---:|
| 0e644e | 336 | 10 | 0 |
| 26de63 | 32951 | 10 | 0 |
| 424e18 | 21818 | 8 | 0 |
| 42d360 | 32193 | 9 | 1 |
| 641659 | 57447 | 1 | 1 |
| 86e8e5 | 8687 | 0 | 6 |
| 92ba6a | 50 | 1 | 0 |
| 9c1c5f | 580 | 10 | 0 |
| a295e9 | 520 | 6 | 0 |
| dd7f5e | 160 | 5 | 1 |

Settings: TP4 x DP2 on GPUs 0-7, online FP8 linear-weight quantization of step225,
FP8 E4M3 KV, `trtllm_mha`, CUDA graphs, context 262,144, DFlash off. KV scales use
the uncalibrated default 1.0. Requests use temperature 1.0, top-p 0.95, top-k -1,
131,072 maximum output tokens, 100-way concurrency, and distinct per-sample seeds.
The server cap is 64 requests per replica. The startup arithmetic check passed 8/8.

Generation produced 5,161,500 completion tokens in 1,461.808 seconds (24.36 minutes),
measured from the first request start to the last response completion. The final
archive audit took 13.136 seconds after fixing quadratic SSE replay; no model
samples were rerun. The original collector source and all original sample files
were retained. No matched BF16 comparison was performed.

Artifacts: `/data/home/ycc/work/runs/fm-pochi-fp8-direct-pass10-20260913-piYoKj/`.
`direct/summary.json` contains the statistics and audit provenance;
`deployment-validation.json` records actual precision, runtime hashes and smoke
results. `direct/samples/` contains each original request, SSE stream and response.
