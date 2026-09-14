# Deployment and evaluation fixes, 2026-09-13

- **FP8 KV backend:** the pinned FA4 path is unsuitable for FP8 KV. The FP8
  launcher uses `trtllm_mha`, preserving Olmo3 attention sinks. FP8 allocation,
  all 64 sink tensors on every rank, CUDA graphs and the 8/8 arithmetic smoke
  were checked before the direct benchmark.
- **Direct concurrency and scoring:** the legacy client used the default
  32-worker executor despite a larger concurrency setting, and its fanout
  summary counted successful samples rather than solved problems. `run_direct.py`
  uses async HTTP and computes pass@k per problem after every expected sample
  and its archived response have been audited.
- **SSE replay performance:** repeatedly splitting off one line copied the entire
  remaining archive on every iteration. A large final audit became quadratic.
  Splitting once per block and replaying gzip streams in 64 KiB chunks makes the
  scan linear. The 100 direct responses were preserved and re-audited in 13.136
  seconds; no model generations were repeated. Original collector and repaired
  auditor hashes are retained in the direct summary.
- **Harness request IDs:** wire IDs include a run/problem namespace, including
  native continuations. Repeated logical IDs from the original engine therefore
  cannot collide across separate runs. Search seeds and logical checkpoint IDs
  retain their original meaning.
- **Harness integer extraction:** original Pochi solutions use display math such
  as `\[\boxed{32951}\]`. The adapter now accepts standard closing math delimiters
  and punctuation after the last boxed integer. It still rejects fractions,
  malformed boxes and a later non-integer box. The extraction-only migration
  checks that the rest of the runner's AST is identical, retains the old
  manifests and updates saved answers without inference.

The upstream `ProblemSearch`, scoring/refinement/selection policies and original
prompt templates remain unchanged. The local CPU adapter suite has 16 passing
tests, including archive replay, per-problem pass@k, interrupted-call resume,
wire ID isolation, selected-proof audit and standard LaTeX answer formatting.
