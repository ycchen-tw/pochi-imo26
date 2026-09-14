# FM-Pochi short-answer harness

Uses the **original** `upstream/evaluation/harness/proof_search.py` engine and
its original prover, verifier, refiner and selector prompts. The engine source
is unchanged. Each question gets the instructions in `short_answer.txt`, asking
for a rigorous solution ending with a boxed integer. The original selector
chooses a solution; the wrapper extracts that solution's final answer.

```bash
cd /data/home/ycc/work/fm-pochi/upstream/b200
source env.sh
RUN_DIR="${ARC_RUNS:-/data/home/ycc/work/runs}/fm-pochi-math-$(date -u +%Y%m%dT%H%M%SZ)"

# Inspect and pin a run without making inference requests.
python run_math_harness.py --input problems.csv --budget medium \
  --run-dir "$RUN_DIR" --prepare-only

# After a compatible Pochi service is running:
python run_math_harness.py --input problems.csv --budget medium \
  --run-dir "$RUN_DIR"
```

Input columns are exactly `id,problem`. Output is `submission.csv` with
`id,answer`. Repeat the same command and run directory to resume: the original
engine reuses completed calls and rounds. Changed input, search configuration,
code, prompts, tokenizer or endpoint require a new run directory.

The AIMO3 public 10 input and reference CSVs are tracked under
`../data/aimo3-reference/`; evaluation does not depend on the former NFS data
directory. Gold answers remain scoring-only inputs.

`--budget medium|high|xhigh` uses the corresponding upstream step225 search
settings verbatim, including its selector settings and temperatures. Medium
uses 32 candidates per round, 16 reviews per candidate, up to 4 rounds,
4 refinement parents and 3 reviews per parent. `--config` accepts an original
schema-12 YAML; only its search section controls this client. The wrapper
connects to `--url` using `--model` and never starts a GPU service.

The FP8 launcher defaults to 262,144 context, matching the original harness
configurations. Multi-parent refinement can have much longer prompts than direct
sampling. The wrapper preserves the original token budgets and does not trim
prompts. GPU end-to-end operation of the full generate-verify-refine short-answer
harness has not yet been measured; direct-mode tests do not validate that flow.

All original calls (including reasoning), prompts, reviews, refinements and
selector ballots are retained under `problems/row-NNNN/`. `final.json` holds the
selected full solution; `answer.json` holds its extracted integer.
Standard closing LaTeX math delimiters after the final box are accepted.
`answer-checkpoint.json` is provisional, and `status.json` reports run progress.
If the selected solution has no final boxed integer, its CSV answer is blank,
the run is marked `completed_with_invalid_answers`, and the exit code is 2.
The wrapper neither guesses an integer nor substitutes a different candidate.

This is an adapter for integer short answers, not a new answer-voting algorithm.
The proof harness's original verification, early stopping, selection and
length-recovery behavior all remain in effect. Gold answers are not solver inputs.

`--seed N` selects an independent harness replicate. Request IDs are namespaced
by run and problem, while the original search's logical call IDs remain unchanged.
`run_harness_campaign.py` supervises the requested medium x4, high x2 and xhigh x1
campaign; it checks progress every 300 seconds, audits each selected solution and
updates the main README and [RESULTS.md](RESULTS.md). Failed jobs keep their
checkpoints, and the supervisor releases only its recorded GPU server on exit.
