# Validation log

## Step 1: evaluation artifacts, 2026-09-17

Implements step 1 of [next-steps-plan.md](next-steps-plan.md).

- 117 unit tests pass, including artifact structure, per-repeat gate
  failures, fallback status accounting, gate reconstruction from saved
  counts, unavailable-provenance marking, no fixture text in artifacts, and
  T1 outcome classification.
- `uv run --extra llm evals/harness.py --t1 --repeat 3 --json evals/results/step1-2026-09-17.json`
  on revision `ca45a45` (clean worktree, nothing unavailable) passed every
  gate on all three repeats. Combined recall 85.1%, precision 97.6%, FPR
  3.2%, lift +31.9 points each run; the same seven confident misses as the
  review run. 51/51 inferences succeeded per repeat.
- Measured T1 latency at concurrency 4, model already loaded: p50 about
  3.0 s, p95 about 4.1 s per message. A separate run that started with the
  model unloaded showed p95 12.2 s on its first repeat and about 4.3 s after.
  Warm and cold behavior differ; benchmarks must report them separately.
- Hardware in the artifact: RTX 5090, 32 GB; qwen3.5 Q4_K_M, 5.7 GB in VRAM.

These numbers describe the synthetic golden set and one machine.

## Review-fix validation, 2026-09-17

All six review findings are addressed. The implementation plan is in
[review-plan.md](review-plan.md).

## Deterministic checks

- Before implementation, new review regressions produced 17 failures.
- After implementation, 95 unit tests pass, including malformed output,
  runtime messages without labels, explanations, process-wide admission,
  initialization races, missing policy, invalid gate inputs, model outage,
  partial model coverage, and an early failing evaluation repeat.
- `uv run evals/harness.py` passes the unchanged T0 quality thresholds and
  the existing 70% escalation ceiling. Escalation increased from 47/78 to
  51/78; recall, precision, and safe-miss ratio are unchanged.
- `uv run tools/check_wheel.py` builds and installs a wheel into a temporary
  location, imports it outside the checkout, checks its packaged policy, and
  verifies that a missing configured policy raises `BlocklistError`.

## Live model gate

Command:

```bash
uv run --extra llm evals/harness.py --t1 --repeat 3 --json docs/eval-results.json
```

All three repeats passed combined precision, recall, FPR, lift, and successful
inference gates. Recall lift was +31.9 percentage points each time. The final
run had 51/51 successful inferences, 85.1% combined recall, 97.6% precision,
and 3.2% FPR. The JSON stores final-run slice metrics and the three recall
lifts; it does not store per-message model responses or all per-run slices.

The [retained report](../evals/results/review-2026-09-17.json) includes the model
digest, client versions, configuration, and dataset hash. No threshold was
lowered. Combined recall is below the historical 87.2% result; the report and
README reflect the current measurement.

Final-run confident misses: `g055`, `g058`, `g059`, `g060`, `g068`, `g071`,
`g076`. These are model classification errors, not fallback-contract failures.
The system remains advisory and the small synthetic set does not establish
production accuracy, latency, or traffic cost.

## CI and operational limits

CI runs deterministic tests, the installed-wheel check, and T0 evaluation.
It does not have an Ollama service. Re-run the live command locally for model,
prompt, schema, or inference-configuration changes and retain the report.

Inference admission is bounded per process; input queues and batch sizes are
the integrating application's responsibility. HTTP inactivity timeouts and a
generation-token cap do not guarantee an end-to-end deadline. Policy loading
is cached, so configure it at startup and restart after policy edits.
