# Evaluation artifacts

Every `--json` run of the harness writes one self-describing artifact here (or
wherever `--json` points). Nothing in an artifact is typed by hand: provenance
the harness could not collect is listed under `provenance.unavailable` with the
reason, never filled in.

Artifacts contain fixture **IDs**, verdict flags, statuses, and latencies. They
never contain message text. The public golden set is synthetic, so its IDs are
shareable; a private replay set evaluated with the same code stays private.

## Reproduce

```bash
uv run evals/harness.py --json evals/results/<name>.json                       # T0, no model
uv run --extra llm evals/harness.py --t1 --repeat 3 --json evals/results/<name>.json
```

The second command needs an Ollama server with the configured model pulled
(`ollama pull qwen3.5`). Settings come from `T1_MODEL`, `OLLAMA_URL`,
`T1_NUM_CTX`, `T1_CONCURRENCY`, and `T1_TIMEOUT_S`; the values in effect are
recorded under `provenance.inference`.

## Compare two runs

```bash
uv run evals/compare.py evals/results/a.json evals/results/b.json
```

Prints what changed in the inputs (revision, hashed files, model digest,
server version, packages, inference settings) and what changed in the outcomes
(gate result, headline rates, T1 status counts, and every message whose final
cell or uncertainty flipped). A metric delta with no input delta is noise or
model nondeterminism; quote both halves together.

## What repeatability means here

Inputs and settings are traceable: the same revision, data hashes, thresholds,
model digest, and inference settings can be re-run. It is **not** a promise of
identical model outputs on different hardware, server builds, or quantizations.
`temperature=0` constrains sampling; it does not make a threaded batch
deterministic. That is why `--repeat` exists and every repeat is gated.

## Schema (version 1)

```text
schema_version   1
provenance
  run_id, timestamp_utc, command
  git          revision, branch, dirty, modified_files   (tracked files only)
  hashes       sha256 of golden set, thresholds, metrics.py, blocklist, t0.py, t1.py
  python, packages
  hardware     machine, processor, cpu_count, os, gpus (nvidia-smi when present)
  model        name, digest, parameter_size, quantization, family,
               server_url, server_version, size_bytes, size_vram_bytes   (T1 runs)
  inference    the settings passed to the client                          (T1 runs)
  unavailable  {field: reason} for anything above that could not be collected
gate
  mode         "t0" | "t1"
  repeats      number of T1 repeats (0 for a T0 run)
  passed       no threshold failed on any repeat
  failures     ["run N: <rule>", ...]
  enforced     false when --no-gate was passed
t0
  elapsed_ms, metrics, gate_failures, messages[]
repeats[]      one per T1 repeat
  index, metrics, gate_failures
  t1           evaluated, status_counts{ok,capacity,transport,validation,model},
               latency_ms{n,p50,p95,max,mean} over all attempts,
               ok_latency_ms over validated results only
  messages[]
summary        headline rates for the final repeat (or the T0 pass), t1_lift_runs,
               t1_status_counts, combined_confident_misses
```

`metrics` holds raw `tp/fp/tn/fn` counts per slice and per decision
(`toxic`, `scam`, `flagged`) alongside the derived rates, plus routing counts
and the T1 lift. `report.metrics_from_dict` rebuilds a `Report` from them, so
`harness.check_thresholds` can be re-applied to any saved repeat and must
reproduce its `gate_failures` (`tests/test_eval_artifact.py` checks this).

Each `messages[]` entry records the fixture ID, expected flags, the T0 flags,
the final flags and tier, the route (`t0` or `t1`), the T1 `status`,
`latency_ms`, and `error` (exception class name only) when escalated, and the
confusion cell of the final `flagged` decision. Every escalated message ends in
exactly one status, so `status_counts` sums to `evaluated`.

## Retained runs

| File | Revision | Purpose |
|---|---|---|
| `review-2026-09-17.json` | pre-`c228bcb` | Pre-schema report from the review pass; provenance block was written by hand. Kept for the numbers, superseded by the runs below. |
| `step1-2026-09-17.json` | see file | First schema-v1 artifact; the baseline for later comparisons. |
