# Case study: preserving uncertainty in AI moderation

**Scope:** a Python classification library and evaluation harness, extracted and rebuilt from a live-chat use case. This public repository contains synthetic fixtures, not viewer data or a deployed chat service.

## Problem and constraints

A deterministic policy is cheap and inspectable, but cannot understand all paraphrases or languages. A model adds capability and introduces latency, capacity limits, malformed responses, and confident errors. The critical distinction is between a clean classification and an inability to classify.

## Decisions

- **Escalate uncertainty explicitly.** `needs_llm=True` means the caller must wait or abstain. Short or positive-sounding text is not sufficient evidence of safety. See [T0](../src/tiermod/t0.py).
- **Make failure preserve information.** T1 requires the original verdict and returns it unchanged on inference failure or overload. A model may add flags but cannot erase a deterministic policy match. See [T1](../src/tiermod/t1.py).
- **Bound active inference.** A process-wide semaphore rejects excess work without building an internal waiting queue. Callers still own batch size, cross-process load, and end-to-end deadlines.
- **Separate software correctness from model quality.** CI checks deterministic behavior, degradation, packaging, and the T0 gate. Model evaluations run separately and retain provenance and per-message outcomes. See [CI](../.github/workflows/eval.yml) and [artifact format](../evals/results/README.md).

## A failure that changed the interface

An earlier batch call omitted an optional fallback argument. When inference failed, it produced a blank verdict: an uncertain message became apparently clean. Making the original verdict mandatory removed that accidental call pattern; regression tests exercise direct and batch failures. [Failure and tests](../tests/test_t1_degradation.py).

A separate review found that positive-keyword routing missed hostile messages outside the golden set despite a perfect safe-miss score on that set. The router now clears only explicit complete trivial messages without inference. [Failure history](failure-modes.md).

## Evidence and limits

The [September 17 recorded evaluation](../evals/results/review-2026-09-17.json) uses 78 synthetic, hand-labeled messages. On the final model run, recall increased from 53.2% for T0 to 85.1% combined; combined precision was 97.6%. Seven positive fixtures still received clean verdicts. These are development-set results, not estimates of production accuracy.

The [holdout protocol](../evals/holdout/README.md) specifies isolated generation, blind verification, human adjudication, and freezing before evaluation. No holdout result is claimed here. [Replay tooling](../evals/replay.py) measures routing and inference behavior on unlabeled inputs; it cannot establish correctness without labels.

## Reproduce the deterministic checks

```bash
uv sync --locked
uv run pytest -q
uv run evals/harness.py
uv run tools/check_wheel.py
```

No model is needed. See the [README](../README.md) for model-evaluation commands and operating constraints.
