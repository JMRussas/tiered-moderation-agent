# Next steps: evidence, integration, and presentation

Status: proposed; implementation has not started.

## Objective

Demonstrate that the moderation pipeline generalizes beyond its development
fixtures, behaves correctly when integrated with a consumer, and offers a
measurable cost/latency benefit. Package that evidence so a hiring reviewer can
reproduce the results and understand the remaining limitations.

The completed correctness work is recorded in [review-plan.md](review-plan.md).
The current baseline and limitations are in [validation.md](validation.md).

## Current baseline

- 95 passing tests and an installed-wheel smoke check.
- 78 synthetic development fixtures, originally authored alongside the lexicon.
- T0: 53.2% recall, 96.2% precision, and 65.4% escalation.
- Three passing live model runs; final combined recall is 85.1%, precision is
  97.6%, and FPR is 3.2%.
- Seven final-run confident misses: `g055`, `g058`, `g059`, `g060`, `g068`,
  `g071`, and `g076`.
- No representative traffic benchmark or reference consumer integration.

These measurements describe this fixture set, not production performance.

## Execution order

| Step | Work | Dependency |
|---|---|---|
| 1 | Automate reproducible evaluation artifacts | Existing harness |
| 2 | Create an independently labeled holdout set | Labeling policy and an independent reviewer |
| 3 | Build and test a reference consumer | Existing message/verdict API |
| 4 | Benchmark traffic, latency, and overload | Artifact format and reference consumer |
| 5 | Analyze model misses and evaluate improvements | Holdout protocol and reproducible baseline |
| 6 | Publish a concise architecture explanation and demo | Verified results from the preceding steps |

Steps 2 and 3 can proceed independently once their contracts are defined.
Avoid model tuning until the baseline and holdout protocol are frozen.

## 1. Automate reproducible model evaluation

### Deliverables

- Extend the harness output with a versioned report schema containing:
  run ID, timestamp, code revision, dirty-worktree status, relevant source/data
  hashes, model name and digest, dependency versions, and inference settings.
- Save metrics for every repeat, rather than only final-run slices and a list
  of recall lifts.
- Record per-message fixture ID, T0/final verdicts, route, latency, and whether
  inference succeeded or fell back. Distinguish capacity rejection from model,
  transport, and validation failures without exposing exception payloads.
- Retain public synthetic-fixture outcomes in shareable artifacts; keep any
  future private replay content out of committed reports.
- Document commands to reproduce and compare two runs. Record available
  hardware/server information for timing comparisons.

### Acceptance criteria

- A single command produces an artifact with no manual provenance editing.
- Tests cover report structure, failed repeats, and fallback accounting.
- Every repeat's gate decision can be reconstructed from its saved metrics.
- Missing provenance is explicitly marked unavailable rather than invented.
- Repeatability means traceable inputs and settings, not a promise of identical
  model outputs across hardware or server versions.

## 2. Establish an independent evaluation set

### Deliverables

- Freeze the existing golden set as a development/regression set.
- Write an annotation guide based on moderation policy, including ambiguous
  cases, acceptable criticism, and disagreement resolution.
- Have an independent reviewer label a new set without seeing implementation
  patterns or predictions. Start with approximately 200-300 examples, and
  expand where slices lack enough positives or negatives to be informative.
- Cover benign conversation, directed hostility, scams, mixed languages,
  unsupported scripts, obfuscation, short messages, positive-prefix attacks,
  and instructions attempting to manipulate the classifier.
- Record provenance, labeling disagreements, category counts, and sampling
  choices. Use consented or independently authored material; label synthetic
  data as synthetic.
- Define development and holdout boundaries before running experiments. Once
  examples guide a fix, treat them as development data and replenish the
  holdout instead of continuing to claim independence.

### Acceptance criteria

- The holdout was not selected or labeled to match existing outputs.
- Overall and slice-level reports include counts and uncertainty estimates;
  small slices are identified as inconclusive where appropriate.
- Thresholds and promotion criteria are agreed before candidate results are
  examined. Existing development gates remain active.
- If an independent reviewer is unavailable, document that blocker and call
  any interim self-authored expansion a development set, not a holdout.

## 3. Implement a reference consumer and integration tests

### Deliverables

- Add a small runnable consumer that accepts messages, invokes the tiers, and
  displays one of: respond, withhold, or await classification.
- Model pending, resolved, expired, and failed processing explicitly. Unknown
  verdicts must never authorize a response.
- Correlate results by message ID and generation/version so delayed results
  cannot authorize a newer message or produce duplicate actions.
- Define a finite queue, response deadline, overload behavior, and cancellation
  behavior. Keep these policies outside the classifier itself.
- Demonstrate normal operation and model failure with fake adapters; keep live
  Ollama use optional for the demo.

### Acceptance criteria

- Deterministic integration tests prove that uncertain, toxic, or scam verdicts
  never trigger the response action.
- Tests cover delayed and out-of-order results, duplicate IDs/events, expired
  messages, model outages, and overload.
- Queue bounds and deadlines are enforced by the consumer. Late inference
  completion cannot trigger an expired action.
- A reviewer can run the example without connecting a real chat account.

## 4. Measure realistic traffic cost and latency

### Deliverables

- Define a replay format and workload provenance. Prefer a privacy-safe sample
  with a justified traffic mix; if only synthetic traffic is available, present
  it as a scenario benchmark rather than representative production evidence.
- Replay ordinary traffic, multilingual traffic, and bursts at increasing
  arrival rates. Include an outage and a slow-model scenario.
- Measure escalation, model calls, completed messages per second, p50/p95
  end-to-end latency, inference latency, queue depth, fallback rate, and expired
  or rejected messages. Report warm and cold behavior separately.
- Compare tiered processing with T0-only and all-model baselines under the same
  workload, hardware, and settings. Account for quality differences alongside
  speed and call savings.
- Use observed model calls or GPU time as cost proxies. Do not imply a dollar
  cost without an explicit pricing/utilization assumption.

### Acceptance criteria

- Reports include offered load as well as completed throughput, so overload
  cannot make latency look artificially good by silently excluding work.
- Every submitted message is accounted for as completed, rejected, expired,
  or still pending at the measurement boundary.
- The benchmark identifies the sustainable load and the point where the
  configured latency/queue limits fail.
- README efficiency claims cite measured workloads and their limitations.

## 5. Investigate remaining model misses

### Deliverables

- Review the seven current misses and classify likely causes: language
  understanding, policy ambiguity, missing context, or inference limitations.
- Inspect explanations and disagreements without assuming every existing
  label is indisputable. Record justified label corrections separately from
  model improvements.
- Add related development examples and benign counterexamples before changing
  prompts or model settings.
- Compare focused candidates against the frozen baseline. Track toxic and scam
  metrics separately, plus routing, latency, and inference failure rate.
- Evaluate the selected candidate on the independent holdout under the agreed
  protocol. Keep an experiment log, including unsuccessful changes.

### Acceptance criteria

- Improvements extend to related examples and the independent evaluation;
  merely fixing the seven named IDs is insufficient.
- Candidate acceptance accounts for false positives and operational cost, not
  recall alone. Any remaining errors stay visible in the published report.
- Do not lower thresholds to declare an experiment successful. A candidate
  that fails the agreed criteria remains an experiment.

## 6. Improve hiring-review presentation

### Deliverables

- Add a compact architecture diagram showing tier routing, uncertainty,
  admission limits, and the consumer's decision boundary.
- Create a roughly two-minute demo showing a trivial message, unresolved
  content, a moderation decision with reasons, and an outage that causes
  abstention.
- Put a reproducible results table and the most important tradeoffs near the
  top of the README. Link the benchmark protocol and failure analysis.
- Explain what is implemented, what is simulated, and which production
  integrations remain outside the repository.

### Acceptance criteria

- A reviewer can understand the design and run a deterministic demonstration
  within a few minutes.
- Every performance or quality claim links to a dated, reproducible artifact.
- The demo visibly preserves uncertainty during failure.

## T2 decision gate

Defer a tool-using T2 agent until evaluation identifies meaningful cases that
cannot be resolved from message text alone. Before building it, require:

- Labeled examples whose decisions depend on viewer history or room state.
- Defined, available context sources and bounded tool permissions.
- A baseline comparison and explicit success criteria for incremental quality,
  latency, and cost.

If the evidence does not justify another tier, retain the simpler system and
document that decision.

## Completion checklist

- [ ] Reproducible artifacts include all repeats and message outcomes.
- [ ] Independent labeling and a holdout protocol are established.
- [ ] Reference consumer passes failure and timing integration tests.
- [ ] Traffic benchmarks account for all submitted work and operational limits.
- [ ] Model-error investigation records improvements and residual failures.
- [ ] README, architecture diagram, and demo match the measured implementation.
- [ ] T2 is either justified by evidence or explicitly deferred.
