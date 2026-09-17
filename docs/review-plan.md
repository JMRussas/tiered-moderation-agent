# Review remediation plan

1. Replace sentiment/length-based clean routing with a small exact-message
   allowlist. Test positive-prefix attacks, short hostility, and unsupported
   scripts. Measure the change in escalation on the existing golden set.
2. Gate combined T1 precision, recall, FPR, lift, and successful coverage on
   every repeat. Reject missing configuration, empty evaluations, missing
   required slices, and undefined required metrics. Keep no-miss routing
   explicitly valid when there are no misses.
3. Package the synthetic blocklist using `importlib.resources`; validate JSON
   shape and nonempty string entries. Fail visibly on configuration errors.
   Verify behavior from a built wheel outside the checkout.
4. Introduce a runtime message input separate from labels. Require explanations
   for flagged T1 results and preserve fallback verdicts on malformed output.
   Enforce a process-wide inference limit and synchronize lazy initialization.
5. Update CI and documentation: distinguish deterministic CI from an explicit
   local model gate, historical model measurements from current measurements,
   and HTTP timeouts from end-to-end deadlines. Remove unsupported retry and
   production-readiness claims.

Validation: targeted regression tests first, full unit suite, deterministic
evaluation, broken-model evaluation, wheel smoke check, and diff inspection.
Live model measurements require an available Ollama server and model; report
their availability separately from deterministic validation.

## Completion

All five steps are implemented. See [validation.md](validation.md) for the
regression results, installed-wheel check, and three passing live model runs.
