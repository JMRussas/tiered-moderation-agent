"""Eval runner. Scores the golden set, prints a report, gates CI.

    uv run evals/harness.py              # T0 only -- no model, no network
    uv run evals/harness.py --t1         # adds the local LLM tier
    uv run evals/harness.py --json out.json

Exits non-zero when a threshold in evals/thresholds.toml is breached, so a
prompt tweak that quietly costs recall fails the build instead of shipping.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from metrics import SLICES, Report, evaluate  # noqa: E402
from tiermod import t0  # noqa: E402
from tiermod.schema import Label, Verdict  # noqa: E402

GOLDEN = ROOT / "evals" / "golden" / "messages.jsonl"
THRESHOLDS = ROOT / "evals" / "thresholds.toml"

VALID_METRICS = {"precision", "recall", "f1", "fpr"}
VALID_ROUTING = {"max_escalation_rate", "min_safe_miss_rate"}


def load_golden(path: Path = GOLDEN) -> list[Label]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            rows.append(Label(**json.loads(line)))
    return rows


# --- Formatting -----------------------------------------------------------

def _pct(x: float | None) -> str:
    return "  --  " if x is None else f"{x * 100:5.1f}%"


def _bar(x: float | None, width: int = 12) -> str:
    if x is None:
        return " " * width
    filled = round(x * width)
    return "#" * filled + "." * (width - filled)


def print_report(
    report: Report, golden: list[Label], *, elapsed_ms: float, t1: bool
) -> None:
    print()
    print("=" * 74)
    print(f"  TIERED MODERATION EVAL   |   {report.routing.total} messages"
          f"   |   T0 {elapsed_ms:.1f}ms total")
    print("=" * 74)

    print(f"\n  {'slice':<16}{'n':>4}  {'precision':>10}{'recall':>9}{'F1':>8}{'FPR':>8}")
    print("  " + "-" * 70)
    for s in report.slices:
        c = s.flagged
        marker = " <" if s.name in ("t0_blind", "all+t1") else ""
        print(
            f"  {s.name:<16}{s.n:>4}  {_pct(c.precision):>10}{_pct(c.recall):>9}"
            f"{_pct(c.f1):>8}{_pct(c.fpr):>8}{marker}"
        )

    r = report.routing
    print("\n  ROUTING - the two numbers the architecture rests on")
    print("  " + "-" * 70)
    print(f"    escalation rate    {_pct(r.escalation_rate)}  {_bar(r.escalation_rate)}"
          f"   {r.escalated}/{r.total} sent to a model")
    print(f"    safe miss rate     {_pct(r.safe_miss_rate)}  {_bar(r.safe_miss_rate)}"
          f"   {r.missed_escalated}/{r.missed_escalated + r.missed_silent} misses were flagged uncertain")

    if r.silent_ids:
        print(f"\n  SILENT MISSES ({len(r.silent_ids)}) - wrong AND confident.")
        print("  These reach a downstream gate as `toxic=False, needs_llm=False`.")
        print("  " + "-" * 70)
        by_id = {l.id: l for l in golden}
        for mid in r.silent_ids:
            lab = by_id[mid]
            print(f"    {mid}  [{lab.lang:>10}] {lab.text[:44]!r}")
            if lab.note:
                print(f"           {lab.note[:60]}")

    if report.combined_confident_misses:
        n = len(report.combined_confident_misses)
        print(f"\n  CONFIDENT MISSES AFTER T1 ({n})"
              " - residual exposure, reported not gated")
        print("    " + ", ".join(report.combined_confident_misses))

    if t1 and report.t1_lift is not None:
        arrow = "+" if report.t1_lift >= 0 else ""
        print(f"\n  T1 LIFT            {arrow}{report.t1_lift * 100:.1f} recall points"
              f"   ({report.t1_evaluated} escalated messages re-scored)")
        runs = report.t1_lift_runs
        if len(runs) > 1:
            lo, hi = min(runs) * 100, max(runs) * 100
            mean = sum(runs) / len(runs) * 100
            print(f"    across {len(runs)} runs   min {lo:.1f}   mean {mean:.1f}"
                  f"   max {hi:.1f}   spread {hi - lo:.1f}")
    print()


# --- Thresholds -----------------------------------------------------------

def check_thresholds(report: Report) -> list[str]:
    if not THRESHOLDS.exists():
        return []
    cfg = tomllib.loads(THRESHOLDS.read_text(encoding="utf-8"))
    failures: list[str] = []

    # A typo in this file used to be a silent no-op -- `getattr(..., None)` for
    # an unknown metric, `continue` for an unknown slice. Both made CI pass
    # having gated nothing, which is the worst possible behaviour for the file
    # whose only job is gating. Unknown keys are now hard errors.
    for name, rules in cfg.get("slice", {}).items():
        if name not in SLICES:
            raise ValueError(
                f"thresholds.toml: unknown slice [slice.{name}]. "
                f"Known: {', '.join(sorted(SLICES))}"
            )
        for metric in rules:
            if metric not in VALID_METRICS:
                raise ValueError(
                    f"thresholds.toml: unknown metric '{metric}' under [slice.{name}]. "
                    f"Known: {', '.join(sorted(VALID_METRICS))}"
                )
    for key in cfg.get("routing", {}):
        if key not in VALID_ROUTING:
            raise ValueError(
                f"thresholds.toml: unknown key '{key}' under [routing]. "
                f"Known: {', '.join(sorted(VALID_ROUTING))}"
            )

    for name, rules in cfg.get("slice", {}).items():
        s = report.slice_by(name)
        if s is None:
            continue
        for metric, floor in rules.items():
            got = getattr(s.flagged, metric, None)
            if metric == "fpr":
                if got is not None and got > floor:
                    failures.append(f"{name}.fpr {got:.3f} > max {floor}")
            elif got is not None and got < floor:
                failures.append(f"{name}.{metric} {got:.3f} < min {floor}")

    routing = cfg.get("routing", {})
    r = report.routing
    if "max_escalation_rate" in routing and r.escalation_rate is not None:
        if r.escalation_rate > routing["max_escalation_rate"]:
            failures.append(
                f"escalation_rate {r.escalation_rate:.3f} > max {routing['max_escalation_rate']}"
            )
    if "min_safe_miss_rate" in routing and r.safe_miss_rate is not None:
        if r.safe_miss_rate < routing["min_safe_miss_rate"]:
            failures.append(
                f"safe_miss_rate {r.safe_miss_rate:.3f} < min {routing['min_safe_miss_rate']}"
            )
    return failures


def to_dict(report: Report) -> dict:
    return {
        "slices": {
            s.name: {
                "n": s.n,
                "precision": s.flagged.precision,
                "recall": s.flagged.recall,
                "f1": s.flagged.f1,
                "fpr": s.flagged.fpr,
                "toxic": {"precision": s.toxic.precision, "recall": s.toxic.recall},
                "scam": {"precision": s.scam.precision, "recall": s.scam.recall},
            }
            for s in report.slices
        },
        "routing": {
            "escalation_rate": report.routing.escalation_rate,
            "safe_miss_rate": report.routing.safe_miss_rate,
            "silent_misses": report.routing.silent_ids,
        },
        "combined_confident_misses": report.combined_confident_misses,
        "t1_lift": report.t1_lift,
        "t1_lift_runs": report.t1_lift_runs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", action="store_true", help="also run the local LLM tier")
    ap.add_argument(
        "--repeat", type=int, default=1, metavar="N",
        help="run the T1 tier N times and report min/mean/max lift. A single "
             "run of a stochastic tier is an anecdote; temperature=0 is not a "
             "determinism guarantee.",
    )
    ap.add_argument("--json", type=Path, help="write machine-readable results here")
    ap.add_argument("--no-gate", action="store_true", help="report only, never fail")
    args = ap.parse_args()

    golden = load_golden()

    start = time.perf_counter()
    verdicts: list[Verdict] = [t0.score(l.text) for l in golden]
    elapsed_ms = (time.perf_counter() - start) * 1000

    pairs = list(zip(golden, verdicts))

    t1_pairs = None
    if args.t1:
        from tiermod import t1 as t1_mod

        escalated = [(l, v) for l, v in pairs if v.needs_llm]
        lifts = []
        for run in range(max(1, args.repeat)):
            print(f"  [T1] run {run + 1}/{args.repeat}: re-scoring "
                  f"{len(escalated)} escalated messages via "
                  f"{t1_mod.model_name()} ...", flush=True)
            # Pairs, not bare labels: T1 must know what to degrade back to.
            t1_pairs = t1_mod.score_batch(escalated)
            lift = evaluate(pairs, t1_pairs=t1_pairs).t1_lift
            if lift is not None:
                lifts.append(lift)


    report = evaluate(pairs, t1_pairs=t1_pairs)
    if args.t1:
        report.t1_lift_runs = lifts
    print_report(report, golden, elapsed_ms=elapsed_ms, t1=args.t1)

    if args.json:
        args.json.write_text(json.dumps(to_dict(report), indent=2), encoding="utf-8")
        print(f"  wrote {args.json}")

    failures = check_thresholds(report)
    if failures:
        print("  THRESHOLD FAILURES")
        for f in failures:
            print(f"    x  {f}")
        print()
        return 0 if args.no_gate else 1
    print("  all thresholds met\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
