"""Compare two evaluation artifacts: what changed in the inputs, and what
changed in the outcomes.

    uv run evals/compare.py evals/results/a.json evals/results/b.json

A metric delta without an input delta is noise or nondeterminism; an input
delta without a metric delta is a change that did not matter. Both halves are
printed so neither can be quoted without the other.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report import Artifact, MessageOutcome  # noqa: E402


def _load(path: Path) -> Artifact:
    return Artifact.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _pct(x: float | None) -> str:
    return "  --  " if x is None else f"{x * 100:5.1f}%"


def _delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "   n/a"
    return f"{(b - a) * 100:+5.1f}"


def _final_messages(artifact: Artifact) -> list[MessageOutcome]:
    return artifact.repeats[-1].messages if artifact.repeats else artifact.t0.messages


def compare(a: Artifact, b: Artifact) -> str:
    out: list[str] = []
    pa, pb = a.provenance, b.provenance

    out.append("INPUTS")
    out.append(f"  {'':<22}{'A':<44}B")
    rows = [
        ("run", pa.run_id[:8], pb.run_id[:8]),
        ("timestamp", pa.timestamp_utc, pb.timestamp_utc),
        ("revision", (pa.git.revision or "unavailable")[:12], (pb.git.revision or "unavailable")[:12]),
        ("dirty", str(pa.git.dirty), str(pb.git.dirty)),
        ("model", pa.model.name if pa.model else "none", pb.model.name if pb.model else "none"),
        ("model digest", (pa.model.digest if pa.model and pa.model.digest else "unavailable")[:12],
         (pb.model.digest if pb.model and pb.model.digest else "unavailable")[:12]),
        ("server", (pa.model.server_version if pa.model else None) or "unavailable",
         (pb.model.server_version if pb.model else None) or "unavailable"),
        ("python", pa.python, pb.python),
    ]
    for name, x, y in rows:
        mark = "" if x == y else "   <- differs"
        out.append(f"  {name:<22}{x:<44}{y}{mark}")
    changed = sorted(k for k in set(pa.hashes) | set(pb.hashes) if pa.hashes.get(k) != pb.hashes.get(k))
    out.append(f"  changed inputs         {', '.join(changed) if changed else 'none'}")
    pkgs = sorted(k for k in set(pa.packages) | set(pb.packages) if pa.packages.get(k) != pb.packages.get(k))
    if pkgs:
        out.append("  changed packages       " + ", ".join(
            f"{k} {pa.packages.get(k)}->{pb.packages.get(k)}" for k in pkgs))
    if pa.inference != pb.inference:
        keys = sorted(k for k in set(pa.inference or {}) | set(pb.inference or {})
                      if (pa.inference or {}).get(k) != (pb.inference or {}).get(k))
        out.append("  changed inference      " + ", ".join(
            f"{k} {(pa.inference or {}).get(k)}->{(pb.inference or {}).get(k)}" for k in keys))
    if pa.unavailable or pb.unavailable:
        out.append(f"  unavailable            A: {sorted(pa.unavailable) or 'none'}  "
                   f"B: {sorted(pb.unavailable) or 'none'}")

    out.append("")
    out.append("OUTCOMES")
    out.append(f"  gate                   A: {'pass' if a.gate['passed'] else 'FAIL'}"
               f"   B: {'pass' if b.gate['passed'] else 'FAIL'}"
               f"   (repeats {a.gate['repeats']} / {b.gate['repeats']})")
    sa, sb = a.summary, b.summary
    out.append(f"  {'metric':<22}{'A':>8}{'B':>8}{'delta':>8}")
    out.append(f"  {'escalation rate':<22}{_pct(sa['escalation_rate']):>8}"
               f"{_pct(sb['escalation_rate']):>8}{_delta(sa['escalation_rate'], sb['escalation_rate']):>8}")
    for stage in ("t0", "final"):
        for metric in ("precision", "recall", "fpr"):
            x = (sa.get(stage) or {}).get(metric)
            y = (sb.get(stage) or {}).get(metric)
            out.append(f"  {stage + ' ' + metric:<22}{_pct(x):>8}{_pct(y):>8}{_delta(x, y):>8}")
    out.append(f"  {'t1 lift':<22}{_pct(sa['t1_lift']):>8}{_pct(sb['t1_lift']):>8}"
               f"{_delta(sa['t1_lift'], sb['t1_lift']):>8}")
    if sa.get("t1_status_counts") or sb.get("t1_status_counts"):
        out.append(f"  t1 statuses            A: {sa.get('t1_status_counts')}")
        out.append(f"                         B: {sb.get('t1_status_counts')}")

    ma = {m.id: m for m in _final_messages(a)}
    mb = {m.id: m for m in _final_messages(b)}
    flips = [i for i in sorted(set(ma) & set(mb)) if ma[i].cell != mb[i].cell
             or ma[i].final.needs_llm != mb[i].final.needs_llm]
    only_a, only_b = sorted(set(ma) - set(mb)), sorted(set(mb) - set(ma))
    out.append("")
    out.append(f"MESSAGES   {len(flips)} changed outcome, {len(only_a)} only in A, {len(only_b)} only in B")
    for i in flips:
        x, y = ma[i], mb[i]
        out.append(f"  {i:<8} {x.cell}{'?' if x.final.needs_llm else ' '} -> "
                   f"{y.cell}{'?' if y.final.needs_llm else ' '}   "
                   f"[{x.status or 't0'} -> {y.status or 't0'}]")
    if flips:
        out.append("  (? = final verdict still carries needs_llm)")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    args = ap.parse_args()
    print(compare(_load(args.a), _load(args.b)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
