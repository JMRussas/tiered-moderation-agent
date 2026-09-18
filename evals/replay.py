"""Replay unlabeled chat through the tiers and report routing, load, and cost.

    uv run evals/replay.py data/replay/<creator>.jsonl --json evals/results/replay-<name>.json
    uv run --extra llm evals/replay.py data/replay/<creator>.jsonl --t1 --t1-limit 500 --json ...

Input: one JSON object per line, {"id": str, "text": str, "ts": ISO-8601,
"session": str | null}. No labels: this answers "what fraction of real traffic
reaches a model, how fast, and what does it cost", not "was the verdict
right". Quality needs the labeled holdout.

The artifact holds aggregates only. Message text, viewer identity, and per-
message rows never enter it unless --per-message is passed, and that output
must stay with the private data.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from report import SCHEMA_VERSION, collect_provenance, summarize_latency  # noqa: E402
from tiermod import t0  # noqa: E402
from tiermod.schema import Message, Verdict  # noqa: E402

LENGTH_BUCKETS = (("1-3", 1, 3), ("4-10", 4, 10), ("11-40", 11, 40), ("41-80", 41, 80), ("81+", 81, 10**9))


def load_replay(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("id"), str) or not isinstance(row.get("text"), str):
            raise ValueError(f"{path}:{n}: needs string id and text")
        rows.append(row)
    if not rows:
        raise ValueError("replay file is empty")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("replay ids must be unique")
    return rows


def _parse_ts(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _bucket(text: str) -> str:
    n = len(text)
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= n <= hi:
            return name
    return "0"


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def _why_escalated(text: str, verdict: Verdict) -> str:
    """Which rule sent this to a model. Order matches t0.score."""
    if t0._has_non_latin_letters(text):
        return "non_latin"
    if verdict.arabizi:
        return "arabizi"
    return "unmatched"


def score_t0(rows: list[dict]) -> tuple[list[Verdict], float]:
    start = time.perf_counter()
    verdicts = [t0.score(r["text"]) for r in rows]
    return verdicts, (time.perf_counter() - start) * 1000


def summarize_t0(rows: list[dict], verdicts: list[Verdict], elapsed_ms: float) -> dict:
    n = len(rows)
    escalated = sum(v.needs_llm for v in verdicts)
    flagged = sum(v.toxic or v.scam for v in verdicts)
    reasons = Counter(r for v in verdicts for r in v.reasons)
    why = Counter(_why_escalated(r["text"], v) for r, v in zip(rows, verdicts) if v.needs_llm)
    resolved_clean = sum(not v.needs_llm and not (v.toxic or v.scam) for v in verdicts)

    buckets: dict[str, dict] = {}
    for r, v in zip(rows, verdicts):
        b = buckets.setdefault(_bucket(r["text"]), {"n": 0, "escalated": 0, "flagged": 0})
        b["n"] += 1
        b["escalated"] += v.needs_llm
        b["flagged"] += v.toxic or v.scam
    for b in buckets.values():
        b["escalation_rate"] = _rate(b["escalated"], b["n"])

    return {
        "n": n,
        "elapsed_ms": elapsed_ms,
        "per_message_us": elapsed_ms * 1000 / n,
        "escalated": escalated,
        "escalation_rate": _rate(escalated, n),
        # Resolved without a model: trivial allowlist hits plus deterministic policy hits.
        "resolved_clean": resolved_clean,
        "resolved_clean_rate": _rate(resolved_clean, n),
        "flagged": flagged,
        "toxic": sum(v.toxic for v in verdicts),
        "scam": sum(v.scam for v in verdicts),
        "question": sum(v.question for v in verdicts),
        "flagged_and_escalated": sum((v.toxic or v.scam) and v.needs_llm for v in verdicts),
        "escalation_cause": dict(why),
        "reasons": dict(reasons.most_common()),
        "length_buckets": {name: buckets[name] for name, _, _ in LENGTH_BUCKETS if name in buckets}
                          | ({"0": buckets["0"]} if "0" in buckets else {}),
    }


def summarize_load(rows: list[dict], verdicts: list[Verdict]) -> dict:
    """Offered load from timestamps: what a consumer would have to keep up with."""
    stamped = [(_parse_ts(r.get("ts")), r.get("session"), v) for r, v in zip(rows, verdicts)]
    stamped = [(ts, s, v) for ts, s, v in stamped if ts is not None]
    if not stamped:
        return {"timestamped": 0}
    stamped.sort(key=lambda x: x[0])
    per_minute = Counter(ts.replace(second=0, microsecond=0) for ts, _, _ in stamped)
    esc_per_minute = Counter(ts.replace(second=0, microsecond=0) for ts, _, v in stamped if v.needs_llm)
    minutes = sorted(per_minute.values(), reverse=True)
    esc_minutes = sorted(esc_per_minute.values(), reverse=True)

    sessions: dict = {}
    for ts, s, v in stamped:
        key = s or ts.date().isoformat()
        entry = sessions.setdefault(key, {"n": 0, "escalated": 0, "first": ts, "last": ts})
        entry["n"] += 1
        entry["escalated"] += v.needs_llm
        entry["first"] = min(entry["first"], ts)
        entry["last"] = max(entry["last"], ts)
    session_rates = [e["escalated"] / e["n"] for e in sessions.values() if e["n"]]
    session_rows = {
        k: {"n": e["n"], "escalated": e["escalated"], "escalation_rate": _rate(e["escalated"], e["n"]),
            "minutes": round((e["last"] - e["first"]).total_seconds() / 60, 1)}
        for k, e in sessions.items()
    }
    return {
        "timestamped": len(stamped),
        "first": stamped[0][0].isoformat(),
        "last": stamped[-1][0].isoformat(),
        "active_minutes": len(per_minute),
        "messages_per_minute": {"p50": median(minutes), "p95": minutes[int(0.05 * len(minutes))],
                                "max": minutes[0]},
        "escalated_per_minute": {"p50": median(esc_minutes) if esc_minutes else 0,
                                 "p95": esc_minutes[int(0.05 * len(esc_minutes))] if esc_minutes else 0,
                                 "max": esc_minutes[0] if esc_minutes else 0},
        "sessions": len(sessions),
        "session_key": "session" if any(s for _, s, _ in stamped) else "date",
        "session_escalation_rate": {"min": min(session_rates), "median": median(session_rates),
                                    "max": max(session_rates)} if session_rates else None,
        "per_session": session_rows,
    }


def run_t1(rows: list[dict], verdicts: list[Verdict], *, limit: int | None, seed: int) -> dict:
    from tiermod import t1 as t1_mod

    escalated = [(r, v) for r, v in zip(rows, verdicts) if v.needs_llm]
    chosen = escalated
    if limit is not None and limit < len(escalated):
        chosen = random.Random(seed).sample(escalated, limit)
    print(f"  [T1] classifying {len(chosen)}/{len(escalated)} escalated messages via "
          f"{t1_mod.model_name()} ...", flush=True)
    inputs = [(Message(id=r["id"], text=r["text"]), v) for r, v in chosen]
    start = time.perf_counter()
    results = t1_mod.classify_batch(inputs)
    wall_ms = (time.perf_counter() - start) * 1000
    outcomes = [o for _, o in results]
    counts = {s: 0 for s in t1_mod.STATUSES}
    for o in outcomes:
        counts[o.status] += 1
    ok = [o for o in outcomes if o.succeeded]
    after = Counter()
    for o in ok:
        after["toxic" if o.verdict.toxic else "scam" if o.verdict.scam else "clean"] += 1
        if o.verdict.toxic and o.verdict.scam:
            after["both"] += 1
    langs = Counter(o.verdict.lang or "(none)" for o in ok)
    return {
        "escalated": len(escalated),
        "evaluated": len(chosen),
        "sampled": len(chosen) < len(escalated),
        "seed": seed,
        "status_counts": counts,
        "success_rate": _rate(len(ok), len(chosen)),
        "wall_ms": wall_ms,
        "throughput_per_s": len(chosen) / (wall_ms / 1000) if wall_ms else None,
        "latency_ms": summarize_latency([o.latency_ms for o in outcomes]).model_dump(),
        "ok_latency_ms": summarize_latency([o.latency_ms for o in ok]).model_dump(),
        # Of the escalated messages a model actually resolved, how many turned
        # out to need a flag. This is the recall T0 would have had no way to get.
        "resolved_as": dict(after),
        "resolved_flag_rate": _rate(after["toxic"] + after["scam"], len(ok)),
        "lang": dict(langs.most_common(20)),
        "inference": t1_mod.inference_settings(),
        "_outcomes": {r["id"]: o for (r, _), o in zip(chosen, outcomes)},
    }


def print_summary(name: str, t0s: dict, load: dict, t1s: dict | None) -> None:
    print()
    print("=" * 74)
    print(f"  REPLAY {name}   |   {t0s['n']} messages   |   T0 {t0s['per_message_us']:.0f} us/msg")
    print("=" * 74)
    print(f"\n  escalated to a model   {t0s['escalation_rate'] * 100:5.1f}%   ({t0s['escalated']}/{t0s['n']})")
    print(f"  resolved clean at T0   {t0s['resolved_clean_rate'] * 100:5.1f}%")
    print(f"  T0 flagged             {t0s['flagged']}   toxic {t0s['toxic']}   scam {t0s['scam']}"
          f"   (of which still escalated: {t0s['flagged_and_escalated']})")
    print("  escalation cause       " + "   ".join(f"{k} {v}" for k, v in t0s["escalation_cause"].items()))
    print(f"\n  {'length':<8}{'n':>7}{'escalated':>11}")
    for k, b in t0s["length_buckets"].items():
        print(f"  {k:<8}{b['n']:>7}{b['escalation_rate'] * 100:>10.1f}%")
    if load.get("timestamped"):
        mpm, epm = load["messages_per_minute"], load["escalated_per_minute"]
        print(f"\n  offered load           p50 {mpm['p50']:.0f}/min   p95 {mpm['p95']}/min   max {mpm['max']}/min"
              f"   over {load['active_minutes']} active minutes, {load['sessions']} {load['session_key']}s")
        print(f"  escalated load         p50 {epm['p50']:.0f}/min   p95 {epm['p95']}/min   max {epm['max']}/min")
        if load["session_escalation_rate"]:
            r = load["session_escalation_rate"]
            print(f"  per-session escalation min {r['min'] * 100:.1f}%  median {r['median'] * 100:.1f}%"
                  f"  max {r['max'] * 100:.1f}%")
    if t1s:
        lat = t1s["latency_ms"]
        counts = " ".join(f"{k}={v}" for k, v in t1s["status_counts"].items() if v)
        print(f"\n  T1 on {t1s['evaluated']}/{t1s['escalated']} escalated"
              f"{' (sampled)' if t1s['sampled'] else ''}   {counts}")
        print(f"  T1 latency             p50 {lat['p50_ms']:.0f}ms   p95 {lat['p95_ms']:.0f}ms"
              f"   max {lat['max_ms']:.0f}ms   throughput {t1s['throughput_per_s']:.2f}/s")
        print(f"  resolved as            {t1s['resolved_as']}   flag rate {t1s['resolved_flag_rate'] * 100:.1f}%")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("replay", type=Path, help="JSONL of {id, text, ts, session?}")
    ap.add_argument("--name", help="label for the report (default: file stem)")
    ap.add_argument("--t1", action="store_true", help="classify the escalated subset with the model")
    ap.add_argument("--t1-limit", type=int, metavar="N",
                    help="classify a random sample of N escalated messages instead of all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, help="write the aggregate artifact here")
    ap.add_argument("--per-message", type=Path, metavar="PATH",
                    help="also write per-message routes/statuses (private; keep with the data)")
    args = ap.parse_args()

    rows = load_replay(args.replay)
    name = args.name or args.replay.stem
    verdicts, elapsed_ms = score_t0(rows)
    t0s = summarize_t0(rows, verdicts, elapsed_ms)
    load = summarize_load(rows, verdicts)
    t1s = run_t1(rows, verdicts, limit=args.t1_limit, seed=args.seed) if args.t1 else None
    print_summary(name, t0s, load, t1s)

    if args.per_message:
        outcomes = (t1s or {}).get("_outcomes", {})
        with args.per_message.open("w", encoding="utf-8", newline="\n") as fh:
            for r, v in zip(rows, verdicts):
                o = outcomes.get(r["id"])
                fh.write(json.dumps({
                    "id": r["id"], "t0_toxic": v.toxic, "t0_scam": v.scam, "needs_llm": v.needs_llm,
                    "reasons": v.reasons, "t1_status": o.status if o else None,
                    "t1_toxic": o.verdict.toxic if o else None, "t1_scam": o.verdict.scam if o else None,
                    "t1_lang": o.verdict.lang if o else None, "latency_ms": o.latency_ms if o else None,
                }) + "\n")
        print(f"  wrote per-message rows to {args.per_message} (private)")

    if args.json:
        public_t1 = {k: v for k, v in (t1s or {}).items() if not k.startswith("_")} or None
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "kind": "replay",
            "name": name,
            "provenance": collect_provenance(
                command=sys.argv, inference=(t1s or {}).get("inference"),
                dataset=args.replay, thresholds=None).model_dump(mode="json"),
            "t0": t0s,
            "load": load,
            "t1": public_t1,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(artifact, indent=2, default=str) + "\n",
                             encoding="utf-8", newline="\n")
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
