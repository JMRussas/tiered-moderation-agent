"""Holdout-set protocol tooling: strip labels for the blind verifier, then
merge generator and verifier labels and list what the adjudicator must decide.

    # from the repository root; H=evals/holdout
    uv run $H/holdout.py strip $H/raw/generated.jsonl > $H/raw/text-only.jsonl
    uv run $H/holdout.py merge $H/raw/generated.jsonl $H/raw/verified.jsonl \
        --author generator-<model> --verifier verifier-<model> > $H/raw/merged.jsonl
    uv run $H/holdout.py freeze $H/raw/merged.jsonl $H/adjudicated.jsonl $H/messages.jsonl

Nothing here reads a classifier or the golden set. `merge` writes a row for
every message with both proposals and a `needs_review` flag; `freeze` applies
the adjudicator's decisions, checks that every flagged row was decided, and
emits the final `Label`-shaped file plus counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

DECISIONS = ("toxic", "scam", "question")
# Fraction of agreed rows the adjudicator still spot-checks.
SPOT_CHECK_EVERY = 5
# Benign rows in these categories exist to tempt a false positive; they are
# the holdout's counterpart of the golden set's `fp_trap` rows.
TRAP_CATEGORIES = {"benign-critical", "benign-hostile-vocab", "benign-offplatform"}
TRAP_TAGS = {"hostile-vocab-benign", "quoted"}

# Required fields per file kind and the type each must have. LLM-authored
# rows drift; a string "false" must not become a positive.
SCHEMAS = {
    "generated": {"id": str, "text": str, "toxic": bool, "scam": bool, "question": bool,
                  "lang": str, "script": str, "category": str, "tags": list},
    "verified": {"id": str, "toxic": bool, "scam": bool, "question": bool},
    "merged": {"id": str, "text": str, "generator": dict, "verifier": dict,
               "needs_review": bool, "category": str, "tags": list, "script": str},
    "adjudicated": {"id": str, "toxic": bool, "scam": bool, "question": bool},
}


def _rows(path: Path, kind: str) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path}: not found")
    rows = []
    schema = SCHEMAS[kind]
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{n}: {exc}") from None
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise SystemExit(f"{path}:{n}: not an object with a string id")
        for field, typ in schema.items():
            if field not in row:
                raise SystemExit(f"{path}:{n} ({row['id']}): missing {field}")
            if not isinstance(row[field], typ) or (typ is not bool and isinstance(row[field], bool)):
                raise SystemExit(f"{path}:{n} ({row['id']}): {field} must be {typ.__name__}, "
                                 f"got {row[field]!r}")
        rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no rows")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{path}: duplicate ids")
    return rows


def _emit(rows) -> None:
    for row in rows:
        sys.stdout.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_strip(args) -> int:
    _emit({"id": r["id"], "text": r["text"]} for r in _rows(args.generated, "generated"))
    return 0


def cmd_merge(args) -> int:
    generated = _rows(args.generated, "generated")
    verified = {r["id"]: r for r in _rows(args.verified, "verified")}
    missing = [r["id"] for r in generated if r["id"] not in verified]
    if missing:
        raise SystemExit(f"verifier output missing ids: {', '.join(missing)}")
    merged = []
    for i, g in enumerate(generated):
        v = verified[g["id"]]
        disagreements = [d for d in DECISIONS if g[d] != v[d]]
        review = (
            bool(disagreements)
            or v.get("confidence") == "low"
            or "ambiguous" in g.get("tags", [])
            or (g["toxic"] or g["scam"]) and g["category"] in ("toxic-subtle", "injection")
            or i % SPOT_CHECK_EVERY == 0
        )
        merged.append({
            "id": g["id"], "text": g["text"],
            "generator": {d: g[d] for d in DECISIONS} | {
                "lang": g.get("lang"), "rationale": g.get("rationale", "")},
            "verifier": {d: v[d] for d in DECISIONS} | {
                "lang": v.get("lang"), "confidence": v.get("confidence"),
                "rationale": v.get("rationale", "")},
            "disagreements": disagreements,
            "needs_review": review,
            "category": g["category"], "tags": g.get("tags", []),
            "script": g.get("script", "latin"), "lang": g.get("lang"),
            "author": args.author, "verifier_id": args.verifier,
        })
    _emit(merged)
    n = len(merged)
    agree = sum(not m["disagreements"] for m in merged)
    print(f"{n} rows; verifier agreed on {agree} ({agree / n:.1%}); "
          f"{sum(m['needs_review'] for m in merged)} flagged for review", file=sys.stderr)
    return 0


def cmd_freeze(args) -> int:
    merged = _rows(args.merged, "merged")
    # The adjudication file must exist, even if empty: a wrong path would
    # otherwise read as "nothing decided yet" and list every flagged row.
    if not args.adjudicated.exists():
        raise SystemExit(f"{args.adjudicated}: not found (create it, empty if needed)")
    decided = ({r["id"]: r for r in _rows(args.adjudicated, "adjudicated")}
               if args.adjudicated.read_text(encoding="utf-8").strip() else {})
    unknown = sorted(set(decided) - {m["id"] for m in merged})
    if unknown:
        raise SystemExit(f"adjudicated ids not in the merged file: {', '.join(unknown)}")
    undecided = [m["id"] for m in merged if m["needs_review"] and m["id"] not in decided]
    if undecided:
        raise SystemExit(f"{len(undecided)} rows need adjudication: {', '.join(undecided[:20])}"
                         + (" ..." if len(undecided) > 20 else ""))
    out = []
    changed = 0
    for m in merged:
        d = decided.get(m["id"])
        labels = {k: d[k] for k in DECISIONS} if d else m["generator"]
        if d and any(d[k] != m["generator"][k] for k in DECISIONS):
            changed += 1
        positive = bool(labels["toxic"] or labels["scam"])
        row = {
            "id": m["id"], "text": m["text"], **{k: bool(labels[k]) for k in DECISIONS},
            "lang": m["lang"], "script": m["script"], "category": m["category"],
            # Golden-set conventions, applied mechanically: `t0_blind` marks
            # POSITIVES whose text is outside Latin processing (the golden set
            # also hand-marks English evasion rows; the holdout does not, so
            # read the holdout by its `category:` slices instead). `fp_trap`
            # marks BENIGN rows written to tempt a false positive.
            "t0_blind": positive and (m["script"] != "latin" or m["lang"] == "ar-arabizi"),
            "fp_trap": not positive and (m["category"] in TRAP_CATEGORIES
                                         or bool(TRAP_TAGS & set(m["tags"]))),
            "note": (d or {}).get("note", ""),
            "tags": sorted(set(m["tags"]) | {"synthetic", m["author"], m["verifier_id"]}
                           | ({"adjudicated"} if d else set())),
        }
        out.append(row)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
                        encoding="utf-8", newline="\n")
    cats = Counter(r["category"] for r in out)
    pos = sum(r["toxic"] or r["scam"] for r in out)
    print(f"wrote {args.out}: {len(out)} rows, {pos} positive, {len(out) - pos} benign; "
          f"{len(decided)} adjudicated, {changed} labels changed from the generator",
          file=sys.stderr)
    for cat, n in sorted(cats.items()):
        print(f"  {cat:<24}{n:>4}", file=sys.stderr)
    return 0


def main() -> int:
    # JSONL is UTF-8 whatever the console thinks; Windows defaults to cp1252.
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    sys.stderr.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("strip", help="text-only JSONL for the blind verifier")
    s.add_argument("generated", type=Path)
    s.set_defaults(fn=cmd_strip)
    m = sub.add_parser("merge", help="combine generator and verifier output")
    m.add_argument("generated", type=Path)
    m.add_argument("verified", type=Path)
    m.add_argument("--author", required=True, help="e.g. generator-claude-opus-5")
    m.add_argument("--verifier", required=True, help="e.g. verifier-claude-sonnet-5")
    m.set_defaults(fn=cmd_merge)
    f = sub.add_parser("freeze", help="apply adjudication and write the Label-shaped file")
    f.add_argument("merged", type=Path)
    f.add_argument("adjudicated", type=Path,
                   help='JSONL of {"id","toxic","scam","question","note"} for reviewed rows')
    f.add_argument("out", type=Path)
    f.set_defaults(fn=cmd_freeze)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
