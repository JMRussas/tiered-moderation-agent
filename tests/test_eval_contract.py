"""Tests for the eval harness itself.

An eval suite that is silently broken is worse than none: it reports green
while measuring nothing. These pin the metric definitions and the fixture's
own integrity.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness import GOLDEN, load_golden
from metrics import Counts, Routing, evaluate
from tiermod import t0
from tiermod.schema import Verdict

ROOT = Path(__file__).resolve().parents[1]


# --- Fixture integrity ----------------------------------------------------

def test_golden_ids_unique():
    ids = [l.id for l in load_golden()]
    assert len(ids) == len(set(ids))


def test_golden_is_valid_jsonl():
    for i, line in enumerate(GOLDEN.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            json.loads(line)  # raises with the line number in context


def test_gated_slice_has_both_classes():
    """A slice with no positives reports recall=None and silently passes any
    threshold. Guard against a fixture edit hollowing out a gate."""
    golden = load_golden()
    readable = [l for l in golden if not l.t0_blind]
    assert sum(l.toxic or l.scam for l in readable) >= 5
    assert sum(not (l.toxic or l.scam) for l in readable) >= 5


def test_no_real_blocklist_terms_committed():
    """The committed list must stay synthetic. If someone points this at a real
    list and commits it, fail loudly."""
    terms = json.loads((ROOT / "src" / "tiermod" / "blocklist.json").read_text(encoding="utf-8"))
    assert all(t.startswith("zzsynth") for t in terms), (
        "src/tiermod/blocklist.json must contain synthetic placeholders only"
    )


# --- Metric definitions ---------------------------------------------------

def test_counts_math():
    c = Counts(tp=3, fp=1, tn=5, fn=1)
    assert c.precision == pytest.approx(0.75)
    assert c.recall == pytest.approx(0.75)
    assert c.f1 == pytest.approx(0.75)
    assert c.fpr == pytest.approx(1 / 6)


def test_counts_undefined_rather_than_zero():
    """No positives means recall is undefined, not 0.0. Reporting 0.0 would
    make an empty slice look like a total failure and trip every gate."""
    assert Counts(tn=4).recall is None
    assert Counts(tn=4).precision is None


def test_safe_miss_rate_definition():
    r = Routing(total=10, escalated=4, missed_escalated=3, missed_silent=1)
    assert r.safe_miss_rate == pytest.approx(0.75)
    assert r.escalation_rate == pytest.approx(0.4)


def test_safe_miss_rate_is_undefined_when_nothing_was_missed():
    """No misses means the ratio has no denominator. `None`, not 1.0 -- a
    perfect-looking 1.0 on an empty run would mask a harness that scored
    nothing at all."""
    assert Routing(total=5, escalated=1).safe_miss_rate is None


# --- The invariant the whole design rests on ------------------------------

def test_no_silent_misses_on_the_golden_set():
    """Every positive T0 fails to catch must carry needs_llm=True.

    This is the same property `routing.min_safe_miss_rate = 1.0` gates in CI,
    asserted here so it fails at the unit level with the offending ids named.
    """
    golden = load_golden()
    pairs = [(l, t0.score(l.text)) for l in golden]
    report = evaluate(pairs)
    assert report.routing.silent_ids == [], (
        "confidently wrong on: " + ", ".join(report.routing.silent_ids)
    )


def test_t1_recovery_is_credited_only_where_it_ran():
    """Combined-tier scoring must take T1's answer on escalated messages and
    leave every other verdict untouched."""
    golden = load_golden()[:6]
    t0_pairs = [(l, Verdict(needs_llm=True)) for l in golden]
    t1_pairs = [(golden[0], Verdict(toxic=True, tier="T1"))]

    report = evaluate(t0_pairs, t1_pairs=t1_pairs)
    combined = report.slice_by("all+t1")
    assert combined is not None
    assert combined.n == len(golden)
    assert report.t1_evaluated == 1


# --- Gate configuration ---------------------------------------------------
# thresholds.toml is the only thing standing between a regression and a green
# build. A typo in it used to be a silent no-op.

def _report():
    golden = load_golden()
    return evaluate([(l, t0.score(l.text)) for l in golden])


def _with_thresholds(tmp_path, monkeypatch, *lines: str):
    import harness

    bad = tmp_path / "thresholds.toml"
    bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(harness, "THRESHOLDS", bad)
    return harness


def test_unknown_metric_in_thresholds_raises(tmp_path, monkeypatch):
    """`recal` instead of `recall` used to gate nothing and pass."""
    harness = _with_thresholds(tmp_path, monkeypatch, "[slice.all]", "recal = 0.9")

    with pytest.raises(ValueError, match="unknown metric"):
        harness.check_thresholds(_report())


def test_unknown_slice_in_thresholds_raises(tmp_path, monkeypatch):
    harness = _with_thresholds(
        tmp_path, monkeypatch, "[slice.t0_readible]", "recall = 0.9"
    )

    with pytest.raises(ValueError, match="unknown slice"):
        harness.check_thresholds(_report())


def test_unknown_routing_key_raises(tmp_path, monkeypatch):
    harness = _with_thresholds(
        tmp_path, monkeypatch, "[routing]", "min_safe_rate = 1.0"
    )

    with pytest.raises(ValueError, match="unknown key"):
        harness.check_thresholds(_report())


def test_shipped_thresholds_are_valid():
    """The committed gate config must itself pass validation."""
    import harness

    assert harness.check_thresholds(_report()) == []


@pytest.mark.parametrize("config", [
    "", "[slices.all]\nrecall = 0.9", "[slice.all]\nrecall = nan",
    "[slice.all]\nrecall = 1.5", "[slice.all]\nrecall = true",
])
def test_invalid_gate_configuration_is_rejected(tmp_path, monkeypatch, config):
    harness = _with_thresholds(tmp_path, monkeypatch, config)
    with pytest.raises(ValueError):
        harness.check_thresholds(_report())


def test_missing_required_slice_fails(tmp_path, monkeypatch):
    harness = _with_thresholds(tmp_path, monkeypatch, "[slice.paraphrase]", "recall = 0.5")
    pairs = [(l, t0.score(l.text)) for l in load_golden() if not l.paraphrase]
    assert "required slice paraphrase" in harness.check_thresholds(evaluate(pairs))[0]


def test_undefined_required_metric_fails(tmp_path, monkeypatch):
    harness = _with_thresholds(tmp_path, monkeypatch, "[slice.all]", "precision = 0.9")
    report = evaluate([(l, Verdict()) for l in load_golden()])
    assert harness.check_thresholds(report) == ["all.precision is undefined"]


def test_no_misses_satisfies_routing_without_fake_ratio(tmp_path, monkeypatch):
    harness = _with_thresholds(tmp_path, monkeypatch, "[routing]", "min_safe_miss_rate = 1.0")
    report = evaluate([(l, Verdict(toxic=l.toxic, scam=l.scam)) for l in load_golden()])
    assert report.routing.safe_miss_rate is None
    assert harness.check_thresholds(report) == []


def test_total_failure_has_zero_f1():
    assert Counts(fn=3).f1 == 0.0


def test_t1_outage_fails_even_though_fallback_is_safe():
    import harness

    pairs = [(l, t0.score(l.text)) for l in load_golden()]
    report = evaluate(pairs, t1_pairs=[(l, v) for l, v in pairs if v.needs_llm])
    assert report.t1_successful == 0
    assert any("success_rate" in f for f in harness.check_thresholds(report))


def test_missing_t1_run_fails_when_required():
    import harness

    assert "T1 evaluation is required but missing" in harness.check_thresholds(
        _report(), require_t1=True
    )


def test_partial_t1_coverage_fails():
    import harness

    pairs = [(l, t0.score(l.text)) for l in load_golden()]
    escalated = [(l, v) for l, v in pairs if v.needs_llm]
    report = evaluate(pairs, t1_pairs=escalated[:1])
    assert any("every escalated message" in f for f in harness.check_thresholds(report))


def test_earlier_failed_repeat_is_not_hidden_by_final_success(monkeypatch, tmp_path):
    import harness
    from tiermod import t1
    from tiermod.schema import Message

    calls = []
    labels = {l.id: l for l in load_golden()}

    def batch(inputs):
        assert all(type(message) is Message for message, _ in inputs)
        calls.append(1)
        return [(m, Verdict(tier="T1", toxic=labels[m.id].toxic if len(calls) > 1 else False,
                            scam=labels[m.id].scam if len(calls) > 1 else False))
                for m, _ in inputs]

    output = tmp_path / "results.json"
    monkeypatch.setattr(t1, "score_batch", batch)
    monkeypatch.setattr("sys.argv", ["harness", "--t1", "--repeat", "2", "--json", str(output)])
    assert harness.main() == 1
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["t1_lift_runs"][0] < 0
    assert saved["t1_lift_runs"][1] > 0.25
    assert all(f.startswith("run 1:") for f in saved["gate_failures"])


@pytest.mark.parametrize("contents", ["", "\n", "// no rows"])
def test_empty_golden_is_rejected(tmp_path, contents):
    path = tmp_path / "empty.jsonl"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_golden(path)
