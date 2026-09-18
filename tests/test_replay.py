"""Replay of unlabeled chat: routing and load aggregates, no text in the artifact."""

from __future__ import annotations

import json

import replay
from harness import load_golden
from tiermod import t1
from tiermod.schema import Verdict


def _replay_file(tmp_path, rows):
    path = tmp_path / "chat.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _rows():
    # Two sessions, one minute apart, golden texts as stand-ins for real chat.
    golden = load_golden()
    rows = []
    for i, label in enumerate(golden):
        minute = i // 10
        rows.append({"id": f"m{i}", "text": label.text,
                     "ts": f"2026-09-01T20:{minute:02d}:{(i % 10) * 5:02d}+00:00",
                     "session": "s1" if i < 40 else "s2"})
    return rows


def test_t0_summary_accounts_for_every_message(tmp_path):
    rows = _rows()
    verdicts, elapsed = replay.score_t0(rows)
    s = replay.summarize_t0(rows, verdicts, elapsed)
    assert s["n"] == len(rows)
    assert s["escalated"] + s["resolved_clean"] + (s["flagged"] - s["flagged_and_escalated"]) == s["n"]
    assert sum(s["escalation_cause"].values()) == s["escalated"]
    assert sum(b["n"] for b in s["length_buckets"].values()) == s["n"]
    assert set(s["escalation_cause"]) <= {"non_latin", "arabizi", "unmatched"}


def test_load_uses_sessions_and_minutes():
    rows = _rows()
    verdicts, _ = replay.score_t0(rows)
    load = replay.summarize_load(rows, verdicts)
    assert load["timestamped"] == len(rows)
    assert load["sessions"] == 2 and load["session_key"] == "session"
    assert load["messages_per_minute"]["max"] == 10
    assert set(load["per_session"]) == {"s1", "s2"}
    assert load["per_session"]["s1"]["n"] == 40


def test_load_without_timestamps_is_explicit():
    rows = [{"id": "a", "text": "hi"}, {"id": "b", "text": "kys"}]
    verdicts, _ = replay.score_t0(rows)
    assert replay.summarize_load(rows, verdicts) == {"timestamped": 0}


def test_artifact_has_no_text_and_per_message_is_opt_in(monkeypatch, tmp_path):
    rows = _rows()
    path = _replay_file(tmp_path, rows)

    def batch(inputs):
        return [(m, t1.Outcome(Verdict(tier="T1", toxic="kys" in m.text, lang="en",
                                       reasons=["stub"] if "kys" in m.text else []), "ok", 5.0))
                for m, _ in inputs]

    monkeypatch.setattr(t1, "classify_batch", batch)
    out = tmp_path / "replay.json"
    per = tmp_path / "per.jsonl"
    monkeypatch.setattr("sys.argv", ["replay", str(path), "--t1", "--t1-limit", "5",
                                     "--json", str(out), "--per-message", str(per)])
    assert replay.main() == 0
    text = out.read_text(encoding="utf-8")
    for r in rows:
        assert r["text"] not in text
    art = json.loads(text)
    assert art["kind"] == "replay" and art["provenance"]["thresholds"] is None
    assert art["t1"]["evaluated"] == 5 and art["t1"]["sampled"] is True
    assert sum(art["t1"]["status_counts"].values()) == 5
    assert "_outcomes" not in art["t1"]
    per_rows = [json.loads(l) for l in per.read_text(encoding="utf-8").splitlines()]
    assert len(per_rows) == len(rows)
    assert sum(r["t1_status"] is not None for r in per_rows) == 5
    assert all("text" not in r for r in per_rows)
