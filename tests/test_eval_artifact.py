"""The evaluation artifact: structure, provenance, and fallback accounting.

An artifact is the only thing a reader has after the run is gone. These tests
pin what it must contain: every repeat, every message outcome, enough counts
to recompute each gate decision, and provenance that says "unavailable" rather
than guessing.
"""

from __future__ import annotations

import json

import pytest

import harness
import report as report_mod
from harness import load_golden
from report import Artifact, metrics_from_dict
from tiermod import t1
from tiermod.schema import Verdict


def _run(monkeypatch, tmp_path, *argv: str) -> tuple[int, Artifact, str]:
    output = tmp_path / "artifact.json"
    monkeypatch.setattr("sys.argv", ["harness", *argv, "--json", str(output)])
    code = harness.main()
    text = output.read_text(encoding="utf-8")
    return code, Artifact.model_validate(json.loads(text)), text


def _stub_t1(monkeypatch, statuses):
    """A T1 whose outcomes cycle through `statuses`. Non-ok outcomes return the
    base verdict, as the real tier does; ok outcomes copy the label."""
    labels = {l.id: l for l in load_golden()}

    def batch(inputs):
        results = []
        for i, (message, base) in enumerate(inputs):
            status = statuses[i % len(statuses)]
            if status == "ok":
                label = labels[message.id]
                verdict = Verdict(tier="T1", toxic=label.toxic, scam=label.scam,
                                  reasons=["stub"] if label.toxic or label.scam else [])
                results.append((message, t1.Outcome(verdict, "ok", 10.0 + i)))
            else:
                results.append((message, t1.Outcome(base, status, 1.0, error="StubError")))
        return results

    monkeypatch.setattr(t1, "classify_batch", batch)


# --- Structure -------------------------------------------------------------

def test_t0_artifact_structure(monkeypatch, tmp_path):
    code, artifact, _ = _run(monkeypatch, tmp_path)
    assert code == 0
    assert artifact.schema_version == 1
    assert artifact.gate["mode"] == "t0" and artifact.gate["passed"] is True
    assert artifact.repeats == []
    assert len(artifact.t0.messages) == len(load_golden())
    assert all(m.route == "t0" and m.status is None for m in artifact.t0.messages)
    # Per-message cells reproduce the slice counts.
    flagged = artifact.t0.metrics["slices"]["all"]["flagged"]
    for cell in ("tp", "fp", "tn", "fn"):
        assert sum(m.cell == cell for m in artifact.t0.messages) == flagged[cell]


def test_every_repeat_is_saved(monkeypatch, tmp_path):
    _stub_t1(monkeypatch, ["ok"])
    code, artifact, _ = _run(monkeypatch, tmp_path, "--t1", "--repeat", "3")
    assert code == 0
    assert [r.index for r in artifact.repeats] == [1, 2, 3]
    assert len(artifact.summary["t1_lift_runs"]) == 3
    for rep in artifact.repeats:
        assert len(rep.messages) == len(load_golden())
        assert rep.metrics["slices"]["all+t1"]["n"] == len(load_golden())


def test_artifact_carries_ids_not_fixture_text(monkeypatch, tmp_path):
    """Fixture text stays in the golden set. A private replay set must never
    be reproduced by a shared report, and the same code writes both."""
    _stub_t1(monkeypatch, ["ok"])
    _, _, text = _run(monkeypatch, tmp_path, "--t1")
    for label in load_golden():
        assert label.text not in text


# --- Fallback accounting -----------------------------------------------------

def test_fallback_accounting_covers_every_escalated_message(monkeypatch, tmp_path):
    statuses = ["ok", "capacity", "transport", "validation", "model"]
    _stub_t1(monkeypatch, statuses)
    code, artifact, _ = _run(monkeypatch, tmp_path, "--t1")
    assert code == 1  # success rate gate
    [rep] = artifact.repeats
    escalated = [m for m in rep.messages if m.route == "t1"]
    assert rep.t1.evaluated == len(escalated) == artifact.t0.metrics["routing"]["escalated"]
    assert set(rep.t1.status_counts) == set(t1.STATUSES)
    assert sum(rep.t1.status_counts.values()) == rep.t1.evaluated
    assert rep.metrics["t1"]["successful"] == rep.t1.status_counts["ok"]
    assert rep.t1.latency_ms.n == rep.t1.evaluated
    assert rep.t1.ok_latency_ms.n == rep.t1.status_counts["ok"]
    for m in escalated:
        if m.status != "ok":
            # Degraded messages keep T0's flags and its uncertainty.
            assert m.final == m.t0 and m.final.needs_llm and m.error == "StubError"
        else:
            assert m.final.tier == "T1" and not m.final.needs_llm and m.error is None
    for m in rep.messages:
        if m.route == "t0":
            assert m.status is None and m.latency_ms is None and m.final == m.t0


def test_gate_failures_recorded_per_repeat(monkeypatch, tmp_path):
    calls = []

    def batch(inputs):
        calls.append(1)
        status = "transport" if len(calls) == 1 else "ok"
        labels = {l.id: l for l in load_golden()}
        return [(m, t1.Outcome(
            base if status != "ok" else Verdict(tier="T1", toxic=labels[m.id].toxic,
                                                scam=labels[m.id].scam),
            status, 1.0)) for m, base in inputs]

    monkeypatch.setattr(t1, "classify_batch", batch)
    code, artifact, _ = _run(monkeypatch, tmp_path, "--t1", "--repeat", "2")
    assert code == 1
    first, second = artifact.repeats
    assert first.t1.status_counts["transport"] == first.t1.evaluated
    assert any("success_rate" in f for f in first.gate_failures)
    assert second.gate_failures == []
    assert artifact.gate["passed"] is False
    assert all(f.startswith("run 1:") for f in artifact.gate["failures"])


def test_no_gate_is_recorded_as_not_enforced(monkeypatch, tmp_path):
    _stub_t1(monkeypatch, ["transport"])
    code, artifact, _ = _run(monkeypatch, tmp_path, "--t1", "--no-gate")
    assert code == 0
    assert artifact.gate["passed"] is False and artifact.gate["enforced"] is False


# --- Reconstruction ----------------------------------------------------------

def test_gate_decisions_reconstruct_from_saved_counts(monkeypatch, tmp_path):
    """Applying today's thresholds to the saved counts must reproduce the
    saved decision exactly. Rates alone could not: 0/0 and rounding both hide."""
    _stub_t1(monkeypatch, ["ok", "ok", "validation"])
    _, artifact, _ = _run(monkeypatch, tmp_path, "--t1", "--repeat", "2")
    assert harness.check_thresholds(metrics_from_dict(artifact.t0.metrics)) == \
        artifact.t0.gate_failures
    for rep in artifact.repeats:
        rebuilt = metrics_from_dict(rep.metrics)
        assert harness.check_thresholds(rebuilt, require_t1=True) == rep.gate_failures
        assert rebuilt.t1_lift == pytest.approx(rep.metrics["t1"]["lift"])
        for name, saved in rep.metrics["slices"].items():
            assert rebuilt.slice_by(name).flagged.recall == saved["flagged"]["recall"]


# --- Provenance --------------------------------------------------------------

def test_provenance_is_collected_not_typed(monkeypatch, tmp_path):
    _, artifact, _ = _run(monkeypatch, tmp_path)
    prov = artifact.provenance
    assert prov.command[1:] == ["--json", str(tmp_path / "artifact.json")]
    assert set(prov.hashes) == set(report_mod.HASHED_INPUTS)
    assert all(len(h) == 64 for h in prov.hashes.values())
    assert prov.packages["pydantic"]
    assert prov.model is None and prov.inference is None  # T0 run: no model
    assert prov.run_id and prov.timestamp_utc.endswith("+00:00")


def test_missing_provenance_is_marked_unavailable(monkeypatch, tmp_path):
    def no_git(*_args):
        raise FileNotFoundError("git")

    def no_server(*_args, **_kwargs):
        raise ConnectionRefusedError()

    monkeypatch.setattr(report_mod, "_git", no_git)
    monkeypatch.setattr(report_mod, "_ollama", no_server)
    monkeypatch.setattr(report_mod, "HASHED_INPUTS", (*report_mod.HASHED_INPUTS, "does/not/exist"))
    _stub_t1(monkeypatch, ["ok"])
    _, artifact, _ = _run(monkeypatch, tmp_path, "--t1")
    prov = artifact.provenance
    assert prov.git.revision is None and prov.git.dirty is None
    assert prov.unavailable["git"] == "FileNotFoundError"
    assert prov.unavailable["hashes.does/not/exist"] == "file not found"
    assert prov.model is not None and prov.model.digest is None
    assert {"model.digest", "model.details", "model.server_version", "model.loaded"} <= \
        set(prov.unavailable)
    assert prov.inference == t1.inference_settings()


def test_model_name_resolves_to_server_tag(monkeypatch):
    seen = []

    def fake(url, path, body=None, timeout=5.0):
        seen.append((path, body))
        if path == "/api/tags":
            return {"models": [{"name": "qwen3.5:latest", "digest": "abc"}]}
        if path == "/api/ps":
            return {"models": [{"name": "qwen3.5:latest", "size": 1, "size_vram": 2}]}
        if path == "/api/show":
            return {"details": {"parameter_size": "9.7B", "quantization_level": "Q4_K_M",
                                "family": "qwen3"}}
        return {"version": "0.0"}

    monkeypatch.setattr(report_mod, "_ollama", fake)
    unavailable = {}
    info = report_mod.collect_model({"model": "ollama:qwen3.5", "server_url": "http://x"},
                                    unavailable)
    assert info.name == "qwen3.5:latest" and info.digest == "abc"
    assert info.size_vram_bytes == 2 and info.quantization == "Q4_K_M"
    assert ("/api/show", {"model": "qwen3.5:latest"}) in seen
    assert unavailable == {}


# --- Comparison --------------------------------------------------------------

def test_compare_reports_flipped_messages_and_changed_inputs(monkeypatch, tmp_path):
    import compare

    _stub_t1(monkeypatch, ["ok"])
    _, good, _ = _run(monkeypatch, tmp_path / "a", "--t1")
    _stub_t1(monkeypatch, ["transport"])
    _, degraded, _ = _run(monkeypatch, tmp_path / "b", "--t1")
    degraded.provenance.hashes["src/tiermod/t1.py"] = "0" * 64

    text = compare.compare(good, degraded)
    recovered = [m.id for m in good.repeats[0].messages
                 if m.route == "t1" and m.cell == "tp" and not m.t0.toxic and not m.t0.scam]
    assert recovered, "stub must recover at least one positive"
    # Every escalated message changed: it went from resolved back to uncertain.
    assert f"MESSAGES   {good.repeats[0].t1.evaluated} changed outcome" in text
    assert all(f"  {i:<8} tp  -> fn?" in text for i in recovered)
    assert "changed inputs         src/tiermod/t1.py" in text
    assert "A: pass   B: FAIL" in text
