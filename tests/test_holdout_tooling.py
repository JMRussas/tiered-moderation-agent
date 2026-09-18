"""The holdout protocol tooling: strip, merge, freeze."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from harness import load_golden

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "evals" / "holdout" / "holdout.py"


def _jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _run(*args):
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)],
                          capture_output=True, text=True, encoding="utf-8")


GENERATED = [
    {"id": "h001", "text": "gg", "toxic": False, "scam": False, "question": False,
     "lang": "en", "script": "latin", "category": "benign-chat", "tags": ["short"],
     "rationale": "hype"},
    {"id": "h002", "text": "wow did you learn to play yesterday", "toxic": True, "scam": False,
     "question": True, "lang": "en", "script": "latin", "category": "toxic-subtle",
     "tags": ["sarcasm"], "rationale": "contempt"},
    {"id": "h003", "text": "send 5 on c a s h a p p", "toxic": False, "scam": True,
     "question": False, "lang": "en", "script": "latin", "category": "scam-obfuscated",
     "tags": ["obfuscated"], "rationale": "payment lure"},
    {"id": "h004", "text": "ليه انتي وحشة كده", "toxic": True, "scam": False, "question": True,
     "lang": "ar", "script": "arabic", "category": "multilingual", "tags": [],
     "rationale": "insult"},
    {"id": "h005", "text": "id kill for that skin", "toxic": False, "scam": False,
     "question": False, "lang": "en", "script": "latin", "category": "benign-hostile-vocab",
     "tags": ["hostile-vocab-benign"], "rationale": "idiom"},
    {"id": "h006", "text": "ok", "toxic": False, "scam": False, "question": False,
     "lang": "en", "script": "latin", "category": "benign-chat", "tags": ["short"],
     "rationale": "ack"},
]
VERIFIED = [
    {"id": "h001", "toxic": False, "scam": False, "question": False, "lang": "en",
     "confidence": "high", "rationale": ""},
    {"id": "h002", "toxic": False, "scam": False, "question": True, "lang": "en",
     "confidence": "medium", "rationale": "read as banter"},   # disagreement
    {"id": "h003", "toxic": False, "scam": True, "question": False, "lang": "en",
     "confidence": "high", "rationale": ""},
    {"id": "h004", "toxic": True, "scam": False, "question": True, "lang": "ar",
     "confidence": "high", "rationale": ""},
    {"id": "h005", "toxic": False, "scam": False, "question": False, "lang": "en",
     "confidence": "low", "rationale": "unsure"},              # low confidence
    {"id": "h006", "toxic": False, "scam": False, "question": False, "lang": "en",
     "confidence": "high", "rationale": ""},
]


def test_strip_keeps_only_id_and_text(tmp_path):
    gen = _jsonl(tmp_path / "g.jsonl", GENERATED)
    out = _run("strip", gen)
    assert out.returncode == 0
    rows = [json.loads(l) for l in out.stdout.splitlines()]
    assert [set(r) for r in rows] == [{"id", "text"}] * len(GENERATED)
    assert "toxic" not in out.stdout and "rationale" not in out.stdout


def test_merge_flags_what_the_adjudicator_must_decide(tmp_path):
    gen = _jsonl(tmp_path / "g.jsonl", GENERATED)
    ver = _jsonl(tmp_path / "v.jsonl", VERIFIED)
    out = _run("merge", gen, ver, "--author", "generator-a", "--verifier", "verifier-b")
    assert out.returncode == 0, out.stderr
    merged = {r["id"]: r for r in map(json.loads, out.stdout.splitlines())}
    assert merged["h002"]["disagreements"] == ["toxic"] and merged["h002"]["needs_review"]
    assert merged["h005"]["needs_review"]                 # low confidence
    assert merged["h001"]["needs_review"]                 # spot check (index 0)
    assert not merged["h004"]["needs_review"] and not merged["h003"]["needs_review"]
    assert "agreed on 5 (83.3%)" in out.stderr


def test_merge_rejects_missing_verifier_rows(tmp_path):
    gen = _jsonl(tmp_path / "g.jsonl", GENERATED)
    ver = _jsonl(tmp_path / "v.jsonl", VERIFIED[:3])
    out = _run("merge", gen, ver, "--author", "a", "--verifier", "b")
    assert out.returncode != 0 and "missing ids: h004" in out.stderr


def test_freeze_requires_every_flagged_row_decided(tmp_path):
    gen = _jsonl(tmp_path / "g.jsonl", GENERATED)
    ver = _jsonl(tmp_path / "v.jsonl", VERIFIED)
    merged = tmp_path / "m.jsonl"
    merged.write_text(_run("merge", gen, ver, "--author", "generator-a",
                           "--verifier", "verifier-b").stdout, encoding="utf-8")
    adjudicated = _jsonl(tmp_path / "a.jsonl", [
        {"id": "h002", "toxic": True, "scam": False, "question": True, "note": "guide: sarcasm"},
    ])
    out = _run("freeze", merged, adjudicated, tmp_path / "messages.jsonl")
    assert out.returncode != 0
    assert "need adjudication" in out.stderr and "h005" in out.stderr and "h001" in out.stderr


def test_freeze_writes_label_rows_the_harness_loads(tmp_path):
    gen = _jsonl(tmp_path / "g.jsonl", GENERATED)
    ver = _jsonl(tmp_path / "v.jsonl", VERIFIED)
    merged = tmp_path / "m.jsonl"
    merged.write_text(_run("merge", gen, ver, "--author", "generator-a",
                           "--verifier", "verifier-b").stdout, encoding="utf-8")
    adjudicated = _jsonl(tmp_path / "a.jsonl", [
        {"id": "h001", "toxic": False, "scam": False, "question": False, "note": ""},
        {"id": "h002", "toxic": True, "scam": False, "question": True, "note": "guide: sarcasm"},
        {"id": "h005", "toxic": False, "scam": False, "question": False, "note": "idiom"},
        {"id": "h006", "toxic": False, "scam": False, "question": False, "note": ""},
    ])
    final = tmp_path / "messages.jsonl"
    out = _run("freeze", merged, adjudicated, final)
    assert out.returncode == 0, out.stderr
    labels = {l.id: l for l in load_golden(final)}
    assert len(labels) == 6
    assert labels["h002"].toxic and labels["h002"].note == "guide: sarcasm"
    assert "adjudicated" in labels["h002"].tags and "adjudicated" not in labels["h003"].tags
    assert labels["h004"].t0_blind and labels["h004"].script == "arabic"
    assert labels["h005"].fp_trap and not labels["h001"].fp_trap
    assert {"synthetic", "generator-a", "verifier-b"} <= set(labels["h003"].tags)
    assert "3 positive, 3 benign; 4 adjudicated, 0 labels changed" in out.stderr
    assert "\r\n" not in final.read_bytes().decode()


def test_rows_are_validated_with_the_row_id(tmp_path):
    bad = [dict(GENERATED[0]), dict(GENERATED[1])]
    bad[1]["toxic"] = "false"          # a string is not a label
    gen = _jsonl(tmp_path / "g.jsonl", bad)
    out = _run("strip", gen)
    assert out.returncode != 0 and "(h002): toxic must be bool, got 'false'" in out.stderr

    del bad[1]["toxic"]
    gen = _jsonl(tmp_path / "g.jsonl", bad)
    out = _run("strip", gen)
    assert out.returncode != 0 and "(h002): missing toxic" in out.stderr

    out = _run("strip", tmp_path / "missing.jsonl")
    assert out.returncode != 0 and "not found" in out.stderr
    out = _run("strip", _jsonl(tmp_path / "empty.jsonl", []))
    assert out.returncode != 0 and "no rows" in out.stderr


def test_freeze_rejects_unknown_ids_and_missing_file(tmp_path):
    gen = _jsonl(tmp_path / "g.jsonl", GENERATED)
    ver = _jsonl(tmp_path / "v.jsonl", VERIFIED)
    merged = tmp_path / "m.jsonl"
    merged.write_text(_run("merge", gen, ver, "--author", "a", "--verifier", "b").stdout,
                      encoding="utf-8")
    out = _run("freeze", merged, tmp_path / "nope.jsonl", tmp_path / "out.jsonl")
    assert out.returncode != 0 and "nope.jsonl: not found" in out.stderr
    adjudicated = _jsonl(tmp_path / "a.jsonl", [
        {"id": "h999", "toxic": True, "scam": False, "question": False, "note": "typo"},
    ])
    out = _run("freeze", merged, adjudicated, tmp_path / "out.jsonl")
    assert out.returncode != 0 and "not in the merged file: h999" in out.stderr


def test_fixture_assertions_follow_golden_conventions(tmp_path):
    """`t0_blind` marks non-Latin POSITIVES; `fp_trap` marks BENIGN trap-shaped
    rows. Both are set from labels and metadata, never from a classifier."""
    rows = [
        {**GENERATED[3], "id": "p1"},                                      # Arabic, toxic
        {**GENERATED[3], "id": "n1", "toxic": False, "question": False},   # Arabic, benign
        {**GENERATED[0], "id": "z1", "lang": "ar-arabizi", "scam": True},  # Arabizi, scam
        {**GENERATED[0], "id": "c1", "category": "benign-critical", "tags": []},
        {**GENERATED[0], "id": "o1", "category": "benign-offplatform", "tags": []},
        {**GENERATED[1], "id": "q1", "tags": ["quoted"]},                  # toxic + quoted
        {**GENERATED[0], "id": "b1"},                                      # plain benign
    ]
    verified = [{"id": r["id"], "toxic": r["toxic"], "scam": r["scam"],
                 "question": r["question"], "confidence": "high"} for r in rows]
    gen, ver = _jsonl(tmp_path / "g.jsonl", rows), _jsonl(tmp_path / "v.jsonl", verified)
    merged = tmp_path / "m.jsonl"
    merged.write_text(_run("merge", gen, ver, "--author", "a", "--verifier", "b").stdout,
                      encoding="utf-8")
    flagged = [json.loads(l)["id"] for l in merged.read_text(encoding="utf-8").splitlines()
               if json.loads(l)["needs_review"]]
    by_id = {r["id"]: r for r in rows}
    adjudicated = _jsonl(tmp_path / "a.jsonl", [
        {"id": i, "toxic": by_id[i]["toxic"], "scam": by_id[i]["scam"],
         "question": by_id[i]["question"], "note": ""} for i in flagged])
    final = tmp_path / "messages.jsonl"
    out = _run("freeze", merged, adjudicated, final)
    assert out.returncode == 0, out.stderr
    labels = {l.id: l for l in load_golden(final)}
    assert labels["p1"].t0_blind and labels["z1"].t0_blind
    assert not labels["n1"].t0_blind and not labels["b1"].t0_blind
    assert labels["c1"].fp_trap and labels["o1"].fp_trap
    assert not labels["q1"].fp_trap and not labels["b1"].fp_trap and not labels["n1"].fp_trap
