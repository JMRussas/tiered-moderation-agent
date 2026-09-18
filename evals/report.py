"""Versioned evaluation artifact: what a run measured, and what it ran.

An eval number without its inputs is a rumor. Every artifact the harness writes
carries enough to say what code, data, thresholds, model, and settings produced
it, and enough raw counts to recompute every gate decision without re-running
anything. Provenance the harness cannot collect is listed under
``provenance.unavailable`` with a reason; it is never guessed.

Repeatability here means traceable inputs and settings. It is not a promise
that a model returns identical answers on different hardware or server builds.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Literal, Sequence

from pydantic import BaseModel, Field

from metrics import Counts, Report, Routing, Slice
from tiermod.schema import Label, Verdict

SCHEMA_VERSION = 1

ROOT = Path(__file__).resolve().parents[1]
# Files whose content changes what an evaluation measures. Paths are relative
# to the repository root and are the keys under ``provenance.hashes``.
HASHED_INPUTS = (
    "evals/golden/messages.jsonl",
    "evals/thresholds.toml",
    "evals/metrics.py",
    "src/tiermod/blocklist.json",
    "src/tiermod/t0.py",
    "src/tiermod/t1.py",
)
PACKAGES = ("pydantic", "langchain", "langchain-core", "langchain-ollama", "ollama", "httpx")

Status = Literal["ok", "capacity", "transport", "validation", "model"]
Cell = Literal["tp", "fp", "tn", "fn"]


# --- Per-message outcomes --------------------------------------------------

class Expected(BaseModel):
    toxic: bool
    scam: bool


class Flags(BaseModel):
    toxic: bool
    scam: bool
    needs_llm: bool
    tier: str


class MessageOutcome(BaseModel):
    """One fixture through the pipeline. IDs only: fixture text stays in the
    golden set, and a future private replay must never end up in a report."""

    id: str
    expected: Expected
    t0: Flags
    final: Flags
    route: Literal["t0", "t1"]
    # T1 fields are None when the message was not escalated.
    status: Status | None = None
    latency_ms: float | None = None
    error: str | None = None
    # On the `flagged` (toxic or scam) decision, against the final verdict.
    cell: Cell


def _cell(predicted: bool, actual: bool) -> Cell:
    if predicted:
        return "tp" if actual else "fp"
    return "fn" if actual else "tn"


def message_outcome(
    label: Label, t0: Verdict, final: Verdict, *,
    status: Status | None = None, latency_ms: float | None = None,
    error: str | None = None,
) -> MessageOutcome:
    return MessageOutcome(
        id=label.id,
        expected=Expected(toxic=label.toxic, scam=label.scam),
        t0=Flags(toxic=t0.toxic, scam=t0.scam, needs_llm=t0.needs_llm, tier=t0.tier),
        final=Flags(toxic=final.toxic, scam=final.scam, needs_llm=final.needs_llm,
                    tier=final.tier),
        route="t1" if status is not None else "t0",
        status=status, latency_ms=latency_ms, error=error,
        cell=_cell(final.toxic or final.scam, label.toxic or label.scam),
    )


# --- Metrics, as raw counts ------------------------------------------------
# Rates are included for reading; counts are what make the gate reproducible.

def _counts_to_dict(c: Counts) -> dict:
    return {"tp": c.tp, "fp": c.fp, "tn": c.tn, "fn": c.fn,
            "precision": c.precision, "recall": c.recall, "f1": c.f1, "fpr": c.fpr}


def _counts_from_dict(d: dict) -> Counts:
    return Counts(tp=d["tp"], fp=d["fp"], tn=d["tn"], fn=d["fn"])


def metrics_to_dict(report: Report) -> dict:
    r = report.routing
    return {
        "slices": {
            s.name: {"n": s.n, "toxic": _counts_to_dict(s.toxic),
                     "scam": _counts_to_dict(s.scam), "flagged": _counts_to_dict(s.flagged)}
            for s in report.slices
        },
        "routing": {
            "total": r.total, "escalated": r.escalated,
            "missed_escalated": r.missed_escalated, "missed_silent": r.missed_silent,
            "silent_ids": list(r.silent_ids),
            "escalation_rate": r.escalation_rate, "safe_miss_rate": r.safe_miss_rate,
        },
        "t1": {
            "requested": report.t1_requested, "evaluated": report.t1_evaluated,
            "successful": report.t1_successful, "lift": report.t1_lift,
        },
        "combined_confident_misses": list(report.combined_confident_misses),
    }


def metrics_from_dict(d: dict) -> Report:
    """Rebuild a `Report` from saved counts, so `check_thresholds` can be
    re-applied to an artifact exactly as it was applied at run time."""
    slices = [
        Slice(name=name, n=s["n"], toxic=_counts_from_dict(s["toxic"]),
              scam=_counts_from_dict(s["scam"]), flagged=_counts_from_dict(s["flagged"]))
        for name, s in d["slices"].items()
    ]
    r = d["routing"]
    routing = Routing(escalated=r["escalated"], total=r["total"],
                      missed_escalated=r["missed_escalated"],
                      missed_silent=r["missed_silent"], silent_ids=list(r["silent_ids"]))
    t1 = d["t1"]
    return Report(slices=slices, routing=routing, t1_lift=t1["lift"],
                  t1_evaluated=t1["evaluated"], t1_requested=t1["requested"],
                  t1_successful=t1["successful"],
                  combined_confident_misses=list(d["combined_confident_misses"]))


# --- Repeats -----------------------------------------------------------------

class LatencySummary(BaseModel):
    n: int
    p50_ms: float | None
    p95_ms: float | None
    max_ms: float | None
    mean_ms: float | None


def summarize_latency(values: Sequence[float]) -> LatencySummary:
    if not values:
        return LatencySummary(n=0, p50_ms=None, p95_ms=None, max_ms=None, mean_ms=None)
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return LatencySummary(n=len(values), p50_ms=median(ordered), p95_ms=p95,
                          max_ms=ordered[-1], mean_ms=mean(ordered))


class T1Accounting(BaseModel):
    """Every escalated message ends in exactly one status; the counts sum to
    `evaluated`. Latency covers every attempt, including failed ones."""

    evaluated: int
    status_counts: dict[Status, int]
    latency_ms: LatencySummary
    # Latency of attempts that produced a validated verdict.
    ok_latency_ms: LatencySummary


class Repeat(BaseModel):
    index: int
    metrics: dict
    gate_failures: list[str]
    t1: T1Accounting
    messages: list[MessageOutcome]


class T0Pass(BaseModel):
    elapsed_ms: float
    metrics: dict
    gate_failures: list[str]
    messages: list[MessageOutcome]


# --- Provenance --------------------------------------------------------------

class GitInfo(BaseModel):
    revision: str | None = None
    branch: str | None = None
    # Tracked files with uncommitted changes. Untracked files are ignored, so
    # writing the artifact itself into the checkout does not mark it dirty.
    dirty: bool | None = None
    modified_files: list[str] = Field(default_factory=list)


class ModelInfo(BaseModel):
    name: str
    digest: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    family: str | None = None
    server_url: str
    server_version: str | None = None
    # From the server's loaded-model listing after the run, when present.
    size_bytes: int | None = None
    size_vram_bytes: int | None = None


class Hardware(BaseModel):
    machine: str | None = None
    processor: str | None = None
    cpu_count: int | None = None
    os: str | None = None
    gpus: list[str] = Field(default_factory=list)


class FileRef(BaseModel):
    path: str      # relative to the repository root when inside it
    sha256: str


class Provenance(BaseModel):
    run_id: str
    timestamp_utc: str
    command: list[str]
    git: GitInfo
    # What was scored and what gated it. These are the two inputs a reader
    # must check before comparing artifacts.
    dataset: FileRef
    thresholds: FileRef
    hashes: dict[str, str]
    python: str
    packages: dict[str, str | None]
    hardware: Hardware
    model: ModelInfo | None = None
    inference: dict | None = None
    # Field name -> why it could not be collected. Never filled in by hand.
    unavailable: dict[str, str] = Field(default_factory=dict)


class Artifact(BaseModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    provenance: Provenance
    gate: dict
    t0: T0Pass
    repeats: list[Repeat]
    summary: dict


# --- Collection --------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    # Not stripped: porcelain status lines begin with a significant space.
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True, timeout=10,
    ).stdout


def collect_git(unavailable: dict[str, str]) -> GitInfo:
    info = GitInfo()
    try:
        info.revision = _git("rev-parse", "HEAD").strip()
        info.branch = _git("rev-parse", "--abbrev-ref", "HEAD").strip()
        status = _git("status", "--porcelain", "--untracked-files=no")
        info.modified_files = sorted(line[3:] for line in status.splitlines() if line)
        info.dirty = bool(info.modified_files)
    except Exception as exc:  # git missing, not a checkout, timeout
        unavailable["git"] = type(exc).__name__
    return info


def collect_packages() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_hardware(unavailable: dict[str, str]) -> Hardware:
    hw = Hardware(machine=platform.machine() or None, processor=platform.processor() or None,
                  cpu_count=os.cpu_count(), os=platform.platform())
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
        hw.gpus = [line.strip() for line in out.splitlines() if line.strip()]
    except Exception as exc:
        unavailable["hardware.gpus"] = type(exc).__name__
    return hw


def _ollama(url: str, path: str, body: dict | None = None, timeout: float = 5.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url.rstrip("/") + path, data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def collect_model(settings: dict, unavailable: dict[str, str]) -> ModelInfo:
    """Query the Ollama server for the digest and loaded-model facts. Each
    field fails independently; the run's own results do not depend on this."""
    name = settings["model"].split(":", 1)[1] if settings["model"].startswith("ollama:") else settings["model"]
    if ":" not in name:
        name += ":latest"
    url = settings["server_url"]
    info = ModelInfo(name=name, server_url=url)
    try:
        info.server_version = _ollama(url, "/api/version").get("version")
    except Exception as exc:
        unavailable["model.server_version"] = type(exc).__name__
    try:
        show = _ollama(url, "/api/show", {"model": name})
        details = show.get("details", {})
        info.parameter_size = details.get("parameter_size")
        info.quantization = details.get("quantization_level")
        info.family = details.get("family")
    except Exception as exc:
        unavailable["model.details"] = type(exc).__name__
    try:
        tags = _ollama(url, "/api/tags").get("models", [])
        info.digest = next((m.get("digest") for m in tags if m.get("name") == name), None)
        if info.digest is None:
            unavailable["model.digest"] = "model not listed by server"
    except Exception as exc:
        unavailable["model.digest"] = type(exc).__name__
    try:
        loaded = _ollama(url, "/api/ps").get("models", [])
        entry = next((m for m in loaded if m.get("name") == name), None)
        if entry is None:
            unavailable["model.loaded"] = "model not loaded at collection time"
        else:
            info.size_bytes = entry.get("size")
            info.size_vram_bytes = entry.get("size_vram")
    except Exception as exc:
        unavailable["model.loaded"] = type(exc).__name__
    return info


def _file_ref(path: Path) -> FileRef:
    path = path.resolve()
    try:
        shown = path.relative_to(ROOT).as_posix()
    except ValueError:
        shown = str(path)
    return FileRef(path=shown, sha256=_sha256(path))


def collect_provenance(*, command: Sequence[str], inference: dict | None,
                       dataset: Path, thresholds: Path) -> Provenance:
    unavailable: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for rel in HASHED_INPUTS:
        path = ROOT / rel
        if path.exists():
            hashes[rel] = _sha256(path)
        else:
            unavailable[f"hashes.{rel}"] = "file not found"
    git = collect_git(unavailable)
    hardware = collect_hardware(unavailable)
    model = collect_model(inference, unavailable) if inference is not None else None
    # Built last: pydantic copies `unavailable`, so every collector runs first.
    return Provenance(
        run_id=str(uuid.uuid4()),
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        command=list(command),
        git=git,
        dataset=_file_ref(dataset),
        thresholds=_file_ref(thresholds),
        hashes=hashes,
        python=sys.version.split()[0],
        packages=collect_packages(),
        hardware=hardware,
        model=model,
        inference=inference,
        unavailable=unavailable,
    )
