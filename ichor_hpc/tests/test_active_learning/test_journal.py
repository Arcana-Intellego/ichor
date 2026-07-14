"""Tests for ichor.hpc.active_learning.daemon.journal."""
import ast
import json
import multiprocessing as mp
import os
from pathlib import Path

import pytest

import ichor.hpc.active_learning.daemon.journal as journal_module

from ichor.hpc.active_learning.daemon.journal import (
    EventTooLargeError,
    JOURNAL_LINE_LIMIT_BYTES,
    KNOWN_EVENT_TYPES,
    JournalCorruptionError,
    append_event,
    iter_events,
    read_events,
)


def test_append_event_writes_one_line(tmp_path):
    j = tmp_path / "journal.ndjson"
    ts = append_event(j, "phase_transition", from_phase="INIT", to_phase="PHASE_A_POLUS")
    raw = j.read_text(encoding="utf-8")
    assert raw.count("\n") == 1
    payload = json.loads(raw.strip())
    assert payload["event"] == "phase_transition"
    assert payload["from_phase"] == "INIT"
    assert payload["to_phase"] == "PHASE_A_POLUS"
    assert payload["ts"] == ts


def test_append_event_creates_parent_directory(tmp_path):
    j = tmp_path / "nested" / "subdir" / "journal.ndjson"
    append_event(j, "test")
    assert j.exists()


def test_append_event_appends_in_order(tmp_path):
    j = tmp_path / "journal.ndjson"
    for i in range(5):
        append_event(j, "step", index=i)
    events = list(iter_events(j))
    assert [e["index"] for e in events] == [0, 1, 2, 3, 4]


def test_append_event_reserved_keys_rejected(tmp_path):
    """`event` is a reserved key inside the JSON payload; passing it via
    **kwargs must raise. `ts` is an explicit parameter rather than a payload
    key, so it cannot collide via this path."""
    j = tmp_path / "journal.ndjson"
    with pytest.raises(ValueError):
        append_event(j, "evt", **{"event": "hijacked"})


def test_event_too_large_raises(tmp_path):
    j = tmp_path / "journal.ndjson"
    big_payload = "x" * (JOURNAL_LINE_LIMIT_BYTES * 2)
    with pytest.raises(EventTooLargeError):
        append_event(j, "huge", blob=big_payload)


def test_event_at_limit_does_not_raise(tmp_path):
    j = tmp_path / "journal.ndjson"
    # Construct a payload that lands just under the limit. The encoded record
    # is ~ 60 bytes of overhead + the blob length; aim for total < 4000.
    blob = "x" * (JOURNAL_LINE_LIMIT_BYTES - 200)
    append_event(j, "small_enough", blob=blob)
    events = list(iter_events(j))
    assert len(events) == 1


def test_iter_events_reports_interior_corruption(tmp_path):
    j = tmp_path / "journal.ndjson"
    j.write_text('{"ts":"2026-01-01T00:00:00Z","event":"ok"}\nbroken\n{"event":"ok2","ts":"2026-01-01T00:00:01Z"}\n')
    with pytest.raises(JournalCorruptionError, match="line 2"):
        list(iter_events(j))


def test_iter_events_returns_nothing_for_missing_file(tmp_path):
    j = tmp_path / "absent.ndjson"
    assert list(iter_events(j)) == []


def test_read_events_filters_by_event_type(tmp_path):
    j = tmp_path / "journal.ndjson"
    append_event(j, "scrub", point="P1")
    append_event(j, "sbatch", job_id="123")
    append_event(j, "scrub", point="P2")
    assert [e["point"] for e in read_events(j, event_type="scrub")] == ["P1", "P2"]
    assert [e["job_id"] for e in read_events(j, event_type=["sbatch"])] == ["123"]


def test_read_events_filters_by_since(tmp_path):
    j = tmp_path / "journal.ndjson"
    append_event(j, "a", x=1, ts="2026-01-01T00:00:00Z")
    append_event(j, "b", x=2, ts="2026-06-01T00:00:00Z")
    append_event(j, "c", x=3, ts="2026-12-01T00:00:00Z")
    out = list(read_events(j, since="2026-05-01T00:00:00Z"))
    assert [e["event"] for e in out] == ["b", "c"]


def test_append_event_retries_short_writes(monkeypatch, tmp_path):
    journal = tmp_path / "journal.ndjson"
    real_write = os.write
    calls = {"count": 0}

    def short_write(fd, data):
        calls["count"] += 1
        if calls["count"] == 1:
            return real_write(fd, data[: max(1, len(data) // 2)])
        return real_write(fd, data)

    monkeypatch.setattr(journal_module.os, "write", short_write)
    append_event(journal, "short_write", value=7)

    assert calls["count"] >= 2
    assert list(iter_events(journal))[0]["value"] == 7


def test_rotation_retains_bounded_complete_segments(tmp_path):
    journal = tmp_path / "journal.ndjson"
    for index in range(8):
        append_event(
            journal,
            "rotate",
            index=index,
            payload="x" * 80,
            max_bytes=220,
            retained_files=3,
        )

    segments = sorted(tmp_path.glob("journal.segment.*.ndjson"))
    assert len(segments) <= 2
    assert journal.is_file()
    events = list(iter_events(journal))
    assert events
    assert events[-1]["index"] == 7


def test_since_filter_compares_instants_not_iso_text(tmp_path):
    journal = tmp_path / "journal.ndjson"
    append_event(journal, "before", ts="2026-01-01T01:00:00+02:00")
    append_event(journal, "after", ts="2026-01-01T00:30:00+00:00")

    events = list(read_events(journal, since="2026-01-01T00:00:00Z"))

    assert [event["event"] for event in events] == ["after"]


def test_known_event_types_cover_static_literal_emitters():
    repo_root = Path(__file__).resolve().parents[3]
    source_root = repo_root / "ichor_hpc" / "ichor" / "hpc" / "active_learning"
    emitted = set()
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            else:
                continue
            if name not in {"append_event", "_journal", "_journal_event"}:
                continue
            event_arg_index = 1 if name == "append_event" else 0
            if len(node.args) <= event_arg_index:
                continue
            event_arg = node.args[event_arg_index]
            if isinstance(event_arg, ast.Constant) and isinstance(event_arg.value, str):
                emitted.add(event_arg.value)

    assert emitted - set(KNOWN_EVENT_TYPES) == set()


def _worker_append(path, count, tag):
    for i in range(count):
        append_event(path, "concurrent", tag=tag, i=i)


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX-only: O_APPEND cross-process atomicity is not guaranteed on Windows",
)
def test_concurrent_append_atomicity(tmp_path):
    """Multi-process appends must not produce malformed lines.

    POSIX guarantees that write(2) of <= PIPE_BUF on an O_APPEND fd is
    atomic with respect to concurrent writers; CSF4 (Linux) honours this.
    Windows _O_APPEND is application-level and does not guarantee atomicity
    across processes; the test is skipped there."""
    j = tmp_path / "journal.ndjson"
    procs = []
    for tag in ("A", "B", "C", "D"):
        p = mp.Process(target=_worker_append, args=(str(j), 50, tag))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=30)
    # Every line must be valid JSON; we may not know order but no corruption.
    with open(j, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert len(lines) == 4 * 50
    for ln in lines:
        json.loads(ln)   # must parse
