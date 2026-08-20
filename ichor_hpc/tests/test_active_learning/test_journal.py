"""Tests for ichor.hpc.active_learning.daemon.journal."""
import ast
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket

import pytest

import ichor.hpc.active_learning.daemon.journal as journal_module

from ichor.hpc.active_learning.daemon.journal import (
    EventTooLargeError,
    JOURNAL_LINE_LIMIT_BYTES,
    KNOWN_EVENT_TYPES,
    JournalCorruptionError,
    append_event,
    inspect_journal_integrity,
    iter_events,
    read_events,
    repair_journal_integrity,
)


def test_append_event_writes_one_line(tmp_path):
    j = tmp_path / "journal.ndjson"
    ts = append_event(j, "phase_transition", from_phase="INIT", to_phase="PHASE_A_DIVERSITY")
    raw = j.read_text(encoding="utf-8")
    assert raw.count("\n") == 1
    payload = json.loads(raw.strip())
    assert payload["event"] == "phase_transition"
    assert payload["from_phase"] == "INIT"
    assert payload["to_phase"] == "PHASE_A_DIVERSITY"
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

    This covers the directory mutex between local Linux processes. Distributed
    filesystem safety comes from server-atomic mkdir, not PIPE_BUF or O_APPEND.
    Windows process spawning does not preserve this test's import fixture."""
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


def _cross_host_overlap_fixture():
    payload = {
        "phase": "AIMALL",
        "iteration": 14,
        "replacement_round": 0,
        "producer_kind": "scheduler",
        "stage": "sge_scheduler_wait",
        "status": "running",
        "elapsed_seconds": 120.0,
        "stage_elapsed_seconds": 120.0,
        "job_id": "898600",
        "attempt_id": "r0000-a0001-13f81f72",
        "completed": 0,
        "total": 1,
        "unit": "tasks",
        "running": 1,
        "pending": 0,
        "failed": 0,
        "missing": 0,
        "scientific_publication_complete": False,
        "scheduler_identity_kind": "sge",
    }
    candidate = journal_module._encode_event(
        "scheduler_progress",
        payload,
        ts="2026-08-20T12:03:12+00:00",
    )
    bad = (
        b'13f81f72", "completed": 0, "total": 1, "unit": "tasks", '
        b'"running": 1, "pending": 0, "failed": 0, "missing": 0, '
        b'"scientific_publication_complete": false, '
        b'"scheduler_identity_kind": "sge"}\n'
    )
    required_previous_length = len(candidate) - len(bad)
    previous = None
    for padding in range(4000):
        trial = journal_module._encode_event(
            "phase_activity_progress",
            {
                "phase": "AIMALL",
                "iteration": 14,
                "producer_kind": "worker",
                "stage": "task",
                "status": "running",
                "padding": "x" * padding,
            },
            ts="2026-08-20T12:02:12+00:00",
        )
        if len(trial) == required_previous_length:
            previous = trial
            break
    assert previous is not None
    return previous + bad + candidate, bad


def test_cross_host_progress_overlap_is_repaired_losslessly(tmp_path):
    journal = tmp_path / "journal.ndjson"
    raw, malformed = _cross_host_overlap_fixture()
    journal.write_bytes(raw)

    report = inspect_journal_integrity(journal)

    assert report.disposition == "recoverable_cross_host_progress_overlap"
    assert report.repairable is True
    assert report.length == len(malformed)
    result = repair_journal_integrity(
        journal,
        report.to_dict(),
        archive_dir=tmp_path / "journal_quarantine",
    )
    assert Path(result["archive_path"]).read_bytes() == raw
    assert inspect_journal_integrity(journal).disposition == "valid"
    assert [event["event"] for event in iter_events(journal)] == [
        "phase_activity_progress",
        "scheduler_progress",
    ]


def test_unknown_interior_corruption_remains_unsafe(tmp_path):
    journal = tmp_path / "journal.ndjson"
    journal.write_text(
        '{"ts":"2026-01-01T00:00:00Z","event":"ok"}\n'
        'unrelated broken bytes\n'
        '{"ts":"2026-01-01T00:00:01Z","event":"ok"}\n',
        encoding="utf-8",
    )

    report = inspect_journal_integrity(journal)

    assert report.disposition == "unsafe"
    assert report.repairable is False


def test_progress_like_suffix_without_matching_identity_remains_unsafe(tmp_path):
    journal = tmp_path / "journal.ndjson"
    raw, malformed = _cross_host_overlap_fixture()
    lines = raw.splitlines(keepends=True)
    journal.write_bytes(
        lines[0] + b"deadbeef" + malformed[8:] + lines[2]
    )

    report = inspect_journal_integrity(journal)

    assert report.disposition == "unsafe"
    assert report.repairable is False


def test_append_refuses_to_cement_torn_tail(tmp_path):
    journal = tmp_path / "journal.ndjson"
    journal.write_bytes(b'{"ts":"2026-01-01T00:00:00Z","event":"partial"')

    with pytest.raises(JournalCorruptionError, match="unterminated tail"):
        append_event(journal, "next")

    assert inspect_journal_integrity(journal).disposition == "recoverable_torn_tail"


def test_complete_but_unterminated_tail_is_recoverable_not_appendable(tmp_path):
    journal = tmp_path / "journal.ndjson"
    journal.write_bytes(b'{"ts":"2026-01-01T00:00:00Z","event":"complete"}')

    report = inspect_journal_integrity(journal)

    assert report.disposition == "recoverable_torn_tail"
    assert report.valid_records == 0
    with pytest.raises(JournalCorruptionError, match="unterminated"):
        list(iter_events(journal))
    with pytest.raises(JournalCorruptionError, match="unterminated tail"):
        append_event(journal, "next")


@pytest.mark.skipif(os.name == "nt", reason="same-host liveness uses Linux /proc")
def test_append_recovers_dead_same_host_directory_lock(tmp_path):
    journal = tmp_path / "journal.ndjson"
    lock_dir = tmp_path / "journal.ndjson.append-lock"
    lock_dir.mkdir()
    (lock_dir / "owner.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "host": socket.gethostname(),
                "pid": 2_147_483_647,
                "process_start_identity": "dead",
                "nonce": "a" * 32,
                "created_at_iso": "2026-01-01T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    append_event(journal, "after_dead_owner", lock_timeout_seconds=1)

    assert not lock_dir.exists()
    assert list(iter_events(journal))[0]["event"] == "after_dead_owner"
