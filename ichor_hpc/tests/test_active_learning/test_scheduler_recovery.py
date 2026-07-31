"""Exact scheduler-terminal evidence and cumulative recovery ledgers."""

from __future__ import annotations

from copy import deepcopy
import getpass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning.daemon import (
    environment_equivalence,
    scheduler_recovery,
)
from ichor.hpc.active_learning.daemon.array_recovery import (
    PARTIAL_RECOVERY_PHASES,
)
from ichor.hpc.active_learning.daemon.live_executor import (
    LIVE_POSTPROCESS_IMPLEMENTED,
    LiveBackendsPhaseExecutor,
)
from ichor.hpc.active_learning.daemon.phase_executor import SBATCH_PHASES
from ichor.hpc.active_learning.daemon.state import atomic_write_json
from ichor.hpc.active_learning.cli import (
    _load_scheduler_recovery_status,
    _status_current_activity,
)
from ichor.hpc.active_learning.daemon.submission_intent import (
    expected_job_name,
    intent_dir,
)
from ichor.hpc.active_learning.daemon.scheduler_recovery import (
    LEGACY_PHASE_RECOVERY_LEDGER_SCHEMA_VERSION,
    LEGACY_SCHEDULER_TERMINAL_RECEIPT_SCHEMA_VERSION,
    classify_terminal_scheduler_evidence,
    classify_unaccepted_scheduler_intent,
    load_scheduler_terminal_receipt,
    phase_recovery_ledger_path,
    read_phase_recovery_ledger,
    read_scheduler_terminal_receipt,
    require_current_phase_recovery_authority,
    scheduler_terminal_recoveries,
    scheduler_terminal_receipt_path,
    write_phase_recovery_ledger,
    write_scheduler_terminal_receipt,
)
from ichor.hpc.active_learning.submit.sacct_poll import (
    JobObservation,
    JobStatus,
)


_QUANTUM_RECOVERY_PHASES = frozenset(
    {
        "INITIAL_GAUSSIAN",
        "INITIAL_AIMALL",
        "INITIAL_REPLACEMENT_GAUSSIAN",
        "INITIAL_REPLACEMENT_AIMALL",
        "GAUSSIAN",
        "AIMALL",
        "REPLACEMENT_GAUSSIAN",
        "REPLACEMENT_AIMALL",
    }
)
_ARIADNE_RECOVERY_PHASES = frozenset({"ARIADNE_ARRAY"})
_FEREBUS_RECOVERY_PHASES = frozenset({"INITIAL_FEREBUS", "FEREBUS"})
_DIVERSITY_RECOVERY_PHASES = frozenset(
    {"PHASE_A_DIVERSITY", "PHASE_B_DIVERSITY"}
)
_SCHEDULER_RECOVERY_PHASES = frozenset(
    _QUANTUM_RECOVERY_PHASES
    | _ARIADNE_RECOVERY_PHASES
    | _FEREBUS_RECOVERY_PHASES
    | _DIVERSITY_RECOVERY_PHASES
)


def test_every_scheduler_phase_has_one_explicit_recovery_family():
    assert len(_SCHEDULER_RECOVERY_PHASES) == 13
    assert SBATCH_PHASES == _SCHEDULER_RECOVERY_PHASES
    assert LIVE_POSTPROCESS_IMPLEMENTED == _SCHEDULER_RECOVERY_PHASES
    assert PARTIAL_RECOVERY_PHASES == (
        _QUANTUM_RECOVERY_PHASES | _ARIADNE_RECOVERY_PHASES
    )
    assert frozenset(environment_equivalence._PHASE_BACKEND) == (
        _SCHEDULER_RECOVERY_PHASES
    )


def _intent(
    *,
    scheduler: str = "slurm",
    expected_tasks: int = 4,
    sequence: int = 1,
    job_id: str = "17923151",
    phase: str = "ARIADNE_ARRAY",
    submission_kind: str = "array",
):
    attempt_id = f"{sequence:032x}"
    submission_identity = (
        "r0000-a" + str(sequence).zfill(4) + "-" + attempt_id[:8]
    )
    timestamp = "2026-07-28T00:00:00+00:00"
    return {
        "schema_version": 2,
        "campaign_uid": "campaign-scheduler-recovery",
        "phase": phase,
        "iteration": 15,
        "replacement_round": 0,
        "attempt_id": attempt_id,
        "attempt_sequence": sequence,
        "submission_identity": submission_identity,
        "scheduler_identity_kind": scheduler,
        "job_id": job_id,
        "expected_job_name": expected_job_name(
            "campaign-scheduler-recovery",
            phase,
            15,
            replacement_round=0,
            attempt_sequence=sequence,
            attempt_id=attempt_id,
            scheduler_identity_kind=scheduler,
        ),
        "status": "FAILED",
        "reason": "user_cancelled_via_stop",
        "submission_kind": submission_kind,
        "expected_tasks": expected_tasks,
        "job_ids_seen": [job_id],
        "submission_metadata": {},
        "environment_generation": sequence,
        "environment_generation_digest_sha256": str(sequence) * 64,
        "created_iso": timestamp,
        "updated_iso": timestamp,
        "updated_at_iso": timestamp,
    }


@pytest.mark.parametrize("scheduler_kind", ["slurm", "sge"])
@pytest.mark.parametrize("phase", sorted(_SCHEDULER_RECOVERY_PHASES))
def test_exact_terminal_classification_covers_every_scheduler_phase(
    tmp_path,
    scheduler_kind,
    phase,
):
    scalar = phase in _DIVERSITY_RECOVERY_PHASES
    expected_tasks = 1 if scalar else 2
    intent = _intent(
        scheduler=scheduler_kind,
        expected_tasks=expected_tasks,
        phase=phase,
        submission_kind=("scalar" if scalar else "array"),
    )
    observations = (
        [
            _observation(
                0,
                JobStatus.COMPLETED,
                job_id=intent["job_id"],
                job_name=intent["expected_job_name"],
            )
        ]
        if scalar
        else [
            _observation(
                0,
                JobStatus.COMPLETED,
                job_id=intent["job_id"],
                job_name=intent["expected_job_name"],
            ),
            _observation(
                1,
                JobStatus.CANCELLED,
                exit_code=(0, 15),
                job_id=intent["job_id"],
                job_name=intent["expected_job_name"],
            ),
        ]
    )
    if scalar:
        observations[0] = JobObservation(
            job_id=intent["job_id"],
            status=JobStatus.COMPLETED,
            exit_code=(0, 0),
            elapsed_seconds=120,
            raw_status=JobStatus.COMPLETED.value,
            job_name=intent["expected_job_name"],
            owner=getpass.getuser(),
        )

    classified = classify_terminal_scheduler_evidence(
        tmp_path,
        intent,
        observations,
        queue_active=False,
    )

    assert classified["phase"] == phase
    assert classified["scheduler_identity_kind"] == scheduler_kind
    assert classified["completed_logical_task_ids"] == [0]
    assert classified["retry_logical_task_ids"] == ([] if scalar else [1])


def _observation(
    task_id: int,
    status: JobStatus,
    *,
    exit_code=(0, 0),
    job_id: str = "17923151",
    job_name: str | None = None,
):
    return JobObservation(
        job_id=job_id + "_" + str(task_id),
        status=status,
        exit_code=exit_code,
        elapsed_seconds=120 + task_id,
        raw_status=status.value,
        job_name=(job_name or _intent(job_id=job_id)["expected_job_name"]),
        owner=getpass.getuser(),
    )


def _source_digest(records):
    return hashlib.sha256(
        ",".join(record["receipt_sha256"] for record in records).encode(
            "ascii"
        )
    ).hexdigest()


def _write_environment_proof(tmp_path, intent, *, current_generation=9):
    payload = {
        "schema_version": environment_equivalence.ENVIRONMENT_EQUIVALENCE_SCHEMA_VERSION,
        "fingerprint_algorithm": (
            environment_equivalence.SCIENTIFIC_FINGERPRINT_ALGORITHM
        ),
        "campaign_uid": intent["campaign_uid"],
        "phase": intent["phase"],
        "iteration": intent["iteration"],
        "replacement_round": intent["replacement_round"],
        "submission_identity": intent["submission_identity"],
        "attempt_id": intent["attempt_id"],
        "job_id": intent["job_id"],
        "scheduler_identity_kind": intent["scheduler_identity_kind"],
        "backend": environment_equivalence._PHASE_BACKEND[intent["phase"]],
        "producer_generation": intent["environment_generation"],
        "producer_generation_digest_sha256": intent[
            "environment_generation_digest_sha256"
        ],
        "current_generation": current_generation,
        "current_generation_digest_sha256": "f" * 64,
        "resource_resolution_path": "resolution.json",
        "resource_resolution_sha256": "e" * 64,
        "checks": {},
        "producer_fingerprint": None,
        "current_fingerprint": None,
        "uncovered_runtime_changes": [],
        "unresolved_dynamic_import_modules": [],
        "equivalent": True,
        "reasons": [],
        "recorded_at_iso": "2026-07-28T00:00:00+00:00",
    }
    payload["proof_sha256"] = environment_equivalence._sha256_json(payload)
    path = environment_equivalence.environment_equivalence_path(
        tmp_path,
        phase=intent["phase"],
        iteration=intent["iteration"],
        replacement_round=intent["replacement_round"],
        submission_identity=intent["submission_identity"],
        current_generation=current_generation,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return {
        "submission_identity": intent["submission_identity"],
        "equivalent": True,
        "reasons": [],
        "path": str(path),
        "sha256": payload["proof_sha256"],
    }


def _write_terminal_source(
    tmp_path,
    intent,
    *,
    completed_task_ids,
):
    history = (
        intent_dir(tmp_path)
        / "history"
        / (
            str(intent["phase"])
            + "-"
            + str(int(intent["iteration"])).zfill(6)
            + "-"
            + str(intent["attempt_id"])
            + ".json"
        )
    )
    history.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(history, intent)
    completed = set(completed_task_ids)
    if intent["submission_kind"] == "scalar":
        status = (
            JobStatus.COMPLETED
            if 0 in completed
            else JobStatus.CANCELLED
        )
        observations = [
            JobObservation(
                job_id=intent["job_id"],
                status=status,
                exit_code=((0, 0) if status is JobStatus.COMPLETED else (0, 15)),
                elapsed_seconds=120,
                raw_status=status.value,
                job_name=intent["expected_job_name"],
                owner=getpass.getuser(),
            )
        ]
    else:
        observations = [
            _observation(
                task_id,
                (
                    JobStatus.COMPLETED
                    if task_id in completed
                    else JobStatus.CANCELLED
                ),
                exit_code=((0, 0) if task_id in completed else (0, 15)),
                job_id=intent["job_id"],
                job_name=intent["expected_job_name"],
            )
            for task_id in range(intent["expected_tasks"])
        ]
    classification = classify_terminal_scheduler_evidence(
        tmp_path,
        intent,
        observations,
        queue_active=False,
    )
    receipt = write_scheduler_terminal_receipt(
        tmp_path,
        intent,
        classification,
    )
    from ichor.hpc.active_learning.daemon.scheduler_recovery import (
        scheduler_terminal_receipt_path,
    )

    path = scheduler_terminal_receipt_path(
        tmp_path,
        phase=intent["phase"],
        iteration=intent["iteration"],
        replacement_round=intent["replacement_round"],
        submission_identity=intent["submission_identity"],
    )
    return {
        "path": str(path),
        "receipt_sha256": receipt["receipt_sha256"],
        "job_id": intent["job_id"],
        "submission_identity": intent["submission_identity"],
    }


def _write_legacy_terminal_source(
    tmp_path,
    intent,
    *,
    completed_task_ids,
):
    source = _write_terminal_source(
        tmp_path,
        intent,
        completed_task_ids=completed_task_ids,
    )
    current_path = Path(source["path"])
    payload = json.loads(current_path.read_text(encoding="utf-8"))
    payload["schema_version"] = (
        LEGACY_SCHEDULER_TERMINAL_RECEIPT_SCHEMA_VERSION
    )
    payload.pop("scheduler_owner", None)
    payload.pop("scheduler_identity_verified", None)
    payload.pop("scheduler_observation_sha256", None)
    payload.pop("receipt_sha256", None)
    payload["receipt_sha256"] = scheduler_recovery._sha256_json(payload)
    legacy_path = scheduler_terminal_receipt_path(
        tmp_path,
        phase=intent["phase"],
        iteration=intent["iteration"],
        replacement_round=intent["replacement_round"],
        submission_identity=intent["submission_identity"],
        schema_version=LEGACY_SCHEDULER_TERMINAL_RECEIPT_SCHEMA_VERSION,
    )
    atomic_write_json(legacy_path, payload)
    current_path.unlink()
    return {
        "path": str(legacy_path),
        "receipt_sha256": payload["receipt_sha256"],
        "job_id": intent["job_id"],
        "submission_identity": intent["submission_identity"],
    }


def test_mixed_slurm_terminal_outcomes_preserve_only_zero_exit_completions(
    tmp_path,
):
    intent = _intent()
    observations = [
        _observation(0, JobStatus.COMPLETED),
        _observation(1, JobStatus.CANCELLED, exit_code=(0, 15)),
        _observation(2, JobStatus.TIMEOUT, exit_code=(0, 0)),
        _observation(3, JobStatus.COMPLETED, exit_code=(7, 0)),
    ]

    classified = classify_terminal_scheduler_evidence(
        tmp_path,
        intent,
        observations,
        queue_active=False,
    )

    assert classified["completed_logical_task_ids"] == [0]
    assert classified["retry_logical_task_ids"] == [1, 2, 3]
    assert classified["n_completed"] == 1
    assert classified["n_retry"] == 3


@pytest.mark.parametrize(
    "job_name,owner,match",
    [
        ("foreign-job", getpass.getuser(), "job name"),
        (_intent()["expected_job_name"], "foreign-owner", "owner"),
        (_intent()["expected_job_name"], None, "owner"),
    ],
)
def test_terminal_classification_requires_exact_scheduler_identity(
    tmp_path,
    job_name,
    owner,
    match,
):
    observation = _observation(0, JobStatus.COMPLETED)
    observation = JobObservation(
        job_id=observation.job_id,
        status=observation.status,
        exit_code=observation.exit_code,
        elapsed_seconds=observation.elapsed_seconds,
        raw_status=observation.raw_status,
        job_id_raw=observation.job_id_raw,
        job_name=job_name,
        owner=owner,
    )
    observations = [observation] + [
        _observation(task_id, JobStatus.CANCELLED, exit_code=(0, 15))
        for task_id in range(1, 4)
    ]

    with pytest.raises(ValueError, match=match):
        classify_terminal_scheduler_evidence(
            tmp_path,
            _intent(),
            observations,
            queue_active=False,
        )


@pytest.mark.parametrize(
    "observations, match",
    [
        (
            [
                _observation(0, JobStatus.COMPLETED),
                _observation(1, JobStatus.CANCELLED),
                _observation(2, JobStatus.CANCELLED),
            ],
            "missing",
        ),
        (
            [
                _observation(0, JobStatus.COMPLETED),
                _observation(1, JobStatus.CANCELLED),
                _observation(2, JobStatus.CANCELLED),
                _observation(4, JobStatus.CANCELLED),
            ],
            "out-of-range",
        ),
        (
            [
                _observation(0, JobStatus.COMPLETED),
                _observation(1, JobStatus.CANCELLED),
                _observation(1, JobStatus.FAILED),
                _observation(3, JobStatus.CANCELLED),
            ],
            "conflicting",
        ),
        (
            [
                _observation(0, JobStatus.COMPLETED),
                _observation(1, JobStatus.CANCELLED),
                _observation(1, JobStatus.CANCELLED),
                _observation(3, JobStatus.CANCELLED),
            ],
            "duplicate",
        ),
    ],
)
def test_incomplete_or_contradictory_terminal_accounting_fails_closed(
    tmp_path,
    observations,
    match,
):
    with pytest.raises(ValueError, match=match):
        classify_terminal_scheduler_evidence(
            tmp_path,
            _intent(),
            observations,
            queue_active=False,
        )


def test_sge_uses_normalised_zero_based_task_ids(tmp_path):
    intent = _intent(scheduler="sge", expected_tasks=2)
    observations = [
        _observation(
            0,
            JobStatus.COMPLETED,
            job_name=intent["expected_job_name"],
        ),
        _observation(
            1,
            JobStatus.CANCELLED,
            exit_code=(137, 0),
            job_name=intent["expected_job_name"],
        ),
    ]

    classified = classify_terminal_scheduler_evidence(
        tmp_path,
        intent,
        observations,
        queue_active=False,
    )

    assert [row["scheduler_task_id"] for row in classified["outcomes"]] == [0, 1]
    assert classified["completed_logical_task_ids"] == [0]
    assert classified["retry_logical_task_ids"] == [1]


def test_sge_deleted_before_start_requires_complete_queued_or_held_evidence(
    tmp_path,
):
    intent = _intent(scheduler="sge", expected_tasks=3)
    classified = classify_terminal_scheduler_evidence(
        tmp_path,
        intent,
        [],
        queue_active=False,
        pre_cancel_rows=[
            {
                "job_id": "17923151_0",
                "state": "qw",
                "job_name": intent["expected_job_name"],
                "owner": getpass.getuser(),
            },
            {
                "job_id": "17923151_1",
                "state": "hqw",
                "job_name": intent["expected_job_name"],
                "owner": getpass.getuser(),
            },
            {
                "job_id": "17923151_2",
                "state": "qw",
                "job_name": intent["expected_job_name"],
                "owner": getpass.getuser(),
            },
        ],
    )
    assert classified["accounting_exception"] == "sge_deleted_before_start"
    assert classified["completed_logical_task_ids"] == []
    assert classified["retry_logical_task_ids"] == [0, 1, 2]

    with pytest.raises(ValueError, match="cardinality|missing|terminal"):
        classify_terminal_scheduler_evidence(
            tmp_path,
            intent,
            [],
            queue_active=False,
            pre_cancel_rows=[
                {
                    "job_id": "17923151_0",
                    "state": "qw",
                    "job_name": intent["expected_job_name"],
                    "owner": getpass.getuser(),
                },
                {
                    "job_id": "17923151_1",
                    "state": "qw",
                    "job_name": intent["expected_job_name"],
                    "owner": getpass.getuser(),
                },
            ],
        )


def test_pre_submit_cancellation_records_all_tasks_without_inventing_job_id(
    tmp_path,
    monkeypatch,
):
    from ichor.hpc.active_learning.daemon import submission_intent

    intent = _intent(expected_tasks=3)
    intent["job_id"] = None
    intent["status"] = "FAILED"
    intent["reason"] = "user_cancelled_before_scheduler_acceptance"
    classification = classify_unaccepted_scheduler_intent(tmp_path, intent)

    assert classification["scheduler_acceptance"] == "not_accepted"
    assert classification["job_id"] is None
    assert classification["completed_logical_task_ids"] == []
    assert classification["retry_logical_task_ids"] == [0, 1, 2]

    written = write_scheduler_terminal_receipt(
        tmp_path,
        intent,
        classification,
    )
    assert load_scheduler_terminal_receipt(tmp_path, intent) == written
    monkeypatch.setattr(
        submission_intent,
        "intent_attempt_records",
        lambda *_args, **_kwargs: [intent],
    )
    recovered = scheduler_terminal_recoveries(
        tmp_path,
        campaign_uid=intent["campaign_uid"],
        phase=intent["phase"],
        iteration=intent["iteration"],
        replacement_round=intent["replacement_round"],
    )
    assert len(recovered) == 1
    assert recovered[0]["receipt"]["job_id"] is None


def test_unstaged_pre_submit_intent_without_task_count_is_not_recovery_evidence(
    tmp_path,
):
    intent = _intent(
        phase="INITIAL_FEREBUS",
        expected_tasks=2,
    )
    intent["status"] = "PRE_SUBMIT"
    intent["job_id"] = None
    intent["job_ids_seen"] = []
    intent.pop("expected_tasks")
    intent.pop("reason")

    assert load_scheduler_terminal_receipt(tmp_path, intent) is None


def test_terminal_receipt_is_idempotent_and_rejects_changed_evidence(tmp_path):
    intent = _intent(expected_tasks=2)
    observations = [
        _observation(0, JobStatus.COMPLETED),
        _observation(1, JobStatus.CANCELLED),
    ]
    classification = classify_terminal_scheduler_evidence(
        tmp_path,
        intent,
        observations,
        queue_active=False,
    )

    first = write_scheduler_terminal_receipt(
        tmp_path,
        intent,
        classification,
    )
    second = write_scheduler_terminal_receipt(
        tmp_path,
        intent,
        classification,
    )
    assert second == first
    assert load_scheduler_terminal_receipt(tmp_path, intent) == first

    changed = deepcopy(classification)
    changed["outcomes"][0]["raw_status"] = "changed"
    with pytest.raises(
        ValueError,
        match="observation digest mismatch|different evidence",
    ):
        write_scheduler_terminal_receipt(
            tmp_path,
            intent,
            changed,
        )


def test_legacy_terminal_receipt_remains_readable_but_forces_retry(tmp_path):
    intent = _intent(expected_tasks=2)
    source = _write_legacy_terminal_source(
        tmp_path,
        intent,
        completed_task_ids=[0],
    )

    receipt = read_scheduler_terminal_receipt(
        source["path"],
        expected_intent=intent,
        campaign_dir=tmp_path,
    )
    assert receipt["schema_version"] == 1
    assert load_scheduler_terminal_receipt(tmp_path, intent) == receipt

    executor = object.__new__(LiveBackendsPhaseExecutor)
    assessments = executor._scheduler_recovery_environment_assessments(
        [{"intent": intent, "receipt": receipt}]
    )
    assert assessments[intent["submission_identity"]] == {
        "equivalent": False,
        "reasons": ["legacy_scheduler_identity_unproven"],
        "proof": None,
        "path": None,
        "sha256": None,
    }

    status = _load_scheduler_recovery_status(
        tmp_path,
        SimpleNamespace(
            campaign_uid=intent["campaign_uid"],
            phase=SimpleNamespace(value=intent["phase"]),
            iteration=intent["iteration"],
            replacement_round=intent["replacement_round"],
        ),
    )
    assert status["state"] == "legacy_unverified"
    assert status["n_reusable"] == 0
    assert status["n_retry"] == 2
    activity = _status_current_activity(
        {
            "phase": intent["phase"],
            "iteration": intent["iteration"],
            "replacement_round": intent["replacement_round"],
            "pending_jobs": {},
            "active_submission_intents": [],
            "_presentation_scheduler_recovery": status,
        }
    )
    assert "cannot safely authorise output reuse" in activity
    assert "2 affected tasks will be retried" in activity


def test_v2_ledger_accepts_legacy_receipt_only_for_safe_retry(tmp_path):
    intent = _intent(expected_tasks=2)
    source = _write_legacy_terminal_source(
        tmp_path,
        intent,
        completed_task_ids=[0],
    )
    ledger = {
        "campaign_uid": intent["campaign_uid"],
        "phase": intent["phase"],
        "iteration": intent["iteration"],
        "replacement_round": intent["replacement_round"],
        "source_receipt_sha256": _source_digest([source]),
        "source_terminal_receipts": [source],
        "environment_equivalences": [
            {
                "submission_identity": intent["submission_identity"],
                "equivalent": False,
                "reasons": ["legacy_scheduler_identity_unproven"],
                "path": None,
                "sha256": None,
            }
        ],
        "reusable_logical_task_ids": [],
        "retry_logical_task_ids": [0, 1],
        "recovery_lineage": [],
    }

    written = write_phase_recovery_ledger(tmp_path, ledger)
    assert written["schema_version"] == 2
    assert written["n_reusable"] == 0
    assert written["n_retry"] == 2

    unsafe = dict(
        ledger,
        reusable_logical_task_ids=[0],
        retry_logical_task_ids=[1],
        recovery_lineage=[],
    )
    with pytest.raises(
        ValueError,
        match="lineage does not cover|cannot authorise reusable output",
    ):
        scheduler_recovery._validate_phase_recovery_ledger(
            {
                **unsafe,
                "schema_version": 2,
                "n_reusable": 1,
                "n_retry": 1,
                "recorded_at_iso": "2026-07-28T00:00:00+00:00",
                "ledger_sha256": "0" * 64,
            },
            campaign_dir=tmp_path,
        )


def test_legacy_phase_ledger_is_readable_but_not_reuse_authority(tmp_path):
    intent = _intent(expected_tasks=2)
    source = _write_legacy_terminal_source(
        tmp_path,
        intent,
        completed_task_ids=[0],
    )
    payload = {
        "schema_version": LEGACY_PHASE_RECOVERY_LEDGER_SCHEMA_VERSION,
        "campaign_uid": intent["campaign_uid"],
        "phase": intent["phase"],
        "iteration": intent["iteration"],
        "replacement_round": intent["replacement_round"],
        "source_receipt_sha256": _source_digest([source]),
        "source_terminal_receipts": [source],
        "environment_equivalences": [
            {
                "submission_identity": intent["submission_identity"],
                "equivalent": False,
                "reasons": ["legacy evidence"],
                "path": None,
                "sha256": None,
            }
        ],
        "reusable_logical_task_ids": [],
        "retry_logical_task_ids": [0, 1],
        "n_reusable": 0,
        "n_retry": 2,
        "recovery_lineage": [],
        "recorded_at_iso": "2026-07-28T00:00:00+00:00",
    }
    payload["ledger_sha256"] = scheduler_recovery._sha256_json(payload)
    path = phase_recovery_ledger_path(
        tmp_path,
        phase=intent["phase"],
        iteration=intent["iteration"],
        replacement_round=intent["replacement_round"],
        schema_version=LEGACY_PHASE_RECOVERY_LEDGER_SCHEMA_VERSION,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)

    legacy = read_phase_recovery_ledger(path)
    assert legacy["schema_version"] == 1
    with pytest.raises(ValueError, match="cannot authorise reusable output"):
        require_current_phase_recovery_authority(legacy)


@pytest.mark.parametrize(
    ("phase", "submission_kind", "reuse_basis"),
    [
        ("ARIADNE_ARRAY", "array", "self_authenticating_output"),
        (
            "PHASE_B_DIVERSITY",
            "scalar",
            "authority_valid_publication",
        ),
    ],
)
def test_phase_authority_can_prove_reuse_after_scheduler_cancellation(
    tmp_path,
    phase,
    submission_kind,
    reuse_basis,
):
    intent = _intent(
        phase=phase,
        submission_kind=submission_kind,
        expected_tasks=1,
    )
    source = _write_terminal_source(
        tmp_path,
        intent,
        completed_task_ids=[],
    )
    equivalence = _write_environment_proof(tmp_path, intent)
    ledger = {
        "campaign_uid": intent["campaign_uid"],
        "phase": phase,
        "iteration": intent["iteration"],
        "replacement_round": intent["replacement_round"],
        "source_receipt_sha256": _source_digest([source]),
        "source_terminal_receipts": [source],
        "environment_equivalences": [equivalence],
        "reusable_logical_task_ids": [0],
        "retry_logical_task_ids": [],
        "recovery_lineage": [
            {
                "logical_task_id": 0,
                "reuse_basis": reuse_basis,
                "attempt_id": intent["attempt_id"],
                "submission_identity": intent["submission_identity"],
                "job_id": intent["job_id"],
                "environment_generation": intent["environment_generation"],
                "environment_generation_digest_sha256": intent[
                    "environment_generation_digest_sha256"
                ],
                "environment_equivalence_path": equivalence["path"],
                "environment_equivalence_sha256": equivalence["sha256"],
            }
        ],
    }

    written = write_phase_recovery_ledger(tmp_path, ledger)
    path = phase_recovery_ledger_path(
        tmp_path,
        phase=phase,
        iteration=intent["iteration"],
        replacement_round=intent["replacement_round"],
    )

    assert read_phase_recovery_ledger(path) == written


def test_quantum_reuse_still_requires_successful_scheduler_outcome(tmp_path):
    intent = _intent(
        phase="GAUSSIAN",
        submission_kind="array",
        expected_tasks=1,
    )
    source = _write_terminal_source(
        tmp_path,
        intent,
        completed_task_ids=[],
    )
    equivalence = _write_environment_proof(tmp_path, intent)
    ledger = {
        "campaign_uid": intent["campaign_uid"],
        "phase": intent["phase"],
        "iteration": intent["iteration"],
        "replacement_round": intent["replacement_round"],
        "source_receipt_sha256": _source_digest([source]),
        "source_terminal_receipts": [source],
        "environment_equivalences": [equivalence],
        "reusable_logical_task_ids": [0],
        "retry_logical_task_ids": [],
        "recovery_lineage": [
            {
                "logical_task_id": 0,
                "reuse_basis": "scheduler_completed_validated_output",
                "attempt_id": intent["attempt_id"],
                "submission_identity": intent["submission_identity"],
                "job_id": intent["job_id"],
                "environment_generation": intent["environment_generation"],
                "environment_generation_digest_sha256": intent[
                    "environment_generation_digest_sha256"
                ],
                "environment_equivalence_path": equivalence["path"],
                "environment_equivalence_sha256": equivalence["sha256"],
            }
        ],
    }

    with pytest.raises(ValueError, match="successful scheduler outcome"):
        write_phase_recovery_ledger(tmp_path, ledger)


def test_phase_recovery_ledger_allows_only_monotonic_receipt_history(tmp_path):
    first_intent = _intent(sequence=1, job_id="17923151")
    first_source = _write_terminal_source(
        tmp_path,
        first_intent,
        completed_task_ids=[0],
    )
    first_equivalence = _write_environment_proof(
        tmp_path,
        first_intent,
    )
    base = {
        "campaign_uid": "campaign-scheduler-recovery",
        "phase": "ARIADNE_ARRAY",
        "iteration": 15,
        "replacement_round": 0,
        "source_receipt_sha256": _source_digest([first_source]),
        "source_terminal_receipts": [first_source],
        "environment_equivalences": [first_equivalence],
        "reusable_logical_task_ids": [0],
        "retry_logical_task_ids": [1, 2, 3],
        "recovery_lineage": [
            {
                "logical_task_id": 0,
                "reuse_basis": "self_authenticating_output",
                "attempt_id": first_intent["attempt_id"],
                "submission_identity": first_intent[
                    "submission_identity"
                ],
                "job_id": first_intent["job_id"],
                "environment_generation": 1,
                "environment_generation_digest_sha256": "1" * 64,
                "environment_equivalence_path": first_equivalence["path"],
                "environment_equivalence_sha256": first_equivalence[
                    "sha256"
                ],
            },
        ],
    }
    write_phase_recovery_ledger(tmp_path, base)

    second_intent = _intent(sequence=2, job_id="17923152")
    second_source = _write_terminal_source(
        tmp_path,
        second_intent,
        completed_task_ids=[0, 1],
    )
    second_equivalence = _write_environment_proof(
        tmp_path,
        second_intent,
    )
    sources = [first_source, second_source]
    cumulative = dict(
        base,
        source_receipt_sha256=_source_digest(sources),
        source_terminal_receipts=sources,
        environment_equivalences=[
            first_equivalence,
            second_equivalence,
        ],
        reusable_logical_task_ids=[0, 1],
        retry_logical_task_ids=[2, 3],
        recovery_lineage=[
            {
                "logical_task_id": 0,
                "reuse_basis": "self_authenticating_output",
                "attempt_id": second_intent["attempt_id"],
                "submission_identity": second_intent[
                    "submission_identity"
                ],
                "job_id": second_intent["job_id"],
                "environment_generation": 2,
                "environment_generation_digest_sha256": "2" * 64,
                "environment_equivalence_path": second_equivalence["path"],
                "environment_equivalence_sha256": second_equivalence[
                    "sha256"
                ],
            },
            {
                "logical_task_id": 1,
                "reuse_basis": "self_authenticating_output",
                "attempt_id": second_intent["attempt_id"],
                "submission_identity": second_intent[
                    "submission_identity"
                ],
                "job_id": second_intent["job_id"],
                "environment_generation": 2,
                "environment_generation_digest_sha256": "2" * 64,
                "environment_equivalence_path": second_equivalence["path"],
                "environment_equivalence_sha256": second_equivalence[
                    "sha256"
                ],
            },
        ],
    )
    written = write_phase_recovery_ledger(tmp_path, cumulative)
    path = phase_recovery_ledger_path(
        tmp_path,
        phase="ARIADNE_ARRAY",
        iteration=15,
        replacement_round=0,
    )
    assert read_phase_recovery_ledger(path) == written

    contradictory = dict(
        cumulative,
        source_receipt_sha256=_source_digest([second_source]),
        source_terminal_receipts=[second_source],
        environment_equivalences=[second_equivalence],
    )
    with pytest.raises(ValueError, match="contradictory"):
        write_phase_recovery_ledger(tmp_path, contradictory)
