"""Durable scheduler-terminal evidence for interrupted phase recovery.

The scheduler receipt records only control-plane facts.  Scientific output
validation is deliberately deferred to the phase executor and recorded in a
separate recovery ledger.
"""
from __future__ import annotations

from ..strict_json import strict_json as json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from ..submit.sacct_poll import (
    JobObservation,
    JobStatus,
    aggregate_states,
)
from .state import CampaignPhase, atomic_write_json
from .submission_intent import intent_submitted_logical_task_ids


SCHEDULER_TERMINAL_RECEIPT_SCHEMA_VERSION = 1
PHASE_RECOVERY_LEDGER_SCHEMA_VERSION = 1
SCHEDULER_TERMINAL_RECEIPT_DIRNAME = "scheduler_terminal_receipts"
PHASE_RECOVERY_LEDGER_DIRNAME = "phase_recovery_ledgers"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SGE_NEVER_STARTED_STATES = frozenset(
    {"qw", "hqw", "hRqw", "s", "S", "T", "ts", "tsS", "tT"}
)
_REUSE_BASES_BY_PHASE = {
    "INITIAL_GAUSSIAN": frozenset({"scheduler_completed_validated_output"}),
    "INITIAL_AIMALL": frozenset({"scheduler_completed_validated_output"}),
    "INITIAL_REPLACEMENT_GAUSSIAN": frozenset(
        {"scheduler_completed_validated_output"}
    ),
    "INITIAL_REPLACEMENT_AIMALL": frozenset(
        {"scheduler_completed_validated_output"}
    ),
    "GAUSSIAN": frozenset({"scheduler_completed_validated_output"}),
    "AIMALL": frozenset({"scheduler_completed_validated_output"}),
    "REPLACEMENT_GAUSSIAN": frozenset(
        {"scheduler_completed_validated_output"}
    ),
    "REPLACEMENT_AIMALL": frozenset(
        {"scheduler_completed_validated_output"}
    ),
    "ARIADNE_ARRAY": frozenset({"self_authenticating_output"}),
    "INITIAL_FEREBUS": frozenset({"scheduler_completed_task_receipt"}),
    "FEREBUS": frozenset({"scheduler_completed_task_receipt"}),
    "PHASE_A_DIVERSITY": frozenset({"authority_valid_publication"}),
    "PHASE_B_DIVERSITY": frozenset({"authority_valid_publication"}),
}
_SCHEDULER_COMPLETION_REUSE_BASES = frozenset(
    {
        "scheduler_completed_validated_output",
        "scheduler_completed_task_receipt",
    }
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _source_receipt_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        ",".join(
            str(record.get("receipt_sha256") or "")
            for record in records
        ).encode("ascii")
    ).hexdigest()


def _operational_root(campaign_dir: Union[str, Path]) -> Path:
    return Path(campaign_dir).expanduser().resolve() / ".DATA" / "ACTIVE_LEARNING"


def _phase_name(value: Any) -> str:
    text = value.value if isinstance(value, CampaignPhase) else str(value or "")
    if text not in {phase.value for phase in CampaignPhase}:
        raise ValueError("scheduler recovery phase is invalid")
    return text


def _exact_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(field + " must be a positive integer")
    return int(value)


def _exact_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(field + " must be a non-negative integer")
    return int(value)


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(field + " must be a non-empty string")
    return value


def _safe_identity(value: Any, field: str) -> str:
    text = _required_text(value, field)
    if not _SAFE_ID_RE.fullmatch(text):
        raise ValueError(field + " contains unsafe characters")
    return text


def _intent_identity(
    intent: Mapping[str, Any],
    *,
    allow_unaccepted: bool = False,
) -> Dict[str, Any]:
    job_id = intent.get("job_id")
    if job_id is None and allow_unaccepted:
        parsed_job_id = None
    else:
        parsed_job_id = _required_text(job_id, "job_id")
    return {
        "campaign_uid": _required_text(intent.get("campaign_uid"), "campaign_uid"),
        "phase": _phase_name(intent.get("phase")),
        "iteration": _exact_nonnegative_int(intent.get("iteration"), "iteration"),
        "replacement_round": _exact_nonnegative_int(
            intent.get("replacement_round", 0),
            "replacement_round",
        ),
        "attempt_id": _safe_identity(intent.get("attempt_id"), "attempt_id"),
        "submission_identity": _safe_identity(
            intent.get("submission_identity"),
            "submission_identity",
        ),
        "scheduler_identity_kind": _required_text(
            intent.get("scheduler_identity_kind") or "slurm",
            "scheduler_identity_kind",
        ).lower(),
        "job_id": parsed_job_id,
        "expected_job_name": _required_text(
            intent.get("expected_job_name"),
            "expected_job_name",
        ),
        "submission_kind": _required_text(
            intent.get("submission_kind"),
            "submission_kind",
        ),
        "expected_tasks": _exact_positive_int(
            intent.get("expected_tasks"),
            "expected_tasks",
        ),
    }


def _logical_task_ids(
    campaign_dir: Union[str, Path],
    intent: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> Tuple[int, ...]:
    if identity["submission_kind"] == "array":
        task_ids = tuple(
            int(value)
            for value in intent_submitted_logical_task_ids(campaign_dir, intent)
        )
    elif identity["submission_kind"] == "scalar":
        if int(identity["expected_tasks"]) != 1:
            raise ValueError("scalar scheduler intent must own exactly one task")
        task_ids = (0,)
    else:
        raise ValueError("submission_kind must be scalar or array")
    if len(task_ids) != int(identity["expected_tasks"]):
        raise ValueError("scheduler task map cardinality mismatch")
    if len(set(task_ids)) != len(task_ids) or any(value < 0 for value in task_ids):
        raise ValueError("scheduler task map contains invalid logical IDs")
    return task_ids


def _scheduler_index(
    observation: JobObservation,
    *,
    parent_job_id: str,
    submission_kind: str,
) -> int:
    if submission_kind == "scalar":
        if observation.job_id != parent_job_id:
            raise ValueError("scalar accounting row does not match its parent JobID")
        return 0
    prefix = parent_job_id + "_"
    if not str(observation.job_id).startswith(prefix):
        raise ValueError("array accounting row does not match its parent JobID")
    suffix = str(observation.job_id)[len(prefix) :]
    if not suffix.isdigit():
        raise ValueError("array accounting row has a malformed task index")
    return int(suffix)


def _pre_cancel_never_started(
    rows: Sequence[Mapping[str, Any]],
    *,
    parent_job_id: str,
    submission_kind: str,
    expected_tasks: int,
) -> bool:
    if len(rows) != int(expected_tasks) or not all(
        str(row.get("state") or "") in _SGE_NEVER_STARTED_STATES
        for row in rows
    ):
        return False
    indices = set()
    for row in rows:
        row_job_id = str(row.get("job_id") or "")
        if submission_kind == "scalar":
            if row_job_id != str(parent_job_id):
                return False
            scheduler_index = 0
        else:
            prefix = str(parent_job_id) + "_"
            if (
                not row_job_id.startswith(prefix)
                or not row_job_id[len(prefix) :].isdigit()
            ):
                return False
            scheduler_index = int(row_job_id[len(prefix) :])
        if scheduler_index in indices:
            return False
        indices.add(scheduler_index)
    return indices == set(range(int(expected_tasks)))


def classify_terminal_scheduler_evidence(
    campaign_dir: Union[str, Path],
    intent: Mapping[str, Any],
    observations: Sequence[JobObservation],
    *,
    queue_active: bool,
    queue_inconclusive: bool = False,
    queue_error: Optional[str] = None,
    pre_cancel_rows: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """Classify one submitted attempt after scheduler ownership has ended.

    Every expected native scheduler task must resolve to exactly one terminal
    accounting outcome.  The sole exception is an SGE job proven to have been
    entirely queued or held before deletion and to have produced no qacct rows.
    """
    identity = _intent_identity(intent)
    if queue_inconclusive:
        raise ValueError(
            "scheduler queue ownership is inconclusive"
            + (": " + str(queue_error) if queue_error else "")
        )
    if queue_active:
        raise ValueError("scheduler job remains active")
    logical_task_ids = _logical_task_ids(campaign_dir, intent, identity)
    scheduler_kind = str(identity["scheduler_identity_kind"])
    if scheduler_kind not in {"slurm", "sge"}:
        raise ValueError("unsupported scheduler identity in submission intent")

    accounting_exception: Optional[str] = None
    if (
        scheduler_kind == "sge"
        and not observations
        and _pre_cancel_never_started(
            pre_cancel_rows,
            parent_job_id=str(identity["job_id"]),
            submission_kind=str(identity["submission_kind"]),
            expected_tasks=int(identity["expected_tasks"]),
        )
    ):
        accounting_exception = "sge_deleted_before_start"
        outcomes = [
            {
                "scheduler_task_id": index,
                "logical_task_id": logical_task_id,
                "status": "CANCELLED",
                "exit_code": None,
                "elapsed_seconds": None,
                "job_id": str(identity["job_id"]) + "_" + str(index),
                "job_id_raw": None,
                "raw_status": "SGE job deleted before any task started",
            }
            for index, logical_task_id in enumerate(logical_task_ids)
        ]
    else:
        observed_scheduler_indices: Dict[
            int, Tuple[JobStatus, Optional[Tuple[int, int]]]
        ] = {}
        for observation in observations:
            scheduler_task_id = _scheduler_index(
                observation,
                parent_job_id=str(identity["job_id"]),
                submission_kind=str(identity["submission_kind"]),
            )
            if scheduler_task_id in observed_scheduler_indices:
                if observed_scheduler_indices[scheduler_task_id] != (
                    observation.status,
                    observation.exit_code,
                ):
                    raise ValueError(
                        "scheduler accounting contains conflicting task rows"
                    )
                raise ValueError(
                    "scheduler accounting contains duplicate task rows"
                )
            observed_scheduler_indices[scheduler_task_id] = (
                observation.status,
                observation.exit_code,
            )
        summary = aggregate_states(
            str(identity["job_id"]),
            observations,
            expected_task_count=int(identity["expected_tasks"]),
            submission_kind=str(identity["submission_kind"]),
            strict_parent_job_id=(scheduler_kind == "slurm"),
        )
        problems = []
        if not summary.is_terminal:
            problems.append("scheduler accounting is not terminal")
        if int(summary.n_missing):
            problems.append(
                "scheduler accounting is missing "
                + str(int(summary.n_missing))
                + " expected task row(s)"
            )
        if summary.conflicting_task_indices:
            problems.append("scheduler accounting contains conflicting task rows")
        if summary.out_of_range_task_indices:
            problems.append("scheduler accounting contains out-of-range task rows")
        if summary.malformed_job_ids:
            problems.append("scheduler accounting contains malformed task identities")
        if int(summary.n_observed) != int(identity["expected_tasks"]):
            problems.append("scheduler accounting task cardinality mismatch")
        if any(not observation.is_terminal for observation in summary.observations):
            problems.append("scheduler accounting contains non-terminal task rows")
        if problems:
            raise ValueError("; ".join(dict.fromkeys(problems)))

        outcomes = []
        for observation in summary.observations:
            scheduler_task_id = _scheduler_index(
                observation,
                parent_job_id=str(identity["job_id"]),
                submission_kind=str(identity["submission_kind"]),
            )
            if not 0 <= scheduler_task_id < len(logical_task_ids):
                raise ValueError("scheduler accounting task index is out of range")
            outcomes.append(
                {
                    "scheduler_task_id": scheduler_task_id,
                    "logical_task_id": int(logical_task_ids[scheduler_task_id]),
                    "status": observation.status.value,
                    "exit_code": (
                        None
                        if observation.exit_code is None
                        else [
                            int(observation.exit_code[0]),
                            int(observation.exit_code[1]),
                        ]
                    ),
                    "elapsed_seconds": (
                        None
                        if observation.elapsed_seconds is None
                        else int(observation.elapsed_seconds)
                    ),
                    "job_id": str(observation.job_id),
                    "job_id_raw": observation.job_id_raw,
                    "raw_status": observation.raw_status,
                }
            )
        outcomes.sort(key=lambda record: int(record["scheduler_task_id"]))

    completed = [
        int(record["logical_task_id"])
        for record in outcomes
        if record["status"] == JobStatus.COMPLETED.value
        and record["exit_code"] == [0, 0]
    ]
    retry = [
        int(record["logical_task_id"])
        for record in outcomes
        if int(record["logical_task_id"]) not in set(completed)
    ]
    task_set_sha256 = hashlib.sha256(
        ",".join(str(value) for value in logical_task_ids).encode("ascii")
    ).hexdigest()
    return {
        **identity,
        "scheduler_acceptance": "accepted",
        "logical_task_ids": list(logical_task_ids),
        "logical_task_set_sha256": task_set_sha256,
        "outcomes": outcomes,
        "completed_logical_task_ids": completed,
        "retry_logical_task_ids": retry,
        "n_completed": len(completed),
        "n_retry": len(retry),
        "accounting_exception": accounting_exception,
    }


def classify_unaccepted_scheduler_intent(
    campaign_dir: Union[str, Path],
    intent: Mapping[str, Any],
) -> Dict[str, Any]:
    """Record a gated PRE_SUBMIT attempt that never reached the scheduler."""
    identity = _intent_identity(intent, allow_unaccepted=True)
    if identity["job_id"] is not None:
        raise ValueError("unaccepted scheduler intent unexpectedly has a JobID")
    logical_task_ids = _logical_task_ids(campaign_dir, intent, identity)
    outcomes = [
        {
            "scheduler_task_id": scheduler_task_id,
            "logical_task_id": logical_task_id,
            "status": JobStatus.CANCELLED.value,
            "exit_code": None,
            "elapsed_seconds": None,
            "job_id": None,
            "job_id_raw": None,
            "raw_status": "scheduler submission stopped before acceptance",
        }
        for scheduler_task_id, logical_task_id in enumerate(logical_task_ids)
    ]
    return {
        **identity,
        "scheduler_acceptance": "not_accepted",
        "logical_task_ids": list(logical_task_ids),
        "logical_task_set_sha256": hashlib.sha256(
            ",".join(str(value) for value in logical_task_ids).encode("ascii")
        ).hexdigest(),
        "outcomes": outcomes,
        "completed_logical_task_ids": [],
        "retry_logical_task_ids": list(logical_task_ids),
        "n_completed": 0,
        "n_retry": len(logical_task_ids),
        "accounting_exception": "submission_stopped_before_acceptance",
    }


def scheduler_terminal_receipt_path(
    campaign_dir: Union[str, Path],
    *,
    phase: Any,
    iteration: int,
    replacement_round: int,
    submission_identity: str,
) -> Path:
    phase_name = _phase_name(phase)
    identity = _safe_identity(submission_identity, "submission_identity")
    filename = (
        phase_name
        + "-"
        + f"{_exact_nonnegative_int(iteration, 'iteration'):06d}"
        + "-r"
        + f"{_exact_nonnegative_int(replacement_round, 'replacement_round'):04d}"
        + "-"
        + identity
        + ".json"
    )
    return (
        _operational_root(campaign_dir)
        / SCHEDULER_TERMINAL_RECEIPT_DIRNAME
        / filename
    )


def _validate_terminal_receipt(
    payload: Mapping[str, Any],
    *,
    expected_intent: Optional[Mapping[str, Any]] = None,
    campaign_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    data = dict(payload)
    if data.get("schema_version") != SCHEDULER_TERMINAL_RECEIPT_SCHEMA_VERSION:
        raise ValueError("scheduler terminal receipt schema version is unsupported")
    acceptance = data.get("scheduler_acceptance")
    if acceptance not in {"accepted", "not_accepted"}:
        raise ValueError("scheduler terminal receipt acceptance state is invalid")
    identity = _intent_identity(
        data,
        allow_unaccepted=(acceptance == "not_accepted"),
    )
    if (identity["job_id"] is None) is not (acceptance == "not_accepted"):
        raise ValueError(
            "scheduler terminal receipt acceptance state contradicts its JobID"
        )
    _required_text(data.get("recorded_at_iso"), "recorded_at_iso")
    logical_task_ids = data.get("logical_task_ids")
    if not isinstance(logical_task_ids, list):
        raise ValueError("scheduler terminal receipt logical task IDs are invalid")
    parsed_ids = [
        _exact_nonnegative_int(value, "logical_task_id")
        for value in logical_task_ids
    ]
    if len(parsed_ids) != identity["expected_tasks"] or len(set(parsed_ids)) != len(parsed_ids):
        raise ValueError("scheduler terminal receipt task map is invalid")
    expected_digest = hashlib.sha256(
        ",".join(str(value) for value in parsed_ids).encode("ascii")
    ).hexdigest()
    if data.get("logical_task_set_sha256") != expected_digest:
        raise ValueError("scheduler terminal receipt task-set digest mismatch")
    outcomes = data.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != len(parsed_ids):
        raise ValueError("scheduler terminal receipt outcomes are incomplete")
    seen_scheduler = set()
    seen_logical = set()
    completed = []
    retry = []
    for record in outcomes:
        if not isinstance(record, Mapping):
            raise ValueError("scheduler terminal receipt outcome is invalid")
        scheduler_id = _exact_nonnegative_int(
            record.get("scheduler_task_id"),
            "scheduler_task_id",
        )
        logical_id = _exact_nonnegative_int(
            record.get("logical_task_id"),
            "logical_task_id",
        )
        if scheduler_id >= len(parsed_ids) or parsed_ids[scheduler_id] != logical_id:
            raise ValueError("scheduler terminal receipt outcome task mapping mismatch")
        if scheduler_id in seen_scheduler or logical_id in seen_logical:
            raise ValueError("scheduler terminal receipt contains duplicate outcomes")
        seen_scheduler.add(scheduler_id)
        seen_logical.add(logical_id)
        status = _required_text(record.get("status"), "outcome.status")
        if status not in {item.value for item in JobStatus if item.value}:
            raise ValueError("scheduler terminal receipt outcome status is invalid")
        exit_code = record.get("exit_code")
        if exit_code is not None and (
            not isinstance(exit_code, list)
            or len(exit_code) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in exit_code
            )
        ):
            raise ValueError("scheduler terminal receipt exit code is invalid")
        if status == JobStatus.COMPLETED.value and exit_code == [0, 0]:
            completed.append(logical_id)
        else:
            retry.append(logical_id)
    if data.get("completed_logical_task_ids") != completed:
        raise ValueError("scheduler terminal receipt completed-task summary mismatch")
    if data.get("retry_logical_task_ids") != retry:
        raise ValueError("scheduler terminal receipt retry-task summary mismatch")
    if data.get("n_completed") != len(completed) or data.get("n_retry") != len(retry):
        raise ValueError("scheduler terminal receipt task counts are invalid")
    digest = data.get("receipt_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError("scheduler terminal receipt digest is invalid")
    unsigned = dict(data)
    unsigned.pop("receipt_sha256", None)
    if _sha256_json(unsigned) != digest:
        raise ValueError("scheduler terminal receipt digest mismatch")
    if expected_intent is not None:
        expected = _intent_identity(
            expected_intent,
            allow_unaccepted=(acceptance == "not_accepted"),
        )
        for key, value in expected.items():
            if identity.get(key) != value:
                raise ValueError(
                    "scheduler terminal receipt does not match submission intent"
                )
        if campaign_dir is not None:
            expected_task_ids = list(
                _logical_task_ids(campaign_dir, expected_intent, expected)
            )
            if parsed_ids != expected_task_ids:
                raise ValueError(
                    "scheduler terminal receipt does not match the immutable "
                    "submitted task map"
                )
    return data


def validate_scheduler_terminal_receipt(
    payload: Mapping[str, Any],
    *,
    expected_intent: Optional[Mapping[str, Any]] = None,
    campaign_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Validate an in-memory terminal receipt."""
    return _validate_terminal_receipt(
        payload,
        expected_intent=expected_intent,
        campaign_dir=campaign_dir,
    )


def write_scheduler_terminal_receipt(
    campaign_dir: Union[str, Path],
    intent: Mapping[str, Any],
    classification: Mapping[str, Any],
) -> Dict[str, Any]:
    acceptance = str(classification.get("scheduler_acceptance") or "")
    identity = _intent_identity(
        intent,
        allow_unaccepted=(acceptance == "not_accepted"),
    )
    payload = {
        "schema_version": SCHEDULER_TERMINAL_RECEIPT_SCHEMA_VERSION,
        **dict(classification),
        "recorded_at_iso": _now_iso(),
    }
    payload["receipt_sha256"] = _sha256_json(payload)
    validated = _validate_terminal_receipt(
        payload,
        expected_intent=intent,
        campaign_dir=campaign_dir,
    )
    path = scheduler_terminal_receipt_path(
        campaign_dir,
        phase=identity["phase"],
        iteration=int(identity["iteration"]),
        replacement_round=int(identity["replacement_round"]),
        submission_identity=str(identity["submission_identity"]),
    )
    if path.exists() or path.is_symlink():
        existing = read_scheduler_terminal_receipt(
            path,
            expected_intent=intent,
            campaign_dir=campaign_dir,
        )
        comparable_existing = dict(existing)
        comparable_new = dict(validated)
        for item in (comparable_existing, comparable_new):
            item.pop("recorded_at_iso", None)
            item.pop("receipt_sha256", None)
        if comparable_existing != comparable_new:
            raise ValueError("scheduler terminal receipt already exists with different evidence")
        return existing
    if path.parent.is_symlink():
        raise ValueError("scheduler terminal receipt directory is a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, validated)
    return validated


def read_scheduler_terminal_receipt(
    path: Union[str, Path],
    *,
    expected_intent: Optional[Mapping[str, Any]] = None,
    campaign_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("scheduler terminal receipt is missing or not a regular file")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("scheduler terminal receipt is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("scheduler terminal receipt must be an object")
    return _validate_terminal_receipt(
        payload,
        expected_intent=expected_intent,
        campaign_dir=campaign_dir,
    )


def load_scheduler_terminal_receipt(
    campaign_dir: Union[str, Path],
    intent: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    path = scheduler_terminal_receipt_path(
        campaign_dir,
        phase=intent.get("phase"),
        iteration=_exact_nonnegative_int(
            intent.get("iteration"),
            "iteration",
        ),
        replacement_round=_exact_nonnegative_int(
            intent.get("replacement_round", 0),
            "replacement_round",
        ),
        submission_identity=_safe_identity(
            intent.get("submission_identity"),
            "submission_identity",
        ),
    )
    if not path.exists() and not path.is_symlink():
        return None
    _intent_identity(intent, allow_unaccepted=True)
    return read_scheduler_terminal_receipt(
        path,
        expected_intent=intent,
        campaign_dir=campaign_dir,
    )


def scheduler_terminal_recoveries(
    campaign_dir: Union[str, Path],
    *,
    campaign_uid: str,
    phase: Any,
    iteration: int,
    replacement_round: int,
) -> Tuple[Dict[str, Any], ...]:
    """Return authenticated terminal receipts and their producer intents."""
    from .submission_intent import intent_attempt_records

    phase_name = _phase_name(phase)
    records = []
    seen = set()
    for intent in intent_attempt_records(
        campaign_dir,
        phase_name,
        int(iteration),
        expected_campaign_uid=str(campaign_uid),
    ):
        if int(intent.get("replacement_round", 0)) != int(replacement_round):
            continue
        identity = str(intent.get("submission_identity") or "")
        if not identity or identity in seen:
            continue
        seen.add(identity)
        receipt = load_scheduler_terminal_receipt(campaign_dir, intent)
        if receipt is None:
            continue
        records.append(
            {
                "intent": dict(intent),
                "receipt": dict(receipt),
                "path": str(
                    scheduler_terminal_receipt_path(
                        campaign_dir,
                        phase=phase_name,
                        iteration=int(iteration),
                        replacement_round=int(replacement_round),
                        submission_identity=identity,
                    )
                ),
            }
        )
    records.sort(
        key=lambda item: (
            int(item["intent"].get("attempt_sequence", 0)),
            str(item["receipt"].get("submission_identity") or ""),
        )
    )
    return tuple(records)


def phase_recovery_ledger_path(
    campaign_dir: Union[str, Path],
    *,
    phase: Any,
    iteration: int,
    replacement_round: int,
) -> Path:
    phase_name = _phase_name(phase)
    filename = (
        phase_name
        + "-"
        + f"{_exact_nonnegative_int(iteration, 'iteration'):06d}"
        + "-r"
        + f"{_exact_nonnegative_int(replacement_round, 'replacement_round'):04d}"
        + ".json"
    )
    return _operational_root(campaign_dir) / PHASE_RECOVERY_LEDGER_DIRNAME / filename


def write_phase_recovery_ledger(
    campaign_dir: Union[str, Path],
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    data = dict(payload)
    data["schema_version"] = PHASE_RECOVERY_LEDGER_SCHEMA_VERSION
    data.setdefault("recorded_at_iso", _now_iso())
    data["phase"] = _phase_name(data.get("phase"))
    data["iteration"] = _exact_nonnegative_int(data.get("iteration"), "iteration")
    data["replacement_round"] = _exact_nonnegative_int(
        data.get("replacement_round", 0),
        "replacement_round",
    )
    _required_text(data.get("campaign_uid"), "campaign_uid")
    _required_text(data.get("source_receipt_sha256"), "source_receipt_sha256")
    reusable = data.get("reusable_logical_task_ids")
    retry = data.get("retry_logical_task_ids")
    if not isinstance(reusable, list) or not isinstance(retry, list):
        raise ValueError("phase recovery ledger task lists are invalid")
    reusable_ids = [
        _exact_nonnegative_int(value, "reusable logical task ID")
        for value in reusable
    ]
    retry_ids = [
        _exact_nonnegative_int(value, "retry logical task ID") for value in retry
    ]
    if set(reusable_ids).intersection(retry_ids):
        raise ValueError("phase recovery ledger task sets overlap")
    data["n_reusable"] = len(reusable_ids)
    data["n_retry"] = len(retry_ids)
    data["ledger_sha256"] = _sha256_json(data)
    data = _validate_phase_recovery_ledger(
        data,
        campaign_dir=campaign_dir,
    )
    path = phase_recovery_ledger_path(
        campaign_dir,
        phase=data["phase"],
        iteration=int(data["iteration"]),
        replacement_round=int(data["replacement_round"]),
    )
    if path.parent.is_symlink():
        raise ValueError("phase recovery ledger directory is a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        existing = read_phase_recovery_ledger(path)
        old_source = str(existing.get("source_receipt_sha256") or "")
        if old_source != str(data["source_receipt_sha256"]):
            old_receipts = {
                str(record.get("receipt_sha256") or "")
                for record in existing.get("source_terminal_receipts", [])
                if isinstance(record, Mapping)
                and str(record.get("receipt_sha256") or "")
            }
            new_receipts = {
                str(record.get("receipt_sha256") or "")
                for record in data.get("source_terminal_receipts", [])
                if isinstance(record, Mapping)
                and str(record.get("receipt_sha256") or "")
            }
            if not old_receipts or not old_receipts.issubset(new_receipts):
                raise ValueError(
                    "phase recovery ledger belongs to contradictory terminal evidence"
                )
    atomic_write_json(path, data)
    return data


def _validate_phase_recovery_ledger(
    payload: Mapping[str, Any],
    *,
    campaign_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    data = dict(payload)
    if data.get("schema_version") != PHASE_RECOVERY_LEDGER_SCHEMA_VERSION:
        raise ValueError("phase recovery ledger schema version is unsupported")
    _required_text(data.get("campaign_uid"), "campaign_uid")
    _phase_name(data.get("phase"))
    _exact_nonnegative_int(data.get("iteration"), "iteration")
    _exact_nonnegative_int(data.get("replacement_round"), "replacement_round")
    source_digest = _required_text(
        data.get("source_receipt_sha256"),
        "source_receipt_sha256",
    )
    if not _SHA256_RE.fullmatch(source_digest):
        raise ValueError("phase recovery source receipt digest is invalid")
    source_receipts = data.get("source_terminal_receipts")
    if not isinstance(source_receipts, list) or not source_receipts:
        raise ValueError("phase recovery terminal receipt list is invalid")
    seen_receipts = set()
    source_by_identity: Dict[str, Mapping[str, Any]] = {}
    for record in source_receipts:
        if not isinstance(record, Mapping):
            raise ValueError("phase recovery terminal receipt record is invalid")
        receipt_digest = _required_text(
            record.get("receipt_sha256"),
            "terminal receipt SHA-256",
        )
        if not _SHA256_RE.fullmatch(receipt_digest):
            raise ValueError("phase recovery terminal receipt digest is invalid")
        if receipt_digest in seen_receipts:
            raise ValueError("phase recovery terminal receipts contain duplicates")
        seen_receipts.add(receipt_digest)
        _required_text(record.get("path"), "terminal receipt path")
        submission_identity = _safe_identity(
            record.get("submission_identity"),
            "terminal receipt submission identity",
        )
        if submission_identity in source_by_identity:
            raise ValueError(
                "phase recovery terminal receipt identities contain duplicates"
            )
        source_by_identity[submission_identity] = record
        job_id = record.get("job_id")
        if not isinstance(job_id, str):
            raise ValueError("phase recovery terminal receipt JobID is invalid")
    if source_digest != _source_receipt_digest(source_receipts):
        raise ValueError("phase recovery source receipt digest mismatch")

    equivalences = data.get("environment_equivalences")
    if not isinstance(equivalences, list):
        raise ValueError("phase recovery environment equivalences are invalid")
    equivalence_by_identity: Dict[str, Mapping[str, Any]] = {}
    for record in equivalences:
        if not isinstance(record, Mapping):
            raise ValueError(
                "phase recovery environment equivalence record is invalid"
            )
        submission_identity = _safe_identity(
            record.get("submission_identity"),
            "environment equivalence submission identity",
        )
        if (
            submission_identity in equivalence_by_identity
            or submission_identity not in source_by_identity
        ):
            raise ValueError(
                "phase recovery environment equivalence identity is invalid"
            )
        if not isinstance(record.get("equivalent"), bool):
            raise ValueError(
                "phase recovery environment equivalence verdict is invalid"
            )
        reasons = record.get("reasons")
        if not isinstance(reasons, list) or any(
            not isinstance(reason, str) or not reason for reason in reasons
        ):
            raise ValueError(
                "phase recovery environment equivalence reasons are invalid"
            )
        proof_path = record.get("path")
        proof_sha256 = record.get("sha256")
        if (proof_path is None) is not (proof_sha256 is None):
            raise ValueError(
                "phase recovery environment equivalence binding is incomplete"
            )
        if proof_path is not None and (
            not isinstance(proof_path, str)
            or not proof_path
            or not isinstance(proof_sha256, str)
            or not _SHA256_RE.fullmatch(proof_sha256)
        ):
            raise ValueError(
                "phase recovery environment equivalence binding is invalid"
            )
        if record["equivalent"] is True and proof_path is None:
            raise ValueError(
                "reusable environment equivalence has no proof binding"
            )
        equivalence_by_identity[submission_identity] = record
    if set(equivalence_by_identity) != set(source_by_identity):
        raise ValueError(
            "phase recovery environment equivalences do not cover every "
            "terminal receipt"
        )

    reusable = data.get("reusable_logical_task_ids")
    retry = data.get("retry_logical_task_ids")
    if not isinstance(reusable, list) or not isinstance(retry, list):
        raise ValueError("phase recovery ledger task lists are invalid")
    reusable_ids = [
        _exact_nonnegative_int(value, "reusable logical task ID")
        for value in reusable
    ]
    retry_ids = [
        _exact_nonnegative_int(value, "retry logical task ID") for value in retry
    ]
    if reusable_ids != sorted(set(reusable_ids)):
        raise ValueError("phase recovery reusable task IDs are invalid")
    if retry_ids != sorted(set(retry_ids)):
        raise ValueError("phase recovery retry task IDs are invalid")
    if set(reusable_ids).intersection(retry_ids):
        raise ValueError("phase recovery ledger task sets overlap")
    all_ids = sorted(reusable_ids + retry_ids)
    if not all_ids or all_ids != list(range(len(all_ids))):
        raise ValueError(
            "phase recovery logical task coverage must be contiguous from zero"
        )
    if (
        data.get("n_reusable") != len(reusable_ids)
        or data.get("n_retry") != len(retry_ids)
    ):
        raise ValueError("phase recovery ledger task counts are invalid")

    lineage = data.get("recovery_lineage", [])
    if not isinstance(lineage, list):
        raise ValueError("phase recovery lineage is invalid")
    lineage_ids = set()
    for record in lineage:
        if not isinstance(record, Mapping):
            raise ValueError("phase recovery lineage record is invalid")
        task_id = _exact_nonnegative_int(
            record.get("logical_task_id"),
            "lineage logical task ID",
        )
        if task_id in lineage_ids or task_id not in set(reusable_ids):
            raise ValueError("phase recovery lineage task identity is invalid")
        lineage_ids.add(task_id)
        reuse_basis = _required_text(
            record.get("reuse_basis"),
            "lineage reuse basis",
        )
        if reuse_basis not in _REUSE_BASES_BY_PHASE.get(
            str(data["phase"]),
            frozenset(),
        ):
            raise ValueError(
                "phase recovery lineage reuse basis is invalid for its phase"
            )
        _safe_identity(record.get("attempt_id"), "lineage attempt ID")
        _safe_identity(
            record.get("submission_identity"),
            "lineage submission identity",
        )
        submission_identity = str(record["submission_identity"])
        source_record = source_by_identity.get(submission_identity)
        equivalence_record = equivalence_by_identity.get(submission_identity)
        if (
            not isinstance(source_record, Mapping)
            or not isinstance(equivalence_record, Mapping)
            or equivalence_record.get("equivalent") is not True
            or str(record.get("job_id") or "")
            != str(source_record.get("job_id") or "")
            or record.get("environment_equivalence_path")
            != equivalence_record.get("path")
            or record.get("environment_equivalence_sha256")
            != equivalence_record.get("sha256")
        ):
            raise ValueError(
                "phase recovery lineage producer binding is invalid"
            )
        _exact_nonnegative_int(
            record.get("environment_generation"),
            "lineage environment generation",
        )
        environment_digest = _required_text(
            record.get("environment_generation_digest_sha256"),
            "lineage environment digest",
        )
        if not _SHA256_RE.fullmatch(environment_digest):
            raise ValueError("phase recovery lineage environment digest is invalid")
    if lineage_ids != set(reusable_ids):
        raise ValueError("phase recovery lineage does not cover every reused task")

    if campaign_dir is not None:
        campaign = Path(campaign_dir).expanduser().resolve()
        from .submission_intent import intent_attempt_records

        intent_by_identity: Dict[str, Mapping[str, Any]] = {}
        for intent in intent_attempt_records(
            campaign,
            str(data["phase"]),
            int(data["iteration"]),
            expected_campaign_uid=str(data["campaign_uid"]),
        ):
            if int(intent.get("replacement_round", 0)) != int(
                data["replacement_round"]
            ):
                continue
            submission_identity = str(
                intent.get("submission_identity") or ""
            )
            if (
                not submission_identity
                or submission_identity in intent_by_identity
            ):
                raise ValueError(
                    "phase recovery submission-intent history is ambiguous"
                )
            intent_by_identity[submission_identity] = intent
        source_sequences = []
        for submission_identity in source_by_identity:
            intent = intent_by_identity.get(submission_identity)
            if not isinstance(intent, Mapping):
                raise ValueError(
                    "phase recovery terminal receipt has no canonical "
                    "submission intent"
                )
            source_sequences.append(
                _exact_positive_int(
                    intent.get("attempt_sequence"),
                    "submission intent attempt sequence",
                )
            )
        if source_sequences != sorted(source_sequences) or len(
            set(source_sequences)
        ) != len(source_sequences):
            raise ValueError(
                "phase recovery terminal receipts are not in canonical "
                "attempt order"
            )

        terminal_receipts_by_identity: Dict[str, Mapping[str, Any]] = {}
        latest_source_by_task: Dict[int, str] = {}
        for submission_identity, record in source_by_identity.items():
            expected_path = scheduler_terminal_receipt_path(
                campaign,
                phase=data["phase"],
                iteration=int(data["iteration"]),
                replacement_round=int(data["replacement_round"]),
                submission_identity=submission_identity,
            )
            recorded_path = Path(str(record["path"])).expanduser()
            if recorded_path.resolve(strict=False) != expected_path.resolve(
                strict=False
            ):
                raise ValueError(
                    "phase recovery terminal receipt path is not canonical"
                )
            receipt = read_scheduler_terminal_receipt(
                expected_path,
                expected_intent=intent_by_identity[submission_identity],
                campaign_dir=campaign,
            )
            if (
                str(receipt.get("receipt_sha256") or "")
                != str(record["receipt_sha256"])
                or str(receipt.get("campaign_uid") or "")
                != str(data["campaign_uid"])
                or str(receipt.get("phase") or "") != str(data["phase"])
                or int(receipt.get("iteration", -1))
                != int(data["iteration"])
                or int(receipt.get("replacement_round", -1))
                != int(data["replacement_round"])
                or str(receipt.get("submission_identity") or "")
                != submission_identity
                or str(receipt.get("job_id") or "")
                != str(record.get("job_id") or "")
            ):
                raise ValueError(
                    "phase recovery terminal receipt authority mismatch"
                )
            terminal_receipts_by_identity[submission_identity] = receipt
            for outcome in receipt.get("outcomes", []):
                latest_source_by_task[
                    int(outcome["logical_task_id"])
                ] = submission_identity
        from .environment_equivalence import (
            environment_equivalence_path,
            read_environment_equivalence,
        )

        for submission_identity, record in equivalence_by_identity.items():
            if record.get("path") is None:
                continue
            proof_path = Path(str(record["path"])).expanduser()
            proof = read_environment_equivalence(proof_path)
            expected_proof_path = environment_equivalence_path(
                campaign,
                phase=data["phase"],
                iteration=int(data["iteration"]),
                replacement_round=int(data["replacement_round"]),
                submission_identity=submission_identity,
                current_generation=int(proof["current_generation"]),
            )
            if (
                proof_path.resolve(strict=False)
                != expected_proof_path.resolve(strict=False)
                or
                str(proof.get("proof_sha256") or "")
                != str(record["sha256"])
                or str(proof.get("campaign_uid") or "")
                != str(data["campaign_uid"])
                or str(proof.get("phase") or "") != str(data["phase"])
                or int(proof.get("iteration", -1))
                != int(data["iteration"])
                or int(proof.get("replacement_round", -1))
                != int(data["replacement_round"])
                or str(proof.get("submission_identity") or "")
                != submission_identity
                or str(proof.get("job_id") or "")
                != str(
                    source_by_identity[submission_identity].get("job_id")
                    or ""
                )
                or bool(proof.get("equivalent", False))
                is not bool(record["equivalent"])
            ):
                raise ValueError(
                    "phase recovery environment equivalence authority mismatch"
                )
            for lineage_record in lineage:
                if (
                    str(lineage_record.get("submission_identity") or "")
                    != submission_identity
                ):
                    continue
                if (
                    str(proof.get("attempt_id") or "")
                    != str(lineage_record.get("attempt_id") or "")
                    or
                    int(proof.get("producer_generation", -1))
                    != int(lineage_record["environment_generation"])
                    or str(
                        proof.get(
                            "producer_generation_digest_sha256"
                        )
                        or ""
                    )
                    != str(
                        lineage_record[
                            "environment_generation_digest_sha256"
                        ]
                    )
                ):
                    raise ValueError(
                        "phase recovery lineage environment authority mismatch"
                    )
        for lineage_record in lineage:
            task_id = int(lineage_record["logical_task_id"])
            submission_identity = str(
                lineage_record["submission_identity"]
            )
            receipt = terminal_receipts_by_identity[submission_identity]
            if latest_source_by_task.get(task_id) != submission_identity:
                raise ValueError(
                    "phase recovery lineage is not the latest scheduler "
                    "producer for its task"
                )
            if (
                str(lineage_record["reuse_basis"])
                in _SCHEDULER_COMPLETION_REUSE_BASES
                and task_id
                not in {
                    int(value)
                    for value in receipt.get(
                        "completed_logical_task_ids",
                        [],
                    )
                }
            ):
                raise ValueError(
                    "phase recovery lineage lacks a successful scheduler "
                    "outcome required by its reuse basis"
                )

    digest = data.get("ledger_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError("phase recovery ledger digest is invalid")
    unsigned = dict(data)
    unsigned.pop("ledger_sha256", None)
    if _sha256_json(unsigned) != digest:
        raise ValueError("phase recovery ledger digest mismatch")
    return data


def read_phase_recovery_ledger(path: Union[str, Path]) -> Dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("phase recovery ledger is missing or not a regular file")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("phase recovery ledger is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("phase recovery ledger must be an object")
    try:
        if (
            candidate.parent.name != PHASE_RECOVERY_LEDGER_DIRNAME
            or candidate.parent.parent.name != "ACTIVE_LEARNING"
            or candidate.parent.parent.parent.name != ".DATA"
        ):
            raise ValueError(
                "phase recovery ledger is outside its canonical directory"
            )
        campaign = candidate.parents[3]
    except IndexError as exc:
        raise ValueError(
            "phase recovery ledger is outside its canonical directory"
        ) from exc
    return _validate_phase_recovery_ledger(
        payload,
        campaign_dir=campaign,
    )


__all__ = [
    "PHASE_RECOVERY_LEDGER_DIRNAME",
    "PHASE_RECOVERY_LEDGER_SCHEMA_VERSION",
    "SCHEDULER_TERMINAL_RECEIPT_DIRNAME",
    "SCHEDULER_TERMINAL_RECEIPT_SCHEMA_VERSION",
    "classify_terminal_scheduler_evidence",
    "classify_unaccepted_scheduler_intent",
    "load_scheduler_terminal_receipt",
    "phase_recovery_ledger_path",
    "read_phase_recovery_ledger",
    "read_scheduler_terminal_receipt",
    "scheduler_terminal_receipt_path",
    "scheduler_terminal_recoveries",
    "validate_scheduler_terminal_receipt",
    "write_phase_recovery_ledger",
    "write_scheduler_terminal_receipt",
]
