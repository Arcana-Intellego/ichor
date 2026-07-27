"""Immutable logical-task membership for Gaussian and AIMAll phases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from ..layout import (
    active_iteration_dir,
    bootstrap_selection_dir,
    parse_staging_pointdir_name,
    staging_phase_dir,
    staging_pointdir_name,
)


GAUSSIAN_PHASES = frozenset(
    {
        "INITIAL_GAUSSIAN",
        "GAUSSIAN",
        "INITIAL_REPLACEMENT_GAUSSIAN",
        "REPLACEMENT_GAUSSIAN",
    }
)
AIMALL_PHASES = frozenset(
    {
        "INITIAL_AIMALL",
        "AIMALL",
        "INITIAL_REPLACEMENT_AIMALL",
        "REPLACEMENT_AIMALL",
    }
)
QUANTUM_PHASES = GAUSSIAN_PHASES | AIMALL_PHASES


@dataclass(frozen=True)
class QuantumLogicalTask:
    logical_task_id: int
    pointdir_name: str
    pointdir: Path
    producer_logical_task_id: int
    candidate_id: str


@dataclass(frozen=True)
class QuantumTaskContract:
    campaign_uid: str
    phase: str
    iteration: int
    replacement_round: int
    staging_dir: Path
    tasks: Tuple[QuantumLogicalTask, ...]

    @property
    def pointdir_names(self) -> Tuple[str, ...]:
        return tuple(task.pointdir_name for task in self.tasks)

    @property
    def logical_total(self) -> int:
        return len(self.tasks)


def gaussian_phase_for_aimall(phase_name: str) -> str:
    mapping = {
        "INITIAL_AIMALL": "INITIAL_GAUSSIAN",
        "AIMALL": "GAUSSIAN",
        "INITIAL_REPLACEMENT_AIMALL": "INITIAL_REPLACEMENT_GAUSSIAN",
        "REPLACEMENT_AIMALL": "REPLACEMENT_GAUSSIAN",
    }
    try:
        return mapping[str(phase_name)]
    except KeyError as exc:
        raise ValueError("phase is not an AIMAll phase: " + str(phase_name)) from exc


def aimall_phase_for_gaussian(phase_name: str) -> str:
    mapping = {
        "INITIAL_GAUSSIAN": "INITIAL_AIMALL",
        "GAUSSIAN": "AIMALL",
        "INITIAL_REPLACEMENT_GAUSSIAN": "INITIAL_REPLACEMENT_AIMALL",
        "REPLACEMENT_GAUSSIAN": "REPLACEMENT_AIMALL",
    }
    try:
        return mapping[str(phase_name)]
    except KeyError as exc:
        raise ValueError("phase is not a Gaussian phase: " + str(phase_name)) from exc


def _campaign_uid_from_state(campaign_dir: Path) -> str:
    from .filesystem import operational_path
    from .state import read_state

    state = read_state(operational_path(campaign_dir, "state.json"))
    value = str(state.campaign_uid).strip()
    if not value:
        raise ValueError("campaign state has no campaign UID")
    return value


def _validate_expected_uid(observed: str, expected: Optional[str]) -> str:
    value = str(observed).strip()
    if not value:
        raise ValueError("quantum task producer has no campaign UID")
    if expected is not None and value != str(expected):
        raise ValueError("quantum task producer campaign UID mismatch")
    return value


def _lexical_points_names(points_file: Path, staging: Path) -> Tuple[str, ...]:
    from .input_staging import _validate_pointdir_basename

    if points_file.is_symlink() or not points_file.is_file():
        raise FileNotFoundError("POINTS.txt is missing or symlinked: " + str(points_file))
    names = []
    seen = set()
    for line_number, raw in enumerate(
        points_file.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        text = raw.strip()
        if not text:
            continue
        candidate = Path(text)
        name = _validate_pointdir_basename(candidate.name)
        expected = staging / name
        if candidate.resolve(strict=False) != expected.resolve(strict=False):
            raise ValueError(
                "POINTS.txt path escapes its staging bucket at line "
                + str(line_number)
            )
        if name in seen:
            raise ValueError("POINTS.txt contains a duplicate pointdir: " + name)
        seen.add(name)
        names.append(name)
    return tuple(names)


def _gaussian_source(
    campaign: Path,
    phase: str,
    iteration: int,
    replacement_round: int,
    expected_campaign_uid: Optional[str],
    staging_override: Optional[Path],
) -> Tuple[str, Path, Tuple[str, ...], Tuple[str, ...]]:
    if "REPLACEMENT" in phase:
        from ..point_allocation import read_point_allocation
        from ..replacement_sampling import (
            read_replacement_sample_strict,
            replacement_round_dir,
        )

        context = "bootstrap" if phase.startswith("INITIAL_") else "active"
        allocation_iteration = 0 if context == "bootstrap" else int(iteration)
        sample = read_replacement_sample_strict(
            campaign,
            context=context,
            iteration=allocation_iteration,
            replacement_round=int(replacement_round),
            expected_campaign_uid=expected_campaign_uid,
        )
        allocation = read_point_allocation(
            sample["point_allocation_manifest"],
            expected_campaign_uid=expected_campaign_uid,
            expected_context=context,
            expected_iteration=allocation_iteration,
        )
        names = tuple(
            staging_pointdir_name(int(record["pointdir_index"]))
            for record in sample["records"]
        )
        candidate_ids = tuple(
            str(record["candidate_id"]) for record in sample["records"]
        )
        staging = replacement_round_dir(
            campaign,
            context=context,
            iteration=allocation_iteration,
            replacement_round=int(replacement_round),
        )
        return str(allocation["campaign_uid"]), staging, names, candidate_ids

    expected_uid = (
        str(expected_campaign_uid)
        if expected_campaign_uid is not None
        else _campaign_uid_from_state(campaign)
    )
    if phase == "INITIAL_GAUSSIAN":
        from ..handoff_manifests import read_phase_a_sample_manifest

        try:
            manifest = read_phase_a_sample_manifest(
                bootstrap_selection_dir(campaign),
                expected_campaign_uid=expected_uid,
            )
        except FileNotFoundError:
            if staging_override is None:
                raise
            return _allocation_embedded_gaussian_source(
                campaign,
                phase=phase,
                iteration=int(iteration),
                expected_campaign_uid=expected_uid,
                staging_override=staging_override,
            )
        records = list(manifest["point_allocation"]["primary"])
    elif phase == "GAUSSIAN":
        from ..handoff_manifests import read_phase_b_selection_manifest

        try:
            manifest = read_phase_b_selection_manifest(
                active_iteration_dir(campaign, int(iteration)),
                expected_iteration=int(iteration),
                expected_campaign_uid=expected_uid,
            )
        except FileNotFoundError:
            if staging_override is None:
                raise
            return _allocation_embedded_gaussian_source(
                campaign,
                phase=phase,
                iteration=int(iteration),
                expected_campaign_uid=expected_uid,
                staging_override=staging_override,
            )
        records = list(manifest["final"])
    else:
        raise ValueError("phase is not a Gaussian phase: " + phase)
    names = tuple(staging_pointdir_name(index) for index in range(len(records)))
    candidate_ids = tuple(str(record["candidate_id"]) for record in records)
    return (
        expected_uid,
        staging_phase_dir(campaign, phase, int(iteration)),
        names,
        candidate_ids,
    )


def _allocation_embedded_gaussian_source(
    campaign: Path,
    *,
    phase: str,
    iteration: int,
    expected_campaign_uid: str,
    staging_override: Optional[Path],
) -> Tuple[str, Path, Tuple[str, ...], Tuple[str, ...]]:
    """Read legacy explicit task identities from an authoritative allocation."""
    from ..point_allocation import point_allocation_path, read_point_allocation

    context = "bootstrap" if phase == "INITIAL_GAUSSIAN" else "active"
    allocation_iteration = 0 if context == "bootstrap" else int(iteration)
    allocation = read_point_allocation(
        point_allocation_path(
            campaign,
            context=context,
            iteration=allocation_iteration,
        ),
        expected_campaign_uid=expected_campaign_uid,
        expected_context=context,
        expected_iteration=allocation_iteration,
    )
    records = []
    for slot in allocation["slots"]:
        attempts = [
            attempt
            for attempt in list(slot.get("attempts") or [])
            if int(attempt.get("round", -1)) == 0
        ]
        if len(attempts) != 1:
            raise ValueError(
                "primary point allocation does not contain one initial task per slot"
            )
        attempt = attempts[0]
        name = str(attempt.get("pointdir_name") or "")
        try:
            pointdir_index = parse_staging_pointdir_name(name)
        except (TypeError, ValueError) as exc:
            raise FileNotFoundError(
                phase + " producer handoff is missing"
            ) from exc
        records.append(
            (
                int(pointdir_index),
                name,
                str(attempt.get("candidate_id") or ""),
            )
        )
    records.sort(key=lambda record: record[0])
    expected_indexes = list(range(len(records)))
    if (
        [record[0] for record in records] != expected_indexes
        or len({record[1] for record in records}) != len(records)
        or len({record[2] for record in records}) != len(records)
        or any(not record[2] for record in records)
    ):
        raise ValueError(
            "point allocation embedded Gaussian task identities are incomplete"
        )
    return (
        str(allocation["campaign_uid"]),
        (
            Path(staging_override)
            if staging_override is not None
            else staging_phase_dir(campaign, phase, int(iteration))
        ),
        tuple(record[1] for record in records),
        tuple(record[2] for record in records),
    )


def quantum_task_contract(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
    *,
    replacement_round: int = 0,
    expected_campaign_uid: Optional[str] = None,
    validate_points_file: bool = True,
    points_file_optional: bool = False,
    staging_override: Optional[Union[str, Path]] = None,
) -> QuantumTaskContract:
    """Resolve one phase's logical tasks without requiring task directories."""
    campaign = Path(campaign_dir)
    phase = str(getattr(phase_name, "value", phase_name))
    if phase not in QUANTUM_PHASES:
        raise ValueError("phase is not a quantum array phase: " + phase)
    if phase in GAUSSIAN_PHASES:
        uid, staging, names, candidate_ids = _gaussian_source(
            campaign,
            phase,
            int(iteration),
            int(replacement_round),
            expected_campaign_uid,
            None if staging_override is None else Path(staging_override),
        )
        producer_ids = tuple(range(len(names)))
    else:
        gaussian_phase = gaussian_phase_for_aimall(phase)
        gaussian = quantum_task_contract(
            campaign,
            gaussian_phase,
            int(iteration),
            replacement_round=int(replacement_round),
            expected_campaign_uid=expected_campaign_uid,
            validate_points_file=False,
            staging_override=staging_override,
        )
        from .input_staging import (
            POINTS_MEMBERSHIP_NONE,
            read_quantum_acceptance_manifest,
        )

        _accepted_paths, manifest = read_quantum_acceptance_manifest(
            gaussian.staging_dir,
            expected_phase=gaussian_phase,
            expected_iteration=int(iteration),
            require_nonempty=False,
            points_membership=POINTS_MEMBERSHIP_NONE,
            require_accepted_payloads=False,
        )
        accepted_names = tuple(str(name) for name in manifest["accepted_pointdirs"])
        disposition_names = set(accepted_names)
        disposition_names.update(
            str(record["pointdir"]) for record in manifest["rejected"]
        )
        gaussian_names = gaussian.pointdir_names
        if (
            int(manifest["n_total"]) != len(gaussian_names)
            or disposition_names != set(gaussian_names)
        ):
            raise ValueError(
                "Gaussian acceptance does not exactly cover its producer task set"
            )
        accepted_set = set(accepted_names)
        expected_accepted = tuple(
            name for name in gaussian_names if name in accepted_set
        )
        if accepted_names != expected_accepted:
            raise ValueError(
                "Gaussian accepted pointdirs are not in producer task order"
            )
        uid = gaussian.campaign_uid
        staging = gaussian.staging_dir
        names = accepted_names
        gaussian_id_by_name = {
            task.pointdir_name: task.logical_task_id for task in gaussian.tasks
        }
        candidate_id_by_name = {
            task.pointdir_name: task.candidate_id for task in gaussian.tasks
        }
        producer_ids = tuple(gaussian_id_by_name[name] for name in names)
        candidate_ids = tuple(candidate_id_by_name[name] for name in names)
    uid = _validate_expected_uid(uid, expected_campaign_uid)
    points_file = staging / "POINTS.txt"
    if bool(validate_points_file):
        if points_file_optional and not points_file.exists():
            pass
        else:
            listed = _lexical_points_names(points_file, staging)
            if listed != names:
                raise ValueError(
                    phase + " POINTS.txt does not match authoritative task order"
                )
    tasks = tuple(
        QuantumLogicalTask(
            logical_task_id=index,
            pointdir_name=name,
            pointdir=staging / name,
            producer_logical_task_id=int(producer_ids[index]),
            candidate_id=str(candidate_ids[index]),
        )
        for index, name in enumerate(names)
    )
    return QuantumTaskContract(
        campaign_uid=uid,
        phase=phase,
        iteration=int(iteration),
        replacement_round=int(replacement_round),
        staging_dir=staging,
        tasks=tasks,
    )


__all__ = [
    "AIMALL_PHASES",
    "GAUSSIAN_PHASES",
    "QUANTUM_PHASES",
    "QuantumLogicalTask",
    "QuantumTaskContract",
    "aimall_phase_for_gaussian",
    "gaussian_phase_for_aimall",
    "quantum_task_contract",
]
