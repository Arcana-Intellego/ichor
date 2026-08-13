"""Partial recovery for daemon-owned Slurm array phases.

The daemon can safely reuse completed task outputs for phases whose array
tasks are independent.  This module keeps that logic outside the FSM: it
discovers the logical task set, validates existing task outputs, writes a
daemon-owned ledger, and emits dense retry task maps for Slurm.
"""
from __future__ import annotations

import hashlib
from ..strict_json import strict_json as json
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from .state import atomic_write_json, atomic_write_text
from ..layout import staging_phase_dir
from ..versioning.manifest import sha256_file


ARRAY_RECOVERY_SCHEMA_VERSION = 2
ARRAY_RECOVERY_DIR_NAME = "array_task_ledgers"
RETRY_TASKS_FILENAME_PREFIX = "RETRY_TASKS"

PARTIAL_RECOVERY_PHASES = frozenset(
    {
        "INITIAL_GAUSSIAN",
        "INITIAL_AIMALL",
        "INITIAL_REPLACEMENT_GAUSSIAN",
        "INITIAL_REPLACEMENT_AIMALL",
        "GAUSSIAN",
        "AIMALL",
        "REPLACEMENT_GAUSSIAN",
        "REPLACEMENT_AIMALL",
        "ARIADNE_ARRAY",
    }
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def supports_partial_array_recovery(phase_name: Any) -> bool:
    return str(getattr(phase_name, "value", phase_name)) in PARTIAL_RECOVERY_PHASES


def array_recovery_dir(campaign_dir: Union[str, Path]) -> Path:
    from .filesystem import operational_path

    return operational_path(campaign_dir, ARRAY_RECOVERY_DIR_NAME)


def array_ledger_path(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
    *,
    replacement_round: Optional[int] = None,
) -> Path:
    phase = str(getattr(phase_name, "value", phase_name))
    safe_phase = phase.replace("/", "_").replace("\\", "_")
    round_suffix = ""
    if "REPLACEMENT" in phase:
        resolved_round = (
            int(replacement_round)
            if replacement_round is not None
            else _replacement_identity(
                campaign_dir,
                phase,
                int(iteration),
            )[0]
        )
        round_suffix = "-r" + str(resolved_round).zfill(4)
    return array_recovery_dir(campaign_dir) / (
        safe_phase + "-" + str(int(iteration)).zfill(6) + round_suffix + ".json"
    )


def retry_task_file_path(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
    *,
    replacement_round: Optional[int] = None,
) -> Path:
    phase = str(getattr(phase_name, "value", phase_name))
    safe_phase = phase.replace("/", "_").replace("\\", "_")
    round_suffix = ""
    if "REPLACEMENT" in phase:
        resolved_round = (
            int(replacement_round)
            if replacement_round is not None
            else _replacement_identity(
                campaign_dir,
                phase,
                int(iteration),
            )[0]
        )
        round_suffix = ".r" + str(resolved_round).zfill(4)
    from .filesystem import operational_path

    return operational_path(
        campaign_dir,
        RETRY_TASKS_FILENAME_PREFIX
        + "."
        + safe_phase
        + "."
        + str(int(iteration)).zfill(6)
        + round_suffix
        + ".txt",
    )


def _sha_payload(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _replacement_identity(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
) -> Tuple[int, Path]:
    from ..point_allocation import (
        pending_attempts,
        point_allocation_path,
        read_point_allocation,
    )
    from ..replacement_sampling import (
        read_replacement_sample_strict,
        replacement_round_dir,
    )
    from .filesystem import operational_path
    from .state import read_state

    context = "bootstrap" if str(phase_name).startswith("INITIAL_") else "active"
    allocation_iteration = 0 if context == "bootstrap" else int(iteration)
    state = read_state(operational_path(campaign_dir, "state.json"))
    allocation = read_point_allocation(
        point_allocation_path(
            campaign_dir,
            context=context,
            iteration=allocation_iteration,
        ),
        expected_campaign_uid=str(state.campaign_uid),
        expected_context=context,
        expected_iteration=allocation_iteration,
    )
    rounds = {
        int(attempt.get("round", -1)) for attempt in pending_attempts(allocation)
    }
    if len(rounds) != 1 or next(iter(rounds)) <= 0:
        raise ValueError("replacement recovery cannot resolve one pending round")
    replacement_round = next(iter(rounds))
    read_replacement_sample_strict(
        campaign_dir,
        context=context,
        iteration=allocation_iteration,
        replacement_round=replacement_round,
        expected_campaign_uid=str(state.campaign_uid),
    )
    return replacement_round, replacement_round_dir(
        campaign_dir,
        context=context,
        iteration=allocation_iteration,
        replacement_round=replacement_round,
    )


def _bucket_dir(campaign_dir: Union[str, Path], phase_name: str, iteration: int) -> Path:
    if "REPLACEMENT" in str(phase_name):
        _replacement_round, round_dir = _replacement_identity(
            campaign_dir,
            phase_name,
            int(iteration),
        )
        return round_dir
    return staging_phase_dir(campaign_dir, phase_name, int(iteration))


def _points_file(campaign_dir: Union[str, Path], phase_name: str, iteration: int) -> Path:
    return _bucket_dir(campaign_dir, phase_name, iteration) / "POINTS.txt"


def _read_point_paths(points_file: Path) -> List[Path]:
    if not Path(points_file).is_file():
        return []
    bucket = Path(points_file).parent
    if Path(points_file).is_symlink():
        raise ValueError("POINTS.txt must not be a symlink")
    out: List[Path] = []
    seen = set()
    for line_number, raw in enumerate(Path(points_file).read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if text:
            candidate = Path(text)
            expected = bucket / candidate.name
            if candidate.resolve(strict=False) != expected.resolve(strict=False):
                raise ValueError("POINTS.txt path escapes its bucket at line " + str(line_number))
            if expected.name in seen:
                raise ValueError("POINTS.txt contains a duplicate pointdir")
            seen.add(expected.name)
            out.append(expected)
    return out


@dataclass(frozen=True)
class _PathAuthority:
    path: Path
    is_directory: bool
    size: int
    mtime_ns: int
    device: int
    inode: int
    sha256: Optional[str]

    def assert_unchanged(self) -> None:
        path = Path(self.path)
        if path.is_symlink():
            raise ValueError(
                "array recovery authority became symlinked: " + str(path)
            )
        if self.is_directory:
            if not path.is_dir():
                raise ValueError(
                    "array recovery output directory changed type: " + str(path)
                )
        elif not path.is_file():
            raise ValueError(
                "array recovery control file changed type: " + str(path)
            )
        before = path.stat()
        observed_sha256 = sha256_file(path) if not self.is_directory else None
        after = path.stat()
        observed = (
            int(after.st_size),
            int(after.st_mtime_ns),
            int(after.st_dev),
            int(after.st_ino),
        )
        expected = (
            int(self.size),
            int(self.mtime_ns),
            int(self.device),
            int(self.inode),
        )
        before_identity = (
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_dev),
            int(before.st_ino),
        )
        if (
            before_identity != expected
            or observed != expected
            or observed_sha256 != self.sha256
        ):
            raise ValueError(
                "array recovery authority changed during validation: "
                + str(path)
            )


def _capture_path_authority(path: Path) -> _PathAuthority:
    source = Path(path)
    if source.is_symlink() or not (source.is_file() or source.is_dir()):
        raise ValueError(
            "array recovery authority is missing or unsafe: " + str(source)
        )
    before = source.stat()
    digest = sha256_file(source) if source.is_file() else None
    after = source.stat()
    before_identity = (
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_dev),
        int(before.st_ino),
    )
    after_identity = (
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_dev),
        int(after.st_ino),
    )
    if before_identity != after_identity:
        raise ValueError(
            "array recovery authority changed while it was captured: "
            + str(source)
        )
    return _PathAuthority(
        path=source.resolve(),
        is_directory=bool(source.is_dir()),
        size=int(after.st_size),
        mtime_ns=int(after.st_mtime_ns),
        device=int(after.st_dev),
        inode=int(after.st_ino),
        sha256=digest,
    )


def _existing_control_files(directory: Path) -> List[Path]:
    root = Path(directory)
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise ValueError(
            "array recovery control directory is unsafe: " + str(root)
        )
    controls = []
    for child in sorted(root.iterdir(), key=lambda value: value.name):
        if child.is_dir() and not child.is_symlink():
            continue
        if child.suffix.lower() in {".json", ".txt", ".yaml", ".yml"}:
            controls.append(child)
    return controls


@dataclass(frozen=True)
class ArrayRecoveryScanContext:
    campaign_dir: Path
    phase: str
    iteration: int
    replacement_round: int
    campaign_uid: str
    task_ids: Tuple[int, ...]
    task_by_id: Mapping[int, Any] = field(repr=False)
    quantum_contract: Optional[Any] = field(default=None, repr=False)
    ariadne_task_map: Optional[Mapping[str, Any]] = field(
        default=None,
        repr=False,
    )
    gaussian_rejected_names: FrozenSet[str] = frozenset()
    gaussian_acceptance_error: Optional[str] = None
    producer_index: Mapping[Tuple[str, str, str], Tuple[Mapping[str, Any], FrozenSet[int]]] = field(
        default_factory=dict,
        repr=False,
    )
    control_authorities: Tuple[_PathAuthority, ...] = field(
        default_factory=tuple,
        repr=False,
    )

    def resolve_receipt_producer(
        self,
        receipt: Mapping[str, Any],
        logical_task_id: int,
    ) -> Mapping[str, Any]:
        expected_identity = (
            str(self.campaign_uid),
            str(self.phase),
            int(self.iteration),
            int(logical_task_id),
        )
        observed_identity = (
            str(receipt.get("campaign_uid") or ""),
            str(receipt.get("phase") or ""),
            receipt.get("iteration"),
            receipt.get("logical_task_id"),
        )
        if observed_identity != expected_identity:
            raise ValueError("quantum task receipt phase identity mismatch")
        key = (
            str(receipt.get("attempt_id") or ""),
            str(receipt.get("submission_identity") or ""),
            str(receipt.get("job_id") or ""),
        )
        match = self.producer_index.get(key)
        if match is None:
            raise ValueError(
                "quantum task receipt does not resolve to one scheduler producer"
            )
        producer, submitted_task_ids = match
        if str(producer.get("status") or "") == "PRE_SUBMIT":
            raise ValueError("quantum task receipt producer was never submitted")
        if producer.get("submission_kind") != "array":
            raise ValueError("quantum task receipt producer is not an array")
        if int(producer.get("replacement_round", -1)) != int(
            self.replacement_round
        ):
            raise ValueError("quantum task receipt replacement round mismatch")
        if int(logical_task_id) not in submitted_task_ids:
            raise ValueError(
                "quantum task receipt logical identity is outside its producer array"
            )
        return producer

    def assert_unchanged(
        self,
        reusable_outputs: Sequence[_PathAuthority] = (),
    ) -> None:
        for authority in self.control_authorities:
            authority.assert_unchanged()
        for authority in reusable_outputs:
            authority.assert_unchanged()


def _intent_control_paths(
    campaign_dir: Path,
    phase: str,
    iteration: int,
) -> List[Path]:
    from .submission_intent import (
        INTENT_HISTORY_DIR_NAME,
        intent_dir,
        intent_path,
    )

    root = intent_dir(campaign_dir)
    current = intent_path(campaign_dir, phase, int(iteration))
    paths = [current] if current.exists() or current.is_symlink() else []
    history = root / INTENT_HISTORY_DIR_NAME
    if history.is_dir() and not history.is_symlink():
        prefix = phase.replace("/", "_").replace("\\", "_")
        prefix += "-" + str(int(iteration)).zfill(6)
        paths.extend(sorted(history.glob(prefix + "-*.json")))
    return paths


def _build_array_recovery_scan_context(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
    *,
    include_producers: bool = True,
    validate_points_file: bool = True,
) -> ArrayRecoveryScanContext:
    from .filesystem import campaign_owned_path, operational_path

    campaign = Path(campaign_dir).resolve()
    phase = str(getattr(phase_name, "value", phase_name))
    if not supports_partial_array_recovery(phase):
        raise ValueError("phase does not support partial array recovery: " + phase)
    control_paths: List[Path] = [operational_path(campaign, "state.json")]
    replacement_round = 0
    quantum_contract = None
    ariadne_task_map = None
    rejected_names: FrozenSet[str] = frozenset()
    acceptance_error = None

    if phase == "ARIADNE_ARRAY":
        from ..handoff_manifests import ariadne_task_map_path
        from ..layout import active_iteration_dir
        from ..seed_identity import read_ariadne_task_map

        iteration_dir = active_iteration_dir(campaign, int(iteration))
        task_map_path = campaign_owned_path(
            campaign,
            ariadne_task_map_path(iteration_dir),
        )
        task_map_authority = _capture_path_authority(task_map_path)
        ariadne_task_map = read_ariadne_task_map(
            iteration_dir,
            expected_iteration=int(iteration),
        )
        task_map_authority.assert_unchanged()
        tasks = tuple(dict(task) for task in ariadne_task_map["tasks"])
        task_by_id = {int(task["array_task_id"]): task for task in tasks}
        task_ids = tuple(task_by_id)
        campaign_uid = str(ariadne_task_map.get("campaign_uid") or "")
        control_paths.append(task_map_path)
        selection = ariadne_task_map.get("selection_manifest")
        if isinstance(selection, Mapping):
            control_paths.append(
                campaign_owned_path(
                    campaign,
                    iteration_dir / str(selection.get("path") or ""),
                )
            )
    else:
        from .quantum_task_contracts import quantum_task_contract

        if "REPLACEMENT" in phase:
            replacement_round, round_dir = _replacement_identity(
                campaign,
                phase,
                int(iteration),
            )
            control_paths.extend(_existing_control_files(round_dir))
        quantum_contract = quantum_task_contract(
            campaign,
            phase,
            int(iteration),
            replacement_round=int(replacement_round),
            validate_points_file=bool(validate_points_file),
        )
        tasks = tuple(quantum_contract.tasks)
        task_by_id = {int(task.logical_task_id): task for task in tasks}
        task_ids = tuple(task_by_id)
        campaign_uid = str(quantum_contract.campaign_uid)
        control_paths.extend(_existing_control_files(quantum_contract.staging_dir))
        if "GAUSSIAN" in phase:
            from .input_staging import (
                POINTS_MEMBERSHIP_NONE,
                quantum_acceptance_manifest_path,
                read_quantum_acceptance_manifest,
            )

            manifest_path = quantum_acceptance_manifest_path(
                quantum_contract.staging_dir,
                phase_name=phase,
            )
            if manifest_path.exists() or manifest_path.is_symlink():
                try:
                    if manifest_path.is_symlink():
                        raise ValueError(
                            "Gaussian acceptance manifest must not be a symlink"
                        )
                    _accepted, manifest = read_quantum_acceptance_manifest(
                        quantum_contract.staging_dir,
                        expected_phase=phase,
                        expected_iteration=int(iteration),
                        require_nonempty=False,
                        points_membership=POINTS_MEMBERSHIP_NONE,
                        require_accepted_payloads=False,
                    )
                    dispositions = list(manifest["accepted_pointdirs"]) + [
                        str(record["pointdir"])
                        for record in list(manifest["rejected"])
                    ]
                    if (
                        int(manifest["n_total"]) != quantum_contract.logical_total
                        or set(dispositions)
                        != set(quantum_contract.pointdir_names)
                        or len(dispositions) != len(set(dispositions))
                    ):
                        raise ValueError(
                            "Gaussian acceptance does not cover its producer task set"
                        )
                    rejected_names = frozenset(
                        str(record["pointdir"])
                        for record in list(manifest["rejected"])
                    )
                except Exception as exc:
                    acceptance_error = (
                        "gaussian_acceptance_invalid: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)[:140]
                    )

    producer_index: Dict[
        Tuple[str, str, str], Tuple[Mapping[str, Any], FrozenSet[int]]
    ] = {}
    if include_producers and phase != "ARIADNE_ARRAY":
        from .submission_intent import (
            intent_attempt_records,
            intent_submitted_logical_task_ids,
        )

        records = intent_attempt_records(
            campaign,
            phase,
            int(iteration),
            expected_campaign_uid=campaign_uid,
        )
        for intent in records:
            submitted = frozenset(
                int(task_id)
                for task_id in intent_submitted_logical_task_ids(campaign, intent)
            )
            key = (
                str(intent.get("attempt_id") or ""),
                str(intent.get("submission_identity") or ""),
                str(intent.get("job_id") or ""),
            )
            if key in producer_index:
                raise ValueError("quantum task producer identity is duplicated")
            producer_index[key] = (dict(intent), submitted)
            metadata = intent.get("submission_metadata")
            bundle = (
                metadata.get("script_bundle")
                if isinstance(metadata, Mapping)
                else None
            )
            if isinstance(bundle, str) and bundle:
                map_path = campaign_owned_path(
                    campaign,
                    Path(bundle) / "array_task_map.json",
                )
                if map_path.exists() or map_path.is_symlink():
                    control_paths.append(map_path)
        control_paths.extend(_intent_control_paths(campaign, phase, int(iteration)))

    unique_controls = []
    seen_controls = set()
    for raw_path in control_paths:
        path = campaign_owned_path(campaign, Path(raw_path))
        key = str(path.resolve(strict=False))
        if key in seen_controls:
            continue
        seen_controls.add(key)
        if path.exists() or path.is_symlink():
            unique_controls.append(_capture_path_authority(path))
    return ArrayRecoveryScanContext(
        campaign_dir=campaign,
        phase=phase,
        iteration=int(iteration),
        replacement_round=int(replacement_round),
        campaign_uid=campaign_uid,
        task_ids=tuple(int(task_id) for task_id in task_ids),
        task_by_id=task_by_id,
        quantum_contract=quantum_contract,
        ariadne_task_map=ariadne_task_map,
        gaussian_rejected_names=rejected_names,
        gaussian_acceptance_error=acceptance_error,
        producer_index=producer_index,
        control_authorities=tuple(unique_controls),
    )


def _bound_array_recovery_scan_context(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
    *,
    include_producers: bool = True,
    validate_points_file: bool = True,
) -> Optional[ArrayRecoveryScanContext]:
    """Return the optimized context for a bound campaign.

    Small direct API fixtures historically omit state.json and inject the
    logical-task and validation helpers.  Keep that programmatic path while
    requiring every real campaign to use the authority-bound context.
    """
    from .filesystem import operational_path

    state_path = operational_path(campaign_dir, "state.json")
    if not state_path.exists() and not state_path.is_symlink():
        return None
    return _build_array_recovery_scan_context(
        campaign_dir,
        phase_name,
        int(iteration),
        include_producers=bool(include_producers),
        validate_points_file=bool(validate_points_file),
    )


def logical_task_ids(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
) -> List[int]:
    phase = str(getattr(phase_name, "value", phase_name))
    if not supports_partial_array_recovery(phase):
        return []
    if phase == "ARIADNE_ARRAY":
        from ..layout import active_iteration_dir
        from ..seed_identity import read_ariadne_task_map

        task_map = read_ariadne_task_map(
            active_iteration_dir(campaign_dir, int(iteration)),
            expected_iteration=int(iteration),
        )
        return [int(task["array_task_id"]) for task in task_map["tasks"]]
    from .quantum_task_contracts import quantum_task_contract

    replacement_round = (
        _replacement_identity(campaign_dir, phase, int(iteration))[0]
        if "REPLACEMENT" in phase
        else 0
    )
    contract = quantum_task_contract(
        campaign_dir,
        phase,
        int(iteration),
        replacement_round=int(replacement_round),
        validate_points_file=True,
    )
    return [task.logical_task_id for task in contract.tasks]


def _pointdir_for_task(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    task_id: int,
) -> Optional[Path]:
    from .quantum_task_contracts import quantum_task_contract

    replacement_round = (
        _replacement_identity(campaign_dir, phase_name, int(iteration))[0]
        if "REPLACEMENT" in phase_name
        else 0
    )
    contract = quantum_task_contract(
        campaign_dir,
        phase_name,
        int(iteration),
        replacement_round=int(replacement_round),
        validate_points_file=True,
    )
    index = int(task_id)
    if index < 0 or index >= len(contract.tasks):
        return None
    return contract.tasks[index].pointdir


def _validate_quantum_task(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    task_id: int,
    *,
    scan_context: Optional[ArrayRecoveryScanContext] = None,
) -> Tuple[bool, str, str]:
    if scan_context is None:
        from .quantum_task_contracts import quantum_task_contract

        replacement_round = (
            _replacement_identity(
                campaign_dir,
                phase_name,
                int(iteration),
            )[0]
            if "REPLACEMENT" in phase_name
            else 0
        )
        contract = quantum_task_contract(
            campaign_dir,
            phase_name,
            int(iteration),
            replacement_round=int(replacement_round),
            validate_points_file=True,
        )
        task = next(
            (
                candidate
                for candidate in contract.tasks
                if int(candidate.logical_task_id) == int(task_id)
            ),
            None,
        )
        if task is None:
            return False, "task_not_in_points_file", ""
        pointdir = task.pointdir
        if "GAUSSIAN" in str(phase_name):
            from .input_staging import (
                POINTS_MEMBERSHIP_NONE,
                quantum_acceptance_manifest_path,
                read_quantum_acceptance_manifest,
            )

            manifest_path = quantum_acceptance_manifest_path(
                contract.staging_dir,
                phase_name=str(phase_name),
            )
            if manifest_path.exists() or manifest_path.is_symlink():
                try:
                    if manifest_path.is_symlink():
                        raise ValueError(
                            "Gaussian acceptance manifest must not be a symlink"
                        )
                    _accepted, manifest = read_quantum_acceptance_manifest(
                        contract.staging_dir,
                        expected_phase=str(phase_name),
                        expected_iteration=int(iteration),
                        require_nonempty=False,
                        points_membership=POINTS_MEMBERSHIP_NONE,
                        require_accepted_payloads=False,
                    )
                    dispositions = list(manifest["accepted_pointdirs"]) + [
                        str(record["pointdir"])
                        for record in list(manifest["rejected"])
                    ]
                    if (
                        int(manifest["n_total"]) != contract.logical_total
                        or set(dispositions) != set(contract.pointdir_names)
                        or len(dispositions) != len(set(dispositions))
                    ):
                        raise ValueError(
                            "Gaussian acceptance does not cover its producer task set"
                        )
                    rejected_names = {
                        str(record["pointdir"])
                        for record in list(manifest["rejected"])
                    }
                    if pointdir.name in rejected_names:
                        return (
                            True,
                            "terminal_gaussian_rejection",
                            str(pointdir.resolve(strict=False)),
                        )
                except Exception as exc:
                    return (
                        False,
                        "gaussian_acceptance_invalid: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)[:140],
                        str(pointdir.resolve(strict=False)),
                    )
        return _validate_quantum_task_path(
            campaign_dir,
            phase_name,
            int(iteration),
            int(task_id),
            pointdir,
            expected_campaign_uid=str(contract.campaign_uid),
            replacement_round=int(replacement_round),
        )

    context = scan_context
    if (
        context.phase != str(phase_name)
        or int(context.iteration) != int(iteration)
        or context.quantum_contract is None
    ):
        raise ValueError("quantum array recovery context identity mismatch")
    task = context.task_by_id.get(int(task_id))
    if task is None:
        return False, "task_not_in_points_file", ""
    pointdir = task.pointdir
    if "GAUSSIAN" in str(phase_name):
        if context.gaussian_acceptance_error is not None:
            return (
                False,
                str(context.gaussian_acceptance_error),
                str(pointdir.resolve(strict=False)),
            )
        if pointdir.name in context.gaussian_rejected_names:
            return (
                True,
                "terminal_gaussian_rejection",
                str(pointdir.resolve(strict=False)),
            )
    return _validate_quantum_task_path(
        campaign_dir,
        phase_name,
        int(iteration),
        int(task_id),
        pointdir,
        expected_campaign_uid=str(context.campaign_uid),
        replacement_round=int(context.replacement_round),
        producer_resolver=context.resolve_receipt_producer,
    )


def _validate_quantum_task_path(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    task_id: int,
    pointdir: Path,
    *,
    expected_campaign_uid: Optional[str] = None,
    replacement_round: Optional[int] = None,
    producer_resolver: Optional[
        Callable[[Mapping[str, Any], int], Mapping[str, Any]]
    ] = None,
) -> Tuple[bool, str, str]:
    if pointdir.is_symlink():
        return False, "pointdir_symlinked", str(pointdir.resolve(strict=False))
    if not pointdir.exists():
        return False, "pointdir_missing", str(pointdir.resolve(strict=False))
    if not pointdir.is_dir():
        return False, "pointdir_wrong_type", str(pointdir.resolve(strict=False))
    try:
        from ichor.core.files.point_directory import PointDirectory
        from .live_executor import validate_aimall_completed, validate_gaussian_completed

        validator = validate_gaussian_completed if "GAUSSIAN" in phase_name else validate_aimall_completed
        ok, reason = validator(PointDirectory(pointdir))
        if ok:
            from .quantum_task_receipts import read_quantum_task_receipt
            from .submission_intent import (
                resolve_quantum_task_receipt_producer,
            )

            try:
                receipt = read_quantum_task_receipt(
                    pointdir,
                    phase_name=phase_name,
                    iteration=int(iteration),
                    logical_task_id=int(task_id),
                )
            except ValueError:
                if "GAUSSIAN" not in phase_name:
                    raise
                from .input_staging import (
                    read_gaussian_task_receipt_after_aimall,
                )
                from .quantum_task_contracts import (
                    aimall_phase_for_gaussian,
                )

                receipt = read_gaussian_task_receipt_after_aimall(
                    pointdir,
                    gaussian_phase=phase_name,
                    aimall_phase=aimall_phase_for_gaussian(phase_name),
                    iteration=int(iteration),
                    logical_task_id=int(task_id),
                )
            resolved_round = (
                int(replacement_round)
                if replacement_round is not None
                else (
                    _replacement_identity(
                        campaign_dir,
                        phase_name,
                        int(iteration),
                    )[0]
                    if "REPLACEMENT" in phase_name
                    else 0
                )
            )
            if producer_resolver is not None:
                producer = producer_resolver(receipt, int(task_id))
            else:
                producer = resolve_quantum_task_receipt_producer(
                    campaign_dir,
                    receipt,
                    expected_campaign_uid=(
                        str(expected_campaign_uid)
                        if expected_campaign_uid is not None
                        else str(receipt["campaign_uid"])
                    ),
                    phase_name=phase_name,
                    iteration=int(iteration),
                    logical_task_id=int(task_id),
                    replacement_round=int(resolved_round),
                )
            observed_identity = (
                str(receipt["campaign_uid"]),
                str(receipt["attempt_id"]),
                str(receipt["submission_identity"]),
                str(receipt["job_id"]),
            )
            expected_identity = (
                str(producer["campaign_uid"]),
                str(producer["attempt_id"]),
                str(producer["submission_identity"]),
                str(producer["job_id"]),
            )
            if observed_identity != expected_identity:
                raise ValueError("quantum task receipt producer identity mismatch")
        return bool(ok), str(reason or ""), str(Path(pointdir).resolve(strict=False))
    except Exception as exc:
        return False, type(exc).__name__ + ": " + str(exc)[:160], str(Path(pointdir).resolve(strict=False))


def scan_aimall_upstream_gaussian_recovery(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
) -> Optional[Dict[str, Any]]:
    """Rewind absent AIMAll pointdirs through their Gaussian producer tasks."""
    phase = str(getattr(phase_name, "value", phase_name))
    from .quantum_task_contracts import (
        AIMALL_PHASES,
        gaussian_phase_for_aimall,
        quantum_task_contract,
    )

    if phase not in AIMALL_PHASES:
        return None
    if "REPLACEMENT" in phase:
        replacement_round, staging = _replacement_identity(
            campaign_dir,
            phase,
            int(iteration),
        )
    else:
        replacement_round = 0
        staging = staging_phase_dir(campaign_dir, phase, int(iteration))
    from .input_staging import quantum_acceptance_manifest_path

    publication = quantum_acceptance_manifest_path(
        staging,
        phase_name=phase,
    )
    if publication.exists() or publication.is_symlink():
        return None
    gaussian_phase = gaussian_phase_for_aimall(phase)
    from .input_staging import (
        POINTS_MEMBERSHIP_NONE,
        _points_file_names,
        read_quantum_acceptance_manifest,
    )

    accepted_paths, gaussian_manifest = read_quantum_acceptance_manifest(
        staging,
        expected_phase=gaussian_phase,
        expected_iteration=int(iteration),
        require_nonempty=False,
        points_membership=POINTS_MEMBERSHIP_NONE,
        require_accepted_payloads=False,
    )
    missing_names = {
        Path(path).name
        for path in accepted_paths
        if not Path(path).exists() and not Path(path).is_symlink()
    }
    if not missing_names:
        return None
    aimall = quantum_task_contract(
        campaign_dir,
        phase,
        int(iteration),
        replacement_round=int(replacement_round),
        validate_points_file=False,
    )
    missing = [
        task for task in aimall.tasks if task.pointdir_name in missing_names
    ]
    if len(missing) != len(missing_names):
        raise ValueError(
            "missing AIMAll pointdirs do not match the accepted Gaussian task set"
        )
    gaussian_context = _bound_array_recovery_scan_context(
        campaign_dir,
        gaussian_phase,
        int(iteration),
        validate_points_file=False,
    )
    gaussian = (
        gaussian_context.quantum_contract
        if gaussian_context is not None
        else quantum_task_contract(
            campaign_dir,
            gaussian_phase,
            int(iteration),
            replacement_round=int(replacement_round),
            validate_points_file=False,
        )
    )
    if gaussian is None:
        raise ValueError("Gaussian recovery task contract is unavailable")
    listed_names = tuple(_points_file_names(aimall.staging_dir))
    if listed_names not in {
        aimall.pointdir_names,
        gaussian.pointdir_names,
    }:
        raise ValueError(
            "AIMAll POINTS.txt matches neither its accepted task set nor the "
            "complete Gaussian producer task set"
        )
    prior_rejected = {
        str(record["pointdir"])
        for record in list(gaussian_manifest.get("rejected") or [])
    }
    missing_producer_ids = {
        int(task.producer_logical_task_id) for task in missing
    }
    tasks = []
    retry_ids = []
    n_complete = 0
    reusable_authorities: List[_PathAuthority] = []
    for task in gaussian.tasks:
        if int(task.logical_task_id) in missing_producer_ids:
            ok = False
            reason = "aimall_pointdir_missing_requires_gaussian"
            output_path = str(task.pointdir.resolve(strict=False))
        elif task.pointdir_name in prior_rejected:
            ok = True
            reason = "prior_gaussian_rejection_preserved"
            output_path = str(task.pointdir.resolve(strict=False))
        else:
            ok, reason, output_path = _validate_quantum_task_path(
                campaign_dir,
                gaussian_phase,
                int(iteration),
                int(task.logical_task_id),
                task.pointdir,
                expected_campaign_uid=str(gaussian.campaign_uid),
                replacement_round=int(replacement_round),
                producer_resolver=(
                    None
                    if gaussian_context is None
                    else gaussian_context.resolve_receipt_producer
                ),
            )
        status = "complete" if ok else "pending"
        if ok:
            n_complete += 1
            if gaussian_context is not None:
                reusable_authorities.extend(
                    _reusable_task_authorities(
                        gaussian_context,
                        task_id=int(task.logical_task_id),
                        output_path=str(output_path),
                        reason=str(reason or ""),
                    )
                )
        else:
            retry_ids.append(int(task.logical_task_id))
        tasks.append(
            {
                "task_id": int(task.logical_task_id),
                "status": status,
                "complete": bool(ok),
                "reason": str(reason or ""),
                "output_path": str(output_path),
                "contract_hash": _sha_payload(
                    {
                        "phase": gaussian_phase,
                        "iteration": int(iteration),
                        "task_id": int(task.logical_task_id),
                        "output_path": str(output_path),
                    }
                ),
            }
        )
    payload = {
        "schema_version": ARRAY_RECOVERY_SCHEMA_VERSION,
        "phase": gaussian_phase,
        "iteration": int(iteration),
        "replacement_round": int(replacement_round),
        "updated_at_iso": _now_iso(),
        "force_resubmit": False,
        "logical_total": int(len(tasks)),
        "n_complete": int(n_complete),
        "n_reuse": int(n_complete),
        "n_retry": int(len(retry_ids)),
        "retry_task_ids": retry_ids,
        "all_complete": False,
        "tasks": tasks,
        "source_phase": phase,
        "upstream_rewind": "missing_aimall_pointdir",
        "missing_aimall_task_ids": [
            int(task.logical_task_id) for task in missing
        ],
    }
    if gaussian_context is not None:
        gaussian_context.assert_unchanged(reusable_authorities)
    return payload


def prepare_aimall_upstream_gaussian_recovery(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
) -> Dict[str, Any]:
    """Restore the complete Gaussian task list for an AIMAll rewind."""
    phase = str(getattr(phase_name, "value", phase_name))
    from .input_staging import _points_file_names, write_points_file
    from .quantum_task_contracts import (
        AIMALL_PHASES,
        gaussian_phase_for_aimall,
        quantum_task_contract,
    )

    if phase not in AIMALL_PHASES:
        raise ValueError("phase is not an AIMAll recovery phase: " + phase)
    replacement_round = (
        _replacement_identity(campaign_dir, phase, int(iteration))[0]
        if "REPLACEMENT" in phase
        else 0
    )
    aimall = quantum_task_contract(
        campaign_dir,
        phase,
        int(iteration),
        replacement_round=int(replacement_round),
        validate_points_file=False,
    )
    missing = [
        task
        for task in aimall.tasks
        if not task.pointdir.exists() and not task.pointdir.is_symlink()
    ]
    if not missing:
        raise ValueError(
            "AIMAll recovery has no absent accepted point directory"
        )
    gaussian_phase = gaussian_phase_for_aimall(phase)
    gaussian = quantum_task_contract(
        campaign_dir,
        gaussian_phase,
        int(iteration),
        replacement_round=int(replacement_round),
        validate_points_file=False,
    )
    listed = tuple(_points_file_names(aimall.staging_dir))
    permitted = {aimall.pointdir_names, gaussian.pointdir_names}
    if listed not in permitted:
        raise ValueError(
            "AIMAll POINTS.txt is inconsistent with its Gaussian producer"
        )
    changed = listed != gaussian.pointdir_names
    if changed:
        write_points_file(
            gaussian.staging_dir,
            [task.pointdir for task in gaussian.tasks],
        )
    return {
        "phase": gaussian_phase,
        "source_phase": phase,
        "iteration": int(iteration),
        "replacement_round": int(replacement_round),
        "changed": bool(changed),
        "points_file": str(
            (gaussian.staging_dir / "POINTS.txt").resolve(strict=False)
        ),
        "logical_total": int(gaussian.logical_total),
        "retry_task_ids": [
            int(task.producer_logical_task_id) for task in missing
        ],
    }


def _validate_ariadne_task(
    campaign_dir: Union[str, Path],
    iteration: int,
    task_id: int,
    *,
    scan_context: Optional[ArrayRecoveryScanContext] = None,
) -> Tuple[bool, str, str]:
    from ..ariadne_outputs import validate_seed_output
    from ..layout import active_iteration_dir, ariadne_seed_dir
    from ..seed_identity import read_ariadne_task_map, task_for_array_task_id

    iter_dir = active_iteration_dir(campaign_dir, int(iteration))
    try:
        if scan_context is None:
            task_map = read_ariadne_task_map(
                iter_dir,
                expected_iteration=int(iteration),
            )
            task = task_for_array_task_id(task_map, int(task_id))
        else:
            if (
                scan_context.phase != "ARIADNE_ARRAY"
                or int(scan_context.iteration) != int(iteration)
                or scan_context.ariadne_task_map is None
            ):
                raise ValueError("ARIADNE recovery context identity mismatch")
            task_map = scan_context.ariadne_task_map
            task = scan_context.task_by_id.get(int(task_id))
            if task is None:
                raise ValueError("ARIADNE task is outside its recovery context")
        seed_dir = ariadne_seed_dir(iter_dir, int(task["seed_id"]))
        output = validate_seed_output(
            seed_dir,
            expected_campaign_uid=str(task_map["campaign_uid"]),
            expected_iteration=int(iteration),
            expected_seed_id=int(task["seed_id"]),
            expected_seed_uid=str(task["seed_uid"]),
            expected_array_task_id=int(task_id),
        )
        if not bool(output.get("task_success", False)):
            return False, "task_success_false", str(seed_dir.resolve(strict=False))
        try:
            task_exit_code = int(output.get("task_exit_code", 1))
        except (TypeError, ValueError):
            return False, "task_exit_code_invalid", str(seed_dir.resolve(strict=False))
        if task_exit_code != 0:
            return (
                False,
                "task_exit_code_" + str(task_exit_code),
                str(seed_dir.resolve(strict=False)),
            )
        return True, "", str(seed_dir.resolve(strict=False))
    except Exception as exc:
        return False, type(exc).__name__ + ": " + str(exc)[:160], str(iter_dir.resolve(strict=False))


def _validate_task(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    task_id: int,
    *,
    scan_context: Optional[ArrayRecoveryScanContext] = None,
) -> Tuple[bool, str, str]:
    if phase_name == "ARIADNE_ARRAY":
        return _validate_ariadne_task(
            campaign_dir,
            int(iteration),
            int(task_id),
            scan_context=scan_context,
        )
    return _validate_quantum_task(
        campaign_dir,
        phase_name,
        int(iteration),
        int(task_id),
        scan_context=scan_context,
    )


def _reusable_task_authorities(
    context: ArrayRecoveryScanContext,
    *,
    task_id: int,
    output_path: str,
    reason: str,
) -> Tuple[_PathAuthority, ...]:
    if reason in {
        "terminal_gaussian_rejection",
        "prior_gaussian_rejection_preserved",
    }:
        return ()
    output = Path(output_path)
    authorities = [_capture_path_authority(output)]
    if context.phase == "ARIADNE_ARRAY":
        from ..ariadne_outputs import SEED_OUTPUT_MANIFEST_FILENAME

        receipt = output / SEED_OUTPUT_MANIFEST_FILENAME
    else:
        from .quantum_task_receipts import (
            AIMALL_TASK_RECEIPT,
            GAUSSIAN_TASK_RECEIPT,
        )

        receipt = output / (
            GAUSSIAN_TASK_RECEIPT
            if "GAUSSIAN" in context.phase
            else AIMALL_TASK_RECEIPT
        )
    authorities.append(_capture_path_authority(receipt))
    return tuple(authorities)


def _report_array_recovery_progress(
    callback: Optional[Callable[..., None]],
    phase: str,
    completed: int,
    total: int,
) -> None:
    if callback is None:
        return
    backend = (
        "ARIADNE"
        if phase == "ARIADNE_ARRAY"
        else "Gaussian"
        if "GAUSSIAN" in phase
        else "AIMAll"
    )
    try:
        callback(
            "array_recovery_validation",
            completed=int(completed),
            total=int(total),
            unit="tasks",
            recovery_backend=backend,
        )
    except Exception:
        return


def scan_array_tasks(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
    *,
    force_resubmit: bool = False,
    forced_retry_task_ids: Optional[Sequence[int]] = None,
    progress_callback: Optional[Callable[..., None]] = None,
    scan_context: Optional[ArrayRecoveryScanContext] = None,
    _authority_sink: Optional[List[_PathAuthority]] = None,
) -> Dict[str, Any]:
    phase = str(getattr(phase_name, "value", phase_name))
    if not supports_partial_array_recovery(phase):
        raise ValueError("phase does not support partial array recovery: " + phase)
    context = scan_context or _bound_array_recovery_scan_context(
        campaign_dir,
        phase,
        int(iteration),
    )
    if context is None:
        task_ids = logical_task_ids(campaign_dir, phase, int(iteration))
    else:
        if (
            context.phase != phase
            or int(context.iteration) != int(iteration)
            or context.campaign_dir != Path(campaign_dir).resolve()
        ):
            raise ValueError("array recovery scan context identity mismatch")
        task_ids = list(context.task_ids)
    forced_retry = {
        int(value) for value in (forced_retry_task_ids or ())
    }
    unknown_forced = forced_retry.difference(int(value) for value in task_ids)
    if unknown_forced:
        raise ValueError(
            "forced retry task IDs are outside the canonical task set: "
            + ", ".join(str(value) for value in sorted(unknown_forced))
        )
    tasks: List[Dict[str, Any]] = []
    n_complete = 0
    reusable_authorities: List[_PathAuthority] = []
    _report_array_recovery_progress(
        progress_callback,
        phase,
        0,
        len(task_ids),
    )
    for task_id in task_ids:
        if context is None:
            ok, reason, output_path = _validate_task(
                campaign_dir,
                phase,
                int(iteration),
                int(task_id),
            )
        else:
            ok, reason, output_path = _validate_task(
                campaign_dir,
                phase,
                int(iteration),
                int(task_id),
                scan_context=context,
            )
        if bool(force_resubmit):
            status = "pending"
            reason = "force_resubmit"
        elif int(task_id) in forced_retry:
            status = "pending"
            reason = "environment_equivalence_unproven"
        elif ok:
            status = "complete"
            n_complete += 1
            if context is not None:
                reusable_authorities.extend(
                    _reusable_task_authorities(
                        context,
                        task_id=int(task_id),
                        output_path=str(output_path),
                        reason=str(reason or ""),
                    )
                )
        else:
            status = "pending"
        tasks.append(
            {
                "task_id": int(task_id),
                "status": status,
                "complete": bool(status == "complete"),
                "reason": str(reason or ""),
                "output_path": str(output_path),
                "contract_hash": _sha_payload(
                    {
                        "phase": phase,
                        "iteration": int(iteration),
                        "task_id": int(task_id),
                        "output_path": str(output_path),
                    }
                ),
            }
        )
        _report_array_recovery_progress(
            progress_callback,
            phase,
            len(tasks),
            len(task_ids),
        )
    if bool(force_resubmit):
        n_complete = 0
    retry_ids = [int(task["task_id"]) for task in tasks if task.get("status") != "complete"]
    payload: Dict[str, Any] = {
        "schema_version": ARRAY_RECOVERY_SCHEMA_VERSION,
        "phase": phase,
        "iteration": int(iteration),
        "replacement_round": (
            int(context.replacement_round)
            if context is not None
            else (
                _replacement_identity(campaign_dir, phase, int(iteration))[0]
                if "REPLACEMENT" in phase
                else 0
            )
        ),
        "updated_at_iso": _now_iso(),
        "force_resubmit": bool(force_resubmit),
        "logical_total": int(len(task_ids)),
        "n_complete": int(n_complete),
        "n_reuse": int(n_complete),
        "n_retry": int(len(retry_ids)),
        "retry_task_ids": retry_ids,
        "all_complete": bool(len(task_ids) > 0 and len(retry_ids) == 0),
        "tasks": tasks,
    }
    if context is not None:
        context.assert_unchanged(reusable_authorities)
    if _authority_sink is not None:
        _authority_sink.extend(reusable_authorities)
    return payload


def write_array_ledger(
    campaign_dir: Union[str, Path],
    payload: Dict[str, Any],
) -> Path:
    phase = str(payload.get("phase"))
    iteration = int(payload.get("iteration"))
    path = array_ledger_path(
        campaign_dir,
        phase,
        iteration,
        replacement_round=int(payload.get("replacement_round", 0)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, _validate_array_ledger(dict(payload), path=path, phase=phase, iteration=iteration))
    return path


def _ledger_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(label + " must be an exact integer >= " + str(minimum))
    return int(value)


def _validate_array_ledger(
    data: Any,
    *,
    path: Path,
    phase: str,
    iteration: int,
) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("array recovery ledger must be a JSON object: " + str(path))
    if _ledger_int(data.get("schema_version"), "array ledger schema_version") != ARRAY_RECOVERY_SCHEMA_VERSION:
        raise ValueError("unsupported array recovery ledger schema: " + str(path))
    if data.get("phase") != str(phase) or not supports_partial_array_recovery(phase):
        raise ValueError("array recovery ledger phase mismatch")
    if _ledger_int(data.get("iteration"), "array ledger iteration") != int(iteration):
        raise ValueError("array recovery ledger iteration mismatch")
    _ledger_int(data.get("replacement_round"), "array ledger replacement_round")
    for key in ("force_resubmit", "all_complete"):
        if not isinstance(data.get(key), bool):
            raise ValueError("array recovery ledger " + key + " must be a boolean")
    logical_total = _ledger_int(data.get("logical_total"), "array ledger logical_total")
    n_complete = _ledger_int(data.get("n_complete"), "array ledger n_complete")
    n_reuse = _ledger_int(data.get("n_reuse"), "array ledger n_reuse")
    n_retry = _ledger_int(data.get("n_retry"), "array ledger n_retry")
    tasks = data.get("tasks")
    retry_ids = data.get("retry_task_ids")
    if not isinstance(tasks, list) or len(tasks) != logical_total:
        raise ValueError("array recovery ledger task cardinality mismatch")
    if not isinstance(retry_ids, list):
        raise ValueError("array recovery retry_task_ids must be a list")
    seen = set()
    observed_retry = []
    observed_complete = 0
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("array recovery task must be an object")
        task_id = _ledger_int(task.get("task_id"), "array recovery task_id")
        if task_id in seen:
            raise ValueError("array recovery task IDs contain duplicates")
        seen.add(task_id)
        status = task.get("status")
        if status not in {"complete", "pending"}:
            raise ValueError("array recovery task status is invalid")
        if task.get("complete") is not (status == "complete"):
            raise ValueError("array recovery task status/complete mismatch")
        if not isinstance(task.get("reason"), str) or not isinstance(
            task.get("output_path"), str
        ):
            raise ValueError("array recovery task diagnostics are invalid")
        digest = task.get("contract_hash")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("array recovery task contract hash is invalid")
        if status == "complete":
            observed_complete += 1
        else:
            observed_retry.append(task_id)
    if seen != set(range(logical_total)):
        raise ValueError("array recovery logical task IDs must be contiguous from zero")
    parsed_retry = [
        _ledger_int(value, "array recovery retry task ID") for value in retry_ids
    ]
    if parsed_retry != observed_retry or len(parsed_retry) != len(set(parsed_retry)):
        raise ValueError("array recovery retry task set mismatch")
    if (
        n_complete != observed_complete
        or n_reuse != observed_complete
        or n_retry != len(observed_retry)
        or n_complete + n_retry != logical_total
    ):
        raise ValueError("array recovery ledger summary counts mismatch")
    if data["all_complete"] is not (logical_total > 0 and n_retry == 0):
        raise ValueError("array recovery all_complete is contradictory")
    return data


def read_array_ledger(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
    *,
    replacement_round: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    path = array_ledger_path(
        campaign_dir,
        phase_name,
        int(iteration),
        replacement_round=replacement_round,
    )
    if not path.is_file():
        return None
    if path.is_symlink():
        raise ValueError("array recovery ledger must not be a symlink: " + str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    data = _validate_array_ledger(
        data,
        path=path,
        phase=str(getattr(phase_name, "value", phase_name)),
        iteration=int(iteration),
    )
    data["path"] = str(path)
    return data


def refresh_array_ledger(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
    *,
    force_resubmit: bool = False,
    forced_retry_task_ids: Optional[Sequence[int]] = None,
    progress_callback: Optional[Callable[..., None]] = None,
    scan_context: Optional[ArrayRecoveryScanContext] = None,
    _authority_sink: Optional[List[_PathAuthority]] = None,
) -> Dict[str, Any]:
    context = scan_context or _bound_array_recovery_scan_context(
        campaign_dir,
        phase_name,
        int(iteration),
    )
    reusable_authorities: List[_PathAuthority] = []
    payload = scan_array_tasks(
        campaign_dir,
        phase_name,
        int(iteration),
        force_resubmit=bool(force_resubmit),
        forced_retry_task_ids=forced_retry_task_ids,
        progress_callback=progress_callback,
        scan_context=context,
        _authority_sink=reusable_authorities,
    )
    if context is not None:
        context.assert_unchanged(reusable_authorities)
    path = write_array_ledger(campaign_dir, payload)
    payload["path"] = str(path)
    if _authority_sink is not None:
        _authority_sink.extend(reusable_authorities)
    return payload


def write_retry_task_file(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
    task_ids: Sequence[int],
    *,
    replacement_round: Optional[int] = None,
) -> Path:
    path = retry_task_file_path(
        campaign_dir,
        phase_name,
        int(iteration),
        replacement_round=replacement_round,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(str(int(task_id)) for task_id in task_ids)
    atomic_write_text(path, body + ("\n" if body else ""))
    return path


def clear_retry_task_file(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
    *,
    replacement_round: Optional[int] = None,
) -> None:
    path = retry_task_file_path(
        campaign_dir,
        phase_name,
        int(iteration),
        replacement_round=replacement_round,
    )
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass


def _ensure_inside_campaign(campaign_dir: Path, target: Path) -> None:
    campaign = campaign_dir.resolve(strict=False)
    resolved = target.resolve(strict=False)
    if resolved != campaign and campaign not in resolved.parents:
        raise ValueError("refusing to archive path outside campaign: " + str(target))


def _move_if_exists(source: Path, target_dir: Path, campaign_dir: Path) -> Optional[str]:
    from .filesystem import campaign_owned_path

    target_dir = campaign_owned_path(campaign_dir, target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_dir = campaign_owned_path(campaign_dir, target_dir)
    source_present = source.exists() or source.is_symlink()
    if source_present and source.is_symlink():
        raise ValueError("refusing to archive symlinked array output: " + str(source))
    source = campaign_owned_path(campaign_dir, source)
    target = target_dir / source.name
    target = campaign_owned_path(campaign_dir, target)
    target_present = target.exists() or target.is_symlink()
    if source_present and target_present:
        raise ValueError(
            "array output and its recovery archive destination both exist: "
            + str(source)
        )
    if not source_present:
        if not target_present:
            return None
        if target.is_symlink():
            raise ValueError(
                "array recovery archive destination is a symlink: "
                + str(target)
            )
        return str(target)
    shutil.move(str(source), str(target))
    return str(target)


def archive_existing_array_task_outputs(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
    *,
    task_ids: Optional[Sequence[int]] = None,
    archive_identity: Optional[str] = None,
) -> List[str]:
    phase = str(getattr(phase_name, "value", phase_name))
    if not supports_partial_array_recovery(phase):
        raise ValueError("phase does not support array output archive: " + phase)
    campaign = Path(campaign_dir)
    scan_context = _bound_array_recovery_scan_context(
        campaign,
        phase,
        int(iteration),
        include_producers=False,
    )
    stamp = str(
        archive_identity
        or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid.uuid4().hex[:8]
        )
    )
    from .filesystem import campaign_owned_path, operational_path

    archive_root = campaign_owned_path(
        campaign,
        operational_path(
            campaign,
            "arr",
            phase,
            str(int(iteration)).zfill(6) + "-" + stamp,
        ),
    )
    archive_receipt = archive_root / "ARCHIVE.json"
    ids = [
        int(value)
        for value in (
            task_ids
            if task_ids is not None
            else (
                scan_context.task_ids
                if scan_context is not None
                else logical_task_ids(campaign, phase, int(iteration))
            )
        )
    ]
    unknown_ids = (
        set()
        if scan_context is None
        else set(ids).difference(scan_context.task_ids)
    )
    if unknown_ids:
        raise ValueError(
            "array output archive task IDs are outside the canonical task set"
        )
    archived: List[str] = []
    if archive_root.exists() or archive_root.is_symlink():
        if archive_root.is_symlink() or not archive_root.is_dir():
            raise ValueError("array output archive is not a regular directory")
        if archive_receipt.is_symlink() or not archive_receipt.is_file():
            try:
                unexpected = list(archive_root.iterdir())
            except OSError as exc:
                raise ValueError(
                    "array output archive cannot be inspected"
                ) from exc
            if unexpected:
                raise ValueError(
                    "non-empty array output archive lacks its recovery receipt"
                )
            # A crash may occur after mkdir and before the first durable
            # transaction record.  An empty deterministic destination proves
            # that no output move has happened, so replay may initialise it.
            existing_receipt = None
        else:
            try:
                existing_receipt = json.loads(
                    archive_receipt.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as exc:
                raise ValueError("array output archive receipt is unreadable") from exc
        if existing_receipt is None:
            pass
        elif (
            not isinstance(existing_receipt, dict)
            or existing_receipt.get("schema_version") != 1
            or existing_receipt.get("phase") != phase
            or existing_receipt.get("iteration") != int(iteration)
            or existing_receipt.get("status") not in {"moving", "complete"}
            or not isinstance(existing_receipt.get("moved"), list)
            or any(
                not isinstance(value, str)
                for value in existing_receipt.get("moved", [])
            )
        ):
            raise ValueError("array output archive receipt is invalid")
        else:
            archived = list(existing_receipt["moved"])
            for value in archived:
                archived_path = Path(value)
                if archived_path.is_symlink() or not archived_path.exists():
                    raise ValueError(
                        "array output archive receipt references a missing or "
                        "symlinked payload: " + str(archived_path)
                    )
                archived_path = campaign_owned_path(campaign, archived_path)
                try:
                    archived_path.relative_to(archive_root)
                except ValueError as exc:
                    raise ValueError(
                        "array output archive receipt references a payload "
                        "outside its archive"
                    ) from exc
    else:
        archive_root.mkdir(parents=True, exist_ok=False)

    def record_archive(status: str) -> None:
        payload = {
            "schema_version": 1,
            "phase": phase,
            "iteration": int(iteration),
            "status": str(status),
            "moved": list(archived),
            "updated_at_iso": _now_iso(),
        }
        atomic_write_json(archive_receipt, payload)

    record_archive("moving")
    for task_id in ids:
        task_archive = campaign_owned_path(
            campaign, archive_root / str(int(task_id)).zfill(6)
        )
        if phase == "ARIADNE_ARRAY":
            from ..layout import active_iteration_dir, ariadne_seed_dir
            from ..seed_identity import read_ariadne_task_map, task_for_array_task_id

            iter_dir = active_iteration_dir(campaign, int(iteration))
            if scan_context is None:
                task_map = read_ariadne_task_map(
                    iter_dir,
                    expected_iteration=int(iteration),
                )
                task = task_for_array_task_id(task_map, int(task_id))
            else:
                task = scan_context.task_by_id[int(task_id)]
            seed_dir = ariadne_seed_dir(iter_dir, int(task["seed_id"]))
            moved = _move_if_exists(seed_dir, task_archive, campaign)
            if moved and moved not in archived:
                archived.append(moved)
                record_archive("moving")
            partial_pattern = "." + seed_dir.name + ".partial-*"
            for candidate in sorted(seed_dir.parent.glob(partial_pattern)):
                moved = _move_if_exists(candidate, task_archive, campaign)
                if moved and moved not in archived:
                    archived.append(moved)
                    record_archive("moving")
            continue
        if scan_context is None:
            pointdir = _pointdir_for_task(
                campaign,
                phase,
                int(iteration),
                int(task_id),
            )
            pdir = None if pointdir is None else Path(pointdir)
        else:
            task = scan_context.task_by_id.get(int(task_id))
            pdir = None if task is None else Path(task.pointdir)
        if pdir is None:
            continue
        if "GAUSSIAN" in phase:
            for candidate in [
                pdir / "input.gau",
                pdir / "input.log",
                pdir / "input.wfn",
                pdir / "GAUSSIAN_TASK_RECEIPT.json",
            ]:
                moved = _move_if_exists(candidate, task_archive, campaign)
                if moved and moved not in archived:
                    archived.append(moved)
                    record_archive("moving")
        elif "AIMALL" in phase:
            patterns = [
                "*_atomicfiles",
                "*.int",
                "*.sum",
                "*.agpviz",
                "*.mgpviz",
                "AIMALL_COMPLETION_RECEIPT.json",
            ]
            for pattern in patterns:
                for candidate in sorted(pdir.glob(pattern)):
                    moved = _move_if_exists(candidate, task_archive, campaign)
                    if moved and moved not in archived:
                        archived.append(moved)
                        record_archive("moving")
    record_archive("complete")
    if scan_context is not None:
        scan_context.assert_unchanged()
    return archived


def prepare_retry_submission(
    campaign_dir: Union[str, Path],
    phase_name: Any,
    iteration: int,
    *,
    force_resubmit: bool = False,
    forced_retry_task_ids: Optional[Sequence[int]] = None,
    progress_callback: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    context = _bound_array_recovery_scan_context(
        campaign_dir,
        phase_name,
        int(iteration),
    )
    reusable_authorities: List[_PathAuthority] = []
    if not bool(force_resubmit):
        try:
            existing = read_array_ledger(
                campaign_dir,
                phase_name,
                int(iteration),
                replacement_round=(
                    None
                    if context is None
                    else int(context.replacement_round)
                ),
            )
        except Exception:
            existing = None
        if isinstance(existing, dict) and bool(existing.get("force_resubmit", False)):
            force_resubmit = True
    payload = refresh_array_ledger(
        campaign_dir,
        phase_name,
        int(iteration),
        force_resubmit=bool(force_resubmit),
        forced_retry_task_ids=forced_retry_task_ids,
        progress_callback=progress_callback,
        scan_context=context,
        _authority_sink=reusable_authorities,
    )
    retry_ids = [int(x) for x in payload.get("retry_task_ids") or []]
    if context is not None:
        context.assert_unchanged(reusable_authorities)
    if retry_ids:
        retry_file = write_retry_task_file(
            campaign_dir,
            phase_name,
            int(iteration),
            retry_ids,
            replacement_round=(
                None if context is None else int(context.replacement_round)
            ),
        )
        payload["retry_task_file"] = str(retry_file)
    else:
        clear_retry_task_file(
            campaign_dir,
            phase_name,
            int(iteration),
            replacement_round=(
                None if context is None else int(context.replacement_round)
            ),
        )
        payload["retry_task_file"] = None
    return payload


def compact_array_recovery_summary(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    summary = {
        "phase": payload.get("phase"),
        "iteration": payload.get("iteration"),
        "logical_total": int(payload.get("logical_total") or 0),
        "n_complete": int(payload.get("n_complete") or 0),
        "n_reuse": int(payload.get("n_reuse") or 0),
        "n_retry": int(payload.get("n_retry") or 0),
        "force_resubmit": bool(payload.get("force_resubmit", False)),
        "retry_task_ids_sample": list(payload.get("retry_task_ids") or [])[:12],
        "retry_task_ids_truncated": bool(len(list(payload.get("retry_task_ids") or [])) > 12),
        "ledger": payload.get("path"),
        "retry_task_file": payload.get("retry_task_file"),
    }
    if payload.get("selected_count") is not None:
        summary["selected_count"] = int(payload["selected_count"])
    if payload.get("ordering_classification") is not None:
        summary["ordering_classification"] = str(
            payload["ordering_classification"]
        )
    if payload.get("producer_job_id") is not None:
        summary["producer_job_id"] = str(payload["producer_job_id"])
    if payload.get("scheduler_jobs_submitted") is not None:
        summary["scheduler_jobs_submitted"] = int(
            payload["scheduler_jobs_submitted"]
        )
    for key in (
        "state",
        "publication_disposition",
        "validation",
        "reason",
        "output_dir",
    ):
        if payload.get(key) is not None:
            summary[key] = str(payload[key])
    return summary


def discover_partial_array_recovery(
    campaign_dir: Union[str, Path],
    *,
    preferred_phase: Optional[Any] = None,
    preferred_iteration: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    candidates: List[Tuple[str, int]] = []
    if preferred_phase is not None and supports_partial_array_recovery(preferred_phase):
        try:
            candidates.append((str(getattr(preferred_phase, "value", preferred_phase)), int(preferred_iteration or 0)))
        except Exception:
            pass
    if not candidates:
        return None
    for phase, iteration in candidates:
        try:
            task_ids = logical_task_ids(campaign_dir, phase, int(iteration))
        except Exception:
            continue
        if not task_ids:
            continue
        try:
            payload = refresh_array_ledger(campaign_dir, phase, int(iteration))
        except Exception:
            continue
        if int(payload.get("logical_total") or 0) > 0:
            return payload
    return None
