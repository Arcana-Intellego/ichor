"""Live scheduler resource resolution for active-learning backend phases.

Backend-specific CPU and memory requests live under ``resources``. Each
request can be explicit or ``auto``. This module resolves those values using
the active cluster profile and producer-owned evidence before any sbatch
script is rendered.
"""
from __future__ import annotations

from ..strict_json import strict_json as json
import math
import re
import csv
import shutil
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .cluster_profile import active_machine, profile_value
from .phase_executor import BackendSubmissionError
from .script_bundles import campaign_owned_path
from ..layout import staging_phase_dir, trained_models_dir
from ..versioning.manifest import sha256_file


RESOURCE_FORMULA_VERSION = "2"


class ResourceEvidenceUnavailable(BackendSubmissionError):
    """Raised when a live resource formula lacks producer-owned evidence."""

    def __init__(self, phase_name: str, detail: str):
        self.phase_name = str(phase_name)
        self.detail = str(detail)
        super().__init__(
            "resource evidence not yet produced for "
            + self.phase_name
            + ": "
            + self.detail
        )


class ResourceEvidenceInvalid(BackendSubmissionError):
    """Raised when producer evidence exists but violates its contract."""

    def __init__(self, phase_name: str, detail: str):
        self.phase_name = str(phase_name)
        self.detail = str(detail)
        super().__init__(
            "resource evidence is invalid for "
            + self.phase_name
            + ": "
            + self.detail
        )


@dataclass(frozen=True)
class ResolvedPhaseResources:
    backend: str
    partition: str
    ntasks: int
    cpus_per_task: int
    mem_per_cpu: str
    estimated_total_memory_gb: float
    partition_memory_per_core_gb: float
    cpus_raw: Any
    mem_per_cpu_raw: Any
    cpu_reason: str
    memory_reason: str
    warnings: Tuple[str, ...] = ()
    extra: Dict[str, Any] = field(default_factory=dict)

    def _persisted_extra(self) -> Dict[str, Any]:
        # Evidence has its own top-level field in resource-resolution records.
        # Keeping the transient copy out of resources and journal events avoids
        # duplicating large FEREBUS dataset inventories.
        return {
            str(key): value
            for key, value in self.extra.items()
            if str(key) != "evidence"
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formula_version": RESOURCE_FORMULA_VERSION,
            "backend": self.backend,
            "partition": self.partition,
            "ntasks": int(self.ntasks),
            "cpus_per_task": int(self.cpus_per_task),
            "mem_per_cpu": self.mem_per_cpu,
            "estimated_total_memory_gb": float(self.estimated_total_memory_gb),
            "partition_memory_per_core_gb": float(
                self.partition_memory_per_core_gb
            ),
            "cpus_raw": self.cpus_raw,
            "mem_per_cpu_raw": self.mem_per_cpu_raw,
            "cpu_reason": self.cpu_reason,
            "memory_reason": self.memory_reason,
            "warnings": list(self.warnings),
            "extra": self._persisted_extra(),
        }

    def journal_payload(self, *, phase_name: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "phase": str(phase_name),
            "backend": self.backend,
            "partition": self.partition,
            "ntasks": int(self.ntasks),
            "cpus_raw": self.cpus_raw,
            "cpus_resolved": int(self.cpus_per_task),
            "mem_per_cpu_raw": self.mem_per_cpu_raw,
            "mem_per_cpu_resolved": self.mem_per_cpu,
            "estimated_total_memory_gb": float(self.estimated_total_memory_gb),
            "partition_memory_per_core_gb": float(self.partition_memory_per_core_gb),
            "cpu_reason": self.cpu_reason,
            "memory_reason": self.memory_reason,
        }
        if self.warnings:
            payload["warnings"] = list(self.warnings)
        journal_extra = self._persisted_extra()
        for verbose_key in ("aimall_task_naat", "aimall_task_demands"):
            journal_extra.pop(verbose_key, None)
        if journal_extra:
            payload["extra"] = journal_extra
        return payload


_SLURM_MEM_RE = re.compile(r"^([1-9][0-9]*)([KMGT]?)$", re.IGNORECASE)


def backend_for_phase(phase_name: str) -> str:
    if phase_name in ("PHASE_A_DIVERSITY", "PHASE_B_DIVERSITY"):
        return "diversity"
    if "GAUSSIAN" in phase_name:
        return "gaussian"
    if "AIMALL" in phase_name:
        return "aimall"
    if phase_name == "ARIADNE_ARRAY":
        return "ariadne"
    if phase_name in ("INITIAL_FEREBUS", "FEREBUS"):
        return "ferebus"
    raise BackendSubmissionError("unknown live backend phase: " + str(phase_name))


def slurm_memory_mib(value: Any) -> float:
    text = str(value).strip().upper()
    match = _SLURM_MEM_RE.fullmatch(text)
    if not match:
        raise BackendSubmissionError(
            "unsupported scheduler memory syntax: " + repr(value)
        )
    amount = int(match.group(1))
    unit = match.group(2) or "M"
    scale = {
        "K": 1.0 / 1024.0,
        "M": 1.0,
        "G": 1024.0,
        "T": 1024.0 * 1024.0,
    }[unit]
    return float(amount) * scale


def _partition_profile(partition: str) -> Optional[Dict[str, Any]]:
    partitions = profile_value("hpc", "partitions", default=None)
    if not isinstance(partitions, dict):
        return None
    raw = partitions.get(str(partition))
    if raw is None:
        raise BackendSubmissionError(
            "partition "
            + repr(str(partition))
            + " is not present in active profile hpc.partitions"
        )
    if not isinstance(raw, dict):
        raise BackendSubmissionError(
            "configured hpc.partitions."
            + str(partition)
            + " must be a mapping"
        )
    return raw


def scheduler_queue_for_partition(partition: str) -> str:
    """Return the native queue for an SGE logical partition."""
    profile = _partition_profile(partition)
    if profile is None:
        return str(partition)
    raw = profile.get("scheduler_queue")
    return str(raw).strip() if raw is not None else str(partition)


def parallel_environment_for_partition(partition: str) -> Optional[str]:
    profile = _partition_profile(partition)
    if profile is None:
        return None
    raw = profile.get("parallel_environment")
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def validate_partition_supported(partition: str) -> None:
    profile = _partition_profile(partition)
    if profile is None:
        return
    if bool(profile.get("daemon_supported", True)) is False:
        raise BackendSubmissionError(
            "partition "
            + repr(str(partition))
            + " is configured but is not supported for ICHOR active-learning "
            "daemon live phases"
        )


def partition_core_range(partition: str) -> Optional[Tuple[int, int]]:
    profile = _partition_profile(partition)
    if profile is not None:
        try:
            min_cores = int(profile.get("min_cpus", 1))
            max_cores = int(profile.get("max_cpus", min_cores))
        except (TypeError, ValueError) as exc:
            raise BackendSubmissionError(
                "configured hpc.partitions."
                + str(partition)
                + " min_cpus/max_cpus must be integers"
            ) from exc
        if min_cores < 1 or max_cores < min_cores:
            raise BackendSubmissionError(
                "configured hpc.partitions."
                + str(partition)
                + " has invalid core range ["
                + str(min_cores)
                + ", "
                + str(max_cores)
                + "]"
            )
        return min_cores, max_cores
    parallel = profile_value("hpc", "parallel_environments", default=None)
    if not isinstance(parallel, dict):
        return None
    raw = parallel.get(str(partition))
    if raw is None:
        raise BackendSubmissionError(
            "partition "
            + repr(str(partition))
            + " is not present in active profile hpc.parallel_environments"
        )
    try:
        lo, hi = list(raw)[:2]
        min_cores = int(lo)
        max_cores = int(hi)
    except (TypeError, ValueError) as exc:
        raise BackendSubmissionError(
            "configured hpc.parallel_environments for partition "
            + repr(str(partition))
            + " must be [min_cores, max_cores]"
        ) from exc
    if min_cores < 1 or max_cores < min_cores:
        raise BackendSubmissionError(
            "configured hpc.parallel_environments for partition "
            + repr(str(partition))
            + " has invalid range ["
            + str(min_cores)
            + ", "
            + str(max_cores)
            + "]"
        )
    return min_cores, max_cores


def partition_memory_per_core_gb(partition: str) -> float:
    profile = _partition_profile(partition)
    if profile is not None:
        raw_profile = profile.get("memory_per_core_gb")
        try:
            value = float(raw_profile)
        except (TypeError, ValueError) as exc:
            raise BackendSubmissionError(
                "configured hpc.partitions."
                + str(partition)
                + ".memory_per_core_gb must be numeric"
            ) from exc
        if not math.isfinite(value) or value <= 0.0:
            raise BackendSubmissionError(
                "configured hpc.partitions."
                + str(partition)
                + ".memory_per_core_gb must be > 0"
            )
        return value
    by_partition = profile_value(
        "hpc",
        "memory_per_core_gb_by_partition",
        default=None,
    )
    if isinstance(by_partition, dict):
        raw = by_partition.get(str(partition))
        if raw is None:
            raw = by_partition.get("default")
        if raw is not None:
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise BackendSubmissionError(
                    "configured hpc.memory_per_core_gb_by_partition for "
                    + str(partition)
                    + " must be numeric"
                ) from exc
            if not math.isfinite(value) or value <= 0.0:
                raise BackendSubmissionError(
                    "configured hpc.memory_per_core_gb_by_partition for "
                    + str(partition)
                    + " must be > 0"
                )
            return value
    raw_default = profile_value("hpc", "memory_per_core_gb", default=None)
    if raw_default is not None:
        try:
            value = float(raw_default)
        except (TypeError, ValueError) as exc:
            raise BackendSubmissionError(
                "configured hpc.memory_per_core_gb must be numeric"
            ) from exc
        if not math.isfinite(value) or value <= 0.0:
            raise BackendSubmissionError("configured hpc.memory_per_core_gb must be > 0")
        return value
    machine = active_machine()
    if machine not in (None, "", "_default"):
        raise BackendSubmissionError(
            "active profile "
            + repr(str(machine))
            + " has no hpc memory-per-core limit for partition "
            + repr(str(partition))
        )
    return 4.0


def _partition_min_max(partition: str) -> Tuple[int, int]:
    validate_partition_supported(partition)
    configured = partition_core_range(partition)
    if configured is None:
        return 1, 10_000
    return configured


def validate_partition_walltime(partition: str, walltime_hours: Union[int, float]) -> None:
    profile = _partition_profile(partition)
    if profile is None:
        return
    raw = profile.get("max_walltime_hours")
    if raw is None:
        return
    try:
        limit = float(raw)
        requested = float(walltime_hours)
    except (TypeError, ValueError) as exc:
        raise BackendSubmissionError(
            "configured hpc.partitions."
            + str(partition)
            + ".max_walltime_hours must be numeric"
        ) from exc
    if (
        not math.isfinite(limit)
        or limit <= 0.0
        or not math.isfinite(requested)
        or requested <= 0.0
    ):
        raise BackendSubmissionError(
            "partition walltime limits and requests must be finite and > 0"
        )
    if requested > limit + 1.0e-9:
        raise BackendSubmissionError(
            "walltime "
            + str(walltime_hours)
            + " hours exceeds partition "
            + repr(str(partition))
            + " maximum "
            + str(limit)
            + " hours"
        )


def _is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "auto"


def _ceil_gb(value: float) -> int:
    return max(1, int(math.ceil(float(value))))


def _format_gb(value: float) -> str:
    return str(_ceil_gb(value)) + "G"


def _validate_core_count(field_name: str, value: int, partition: str) -> None:
    configured = partition_core_range(partition)
    if configured is None:
        return
    min_cores, max_cores = configured
    if value < min_cores or value > max_cores:
        machine = active_machine() or "active profile"
        raise BackendSubmissionError(
            field_name
            + "="
            + str(value)
            + " is invalid for partition "
            + repr(str(partition))
            + " on "
            + str(machine)
            + "; configured range is ["
            + str(min_cores)
            + ", "
            + str(max_cores)
            + "]. Use a core count inside that range or choose a compatible partition."
        )


def _explicit_cpu(field_name: str, raw: Any, partition: str) -> int:
    if isinstance(raw, bool):
        raise BackendSubmissionError(field_name + " must be a positive integer or auto")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise BackendSubmissionError(field_name + " must be a positive integer or auto") from exc
    if value <= 0:
        raise BackendSubmissionError(field_name + " must be > 0")
    _validate_core_count(field_name, value, partition)
    return value


def _file_evidence(path: Path) -> Dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("resource evidence is not a regular file: " + str(source))
    return {
        "path": str(source.resolve()),
        "size": int(source.stat().st_size),
        "sha256": sha256_file(source),
    }


@dataclass(frozen=True)
class _AriadneFileAuthority:
    path: Path
    size: int
    mtime_ns: int
    device: int
    inode: int
    sha256: str

    def evidence(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "size": int(self.size),
            "sha256": str(self.sha256),
        }

    def assert_unchanged(self) -> None:
        source = Path(self.path)
        if source.is_symlink() or not source.is_file():
            raise ValueError(
                "ARIADNE resource authority file is missing or unsafe: "
                + str(source)
            )
        stat = source.stat()
        observed = (
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_dev),
            int(stat.st_ino),
        )
        expected = (
            int(self.size),
            int(self.mtime_ns),
            int(self.device),
            int(self.inode),
        )
        if observed != expected:
            raise ValueError(
                "ARIADNE resource authority changed before submission: "
                + str(source)
            )


def _capture_ariadne_file_authority(
    path: Path,
    *,
    known_sha256: Optional[str] = None,
) -> _AriadneFileAuthority:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(
            "ARIADNE resource authority is not a regular file: " + str(source)
        )
    before = source.stat()
    digest = str(known_sha256) if known_sha256 is not None else sha256_file(source)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("ARIADNE resource authority SHA-256 is invalid")
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
            "ARIADNE resource authority changed while it was inspected: "
            + str(source)
        )
    return _AriadneFileAuthority(
        path=source.resolve(),
        size=int(after.st_size),
        mtime_ns=int(after.st_mtime_ns),
        device=int(after.st_dev),
        inode=int(after.st_ino),
        sha256=digest,
    )


def _canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _synthetic_evidence(
    backend: str,
    phase_name: str,
    *,
    n_atoms_override: Optional[int],
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return bounded dimensions for unit tests and explicit dry-run use only."""
    n_atoms = int(n_atoms_override or 12)
    if backend == "diversity":
        return {
            "source": "synthetic_test_evidence",
            "n_frames": 100,
            "n_atoms": n_atoms,
            "atom_order": ["X"] * n_atoms,
        }
    if backend in {"gaussian", "aimall"}:
        return {
            "source": "synthetic_test_evidence",
            "n_tasks": 1,
            "max_n_atoms": n_atoms,
            "max_n_primitives": n_atoms * 40,
        }
    if backend == "ariadne":
        mode = str(
            getattr(
                getattr(getattr(config, "acquisition", object()), "gradient", object()),
                "mode",
                "active_fd",
            )
        )
        dimension = (
            int(
                getattr(
                    getattr(getattr(config, "acquisition", object()), "subspace", object()),
                    "max_subspace_dim",
                    6,
                )
            )
            if mode == "active_fd"
            else 3 * n_atoms
        )
        return {
            "source": "synthetic_test_evidence",
            "n_tasks": 1,
            "n_atoms": n_atoms,
            "model_bytes": 0,
            "gradient_dimension": int(dimension),
        }
    if backend == "ferebus":
        return {
            "source": "synthetic_test_evidence",
            "n_tasks": 1,
            "max_train_rows": 100,
            "max_internal_rows": 0,
            "max_external_rows": 0,
            "max_total_rows": 100,
            "max_features": 32,
        }
    raise BackendSubmissionError("unsupported synthetic resource backend: " + backend)


def _pool_evidence(campaign_dir: Path) -> Dict[str, Any]:
    from ..acquisition.trajectory_pool import (
        POOL_MANIFEST_FILENAME,
        POOL_SUBDIR,
        TrajectoryPool,
    )

    pool = TrajectoryPool.load(campaign_dir)
    manifest_path = campaign_dir / POOL_SUBDIR / POOL_MANIFEST_FILENAME
    if int(pool.manifest.n_frames) <= 0 or int(pool.manifest.natoms) <= 0:
        raise ValueError("trajectory pool resource evidence is empty")
    pool_path = campaign_owned_path(campaign_dir, pool.canonical_path)
    manifest_path = campaign_owned_path(campaign_dir, manifest_path)
    from ..custom_bootstrap import (
        custom_bootstrap_manifest_path,
        read_custom_bootstrap_manifest,
    )

    bootstrap_path = custom_bootstrap_manifest_path(campaign_dir)
    if bootstrap_path.is_symlink() or not bootstrap_path.is_file():
        raise FileNotFoundError(
            "committed bootstrap manifest is not yet available: "
            + str(bootstrap_path)
        )
    bootstrap = read_custom_bootstrap_manifest(campaign_dir)
    raw_excluded = bootstrap.get("excluded_pool_frame_ids")
    if not isinstance(raw_excluded, list):
        raise ValueError("bootstrap excluded_pool_frame_ids must be a list")
    excluded = []
    for value in raw_excluded:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("bootstrap excluded pool IDs must be exact integers")
        if value < 0 or value >= int(pool.manifest.n_frames):
            raise ValueError("bootstrap excluded pool ID is outside the pool")
        excluded.append(value)
    if len(excluded) != len(set(excluded)):
        raise ValueError("bootstrap excluded pool IDs contain duplicates")
    excluded_set = set(excluded)
    eligible = [
        index
        for index in range(int(pool.manifest.n_frames))
        if index not in excluded_set
    ]
    if not eligible:
        raise ValueError("bootstrap exclusions leave no Phase A candidates")
    return {
        "source": "trajectory_pool_manifest",
        "pool": _file_evidence(pool_path),
        "manifest": _file_evidence(manifest_path),
        "bootstrap_manifest": _file_evidence(bootstrap_path),
        "full_pool_n_frames": int(pool.manifest.n_frames),
        "excluded_pool_frame_ids": excluded,
        "excluded_pool_frame_count": len(excluded),
        "eligible_pool_frame_ids_sha256": _canonical_json_sha256(eligible),
        "n_frames": len(eligible),
        "n_atoms": int(pool.manifest.natoms),
        "atom_order": list(pool.manifest.atom_types),
        "coordinate_dimension": int(3 * pool.manifest.natoms),
    }


def _phase_b_evidence_with_config(
    campaign_dir: Path,
    iteration: int,
    config: Any,
) -> Dict[str, Any]:
    from ..handoff_manifests import (
        HandoffManifestError,
        authoritative_ariadne_candidate_frames,
        ariadne_batch_decision_path,
        ariadne_results_path,
    )
    from ..layout import active_iteration_dir
    from ..sampling.diversity import _phase_b_landing_safety_filter
    from ..sampling_protocol import load_sampling_protocol

    iter_dir = campaign_owned_path(
        campaign_dir,
        active_iteration_dir(campaign_dir, int(iteration)),
    )
    results_path = ariadne_results_path(iter_dir)
    decision_path = ariadne_batch_decision_path(iter_dir)
    for label, control_path in (
        ("ARIADNE results manifest", results_path),
        ("ARIADNE batch decision", decision_path),
    ):
        if control_path.is_symlink():
            raise ResourceEvidenceInvalid(
                "PHASE_B_DIVERSITY",
                label + " is a symlink: " + str(control_path),
            )
        if not control_path.is_file():
            raise ResourceEvidenceUnavailable(
                "PHASE_B_DIVERSITY",
                label + " is not yet published: " + str(control_path),
            )
    try:
        protocol = load_sampling_protocol(
            campaign_dir,
            config,
            iteration=int(iteration),
        )
        payload, frames, records = authoritative_ariadne_candidate_frames(
            campaign_dir,
            int(iteration),
        )
    except (ResourceEvidenceUnavailable, ResourceEvidenceInvalid):
        raise
    except FileNotFoundError as exc:
        raise ResourceEvidenceInvalid(
            "PHASE_B_DIVERSITY",
            "published ARIADNE handoff is missing accepted evidence: "
            + type(exc).__name__
            + ": "
            + str(exc),
        ) from exc
    except HandoffManifestError as exc:
        raise ResourceEvidenceInvalid(
            "PHASE_B_DIVERSITY",
            "published ARIADNE handoff is invalid: " + str(exc),
        ) from exc
    frames, records, safety_filter = _phase_b_landing_safety_filter(
        frames,
        records,
    )
    if not frames:
        raise ValueError("ARIADNE results contain no accepted Phase B candidates")
    n_atoms = len(frames[0])
    if any(len(frame) != n_atoms for frame in frames):
            raise ValueError("ARIADNE accepted candidates disagree on atom count")
    result_files = [
        _file_evidence(
            campaign_owned_path(campaign_dir, Path(str(record["result_json"])))
        )
        for record in records
    ]
    protocol_files = []
    for path in (
        protocol.manifest_path,
        protocol.audit_manifest_path,
        protocol.scale_model_path,
    ):
        if path is not None:
            protocol_files.append(_file_evidence(Path(path)))
    return {
        "source": "validated_phase_b_handoff",
        "manifest": _file_evidence(results_path),
        "batch_decision": _file_evidence(decision_path),
        "sampling_protocol_files": protocol_files,
        "accepted_result_files": result_files,
        "accepted_seed_ids": [int(record["seed_id"]) for record in records],
        "safety_filter": safety_filter,
        "n_frames": len(frames),
        "n_atoms": int(n_atoms or 0),
        "coordinate_dimension": int(3 * int(n_atoms or 0)),
    }


def _strict_pointdirs(
    campaign_dir: Path,
    staging: Path,
    required_file: str,
) -> Tuple[Path, List[Path]]:
    staging_path = Path(staging)
    root = campaign_owned_path(
        campaign_dir,
        staging_path,
    )
    points_path = root / "POINTS.txt"
    if points_path.is_symlink() or not points_path.is_file():
        raise FileNotFoundError("POINTS.txt is missing: " + str(points_path))
    pointdirs: List[Path] = []
    seen = set()
    for line_number, raw in enumerate(
        points_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = raw.strip()
        if not text:
            continue
        pointdir = Path(text)
        if not pointdir.is_absolute():
            pointdir = root / pointdir
        resolved = pointdir.resolve(strict=False)
        if resolved.parent != root:
            raise ValueError(
                "POINTS.txt entry is not a direct staging child at line "
                + str(line_number)
            )
        if resolved in seen:
            raise ValueError("POINTS.txt contains a duplicate pointdir")
        seen.add(resolved)
        if pointdir.is_symlink() or resolved.is_symlink() or not resolved.is_dir():
            raise ValueError("POINTS.txt entry is not a regular pointdir: " + str(pointdir))
        required = resolved / required_file
        if required.is_symlink() or not required.is_file():
            raise FileNotFoundError(
                required_file + " is missing from staged pointdir: " + str(resolved)
            )
        pointdirs.append(resolved)
    if not pointdirs:
        raise ValueError("POINTS.txt contains no staged pointdirs")
    return points_path, pointdirs


def _strict_pointdir_sequence(
    campaign_dir: Path,
    staging: Path,
    required_file: str,
    pointdirs: Sequence[Path],
) -> List[Path]:
    """Validate a prepared point sequence before POINTS.txt publication."""
    root = campaign_owned_path(campaign_dir, Path(staging))
    resolved_pointdirs: List[Path] = []
    seen = set()
    for task_index, raw_pointdir in enumerate(pointdirs):
        pointdir = Path(raw_pointdir)
        if not pointdir.is_absolute():
            pointdir = root / pointdir
        resolved = pointdir.resolve(strict=False)
        if resolved.parent != root:
            raise ValueError(
                "prepared quantum pointdir is not a direct staging child at "
                "task "
                + str(task_index)
            )
        if resolved in seen:
            raise ValueError("prepared quantum pointdirs contain a duplicate")
        seen.add(resolved)
        if pointdir.is_symlink() or resolved.is_symlink() or not resolved.is_dir():
            raise ValueError(
                "prepared quantum pointdir is not a regular directory: "
                + str(pointdir)
            )
        required = resolved / required_file
        if required.is_symlink() or not required.is_file():
            raise FileNotFoundError(
                required_file
                + " is missing from prepared pointdir: "
                + str(resolved)
            )
        resolved_pointdirs.append(resolved)
    if not resolved_pointdirs:
        raise ValueError("prepared quantum pointdir sequence is empty")
    return resolved_pointdirs


def _gjf_atom_order(path: Path) -> Tuple[str, ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start: Optional[int] = None
    for index, line in enumerate(lines):
        fields = line.split()
        if len(fields) >= 2:
            try:
                int(fields[0])
                int(fields[1])
            except ValueError:
                continue
            start = index + 1
            break
    if start is None:
        raise ValueError("Gaussian input has no charge/multiplicity line: " + str(path))
    atoms = []
    for line in lines[start:]:
        if not line.strip():
            break
        fields = line.split()
        if len(fields) < 4:
            raise ValueError("Gaussian coordinate row is malformed: " + str(path))
        atoms.append(str(fields[0]))
    if not atoms:
        raise ValueError("Gaussian input has no coordinates: " + str(path))
    return tuple(atoms)


def _reject_gaussian_input_resource_directives(path: Path) -> None:
    """Reject per-input resources; the scheduler owns Gaussian limits."""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("%"):
            if stripped:
                break
            continue
        key = stripped[1:].split("=", 1)[0].strip().lower()
        if key in {"mem", "nprocshared"}:
            raise ValueError(
                "Gaussian input must not contain %Mem or %NProcShared; "
                "the immutable scheduler resource resolution supplies GAUSS_MDEF "
                "and GAUSS_PDEF: "
                + str(path)
            )


def wfn_primitive_count(path: Path) -> int:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for _ in range(12):
            line = handle.readline()
            if not line:
                break
            match = re.search(r"\b([0-9]+)\s+PRIMITIVES\b", line)
            if match:
                value = int(match.group(1))
                if value > 0:
                    return value
    raise ValueError("WFN primitive count is unreadable: " + str(path))


def _quantum_evidence(
    campaign_dir: Path,
    phase_name: str,
    iteration: int,
    *,
    replacement_round: int,
    staging_dir: Optional[Path],
    pointdirs_override: Optional[Sequence[Path]] = None,
    points_evidence_override: Optional[Dict[str, Any]] = None,
    inspect_aimall_task_metadata: bool = True,
) -> Dict[str, Any]:
    staging = _staging_dir(
        campaign_dir,
        phase_name,
        int(iteration),
        replacement_round=int(replacement_round),
        staging_dir=staging_dir,
    )
    if staging is None:
        raise FileNotFoundError("quantum staging directory cannot be resolved")
    required = "input.wfn" if "AIMALL" in str(phase_name) else "input.gjf"
    if pointdirs_override is None:
        points_path, pointdirs = _strict_pointdirs(
            campaign_dir,
            staging,
            required,
        )
        points_evidence = _file_evidence(points_path)
    else:
        points_path = Path(staging) / "POINTS.txt"
        pointdirs = _strict_pointdir_sequence(
            campaign_dir,
            staging,
            required,
            pointdirs_override,
        )
        points_evidence = dict(points_evidence_override or {})
        expected_points_path = str(points_path.resolve())
        if (
            set(points_evidence) != {"path", "size", "sha256"}
            or points_evidence.get("path") != expected_points_path
            or isinstance(points_evidence.get("size"), bool)
            or not isinstance(points_evidence.get("size"), int)
            or int(points_evidence["size"]) < 0
            or not isinstance(points_evidence.get("sha256"), str)
            or len(str(points_evidence["sha256"])) != 64
        ):
            raise ValueError("prepared POINTS.txt resource evidence is invalid")
    atom_orders = []
    primitive_counts = []
    aimall_task_naat = []
    aimall_task_metadata = []
    files = []
    for pointdir in pointdirs:
        gjf = pointdir / "input.gjf"
        if gjf.is_symlink() or not gjf.is_file():
            raise FileNotFoundError(
                "input.gjf is missing from staged pointdir: " + str(pointdir)
            )
        atom_orders.append(_gjf_atom_order(gjf))
        _reject_gaussian_input_resource_directives(gjf)
        files.append(_file_evidence(gjf))
        if "AIMALL" in str(phase_name):
            wfn = pointdir / "input.wfn"
            primitive_counts.append(wfn_primitive_count(wfn))
            files.append(_file_evidence(wfn))
    if not atom_orders:
        raise ValueError("quantum staging has no readable Gaussian geometries")
    max_atoms = max(len(order) for order in atom_orders)
    evidence: Dict[str, Any] = {
        "source": "quantum_staging_points",
        "staging_root": str(Path(staging).resolve()),
        "points": points_evidence,
        "n_tasks": len(pointdirs),
        "max_n_atoms": int(max_atoms),
        "atom_orders": [list(order) for order in atom_orders],
        "inputs": files,
    }
    if primitive_counts:
        evidence["primitive_counts"] = primitive_counts
        evidence["max_n_primitives"] = max(primitive_counts)
        task_paths = [pointdir / "AIMALL_TASK.json" for pointdir in pointdirs]
        task_exists = [path.is_file() and not path.is_symlink() for path in task_paths]
        if inspect_aimall_task_metadata:
            if any(task_exists) and not all(task_exists):
                raise ValueError(
                    "AIMAll task metadata is only present for part of the staged array"
                )
        if inspect_aimall_task_metadata and all(task_exists):
            for index, task_path in enumerate(task_paths):
                try:
                    task_payload = json.loads(task_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise ValueError(
                        "AIMAll task metadata is unreadable: " + str(task_path)
                    ) from exc
                if (
                    not isinstance(task_payload, dict)
                    or isinstance(task_payload.get("schema_version"), bool)
                    or task_payload.get("schema_version") != 2
                ):
                    raise ValueError(
                        "AIMAll task metadata has an unsupported schema: "
                        + str(task_path)
                    )
                atom_count = int(task_payload.get("atom_count", -1))
                primitive_count = int(task_payload.get("primitive_count", -1))
                nproc = int(task_payload.get("nproc", -1))
                naat = int(task_payload.get("naat", -1))
                if atom_count != len(atom_orders[index]):
                    raise ValueError("AIMAll task metadata atom count has drifted")
                if task_payload.get("pointdir") != pointdirs[index].name:
                    raise ValueError("AIMAll task metadata pointdir has drifted")
                expected_atom_names = task_payload.get("expected_atom_names")
                if (
                    not isinstance(expected_atom_names, list)
                    or len(expected_atom_names) != atom_count
                    or len(expected_atom_names) != len(set(expected_atom_names))
                    or any(
                        not isinstance(name, str) or not name
                        for name in expected_atom_names
                    )
                ):
                    raise ValueError("AIMAll task atom identities are invalid")
                if primitive_count != int(primitive_counts[index]):
                    raise ValueError(
                        "AIMAll task metadata primitive count has drifted"
                    )
                if nproc <= 0 or naat <= 0 or naat > nproc or naat > atom_count:
                    raise ValueError("AIMAll task metadata worker counts are invalid")
                for binding_name in (
                    "wfn_method_receipt",
                    "gaussian_task_receipt",
                ):
                    binding = task_payload.get(binding_name)
                    if not isinstance(binding, dict):
                        raise ValueError("AIMAll task receipt binding is invalid")
                    binding_path = pointdirs[index] / str(binding.get("path") or "")
                    if (
                        binding_path.parent != pointdirs[index]
                        or binding_path.is_symlink()
                        or not binding_path.is_file()
                        or sha256_file(binding_path) != str(binding.get("sha256") or "")
                    ):
                        raise ValueError("AIMAll task receipt binding has drifted")
                if sha256_file(pointdirs[index] / "input.gjf") != str(
                    task_payload.get("gjf_sha256") or ""
                ):
                    raise ValueError("AIMAll task GJF binding has drifted")
                aimall_task_naat.append(naat)
                aimall_task_metadata.append(_file_evidence(task_path))
            evidence["aimall_task_naat"] = aimall_task_naat
            evidence["aimall_task_metadata"] = aimall_task_metadata
    acceptance = sorted(Path(staging).glob("accepted_pointdirs*.json"))
    if acceptance:
        evidence["acceptance_manifests"] = [
            _file_evidence(path) for path in acceptance
        ]
    return evidence


def prepared_quantum_resource_evidence(
    campaign_dir: Union[str, Path],
    *,
    phase_name: str,
    iteration: int,
    replacement_round: int = 0,
    staging_dir: Path,
    pointdirs: Sequence[Path],
) -> Dict[str, Any]:
    """Build resource evidence before publishing a filtered POINTS.txt."""
    encoded = (
        "\n".join(str(Path(pointdir).resolve()) for pointdir in pointdirs) + "\n"
    ).encode("utf-8")
    points_path = Path(staging_dir) / "POINTS.txt"
    points_evidence = {
        "path": str(points_path.resolve()),
        "size": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    return _quantum_evidence(
        Path(campaign_dir),
        str(phase_name),
        int(iteration),
        replacement_round=int(replacement_round),
        staging_dir=Path(staging_dir),
        pointdirs_override=pointdirs,
        points_evidence_override=points_evidence,
        inspect_aimall_task_metadata=False,
    )


def _directory_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            total += int(path.stat().st_size)
    return total


def _manifest_directory_bytes(campaign_dir: Path, root: Path) -> int:
    from ..versioning.manifest import MANIFEST_FILENAME, read_manifest

    manifest = read_manifest(root)
    total = 0
    for relative in sorted(manifest):
        path = campaign_owned_path(campaign_dir, root / relative)
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                "trained-model manifest references a missing or unsafe file: "
                + str(path)
            )
        total += int(path.stat().st_size)
    manifest_path = campaign_owned_path(campaign_dir, root / MANIFEST_FILENAME)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("trained-model directory manifest is missing or unsafe")
    return int(total + manifest_path.stat().st_size)


def _report_resource_progress(
    callback: Optional[Callable[..., None]],
    stage: str,
    **payload: Any,
) -> None:
    if callback is None:
        return
    try:
        callback(str(stage), **payload)
    except Exception:
        return


@dataclass(frozen=True)
class AriadneResourceAuthorityContext:
    campaign_dir: Path
    campaign_uid: str
    iteration: int
    replacement_round: int
    task_map: Mapping[str, Any] = field(repr=False)
    task_map_file: Path
    pool_manifest: Any = field(repr=False)
    pool_path: Path
    model_set: Any = field(repr=False)
    model_bytes: int
    validation_source: str
    authority_files: Tuple[_AriadneFileAuthority, ...] = field(repr=False)
    model_payload_bindings: Tuple[Any, ...] = field(repr=False)
    artifact_snapshot: Optional[Any] = field(default=None, repr=False, compare=False)

    @property
    def task_map_evidence(self) -> Dict[str, Any]:
        return self.authority_files[0].evidence()

    @property
    def pool_evidence(self) -> Dict[str, Any]:
        return self.authority_files[1].evidence()

    @property
    def pool_manifest_evidence(self) -> Dict[str, Any]:
        return self.authority_files[2].evidence()

    def assert_unchanged(self, *, verify_model_payloads: bool = True) -> None:
        from ..versioning.reference_data import ReferenceDataVersioning
        from ..versioning.trained_models import (
            TrainedModelVersioning,
            assert_current_model_payloads_unchanged,
        )

        if self.artifact_snapshot is not None:
            expected_references = getattr(
                self.artifact_snapshot,
                "committed_reference_data_versions",
                None,
            )
            expected_models = getattr(
                self.artifact_snapshot,
                "committed_model_versions",
                None,
            )
            if expected_references is not None and tuple(
                ReferenceDataVersioning(
                    self.campaign_dir / "QM_REFERENCE_DATA"
                ).list_committed_versions()
            ) != tuple(expected_references):
                raise ValueError(
                    "committed reference-data inventory changed before "
                    "ARIADNE submission"
                )
            if expected_models is not None and tuple(
                TrainedModelVersioning(
                    trained_models_dir(self.campaign_dir)
                ).list_committed_versions()
            ) != tuple(expected_models):
                raise ValueError(
                    "committed model inventory changed before ARIADNE submission"
                )
        version = int(self.model_set.version)
        if ReferenceDataVersioning(
            self.campaign_dir / "QM_REFERENCE_DATA"
        ).current_version() != version:
            raise ValueError(
                "current reference-data pointer changed before ARIADNE submission"
            )
        if TrainedModelVersioning(
            trained_models_dir(self.campaign_dir)
        ).current_version() != version:
            raise ValueError(
                "current trained-model pointer changed before ARIADNE submission"
            )
        for binding in self.authority_files:
            binding.assert_unchanged()
        if verify_model_payloads:
            assert_current_model_payloads_unchanged(
                self.campaign_dir,
                self.model_set,
                self.model_payload_bindings,
            )


def build_ariadne_resource_authority_context(
    campaign_dir: Union[str, Path],
    iteration: int,
    *,
    expected_campaign_uid: Optional[str] = None,
    replacement_round: int = 0,
    expected_models_version: Optional[int] = None,
    artifact_snapshot: Optional[Any] = None,
    progress_callback: Optional[Callable[..., None]] = None,
) -> AriadneResourceAuthorityContext:
    from ..acquisition.trajectory_pool import (
        POOL_MANIFEST_FILENAME,
        POOL_SUBDIR,
        POOL_XYZ_FILENAME,
        TrajectoryPoolManifest,
    )
    from ..handoff_manifests import ariadne_task_map_path
    from ..layout import active_iteration_dir
    from ..seed_identity import read_ariadne_task_map
    from ..versioning.reference_data import ReferenceDataVersioning
    from ..versioning.trained_models import (
        TrainedModelVersioning,
        _verify_current_model_payloads,
        resolve_trained_model_set,
        trained_model_set_path,
    )

    campaign = Path(campaign_dir).resolve()
    if int(replacement_round) != 0:
        raise ValueError("ARIADNE resource authority requires replacement round zero")
    iter_dir = campaign_owned_path(
        campaign,
        active_iteration_dir(campaign, int(iteration)),
    )
    task_map_file = campaign_owned_path(campaign, ariadne_task_map_path(iter_dir))
    task_map_authority = _capture_ariadne_file_authority(task_map_file)
    task_map = read_ariadne_task_map(
        iter_dir,
        expected_iteration=int(iteration),
    )
    task_map_authority.assert_unchanged()
    campaign_uid = str(task_map.get("campaign_uid") or "")
    if not campaign_uid:
        raise ValueError("ARIADNE task map has no campaign UID")
    if (
        expected_campaign_uid is not None
        and campaign_uid != str(expected_campaign_uid)
    ):
        raise ValueError("ARIADNE task-map campaign UID mismatch")
    _report_resource_progress(
        progress_callback,
        "ariadne_task_map_validation",
        completed=1,
        total=1,
        unit="checks",
        validation_step="task_map",
    )

    manifest_path = campaign_owned_path(
        campaign,
        POOL_SUBDIR / POOL_MANIFEST_FILENAME,
    )
    pool_path = campaign_owned_path(campaign, POOL_XYZ_FILENAME)
    manifest_authority = _capture_ariadne_file_authority(manifest_path)
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            pool_manifest = TrajectoryPoolManifest.from_dict(json.load(handle))
    finally:
        manifest_authority.assert_unchanged()
    if Path(pool_manifest.canonical_path).resolve() != pool_path.resolve():
        raise ValueError(
            "trajectory pool manifest canonical path does not identify pool.xyz"
        )
    pool_authority = _capture_ariadne_file_authority(pool_path)
    if pool_authority.sha256 != str(pool_manifest.sha256):
        raise ValueError("trajectory pool SHA does not match its manifest")
    if str(task_map["trajectory_sha256"]) != pool_authority.sha256:
        raise ValueError("ARIADNE task map trajectory SHA does not match the pool")
    selection_binding = task_map.get("selection_manifest")
    if not isinstance(selection_binding, Mapping):
        raise ValueError("ARIADNE task map selection binding is missing")
    selection_path = campaign_owned_path(
        campaign,
        iter_dir / str(selection_binding.get("path") or ""),
    )
    selection_authority = _capture_ariadne_file_authority(
        selection_path,
        known_sha256=str(selection_binding.get("sha256") or ""),
    )
    _report_resource_progress(
        progress_callback,
        "ariadne_trajectory_pool_validation",
        completed=1,
        total=1,
        unit="checks",
        validation_step="trajectory_pool",
    )

    version = int(task_map["models_version"])
    if expected_models_version is not None and version != int(
        expected_models_version
    ):
        raise ValueError("ARIADNE task-map model version differs from campaign state")
    if ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA").current_version() != version:
        raise ValueError("current reference-data pointer does not match ARIADNE models")
    if TrainedModelVersioning(trained_models_dir(campaign)).current_version() != version:
        raise ValueError("current trained-model pointer does not match ARIADNE models")
    if artifact_snapshot is None:
        model_set = resolve_trained_model_set(
            campaign,
            version,
            verification="metadata",
        )
        validation_source = "legacy_full_chain"
    else:
        reference_view = artifact_snapshot.reference_view(version)
        model_set = artifact_snapshot.model_set(version)
        if str(reference_view.campaign_uid) != campaign_uid:
            raise ValueError("ARIADNE reference-data campaign UID mismatch")
        if (
            int(model_set.reference_data_version) != version
            or str(model_set.reference_data_head_manifest_sha256)
            != str(reference_view.head_manifest_sha256)
            or str(model_set.reference_data_view_sha256)
            != str(reference_view.cumulative_view_sha256)
        ):
            raise ValueError("ARIADNE model/reference authority mismatch")
        validation_source = "snapshot_current_delta"
    if str(model_set.campaign_uid) != campaign_uid:
        raise ValueError("ARIADNE trained-model campaign UID mismatch")
    if str(model_set.head_manifest_sha256) != str(
        task_map["model_manifest_sha256"]
    ):
        raise ValueError("ARIADNE task map model-manifest SHA mismatch")
    if str(model_set.model_set_sha256) != str(task_map["model_set_sha256"]):
        raise ValueError("ARIADNE task map scientific model-set SHA mismatch")
    model_payload_bindings = _verify_current_model_payloads(campaign, model_set)
    model_manifest_authority = _capture_ariadne_file_authority(
        trained_model_set_path(model_set.root),
        known_sha256=str(model_set.head_manifest_sha256),
    )
    try:
        model_bytes = _manifest_directory_bytes(campaign, model_set.root)
    except FileNotFoundError:
        if artifact_snapshot is not None:
            raise
        model_bytes = _directory_bytes(model_set.root)
    _report_resource_progress(
        progress_callback,
        "ariadne_current_model_validation",
        completed=1,
        total=1,
        unit="checks",
        validation_step="current_model",
        validation_source=validation_source,
    )
    context = AriadneResourceAuthorityContext(
        campaign_dir=campaign,
        campaign_uid=campaign_uid,
        iteration=int(iteration),
        replacement_round=int(replacement_round),
        task_map=dict(task_map),
        task_map_file=task_map_file.resolve(),
        pool_manifest=pool_manifest,
        pool_path=pool_path.resolve(),
        model_set=model_set,
        model_bytes=int(model_bytes),
        validation_source=validation_source,
        authority_files=(
            task_map_authority,
            pool_authority,
            manifest_authority,
            selection_authority,
            model_manifest_authority,
        ),
        model_payload_bindings=tuple(model_payload_bindings),
        artifact_snapshot=artifact_snapshot,
    )
    context.assert_unchanged(verify_model_payloads=False)
    return context


def _ariadne_evidence(
    campaign_dir: Path,
    iteration: int,
    config: Any,
    *,
    progress_callback: Optional[Callable[..., None]] = None,
    authority_context: Optional[AriadneResourceAuthorityContext] = None,
) -> Dict[str, Any]:
    if authority_context is None:
        from ..acquisition.trajectory_pool import (
            POOL_MANIFEST_FILENAME,
            POOL_SUBDIR,
            POOL_XYZ_FILENAME,
            TrajectoryPoolManifest,
        )
        from ..handoff_manifests import ariadne_task_map_path
        from ..layout import active_iteration_dir
        from ..seed_identity import read_ariadne_task_map
        from ..versioning.trained_models import resolve_trained_model_set

        iter_dir = campaign_owned_path(
            campaign_dir,
            active_iteration_dir(campaign_dir, int(iteration)),
        )
        task_map_file = ariadne_task_map_path(iter_dir)
        if task_map_file.is_symlink() or not task_map_file.is_file():
            raise FileNotFoundError(
                "ARIADNE task map is not yet available: " + str(task_map_file)
            )
        task_map = read_ariadne_task_map(
            iter_dir,
            expected_iteration=int(iteration),
        )
        _report_resource_progress(
            progress_callback,
            "ariadne_task_map_validation",
            completed=1,
            total=1,
            unit="checks",
            validation_step="task_map",
        )
        manifest_path = campaign_owned_path(
            campaign_dir,
            POOL_SUBDIR / POOL_MANIFEST_FILENAME,
        )
        pool_path = campaign_owned_path(campaign_dir, POOL_XYZ_FILENAME)
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise FileNotFoundError(
                "trajectory pool manifest is not a regular file: "
                + str(manifest_path)
            )
        if pool_path.is_symlink() or not pool_path.is_file():
            raise FileNotFoundError(
                "canonical trajectory pool is not a regular file: "
                + str(pool_path)
            )
        with open(manifest_path, "r", encoding="utf-8") as handle:
            pool_manifest = TrajectoryPoolManifest.from_dict(json.load(handle))
        if Path(pool_manifest.canonical_path).resolve() != pool_path.resolve():
            raise ValueError(
                "trajectory pool manifest canonical path does not identify pool.xyz"
            )
        pool_sha256 = sha256_file(pool_path)
        if pool_sha256 != str(pool_manifest.sha256):
            raise ValueError("trajectory pool SHA does not match its manifest")
        if str(task_map["trajectory_sha256"]) != pool_sha256:
            raise ValueError(
                "ARIADNE task map trajectory SHA does not match the pool"
            )
        _report_resource_progress(
            progress_callback,
            "ariadne_trajectory_pool_validation",
            completed=1,
            total=1,
            unit="checks",
            validation_step="trajectory_pool",
        )
        version = int(task_map["models_version"])
        model_set = resolve_trained_model_set(
            campaign_dir,
            version,
            verification="metadata",
        )
        if str(model_set.head_manifest_sha256) != str(
            task_map["model_manifest_sha256"]
        ):
            raise ValueError("ARIADNE task map model-manifest SHA mismatch")
        if str(model_set.model_set_sha256) != str(task_map["model_set_sha256"]):
            raise ValueError("ARIADNE task map scientific model-set SHA mismatch")
        _report_resource_progress(
            progress_callback,
            "ariadne_current_model_validation",
            completed=1,
            total=1,
            unit="checks",
            validation_step="current_model",
            validation_source="legacy_full_chain",
        )
        task_map_evidence = _file_evidence(task_map_file)
        pool_evidence = {
            "path": str(pool_path.resolve()),
            "size": int(pool_path.stat().st_size),
            "sha256": pool_sha256,
        }
        pool_manifest_evidence = _file_evidence(manifest_path)
        model_bytes = _directory_bytes(model_set.root)
        model_authority_source = "legacy_full_chain"
    else:
        context = authority_context
        if context.campaign_dir != Path(campaign_dir).resolve():
            raise ValueError("ARIADNE resource authority campaign changed")
        if int(context.iteration) != int(iteration):
            raise ValueError("ARIADNE resource authority iteration changed")
        context.assert_unchanged(verify_model_payloads=False)
        task_map = context.task_map
        pool_manifest = context.pool_manifest
        model_set = context.model_set
        pool_path = context.pool_path
        version = int(model_set.version)
        task_map_evidence = context.task_map_evidence
        pool_evidence = context.pool_evidence
        pool_manifest_evidence = context.pool_manifest_evidence
        model_bytes = int(context.model_bytes)
        model_authority_source = str(context.validation_source)
    tasks = list(task_map["tasks"])
    if not tasks:
        raise ValueError("ARIADNE task map contains no tasks")
    n_pool_frames = int(pool_manifest.n_frames)
    for task in tasks:
        raw_index = task.get("pool_row_index_zero_based")
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError("ARIADNE task-map pool row must be an exact integer")
        if raw_index < 0 or raw_index >= n_pool_frames:
            raise ValueError("ARIADNE task-map pool row is outside the pool")
    n_atoms = int(pool_manifest.natoms)
    mode = str(getattr(config.acquisition.gradient, "mode", "active_fd"))
    if mode == "active_fd":
        configured_max = int(config.acquisition.subspace.max_subspace_dim)
        dimension = max(1, min(configured_max, 3 * n_atoms))
        dimensions = [dimension for _task in tasks]
        _report_resource_progress(
            progress_callback,
            "ariadne_resource_bound",
            completed=0,
            total=len(tasks),
            unit="tasks",
            gradient_dimension=int(dimension),
        )
        _report_resource_progress(
            progress_callback,
            "ariadne_resource_bound",
            completed=len(tasks),
            total=len(tasks),
            unit="tasks",
            gradient_dimension=int(dimension),
        )
        dimension_source = "configured_safe_upper_bound"
    else:
        dimensions = [int(3 * n_atoms) for _task in tasks]
        dimension = int(3 * n_atoms)
        dimension_source = "exact_cartesian_dimension"
    decoded_coordinate_bytes = int(n_pool_frames * n_atoms * 3 * 8)
    # The current runner eagerly creates Python Atoms/Atom objects for the
    # complete pool in every array member.  This conservative object allowance
    # is explicit and can later be replaced by measured telemetry or a bounded
    # indexed pool implementation.
    decoded_object_allowance_bytes = int(
        n_pool_frames * 1024 + n_pool_frames * n_atoms * 1024
    )

    return {
        "source": "ariadne_task_map_and_model_set",
        "task_map": task_map_evidence,
        "pool": pool_evidence,
        "pool_manifest": pool_manifest_evidence,
        "pool_n_frames": n_pool_frames,
        "pool_file_bytes": int(pool_path.stat().st_size),
        "decoded_coordinate_bytes": decoded_coordinate_bytes,
        "decoded_object_allowance_bytes": decoded_object_allowance_bytes,
        "decoded_pool_estimate_bytes": int(
            pool_path.stat().st_size
            + decoded_coordinate_bytes
            + decoded_object_allowance_bytes
        ),
        "n_tasks": int(task_map["n_tasks"]),
        "n_atoms": int(n_atoms),
        "gradient_dimension": int(dimension),
        "models_version": version,
        "model_manifest_sha256": str(model_set.head_manifest_sha256),
        "model_set_sha256": str(model_set.model_set_sha256),
        "model_bytes": int(model_bytes),
        "gradient_dimensions": dimensions,
        "gradient_dimension_source": dimension_source,
        "model_authority_source": model_authority_source,
    }


def _csv_feature_count(path: Path) -> int:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("FEREBUS dataset CSV is empty: " + str(path)) from exc
    if len(header) < 2:
        raise ValueError("FEREBUS dataset CSV has no feature/property split")
    return len(header) - 1


def _ferebus_evidence(campaign_dir: Path) -> Dict[str, Any]:
    from .input_staging import read_ferebus_manifest, resolve_ferebus_task_path

    staging = campaign_owned_path(
        campaign_dir,
        trained_models_dir(campaign_dir) / "iteration-staging",
    )
    payload = read_ferebus_manifest(staging, verify_dataset_files=True)
    maxima = {"train": 0, "int_val": 0, "ext_val": 0}
    max_features = 0
    datasets = []
    task_dimensions = []
    runtime_files = [staging / "commands", staging / "list.txt"]
    generated_script = staging / "runFerebus.sh"
    runtime_prepared = (
        generated_script.is_file()
        and not generated_script.is_symlink()
        and all(path.is_file() and not path.is_symlink() for path in runtime_files)
    )
    for task in payload["tasks"]:
        counts = task["row_counts"]
        for split in maxima:
            maxima[split] = max(maxima[split], int(counts[split]))
        task_feature_counts = []
        for split, field_name in (
            ("train", "training_csv"),
            ("int_val", "int_validation_csv"),
            ("ext_val", "ext_validation_csv"),
        ):
            expected = resolve_ferebus_task_path(staging, task[field_name], field_name)
            if not expected.is_file():
                if runtime_prepared:
                    raise FileNotFoundError(
                        "prepared FEREBUS runtime dataset is missing: "
                        + str(expected)
                    )
                expected = resolve_ferebus_task_path(
                    staging,
                    str(task["property"]) + "/" + Path(str(task[field_name])).name,
                    field_name + "_pre_pyferebus",
                )
            feature_count = _csv_feature_count(expected)
            task_feature_counts.append(feature_count)
            max_features = max(max_features, feature_count)
            datasets.append(_file_evidence(expected))
        if len(set(task_feature_counts)) != 1:
            raise ValueError(
                "FEREBUS train/internal/external feature counts disagree for task "
                + str(task.get("task_index"))
            )
        task_dimensions.append(
            {
                "task_index": int(task["task_index"]),
                "property": str(task["property"]),
                "atom": str(task["atom"]),
                "n_train": int(counts["train"]),
                "n_internal": int(counts["int_val"]),
                "n_external": int(counts["ext_val"]),
                "n_total": int(counts["train"])
                + int(counts["int_val"])
                + int(counts["ext_val"]),
                "n_features": int(task_feature_counts[0]),
            }
        )
    total = sum(maxima.values())
    if total <= 0 or max_features <= 0:
        raise ValueError("FEREBUS resource dimensions are empty")
    manifest_path = staging / "FEREBUS_TASKS.json"
    evidence = {
        "source": "ferebus_task_manifest_and_datasets",
        "manifest": _file_evidence(manifest_path),
        "n_tasks": int(payload["n_tasks"]),
        "reference_data_version": int(payload["reference_data_version"]),
        "max_train_rows": maxima["train"],
        "max_internal_rows": maxima["int_val"],
        "max_external_rows": maxima["ext_val"],
        "max_total_rows": total,
        "max_features": max_features,
        "task_dimensions": task_dimensions,
        "datasets": datasets,
        "runtime_tree_prepared": bool(runtime_prepared),
    }
    if runtime_prepared:
        evidence["runtime_files"] = [_file_evidence(path) for path in runtime_files]
        generated_configs = []
        for task in payload["tasks"]:
            generated = task.get("generated_config")
            if not isinstance(generated, dict):
                raise ValueError("prepared FEREBUS task has no generated config binding")
            config_path = resolve_ferebus_task_path(
                staging,
                str(generated.get("path") or ""),
                "generated_config",
            )
            generated_configs.append(_file_evidence(config_path))
        evidence["generated_configs"] = generated_configs
    return evidence


def collect_resource_evidence(
    *,
    phase_name: str,
    config: Any,
    campaign_dir: Optional[Path],
    iteration: int,
    replacement_round: int = 0,
    staging_dir: Optional[Path] = None,
    n_atoms_override: Optional[int] = None,
    require_evidence: bool = True,
    progress_callback: Optional[Callable[..., None]] = None,
    ariadne_authority_context: Optional[
        AriadneResourceAuthorityContext
    ] = None,
) -> Dict[str, Any]:
    backend = backend_for_phase(phase_name)
    if campaign_dir is None:
        if require_evidence:
            raise ResourceEvidenceUnavailable(phase_name, "campaign directory is absent")
        return _synthetic_evidence(
            backend,
            phase_name,
            n_atoms_override=n_atoms_override,
            config=config,
        )
    try:
        if backend == "diversity":
            return (
                _pool_evidence(campaign_dir)
                if phase_name == "PHASE_A_DIVERSITY"
                else _phase_b_evidence_with_config(
                    campaign_dir, int(iteration), config
                )
            )
        if backend in {"gaussian", "aimall"}:
            return _quantum_evidence(
                campaign_dir,
                phase_name,
                int(iteration),
                replacement_round=int(replacement_round),
                staging_dir=staging_dir,
            )
        if backend == "ariadne":
            return _ariadne_evidence(
                campaign_dir,
                int(iteration),
                config,
                progress_callback=progress_callback,
                authority_context=ariadne_authority_context,
            )
        if backend == "ferebus":
            return _ferebus_evidence(campaign_dir)
    except (ResourceEvidenceUnavailable, ResourceEvidenceInvalid):
        raise
    except FileNotFoundError as exc:
        if require_evidence or campaign_dir is not None:
            raise ResourceEvidenceUnavailable(
                phase_name, type(exc).__name__ + ": " + str(exc)
            ) from exc
    except Exception as exc:
        if require_evidence or campaign_dir is not None:
            raise ResourceEvidenceInvalid(
                phase_name, type(exc).__name__ + ": " + str(exc)
            ) from exc
    return _synthetic_evidence(
        backend,
        phase_name,
        n_atoms_override=n_atoms_override,
        config=config,
    )


def _submitted_array_evidence(
    evidence: Dict[str, Any],
    backend: str,
    task_ids: Sequence[int],
) -> Dict[str, Any]:
    """Restrict task-dependent dimensions to a validated dense retry map."""
    logical_total = int(evidence.get("n_tasks", -1))
    if logical_total <= 0:
        raise BackendSubmissionError(
            "resource evidence has no logical task count for partial-array retry"
        )
    parsed: List[int] = []
    for raw in task_ids:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise BackendSubmissionError(
                "partial-array logical task IDs must be non-negative integers"
            )
        task_id = raw
        if task_id < 0 or task_id >= logical_total:
            raise BackendSubmissionError(
                "partial-array logical task ID "
                + str(task_id)
                + " is outside producer evidence range 0.."
                + str(logical_total - 1)
            )
        parsed.append(task_id)
    if not parsed:
        raise BackendSubmissionError("partial-array retry contains no tasks")
    if len(set(parsed)) != len(parsed):
        raise BackendSubmissionError(
            "partial-array retry contains duplicate logical task IDs"
        )

    selected = dict(evidence)
    selected["logical_n_tasks"] = logical_total
    selected["submitted_logical_task_ids"] = parsed
    selected["n_tasks"] = len(parsed)
    if backend in {"gaussian", "aimall"}:
        atom_orders = list(evidence.get("atom_orders") or [])
        if len(atom_orders) != logical_total:
            raise BackendSubmissionError(
                "quantum resource evidence atom-order count does not match logical tasks"
            )
        selected_orders = [atom_orders[index] for index in parsed]
        selected["atom_orders"] = selected_orders
        selected["max_n_atoms"] = max(len(order) for order in selected_orders)
        primitive_counts = list(evidence.get("primitive_counts") or [])
        if backend == "aimall":
            if len(primitive_counts) != logical_total:
                raise BackendSubmissionError(
                    "AIMAll primitive-count evidence does not match logical tasks"
                )
            selected_primitives = [primitive_counts[index] for index in parsed]
            selected["primitive_counts"] = selected_primitives
            selected["max_n_primitives"] = max(selected_primitives)
            frozen_naat = list(evidence.get("aimall_task_naat") or [])
            if frozen_naat:
                if len(frozen_naat) != logical_total:
                    raise BackendSubmissionError(
                        "AIMAll frozen naat evidence does not match logical tasks"
                    )
                selected["aimall_task_naat"] = [
                    int(frozen_naat[index]) for index in parsed
                ]
    elif backend == "ariadne":
        dimensions = list(evidence.get("gradient_dimensions") or [])
        if len(dimensions) != logical_total:
            raise BackendSubmissionError(
                "ARIADNE gradient-dimension evidence does not match logical tasks"
            )
        selected_dimensions = [int(dimensions[index]) for index in parsed]
        selected["gradient_dimensions"] = selected_dimensions
        selected["gradient_dimension"] = max(selected_dimensions)
    return selected


def _staging_dir(
    campaign_dir: Optional[Path],
    phase_name: str,
    iteration: int,
    *,
    replacement_round: int = 0,
    staging_dir: Optional[Path] = None,
) -> Optional[Path]:
    if staging_dir is not None:
        return Path(staging_dir)
    if campaign_dir is None:
        return None
    staging = staging_phase_dir(campaign_dir, phase_name, int(iteration))
    if "REPLACEMENT" in str(phase_name):
        if int(replacement_round) <= 0:
            return None
        from ..replacement_sampling import replacement_round_dir

        staging = replacement_round_dir(
            campaign_dir,
            context=(
                "bootstrap"
                if str(phase_name).startswith("INITIAL_")
                else "active"
            ),
            iteration=int(iteration),
            replacement_round=int(replacement_round),
        )
    return staging


def _basis_factor(config: Any) -> float:
    basis = str(getattr(config.gaussian, "basis_set", "")).lower()
    factor = 1.0
    if "sto" in basis or "minimal" in basis:
        factor = 0.5
    elif "6-31" in basis or "6-311" in basis:
        factor = 1.0
    elif "cc-pvdz" in basis or "def2-svp" in basis:
        factor = 1.3
    elif "cc-pvtz" in basis or "def2-tz" in basis:
        factor = 2.0
    if "aug" in basis or "+" in basis:
        factor *= 1.3
    return factor


def _method_factor(config: Any) -> float:
    method = (
        str(getattr(config.gaussian, "method", ""))
        + " "
        + " ".join(str(value) for value in config.gaussian.extra_route_keywords)
    ).lower()
    if "mp2" in method or "ccsd" in method or "casscf" in method:
        return 1.5
    return 1.0


def _aimall_regime(n_atoms: int, n_primitives: Optional[int]) -> str:
    primitives = n_primitives if n_primitives is not None else n_atoms * 40
    if n_atoms <= 12 and primitives <= 400:
        return "small"
    if n_atoms <= 40 and primitives <= 1500:
        return "medium"
    return "large"


def _aimall_task_dimensions(evidence: Dict[str, Any]) -> List[Tuple[int, int]]:
    orders = list(evidence.get("atom_orders") or [])
    primitives = list(evidence.get("primitive_counts") or [])
    if orders and len(orders) == len(primitives):
        return [
            (len(order), int(primitive_count))
            for order, primitive_count in zip(orders, primitives)
        ]
    return [
        (
            int(evidence["max_n_atoms"]),
            int(
                evidence.get("max_n_primitives")
                or int(evidence["max_n_atoms"]) * 40
            ),
        )
    ]


def _aimall_task_demands(
    config: Any,
    scientific_cpus: int,
    evidence: Dict[str, Any],
) -> List[Dict[str, Any]]:
    demands = []
    dimensions = _aimall_task_dimensions(evidence)
    frozen_naat = list(evidence.get("aimall_task_naat") or [])
    if frozen_naat and len(frozen_naat) != len(dimensions):
        raise BackendSubmissionError(
            "AIMAll frozen naat evidence does not cover every staged task"
        )
    for index, (n_atoms, n_primitives) in enumerate(dimensions):
        regime = _aimall_regime(n_atoms, n_primitives)
        per_atom_gb = {"small": 2.4, "medium": 4.0, "large": 8.0}[regime]
        naat = (
            int(frozen_naat[index])
            if frozen_naat
            else resolve_aimall_naat(
                config,
                int(scientific_cpus),
                int(n_atoms),
                int(n_primitives),
            )
        )
        if naat <= 0 or naat > int(scientific_cpus) or naat > int(n_atoms):
            raise BackendSubmissionError(
                "AIMAll frozen naat="
                + str(naat)
                + " exceeds the scientific CPU or atom count for a staged task"
            )
        demands.append(
            {
                "n_atoms": int(n_atoms),
                "n_primitives": int(n_primitives),
                "regime": regime,
                "naat": int(naat),
                "unprotected_memory_gb": 1.0 + float(naat) * per_atom_gb,
            }
        )
    return demands


def _estimate_backend_memory_gb(
    backend: str,
    phase_name: str,
    config: Any,
    campaign_dir: Optional[Path],
    iteration: int,
    candidate_cpus: int,
    partition_gb: float,
    replacement_round: int = 0,
    staging_dir: Optional[Path] = None,
    n_atoms_override: Optional[int] = None,
    evidence: Optional[Dict[str, Any]] = None,
    active_workers: Optional[int] = None,
) -> Tuple[float, str, Dict[str, Any]]:
    extra: Dict[str, Any] = {}
    evidence = dict(evidence or {})
    safety = float(
        getattr(config.resources, "memory_estimate_safety_factor", 1.25)
    )
    if backend == "diversity":
        n = int(evidence["n_frames"])
        pairs = int(n * (n - 1) // 2)
        store_bytes = int(8 * pairs)
        store_gb = float(store_bytes) / (1024.0 ** 3)
        workers = max(1, int(active_workers or candidate_cpus))
        descriptor = str(getattr(getattr(config, "phase_b", object()), "descriptor", "rmsd_massweight"))
        store_mode = str(evidence.get("distance_store_mode") or "memory")
        resident_store_gb = store_gb if store_mode == "memory" else min(store_gb, 0.25)
        raw_gb = 1.0 + resident_store_gb + 0.25 * float(workers)
        total_gb = raw_gb * safety
        extra.update({
            "n_frames": int(n),
            "n_pairs": int(pairs),
            "descriptor": descriptor,
            "condensed_store_bytes": store_bytes,
            "distance_store_mode": store_mode,
            "active_workers": workers,
            "unprotected_total_memory_gb": raw_gb,
        })
        return total_gb, "diversity_condensed_distance_store", extra
    if backend == "gaussian":
        total_gb = max(float(candidate_cpus) * float(partition_gb), 1.0)
        return total_gb, "gaussian_partition_memory_for_gauss_mdef", extra
    if backend == "aimall":
        demands = _aimall_task_demands(config, candidate_cpus, evidence)
        most_demanding = max(
            demands,
            key=lambda item: (
                float(item["unprotected_memory_gb"]),
                int(item["n_primitives"]),
                int(item["n_atoms"]),
            ),
        )
        raw_gb = float(most_demanding["unprotected_memory_gb"])
        total_gb = raw_gb * safety
        extra.update({
            "n_atoms": int(most_demanding["n_atoms"]),
            "n_primitives": int(most_demanding["n_primitives"]),
            "aimall_regime": str(most_demanding["regime"]),
            "naat_resolved": int(most_demanding["naat"]),
            "active_workers": max(int(item["naat"]) for item in demands),
            "aimall_task_naat": [int(item["naat"]) for item in demands],
            "aimall_task_demands": demands,
            "unprotected_total_memory_gb": raw_gb,
        })
        return total_gb, "aimall_concurrent_atomic_integrations", extra
    if backend == "ariadne":
        workers = max(1, int(active_workers or candidate_cpus))
        model_bytes = int(evidence["model_bytes"])
        model_size = float(model_bytes) / (1024.0 ** 3)
        decoded_pool_bytes = int(evidence.get("decoded_pool_estimate_bytes", 0))
        decoded_pool_gb = float(decoded_pool_bytes) / (1024.0 ** 3)
        raw_gb = (
            1.5
            + 4.0 * model_size
            + decoded_pool_gb
            + 0.5 * float(workers)
        )
        total_gb = raw_gb * safety
        extra.update({
            "model_bytes": model_bytes,
            "model_dir_size_gb": float(model_size),
            "active_workers": int(workers),
            "gradient_dimension": int(evidence["gradient_dimension"]),
            "decoded_pool_estimate_bytes": decoded_pool_bytes,
            "decoded_pool_estimate_gb": decoded_pool_gb,
            "unprotected_total_memory_gb": raw_gb,
        })
        return total_gb, "ariadne_gradient_worker_model_memory", extra
    if backend == "ferebus":
        nagents = int(getattr(config.ferebus, "nagents", 20))
        dimensions = list(evidence.get("task_dimensions") or [])
        if not dimensions:
            dimensions = [
                {
                    "n_train": int(evidence["max_train_rows"]),
                    "n_internal": int(evidence["max_internal_rows"]),
                    "n_external": int(evidence["max_external_rows"]),
                    "n_total": int(evidence["max_total_rows"]),
                    "n_features": int(evidence["max_features"]),
                }
            ]
        evaluated = []
        omp_threads = max(1, int(active_workers or candidate_cpus))
        batch_size = 100
        for item in dimensions:
            n_train = int(item["n_train"])
            n_internal = int(item["n_internal"])
            n_external = int(item["n_external"])
            n_total = int(item["n_total"])
            features = int(item["n_features"])
            distance_tensor_bytes = 8 * features * n_train * n_train
            distance_construction_peak_bytes = 2 * distance_tensor_bytes
            threaded_estimator_bytes = 8 * omp_threads * (
                2 * n_train * n_train
                + n_train * n_internal
                + n_train * batch_size
                + 8 * n_train
                + 4 * n_internal
                + 4 * batch_size
            )
            validation_kernel_bytes = 8 * (
                n_train * n_internal + n_train * n_external
            )
            dataset_bytes = 8 * n_total * (features + 1)
            training_peak_bytes = (
                distance_tensor_bytes
                + threaded_estimator_bytes
                + validation_kernel_bytes
                + dataset_bytes
            )
            working_peak_bytes = max(
                distance_construction_peak_bytes,
                training_peak_bytes,
            )
            components = {
                "distance_tensor_bytes": int(distance_tensor_bytes),
                "distance_construction_peak_bytes": int(
                    distance_construction_peak_bytes
                ),
                "threaded_estimator_bytes": int(threaded_estimator_bytes),
                "validation_kernel_bytes": int(validation_kernel_bytes),
                "dataset_bytes": int(dataset_bytes),
                "working_peak_bytes": int(working_peak_bytes),
                "omp_threads": int(omp_threads),
                "batch_size": int(batch_size),
            }
            evaluated.append((working_peak_bytes, item, components))
        working_peak_bytes, demanding, components = max(
            evaluated,
            key=lambda value: (int(value[0]), int(value[1]["n_train"])),
        )
        n_train = int(demanding["n_train"])
        n_internal = int(demanding["n_internal"])
        n_external = int(demanding["n_external"])
        n_total = int(demanding["n_total"])
        features = int(demanding["n_features"])
        raw_gb = 1.0 + float(working_peak_bytes) / (1024.0 ** 3)
        total_gb = raw_gb * safety
        extra.update({
            "n_train": n_train,
            "n_internal": n_internal,
            "n_external": n_external,
            "n_total": n_total,
            "n_features": features,
            "ferebus_memory_components": dict(components),
            "working_peak_bytes": int(working_peak_bytes),
            "most_demanding_task": dict(demanding),
            "active_workers": int(nagents),
            "unprotected_total_memory_gb": raw_gb,
        })
        return total_gb, "ferebus_source_allocation_peak", extra
    return 1.0, "default_minimal_backend_memory", extra


def resolve_aimall_naat(
    config: Any,
    resolved_cpus: int,
    n_atoms: int,
    n_primitives: Optional[int] = None,
) -> int:
    raw = getattr(getattr(config, "aimall", object()), "naat", "auto")
    if not (isinstance(raw, str) and raw.strip().lower() == "auto"):
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise BackendSubmissionError(
                "aimall.naat must be 'auto' or an exact positive integer"
            )
        if raw <= 0:
            raise BackendSubmissionError("aimall.naat must be > 0")
        if raw > int(n_atoms):
            raise BackendSubmissionError(
                "aimall.naat="
                + str(raw)
                + " exceeds staged atom count "
                + str(int(n_atoms))
            )
        if raw > int(resolved_cpus):
            raise BackendSubmissionError(
                "aimall.naat="
                + str(raw)
                + " exceeds resolved scientific CPUs "
                + str(int(resolved_cpus))
            )
        return raw
    regime = _aimall_regime(int(n_atoms), n_primitives)
    if regime == "small":
        return max(1, min(int(n_atoms), int(resolved_cpus)))
    if regime == "medium":
        return max(1, min(int(n_atoms), int(resolved_cpus) // 2))
    return max(1, min(int(n_atoms), int(resolved_cpus) // 4))


def _auto_cpu_target(
    backend: str,
    phase_name: str,
    config: Any,
    campaign_dir: Optional[Path],
    iteration: int,
    partition_min: int,
    partition_max: int,
    partition_gb: float,
    replacement_round: int = 0,
    staging_dir: Optional[Path] = None,
    n_atoms_override: Optional[int] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> Tuple[int, str, Dict[str, Any], float, str]:
    extra: Dict[str, Any] = {}
    evidence = dict(evidence or {})
    if backend == "diversity":
        n_frames = int(evidence["n_frames"])
        pairs = int(n_frames * (n_frames - 1) // 2)
        target_pairs = int(config.resources.diversity.target_pairs_per_worker)
        wanted = max(1, int(math.ceil(float(pairs) / float(target_pairs))))
        active = min(
            wanted,
            int(config.resources.diversity.auto_max_workers),
            int(partition_max),
        )
        target = max(partition_min, active)
        extra.update({
            "n_frames": n_frames,
            "n_pairs": pairs,
            "active_workers": active,
            "worker_target_before_caps": wanted,
        })
        return target, "diversity_pairs_per_worker", extra, 0.0, "diversity_condensed_distance_store"
    if backend == "gaussian":
        n_atoms = int(evidence["max_n_atoms"])
        weighted = float(n_atoms) * _basis_factor(config) * _method_factor(config)
        if weighted <= 30.0:
            target = partition_min
        elif weighted <= 80.0:
            target = 4
        elif weighted <= 160.0:
            target = 8
        else:
            target = 16
        extra.update({"n_atoms": int(n_atoms), "gaussian_weighted_size": float(weighted)})
        return min(max(target, partition_min), partition_max), "gaussian_size_basis_throughput", extra, 0.0, "gaussian_partition_memory_for_gauss_mdef"
    if backend == "aimall":
        n_atoms = int(evidence["max_n_atoms"])
        n_primitives = int(evidence.get("max_n_primitives") or n_atoms * 40)
        regime = _aimall_regime(n_atoms, n_primitives)
        if regime == "small":
            target = min(n_atoms, 8)
        elif regime == "medium":
            target = 12
        else:
            target = 16
        if not (isinstance(getattr(config.aimall, "naat", "auto"), str) and str(getattr(config.aimall, "naat")).strip().lower() == "auto"):
            target = max(target, int(config.aimall.naat))
        frozen_naat = list(evidence.get("aimall_task_naat") or [])
        if frozen_naat:
            frozen_max = max(int(value) for value in frozen_naat)
            if frozen_max > partition_max:
                raise BackendSubmissionError(
                    "AIMAll frozen naat="
                    + str(frozen_max)
                    + " exceeds partition maximum "
                    + str(partition_max)
                )
            target = max(target, frozen_max)
        target = min(max(target, partition_min), partition_max)
        extra.update({"n_atoms": int(n_atoms), "n_primitives": n_primitives, "aimall_regime": regime})
        active = resolve_aimall_naat(config, target, n_atoms, n_primitives)
        extra["active_workers"] = int(active)
        return target, "aimall_wavefunction_size_parallel_atoms", extra, 0.0, "aimall_concurrent_atomic_integrations"
    if backend == "ariadne":
        grad_backend = str(getattr(config.resources, "gradient_parallel_backend", "process"))
        if grad_backend == "serial":
            target = partition_min
            reason = "ariadne_serial_gradient_backend"
        else:
            dim = int(evidence["gradient_dimension"])
            target = dim
            reason = "ariadne_active_fd_direction_workers"
        if target > partition_max:
            extra["cpu_cap_warning"] = "auto target capped at partition maximum"
        return min(max(target, partition_min), partition_max), reason, extra, 0.0, "ariadne_gradient_worker_model_memory"
    if backend == "ferebus":
        target = int(getattr(config.ferebus, "nagents", 20))
        if target > partition_max:
            raise BackendSubmissionError(
                "resources.ferebus.cpus_per_task:auto resolves to ferebus.nagents="
                + str(target)
                + " but partition allows at most "
                + str(partition_max)
                + " cores"
            )
        return max(target, partition_min), "ferebus_grey_wolf_agents", {"nagents": int(target)}, 0.0, "ferebus_agent_kernel_training_memory"
    return partition_min, "partition_minimum", extra, 1.0, "default_minimal_backend_memory"


def resolve_phase_resources(
    *,
    phase_name: str,
    config: Any,
    partition: Optional[str] = None,
    campaign_dir: Optional[Any] = None,
    iteration: int = 0,
    array_size: Optional[int] = None,
    replacement_round: int = 0,
    staging_dir: Optional[Any] = None,
    n_atoms_override: Optional[int] = None,
    expected_models_version: Optional[int] = None,
    expected_reference_data_version: Optional[int] = None,
    submitted_task_ids: Optional[Sequence[int]] = None,
    require_evidence: bool = True,
    evidence_override: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[..., None]] = None,
    ariadne_authority_context: Optional[
        AriadneResourceAuthorityContext
    ] = None,
) -> ResolvedPhaseResources:
    resources = config.resources
    backend = backend_for_phase(phase_name)
    part = str(partition if partition is not None else resources.partition_for(phase_name))
    raw_cpu = resources.cpus_for(phase_name)
    raw_mem = resources.mem_per_cpu_for(phase_name)
    partition_min, partition_max = _partition_min_max(part)
    explicit_cpus: Optional[int] = None
    if not _is_auto(raw_cpu):
        explicit_cpus = _explicit_cpu(
            "resources." + backend + ".cpus_per_task",
            raw_cpu,
            part,
        )
    partition_gb = partition_memory_per_core_gb(part)
    campaign_path = Path(campaign_dir) if campaign_dir is not None else None
    if n_atoms_override is not None and int(n_atoms_override) <= 0:
        raise BackendSubmissionError("n_atoms_override must be > 0")
    evidence = (
        dict(evidence_override)
        if evidence_override is not None
        else collect_resource_evidence(
            phase_name=phase_name,
            config=config,
            campaign_dir=campaign_path,
            iteration=int(iteration),
            replacement_round=int(replacement_round),
            staging_dir=None if staging_dir is None else Path(staging_dir),
            n_atoms_override=n_atoms_override,
            require_evidence=bool(require_evidence),
            progress_callback=progress_callback,
            ariadne_authority_context=ariadne_authority_context,
        )
    )
    if not evidence or not isinstance(evidence.get("source"), str):
        raise BackendSubmissionError("resource evidence override is malformed")
    if submitted_task_ids is not None:
        if backend not in {"gaussian", "aimall", "ariadne", "ferebus"}:
            raise BackendSubmissionError(
                "partial-array task IDs are unsupported for backend " + backend
            )
        evidence = _submitted_array_evidence(
            evidence,
            backend,
            submitted_task_ids,
        )
    if (
        require_evidence
        and array_size is not None
        and backend in {"gaussian", "aimall", "ariadne", "ferebus"}
    ):
        evidence_tasks = int(evidence.get("n_tasks", -1))
        if evidence_tasks != int(array_size):
            raise BackendSubmissionError(
                "resource evidence task count "
                + str(evidence_tasks)
                + " does not match submitted array size "
                + str(int(array_size))
            )
    if int(evidence.get("n_tasks", 1)) <= 0:
        raise BackendSubmissionError(
            "resource evidence contains no tasks for " + str(phase_name)
        )
    _report_resource_progress(
        progress_callback,
        "resource_rules",
        completed=0,
        total=1,
        unit="steps",
    )
    if backend == "ariadne" and expected_models_version is not None:
        try:
            observed_models_version = int(evidence.get("models_version"))
        except (TypeError, ValueError) as exc:
            raise BackendSubmissionError(
                "ARIADNE resource evidence has no valid models_version"
            ) from exc
        if observed_models_version != int(expected_models_version):
            raise BackendSubmissionError(
                "ARIADNE resource evidence models_version "
                + str(observed_models_version)
                + " does not match daemon state.models_version "
                + str(int(expected_models_version))
            )
    if backend == "ferebus" and expected_reference_data_version is not None:
        try:
            observed_reference_data_version = int(
                evidence.get("reference_data_version")
            )
        except (TypeError, ValueError) as exc:
            raise BackendSubmissionError(
                "FEREBUS resource evidence has no valid reference_data_version"
            ) from exc
        if observed_reference_data_version != int(
            expected_reference_data_version
        ):
            raise BackendSubmissionError(
                "FEREBUS resource evidence reference_data_version "
                + str(observed_reference_data_version)
                + " does not match daemon state.reference_data_version "
                + str(int(expected_reference_data_version))
            )
    warnings: List[str] = []
    extra: Dict[str, Any] = {}

    if _is_auto(raw_cpu):
        cpus, cpu_reason, cpu_extra, _unused_estimate, _unused_reason = _auto_cpu_target(
            backend,
            phase_name,
            config,
            campaign_path,
            int(iteration),
            partition_min,
            partition_max,
            partition_gb,
            int(replacement_round),
            None if staging_dir is None else Path(staging_dir),
            n_atoms_override,
            evidence,
        )
        extra.update(cpu_extra)
        _validate_core_count("resources." + backend + ".cpus_per_task", int(cpus), part)
    else:
        if explicit_cpus is None:  # pragma: no cover - guarded above
            raise BackendSubmissionError("explicit CPU resolution was not initialised")
        cpus = explicit_cpus
        cpu_reason = "explicit"

    scientific_cpus = int(cpus)
    if backend == "diversity":
        pairs = int(evidence["n_frames"] * (evidence["n_frames"] - 1) // 2)
        wanted = max(
            1,
            int(math.ceil(
                float(pairs)
                / float(config.resources.diversity.target_pairs_per_worker)
            )),
        )
        active_workers = min(
            wanted,
            int(config.resources.diversity.auto_max_workers),
            int(scientific_cpus),
        )
        store_bytes = int(8 * pairs)
        if _is_auto(raw_mem):
            prospective_allocation_bytes = (
                float(scientific_cpus)
                * float(partition_gb)
                * 1024.0 ** 3
            )
        else:
            prospective_allocation_bytes = (
                float(scientific_cpus)
                * slurm_memory_mib(raw_mem)
                * 1024.0 ** 2
            )
        fraction = float(
            config.resources.diversity.in_memory_distance_store_fraction
        )
        in_memory = (
            store_bytes <= prospective_allocation_bytes
            and store_bytes <= prospective_allocation_bytes * fraction
        )
        evidence["distance_store_mode"] = "memory" if in_memory else "file"
        if not in_memory:
            required_bytes = int(math.ceil(1.25 * float(store_bytes)))
            evidence["scratch_required_bytes"] = required_bytes
            if campaign_path is not None:
                free_bytes = int(shutil.disk_usage(campaign_path).free)
                evidence["scratch_free_bytes_at_resolution"] = free_bytes
                if free_bytes < required_bytes:
                    raise BackendSubmissionError(
                        "file-backed diversity distances require "
                        + str(required_bytes)
                        + " free bytes under the campaign filesystem, but only "
                        + str(free_bytes)
                        + " are available"
                    )
    elif backend == "aimall":
        active_workers = max(
            int(item["naat"])
            for item in _aimall_task_demands(
                config,
                scientific_cpus,
                evidence,
            )
        )
    elif backend == "ariadne":
        gradient_backend = str(config.resources.gradient_parallel_backend)
        if gradient_backend == "serial":
            active_workers = 1
        else:
            active_workers = min(
                int(scientific_cpus), int(evidence["gradient_dimension"])
            )
    elif backend == "ferebus":
        active_workers = int(config.ferebus.nagents)
        if scientific_cpus < active_workers:
            raise BackendSubmissionError(
                "FEREBUS requires at least ferebus.nagents="
                + str(active_workers)
                + " allocated CPUs; resolved "
                + str(scientific_cpus)
            )
    else:
        active_workers = scientific_cpus

    estimated_total, memory_reason, mem_extra = _estimate_backend_memory_gb(
        backend,
        phase_name,
        config,
        campaign_path,
        int(iteration),
        int(scientific_cpus),
        float(partition_gb),
        int(replacement_round),
        None if staging_dir is None else Path(staging_dir),
        n_atoms_override,
        evidence,
        active_workers,
    )
    extra.update(mem_extra)
    for key in (
        "scratch_required_bytes",
        "scratch_free_bytes_at_resolution",
    ):
        if key in evidence:
            extra[key] = int(evidence[key])

    memory_only_cpus = max(0, int(scientific_cpus) - int(active_workers))
    if (
        backend != "gaussian"
        and _is_auto(raw_cpu)
        and float(estimated_total) > float(cpus) * float(partition_gb)
    ):
        needed = int(math.ceil(float(estimated_total) / float(partition_gb)))
        cpus = min(partition_max, max(int(cpus), needed))
        memory_only_cpus = max(0, int(cpus) - int(active_workers))

    if _is_auto(raw_mem):
        if backend == "gaussian":
            mem_gb = float(partition_gb)
            memory_reason = "gaussian_partition_memory_for_gauss_mdef"
            estimated_total = float(cpus) * mem_gb
        else:
            mem_gb = max(1.0, math.ceil(float(estimated_total) / max(float(cpus), 1.0)))
            if mem_gb > partition_gb + 1.0e-9:
                raise BackendSubmissionError(
                    "resources."
                    + backend
                    + ".mem_per_cpu:auto estimates "
                    + str(round(mem_gb, 3))
                    + " GB/core for "
                    + phase_name
                    + ", exceeding partition "
                    + repr(part)
                    + " cap "
                    + str(partition_gb)
                    + " GB/core. Use a higher-memory partition or explicit resources."
                )
        mem_per_cpu = _format_gb(mem_gb)
    else:
        mem_per_cpu = str(raw_mem).strip()
        requested_gb = slurm_memory_mib(mem_per_cpu) / 1024.0
        if requested_gb > partition_gb + 1.0e-9:
            raise BackendSubmissionError(
                "resources."
                + backend
                + ".mem_per_cpu "
                + mem_per_cpu
                + " exceeds configured profile memory cap for partition "
                + repr(part)
                + " ("
                + str(partition_gb)
                + " GB/core)"
            )
        if backend == "gaussian":
            estimated_total = float(requested_gb) * float(cpus)
            memory_reason = "gaussian_explicit_allocation_for_gauss_mdef"
    allocated_gb = (slurm_memory_mib(mem_per_cpu) / 1024.0) * float(cpus)
    if float(estimated_total) > allocated_gb + 1.0e-9:
        raise BackendSubmissionError(
            "resources."
            + backend
            + ".mem_per_cpu="
            + str(mem_per_cpu)
            + " requests "
            + str(round(allocated_gb, 3))
            + " GB total for "
            + str(phase_name)
            + ", below the protected estimate "
            + str(round(float(estimated_total), 3))
            + " GB. Increase the request, choose auto, or use a higher-memory partition."
        )
    concurrency = 1
    if array_size is not None:
        throttle = getattr(resources, "array_concurrency_limit", None)
        concurrency = min(
            int(array_size),
            int(throttle) if throttle is not None else int(array_size),
        )
    scratch_modes = {
        "diversity": (
            "file_backed_condensed_distances"
            if str(extra.get("distance_store_mode")) == "file"
            else "in_memory_condensed_distances"
        ),
        "gaussian": "gaussian_task_scratch",
        "aimall": "temporary_environment_pointdir_outputs",
        "ariadne": "temporary_environment_canonical_results",
        "ferebus": "temporary_environment_staging_runtime",
    }
    expected_scratch_bytes: Optional[int]
    scratch_requirement_exact = backend == "diversity"
    if backend == "diversity":
        expected_scratch_bytes = (
            int(extra.get("condensed_store_bytes", 0))
            if str(extra.get("distance_store_mode")) == "file"
            else 0
        )
    else:
        expected_scratch_bytes = None
    extra.update({
        "active_workers": int(active_workers),
        "memory_only_cpus": int(memory_only_cpus),
        "memory_estimate_safety_factor": float(
            config.resources.memory_estimate_safety_factor
        ),
        "allocated_cpus": int(cpus),
        "array_size": None if array_size is None else int(array_size),
        "array_concurrency": int(concurrency),
        "per_task_allocation_gb": float(allocated_gb),
        "peak_allocation_gb": float(allocated_gb) * float(concurrency),
        "scratch_mode": scratch_modes[backend],
        "expected_scratch_bytes": expected_scratch_bytes,
        "scratch_requirement_exact": bool(scratch_requirement_exact),
        "profile_limits": {
            "partition_min_cpus": int(partition_min),
            "partition_max_cpus": int(partition_max),
            "partition_memory_per_core_gb": float(partition_gb),
        },
        "evidence": evidence,
    })
    scheduler = str(profile_value("hpc", "scheduler", default="slurm") or "slurm")
    extra["scheduler"] = scheduler
    if scheduler == "sge":
        extra["scheduler_queue"] = scheduler_queue_for_partition(part)
        extra["parallel_environment"] = parallel_environment_for_partition(part)
    if campaign_path is not None:
        try:
            extra["campaign_filesystem"] = {
                "path": str(campaign_path.resolve()),
                "free_bytes_at_resolution": int(
                    shutil.disk_usage(campaign_path).free
                ),
            }
        except OSError as exc:
            warnings.append(
                "campaign filesystem free space could not be recorded: "
                + type(exc).__name__
                + ": "
                + str(exc)
            )
        try:
            campaign_path.resolve().relative_to(Path.home().resolve())
        except ValueError:
            pass
        else:
            warnings.append(
                "campaign root is under $HOME; use a cluster campaign filesystem for live work"
            )
    _report_resource_progress(
        progress_callback,
        "resource_rules",
        completed=1,
        total=1,
        unit="steps",
    )
    return ResolvedPhaseResources(
        backend=backend,
        partition=part,
        ntasks=1,
        cpus_per_task=int(cpus),
        mem_per_cpu=mem_per_cpu,
        estimated_total_memory_gb=float(estimated_total),
        partition_memory_per_core_gb=float(partition_gb),
        cpus_raw=raw_cpu,
        mem_per_cpu_raw=raw_mem,
        cpu_reason=cpu_reason,
        memory_reason=memory_reason,
        warnings=tuple(warnings),
        extra=extra,
    )


def gaussian_mdef(config: Any, resolved: ResolvedPhaseResources) -> str:
    """Render a positive Gaussian memory limit within the scheduler allocation."""
    allocated_mib = slurm_memory_mib(resolved.mem_per_cpu) * float(max(1, int(resolved.cpus_per_task)))
    usable_mib = allocated_mib * float(
        config.resources.gaussian_memory_fraction_of_slurm_for()
    )
    whole_mib = int(math.floor(usable_mib))
    if whole_mib < 1:
        raise BackendSubmissionError(
            "resolved Gaussian allocation is too small to express a positive "
            "GAUSS_MDEF within the configured scheduler memory fraction"
        )
    whole_gib = whole_mib // 1024
    rendered = str(whole_gib) + "GB" if whole_gib >= 1 else str(whole_mib) + "MB"
    rendered_mib = float(whole_gib * 1024 if whole_gib >= 1 else whole_mib)
    if rendered_mib > usable_mib + 1.0e-9:
        raise BackendSubmissionError(
            "rendered GAUSS_MDEF exceeds its protected scheduler allocation"
        )
    return rendered
