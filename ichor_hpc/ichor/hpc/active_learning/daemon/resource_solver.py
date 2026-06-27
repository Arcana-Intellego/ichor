"""Live Slurm resource resolution for active-learning backend phases.

Campaign schema v3 stores backend-specific CPU and memory requests under
``resources``. Each request can be explicit or ``auto``. This module resolves
those values using the active cluster profile, staged artefacts, and
backend-specific scaling heuristics before any sbatch script is rendered.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cluster_profile import active_machine, profile_value
from .phase_executor import BackendSubmissionError


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
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


_SLURM_MEM_RE = re.compile(r"^([1-9][0-9]*)([KMGT]?)$", re.IGNORECASE)


def backend_for_phase(phase_name: str) -> str:
    if phase_name in ("PHASE_A_POLUS", "PHASE_B_POLUS"):
        return "polus"
    if phase_name in ("INITIAL_GAUSSIAN", "GAUSSIAN"):
        return "gaussian"
    if phase_name in ("INITIAL_AIMALL", "AIMALL"):
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
        raise BackendSubmissionError("unsupported Slurm memory syntax: " + repr(value))
    amount = int(match.group(1))
    unit = match.group(2) or "M"
    scale = {
        "K": 1.0 / 1024.0,
        "M": 1.0,
        "G": 1024.0,
        "T": 1024.0 * 1024.0,
    }[unit]
    return float(amount) * scale


def gaussian_memory_mib(value: Any) -> float:
    text = str(value).strip().upper()
    match = re.fullmatch(r"([1-9][0-9]*)([KMGT]?)(?:B|W)?", text)
    if not match:
        raise BackendSubmissionError(
            "unsupported Gaussian memory syntax: " + repr(value)
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


def partition_core_range(partition: str) -> Optional[Tuple[int, int]]:
    parallel = profile_value("hpc", "parallel_environments", default=None)
    if not isinstance(parallel, dict):
        return None
    raw = parallel.get(str(partition))
    if raw is None:
        return None
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
            if value <= 0.0:
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
        if value <= 0.0:
            raise BackendSubmissionError("configured hpc.memory_per_core_gb must be > 0")
        return value
    return 4.0


def _partition_min_max(partition: str) -> Tuple[int, int]:
    configured = partition_core_range(partition)
    if configured is None:
        return 1, 10_000
    return configured


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


def _count_xyz_frames(path: Path) -> Tuple[int, Optional[int]]:
    if not path.is_file():
        return 0, None
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    n_frames = 0
    first_natoms: Optional[int] = None
    while i < len(lines):
        try:
            natoms = int(lines[i].strip())
        except ValueError:
            break
        if first_natoms is None:
            first_natoms = natoms
        n_frames += 1
        i += max(2 + natoms, 1)
    return n_frames, first_natoms


def _campaign_pool_size(campaign_dir: Optional[Path]) -> Tuple[int, Optional[int]]:
    if campaign_dir is None:
        return 0, None
    return _count_xyz_frames(Path(campaign_dir) / ".DATA" / "TRAJECTORY" / "pool.xyz")


def _iter_dir(campaign_dir: Optional[Path], iteration: int) -> Optional[Path]:
    if campaign_dir is None:
        return None
    return Path(campaign_dir) / "7_ACTIVE_LEARNING" / ("iteration-" + str(int(iteration)).zfill(4))


def _phase_b_candidate_count(campaign_dir: Optional[Path], iteration: int) -> int:
    idir = _iter_dir(campaign_dir, iteration)
    if idir is None:
        return 0
    path = idir / "ARIADNE_RESULTS.json"
    if not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    records = data.get("accepted")
    if not isinstance(records, list):
        records = data.get("records")
    return len(records) if isinstance(records, list) else 0


def _staging_dir(campaign_dir: Optional[Path], phase_name: str, iteration: int) -> Optional[Path]:
    if campaign_dir is None:
        return None
    bucket = "initial" if str(phase_name).startswith("INITIAL_") else "iter_" + str(int(iteration))
    return Path(campaign_dir) / ".DATA" / "STAGING" / bucket


def _pointdirs_from_points_file(staging: Optional[Path]) -> List[Path]:
    if staging is None:
        return []
    points = staging / "POINTS.txt"
    if not points.is_file():
        return []
    out: List[Path] = []
    for line in points.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text:
            continue
        p = Path(text)
        if not p.is_absolute():
            p = staging / p
        out.append(p)
    return out


def _natoms_from_gjf(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start: Optional[int] = None
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) >= 2:
            try:
                int(parts[0])
                int(parts[1])
            except ValueError:
                continue
            start = i + 1
            break
    if start is None:
        return None
    count = 0
    for line in lines[start:]:
        if not line.strip():
            break
        count += 1
    return count or None


def _staged_natoms(campaign_dir: Optional[Path], phase_name: str, iteration: int) -> Optional[int]:
    staging = _staging_dir(campaign_dir, phase_name, iteration)
    for pd in _pointdirs_from_points_file(staging):
        natoms = _natoms_from_gjf(pd / "input.gjf")
        if natoms:
            return natoms
    return None


def _model_dir_size_gb(campaign_dir: Optional[Path]) -> float:
    if campaign_dir is None:
        return 0.0
    models = Path(campaign_dir) / "6_TRAINED_MODELS"
    current = models / "current"
    root = current if current.exists() else models
    total = 0
    if root.exists():
        for path in root.rglob("*"):
            if path.is_file():
                try:
                    total += int(path.stat().st_size)
                except OSError:
                    pass
    return float(total) / (1024.0 ** 3)


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
    method = (str(getattr(config.gaussian, "method", "")) + " " + str(getattr(config.gaussian, "extra_keywords", ""))).lower()
    if "mp2" in method or "ccsd" in method or "casscf" in method:
        return 1.5
    return 1.0


def _ferebus_rows_features(campaign_dir: Optional[Path], phase_name: str, iteration: int) -> Tuple[int, int]:
    if campaign_dir is None:
        manifest = None
    else:
        manifest = (
            Path(campaign_dir)
            / "6_TRAINED_MODELS"
            / "iteration-staging"
            / "FEREBUS_TASKS.json"
        )
    if manifest is None or not manifest.is_file():
        return 100, 32
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return 100, 32
    max_rows = 0
    max_features = 32
    for task in data.get("tasks", []) if isinstance(data.get("tasks"), list) else []:
        counts = task.get("row_counts", {}) if isinstance(task, dict) else {}
        if isinstance(counts, dict):
            total = 0
            for value in counts.values():
                try:
                    total += int(value)
                except (TypeError, ValueError):
                    pass
            max_rows = max(max_rows, total)
        alf = task.get("alf_1_indexed", []) if isinstance(task, dict) else []
        if isinstance(alf, list):
            max_features = max(max_features, max(1, len(alf) * 8))
    return max(max_rows, 1), max(max_features, 1)


def _aimall_regime(n_atoms: int, n_primitives: Optional[int]) -> str:
    primitives = n_primitives if n_primitives is not None else n_atoms * 40
    if n_atoms <= 12 and primitives <= 400:
        return "small"
    if n_atoms <= 40 or primitives <= 1500:
        return "medium"
    return "large"


def _estimate_backend_memory_gb(
    backend: str,
    phase_name: str,
    config: Any,
    campaign_dir: Optional[Path],
    iteration: int,
    candidate_cpus: int,
    partition_gb: float,
) -> Tuple[float, str, Dict[str, Any]]:
    extra: Dict[str, Any] = {}
    if backend == "polus":
        if phase_name == "PHASE_A_POLUS":
            n, natoms = _campaign_pool_size(campaign_dir)
        else:
            n = _phase_b_candidate_count(campaign_dir, iteration)
            natoms = None
        n = max(1, int(n or 100))
        descriptor = str(getattr(getattr(config, "phase_b", object()), "descriptor", "rmsd_massweight"))
        if phase_name == "PHASE_A_POLUS":
            factor = 2.5
        elif descriptor == "hybrid_alf_rmsd":
            feature_dim = max(16, int(natoms or 12) * 12)
            factor = max(3.0, float(feature_dim))
        else:
            factor = 2.5
        total_gb = 1.0 + (8.0 * float(n) * float(n) * factor) / (1024.0 ** 3)
        extra.update({"n_frames": int(n), "descriptor": descriptor})
        return total_gb, "polus_pairwise_distance_matrix", extra
    if backend == "gaussian":
        total_gb = max(float(candidate_cpus) * float(partition_gb), 1.0)
        return total_gb, "gaussian_partition_memory_for_gauss_mdef", extra
    if backend == "aimall":
        n_atoms = int(_staged_natoms(campaign_dir, phase_name, iteration) or 12)
        n_primitives = None
        regime = _aimall_regime(n_atoms, n_primitives)
        if regime == "small":
            per_atom = 2.4
        elif regime == "medium":
            per_atom = 4.0
        else:
            per_atom = 8.0
        naat = resolve_aimall_naat(config, candidate_cpus, n_atoms, n_primitives)
        total_gb = 1.0 + float(naat) * per_atom
        extra.update({
            "n_atoms": int(n_atoms),
            "n_primitives": n_primitives,
            "aimall_regime": regime,
            "naat_resolved": int(naat),
        })
        return total_gb, "aimall_concurrent_atomic_integrations", extra
    if backend == "ariadne":
        workers = max(1, int(candidate_cpus))
        model_size = _model_dir_size_gb(campaign_dir)
        per_worker = max(2.0, 4.0 * float(model_size))
        total_gb = 1.5 + float(workers) * per_worker
        extra.update({"model_dir_size_gb": float(model_size), "workers": int(workers)})
        return total_gb, "ariadne_gradient_worker_model_memory", extra
    if backend == "ferebus":
        rows, features = _ferebus_rows_features(campaign_dir, phase_name, iteration)
        nagents = int(getattr(config.ferebus, "nagents", 20))
        kernel_gb = (8.0 * float(rows) * float(rows)) / (1024.0 ** 3)
        dataset_gb = (8.0 * float(rows) * float(features)) / (1024.0 ** 3)
        per_agent = 1.0 + 3.0 * kernel_gb + dataset_gb
        total_gb = float(nagents) * per_agent
        extra.update({"max_rows_per_task": int(rows), "n_features_estimate": int(features), "nagents": int(nagents)})
        return total_gb, "ferebus_agent_kernel_training_memory", extra
    return 1.0, "default_minimal_backend_memory", extra


def resolve_aimall_naat(
    config: Any,
    resolved_cpus: int,
    n_atoms: int,
    n_primitives: Optional[int] = None,
) -> int:
    raw = getattr(getattr(config, "aimall", object()), "naat", "auto")
    if not (isinstance(raw, str) and raw.strip().lower() == "auto"):
        return max(1, min(int(raw), int(n_atoms), int(resolved_cpus)))
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
) -> Tuple[int, str, Dict[str, Any], float, str]:
    extra: Dict[str, Any] = {}
    if backend == "polus":
        estimated, mem_reason, mem_extra = _estimate_backend_memory_gb(
            backend, phase_name, config, campaign_dir, iteration, partition_min, partition_gb
        )
        target = max(partition_min, int(math.ceil(estimated / max(partition_gb, 1.0e-9))))
        return min(target, partition_max), "partition_min_plus_polus_memory_fit", mem_extra, estimated, mem_reason
    if backend == "gaussian":
        n_atoms = _staged_natoms(campaign_dir, phase_name, iteration)
        if n_atoms is None:
            _nframes, n_atoms = _campaign_pool_size(campaign_dir)
        n_atoms = int(n_atoms or 12)
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
        n_atoms = int(_staged_natoms(campaign_dir, phase_name, iteration) or 12)
        n_primitives = None
        regime = _aimall_regime(n_atoms, n_primitives)
        if regime == "small":
            target = min(n_atoms, 8)
        elif regime == "medium":
            target = 12
        else:
            target = 16
        if not (isinstance(getattr(config.aimall, "naat", "auto"), str) and str(getattr(config.aimall, "naat")).strip().lower() == "auto"):
            target = max(target, int(config.aimall.naat))
        target = min(max(target, partition_min), partition_max)
        extra.update({"n_atoms": int(n_atoms), "n_primitives": n_primitives, "aimall_regime": regime})
        estimated, mem_reason, mem_extra = _estimate_backend_memory_gb(
            backend, phase_name, config, campaign_dir, iteration, target, partition_gb
        )
        extra.update(mem_extra)
        if estimated > target * partition_gb and target < partition_max:
            target = min(partition_max, max(target, int(math.ceil(estimated / max(partition_gb, 1.0e-9)))))
            estimated, mem_reason, mem_extra = _estimate_backend_memory_gb(
                backend, phase_name, config, campaign_dir, iteration, target, partition_gb
            )
            extra.update(mem_extra)
        return target, "aimall_wavefunction_size_parallel_atoms", extra, estimated, mem_reason
    if backend == "ariadne":
        grad_backend = str(getattr(config.resources, "gradient_parallel_backend", "process"))
        mode = str(getattr(getattr(config.acquisition, "gradient", object()), "mode", "active_fd"))
        if grad_backend == "serial":
            target = partition_min
            reason = "ariadne_serial_gradient_backend"
        elif mode == "active_fd":
            dim = int(getattr(config.acquisition.subspace, "max_subspace_dim", 6))
            target = dim
            reason = "ariadne_active_fd_direction_workers"
        else:
            n_atoms = int(_staged_natoms(campaign_dir, phase_name, iteration) or 12)
            target = 3 * n_atoms
            reason = "ariadne_cartesian_fd_component_workers"
            extra["n_atoms"] = int(n_atoms)
        if target > partition_max:
            extra["cpu_cap_warning"] = "auto target capped at partition maximum"
        return min(max(target, partition_min), partition_max), reason, extra, 0.0, "ariadne_gradient_worker_model_memory"
    if backend == "ferebus":
        target = int(getattr(config.ferebus, "nagents", 20))
        if target > partition_max:
            raise BackendSubmissionError(
                "resources.ferebus_cpus_per_task:auto resolves to ferebus.nagents="
                + str(target)
                + " but partition "
                + repr(str(getattr(config.resources, "partition", "")))
                + " allows at most "
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
) -> ResolvedPhaseResources:
    resources = config.resources
    part = str(partition if partition is not None else resources.partition)
    backend = backend_for_phase(phase_name)
    raw_cpu = getattr(resources, backend + "_cpus_per_task")
    raw_mem = getattr(resources, backend + "_mem_per_cpu")
    partition_min, partition_max = _partition_min_max(part)
    partition_gb = partition_memory_per_core_gb(part)
    campaign_path = Path(campaign_dir) if campaign_dir is not None else None
    warnings: List[str] = []
    extra: Dict[str, Any] = {}
    estimated_from_cpu = 0.0
    memory_reason_from_cpu = "not_estimated_during_cpu_resolution"

    if _is_auto(raw_cpu):
        cpus, cpu_reason, cpu_extra, estimated_from_cpu, memory_reason_from_cpu = _auto_cpu_target(
            backend,
            phase_name,
            config,
            campaign_path,
            int(iteration),
            partition_min,
            partition_max,
            partition_gb,
        )
        extra.update(cpu_extra)
        _validate_core_count("resources." + backend + "_cpus_per_task", int(cpus), part)
    else:
        cpus = _explicit_cpu("resources." + backend + "_cpus_per_task", raw_cpu, part)
        cpu_reason = "explicit"

    if estimated_from_cpu > 0.0:
        estimated_total, memory_reason, mem_extra = (
            estimated_from_cpu,
            memory_reason_from_cpu,
            {},
        )
    else:
        estimated_total, memory_reason, mem_extra = _estimate_backend_memory_gb(
            backend,
            phase_name,
            config,
            campaign_path,
            int(iteration),
            int(cpus),
            float(partition_gb),
        )
    extra.update(mem_extra)

    if _is_auto(raw_mem):
        if backend == "gaussian":
            mem_gb = float(partition_gb)
            memory_reason = "gaussian_partition_memory_for_gauss_mdef"
            estimated_total = float(cpus) * mem_gb
        else:
            mem_gb = max(1.0, math.ceil(float(estimated_total) / max(float(cpus), 1.0)))
            if mem_gb > partition_gb + 1.0e-9:
                if _is_auto(raw_cpu) and backend in ("polus", "aimall"):
                    needed = min(partition_max, max(int(cpus), int(math.ceil(float(estimated_total) / max(partition_gb, 1.0e-9)))))
                    if needed != cpus:
                        cpus = needed
                        mem_gb = max(1.0, math.ceil(float(estimated_total) / max(float(cpus), 1.0)))
                if mem_gb > partition_gb + 1.0e-9:
                    raise BackendSubmissionError(
                        "resources."
                        + backend
                        + "_mem_per_cpu:auto estimates "
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
                + "_mem_per_cpu "
                + mem_per_cpu
                + " exceeds configured profile memory cap for partition "
                + repr(part)
                + " ("
                + str(partition_gb)
                + " GB/core)"
            )
    allocated_gb = (slurm_memory_mib(mem_per_cpu) / 1024.0) * float(cpus)
    if float(estimated_total) > allocated_gb + 1.0e-9:
        if (
            not _is_auto(raw_mem)
            and bool(
                getattr(
                    config.resources,
                    "fail_on_memory_estimate_exceeds_request",
                    True,
                )
            )
        ):
            raise BackendSubmissionError(
                "resources."
                + backend
                + "_mem_per_cpu="
                + str(mem_per_cpu)
                + " requests "
                + str(round(allocated_gb, 3))
                + " GB total for "
                + str(phase_name)
                + ", below the estimated "
                + str(round(float(estimated_total), 3))
                + " GB. Increase the request, choose auto, or set "
                + "resources.fail_on_memory_estimate_exceeds_request=false."
            )
        warnings.append("estimated memory exceeds requested allocation")
    if backend == "aimall":
        n_atoms = int(extra.get("n_atoms") or _staged_natoms(campaign_path, phase_name, int(iteration)) or 1)
        extra["naat_resolved"] = int(resolve_aimall_naat(config, int(cpus), n_atoms, extra.get("n_primitives")))
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


def gaussian_mdef_gb(config: Any, resolved: ResolvedPhaseResources) -> int:
    allocated_mib = slurm_memory_mib(resolved.mem_per_cpu) * float(max(1, int(resolved.cpus_per_task)))
    usable_mib = allocated_mib * float(config.resources.gaussian_memory_fraction_of_slurm)
    return max(1, int(math.floor(usable_mib / 1024.0)))


def validate_gaussian_link0_memory(config: Any, resolved: ResolvedPhaseResources) -> None:
    if str(config.resources.gaussian_memory_mode) != "link0":
        return
    gaussian_mib = gaussian_memory_mib(config.resources.gaussian_link0_mem)
    allocated_mib = slurm_memory_mib(resolved.mem_per_cpu) * float(max(1, int(resolved.cpus_per_task)))
    limit_mib = float(config.resources.gaussian_memory_fraction_of_slurm) * allocated_mib
    if gaussian_mib > limit_mib + 1.0e-9:
        raise BackendSubmissionError(
            "resources.gaussian_link0_mem "
            + repr(str(config.resources.gaussian_link0_mem))
            + " exceeds "
            + str(config.resources.gaussian_memory_fraction_of_slurm)
            + " of the resolved Link0 Gaussian Slurm allocation "
            + "(resources.gaussian_mem_per_cpu="
            + repr(str(resolved.mem_per_cpu))
            + ", resources.gaussian_cpus_per_task="
            + str(int(resolved.cpus_per_task))
            + ")"
        )
