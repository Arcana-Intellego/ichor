"""Per-point input staging for the live sbatch phases.

Lays out the POINT_<k>.pointdir tree (and the POINTS.txt the array jobs index
into) on the login node before submit, reusing the core file writers. Nothing
here talks to SLURM -- it only writes files.

Layout, matching what the postprocess parsers expect:

    .DATA/STAGING/<bucket>/
        POINT_0000.pointdir/input.gjf      # gaussian input (writes input.wfn)
        POINT_0001.pointdir/input.gjf
        ...
        POINTS.txt                         # one absolute pointdir path per line

<bucket> is "initial" for the INITIAL_* phases and "iter_<N>" otherwise.
"""
from __future__ import annotations

import json
import os  # stage_ferebus_inputs cd's into the staging dir to export csvs; this was missing and only bit on a live run
import re
import shutil
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ichor.core.atoms import Atoms
from ichor.core.files import PointDirectory
from ichor.core.files.xyz import Trajectory

from .resource_solver import (
    resolve_aimall_naat,
    resolve_phase_resources,
    validate_gaussian_link0_memory,
)
from .state import atomic_write_json


QUANTUM_ACCEPTANCE_MANIFEST = "accepted_pointdirs.json"
QUANTUM_ACCEPTANCE_SCHEMA_VERSION = 1
POINTDIR_BASENAME_RE = re.compile(r"^POINT_\d{4}\.pointdir$")
AIMALL_TASK_METADATA = "AIMALL_TASK.json"
AIMALL_TASK_METADATA_SCHEMA_VERSION = 1
FEREBUS_TASK_MANIFEST = "FEREBUS_TASKS.json"
FEREBUS_TASK_SCHEMA_VERSION = 2
FEREBUS_JOB_DETAILS = "job-details"
SAFE_PATH_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def bucket_dir(campaign_dir, phase_name: str, iteration: int) -> Path:
    bucket = "initial" if phase_name.startswith("INITIAL_") else ("iter_" + str(iteration))
    return Path(campaign_dir) / ".DATA" / "STAGING" / bucket


def _load_frames(sample_xyz) -> List[Atoms]:
    traj = Trajectory(Path(sample_xyz))
    traj.read()
    return [atoms.copy() for atoms in traj]


def write_points_file(staging_dir: Path, pointdirs: Sequence[Path]) -> Path:
    """One absolute pointdir path per line; the array sed-lookup reads this."""
    points_file = Path(staging_dir) / "POINTS.txt"
    body = "\n".join(str(Path(p).resolve()) for p in pointdirs)
    # force LF. the array sbatch sed-reads this on a linux node, and if we ever stage from
    # windows the default crlf leaves a trailing \r so cd "$POINT_DIR" quietly breaks.
    points_file.write_text(body + ("\n" if body else ""), encoding="utf-8", newline="\n")
    return points_file


def _pointdir_name(pointdir: Any) -> str:
    path = getattr(pointdir, "path", pointdir)
    return Path(path).name


def validate_safe_path_token(label: str, value: str) -> None:
    text = str(value)
    if not SAFE_PATH_TOKEN_RE.fullmatch(text):
        raise ValueError(
            label
            + " must be a safe path token matching ^[A-Za-z0-9][A-Za-z0-9_.-]*$: "
            + repr(text)
        )


def _is_relative_to(path: Path, parent: Path) -> bool:
    resolved = Path(path).resolve(strict=False)
    root = Path(parent).resolve(strict=False)
    return resolved == root or root in resolved.parents


def _reject_symlink_ancestors(path: Path, stop_at: Path) -> None:
    stop = Path(stop_at).resolve(strict=False)
    current = Path(path)
    for candidate in [current] + list(current.parents):
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            resolved = candidate.absolute()
        if resolved == stop:
            break
        if candidate.exists() and candidate.is_symlink():
            raise OSError("refusing to clean path below symlink: " + str(candidate))


def _checked_rmtree(
    path: Path,
    *,
    campaign_dir: Optional[Path] = None,
    allowed_roots: Iterable[Path] = (),
) -> None:
    target = Path(path)
    if not target.exists():
        return
    if target.is_symlink():
        raise OSError("refusing to remove symlinked staging path: " + str(target))
    if campaign_dir is not None:
        campaign = Path(campaign_dir).resolve(strict=False)
        resolved = target.resolve(strict=False)
        if not _is_relative_to(resolved, campaign):
            raise OSError("refusing to remove path outside campaign: " + str(target))
        _reject_symlink_ancestors(target, campaign)
        roots = [Path(root).resolve(strict=False) for root in allowed_roots]
        if roots and not any(_is_relative_to(resolved, root) for root in roots):
            raise OSError("refusing to remove path outside allowed staging roots: " + str(target))
    shutil.rmtree(str(target), ignore_errors=False)
    if target.exists():
        raise OSError("failed to remove stale staging directory: " + str(target))


def _reject_symlink_tree(root: Path) -> None:
    root = Path(root)
    if root.is_symlink():
        raise ValueError("refusing to copy symlinked path: " + str(root))
    for child in root.rglob("*"):
        if child.is_symlink():
            raise ValueError("refusing to copy tree containing symlink: " + str(child))


def _copytree_no_symlinks(src: Path, dest: Path) -> None:
    _reject_symlink_tree(Path(src))
    shutil.copytree(str(src), str(dest), symlinks=False)


def _validate_pointdir_basename(name: str) -> str:
    text = str(name).strip()
    if Path(text).name != text or not POINTDIR_BASENAME_RE.fullmatch(text):
        raise ValueError("unsafe pointdir name in manifest: " + repr(name))
    return text


def _points_file_names(staging_dir: Path) -> List[str]:
    points_file = Path(staging_dir) / "POINTS.txt"
    if not points_file.is_file():
        raise FileNotFoundError("POINTS.txt missing in quantum staging: " + str(points_file))
    names: List[str] = []
    seen = set()
    for line_no, raw in enumerate(points_file.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if not text:
            continue
        path = Path(text)
        name = _validate_pointdir_basename(path.name)
        if name in seen:
            raise ValueError("duplicate pointdir in POINTS.txt line " + str(line_no) + ": " + name)
        expected = Path(staging_dir) / name
        try:
            resolved = path.resolve(strict=False)
            expected_resolved = expected.resolve(strict=False)
        except OSError as exc:
            raise ValueError("POINTS.txt path cannot be resolved at line " + str(line_no)) from exc
        if resolved != expected_resolved:
            raise ValueError(
                "POINTS.txt path does not point inside staging at line "
                + str(line_no)
                + ": "
                + str(path)
            )
        seen.add(name)
        names.append(name)
    return names


def quantum_acceptance_manifest_path(
    staging_dir: Path,
    *,
    phase_name: Optional[str] = None,
) -> Path:
    staging = Path(staging_dir)
    if phase_name:
        validate_safe_path_token("quantum acceptance phase", str(phase_name))
        return staging / ("accepted_pointdirs." + str(phase_name) + ".json")
    return staging / QUANTUM_ACCEPTANCE_MANIFEST


def write_quantum_acceptance_manifest(
    staging_dir: Path,
    *,
    phase_name: str,
    iteration: int,
    accepted: Sequence[Any],
    rejected: Sequence[Tuple[str, str]],
) -> Path:
    """Persist the exact validated quantum handoff for downstream live stages."""
    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    accepted_names = [_validate_pointdir_basename(_pointdir_name(p)) for p in accepted]
    rejected_payload = [
        {"pointdir": _validate_pointdir_basename(str(name)), "reason": str(reason)}
        for name, reason in rejected
    ]
    payload: Dict[str, Any] = {
        "schema_version": QUANTUM_ACCEPTANCE_SCHEMA_VERSION,
        "phase": str(phase_name),
        "iteration": int(iteration),
        "accepted_pointdirs": accepted_names,
        "rejected": rejected_payload,
        "n_total": int(len(accepted_names) + len(rejected_payload)),
    }
    phase_path = quantum_acceptance_manifest_path(staging, phase_name=phase_name)
    atomic_write_json(phase_path, payload)
    # Keep the legacy filename as a compatibility alias for older tools, while
    # phase-aware readers prefer the immutable phase-specific handoff.
    atomic_write_json(quantum_acceptance_manifest_path(staging), payload)
    return phase_path


def read_quantum_acceptance_manifest(
    staging_dir: Path,
    *,
    expected_phase: str,
    expected_iteration: int,
    require_nonempty: bool = True,
    require_points_file_membership: bool = False,
) -> Tuple[List[Path], Dict[str, Any]]:
    """Read and validate the live quantum acceptance manifest.

    Downstream consumers use this instead of globbing POINT_*.pointdir so
    rejected quantum outputs cannot be silently committed or reprocessed.
    """
    staging = Path(staging_dir)
    phase_path = quantum_acceptance_manifest_path(staging, phase_name=expected_phase)
    used_legacy_alias = not phase_path.is_file()
    path = phase_path if phase_path.is_file() else quantum_acceptance_manifest_path(staging)
    if not path.is_file():
        raise FileNotFoundError("quantum acceptance manifest missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("quantum acceptance manifest unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise ValueError("quantum acceptance manifest must be a JSON object: " + str(path))
    if int(data.get("schema_version", -1)) != QUANTUM_ACCEPTANCE_SCHEMA_VERSION:
        raise ValueError("unsupported quantum acceptance manifest schema: " + str(path))
    if data.get("phase") != expected_phase:
        raise ValueError(
            "quantum acceptance manifest phase mismatch: expected "
            + expected_phase + " got " + str(data.get("phase"))
        )
    try:
        iteration = int(data.get("iteration"))
    except (TypeError, ValueError) as exc:
        raise ValueError("quantum acceptance manifest iteration is not an integer") from exc
    if iteration != int(expected_iteration):
        raise ValueError(
            "quantum acceptance manifest iteration mismatch: expected "
            + str(int(expected_iteration)) + " got " + str(iteration)
        )
    accepted = data.get("accepted_pointdirs")
    if not isinstance(accepted, list):
        raise ValueError("quantum acceptance manifest accepted_pointdirs must be a list")
    rejected = data.get("rejected", [])
    if not isinstance(rejected, list):
        raise ValueError("quantum acceptance manifest rejected must be a list")
    n_total = data.get("n_total")
    if n_total is not None:
        try:
            parsed_total = int(n_total)
        except (TypeError, ValueError) as exc:
            raise ValueError("quantum acceptance manifest n_total is not an integer") from exc
        if parsed_total != len(accepted) + len(rejected):
            raise ValueError("quantum acceptance manifest n_total does not match payload lengths")

    seen = set()
    resolved: List[Path] = []
    points_names = set(_points_file_names(staging)) if require_points_file_membership else None
    for raw_name in accepted:
        if not isinstance(raw_name, str):
            raise ValueError("accepted pointdir name is not a string")
        name = _validate_pointdir_basename(raw_name)
        if name in seen:
            raise ValueError("duplicate accepted pointdir in manifest: " + name)
        if points_names is not None and name not in points_names:
            raise ValueError("accepted pointdir is not present in POINTS.txt: " + name)
        seen.add(name)
        pointdir = staging / name
        if not pointdir.is_dir():
            raise FileNotFoundError(
                "accepted pointdir listed in manifest is missing: " + str(pointdir)
            )
        resolved.append(pointdir)
    if require_nonempty and not resolved:
        raise ValueError("quantum acceptance manifest accepted_pointdirs is empty: " + str(path))
    if used_legacy_alias:
        # Migrate old in-flight campaigns before the next phase overwrites the
        # legacy alias with its own acceptance payload.
        atomic_write_json(phase_path, data)
    return resolved, data


def hash_pointdir_tree(pointdir: Path) -> str:
    root = Path(pointdir)
    _reject_symlink_tree(root)
    digest = hashlib.sha256()
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: str(p.relative_to(root))):
        rel = str(path.relative_to(root)).replace("\\", "/")
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def ferebus_manifest_path(staging_dir: Path) -> Path:
    return Path(staging_dir) / FEREBUS_TASK_MANIFEST


def read_ferebus_manifest(staging_dir: Path) -> Dict[str, Any]:
    path = ferebus_manifest_path(Path(staging_dir))
    if not path.is_file():
        raise FileNotFoundError("FEREBUS task manifest missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("FEREBUS task manifest unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise ValueError("FEREBUS task manifest must be a JSON object: " + str(path))
    if int(data.get("schema_version", -1)) != FEREBUS_TASK_SCHEMA_VERSION:
        raise ValueError("unsupported FEREBUS task manifest schema: " + str(path))
    try:
        reference_data_version = int(data["reference_data_version"])
        n_reference_points = int(data["n_reference_points"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("FEREBUS task manifest reference-data binding is invalid") from exc
    if reference_data_version < 0 or n_reference_points <= 0:
        raise ValueError("FEREBUS task manifest reference-data values are invalid")
    row_order = data.get("pointdir_row_order")
    if not isinstance(row_order, list) or len(row_order) != n_reference_points:
        raise ValueError("FEREBUS task manifest row order/count is invalid")
    for field_name in (
        "reference_data_head_manifest_sha256",
        "reference_data_view_sha256",
    ):
        value = str(data.get(field_name) or "")
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("FEREBUS task manifest " + field_name + " is invalid")
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("FEREBUS task manifest has no tasks: " + str(path))
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("FEREBUS task manifest contains a non-object task")
        counts = task.get("row_counts")
        if not isinstance(counts, dict):
            raise ValueError("FEREBUS task manifest row_counts is invalid")
        try:
            task_total = sum(
                int(counts[split]) for split in ("train", "int_val", "ext_val")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("FEREBUS task manifest row_counts is invalid") from exc
        if task_total != n_reference_points:
            raise ValueError(
                "FEREBUS task row count does not match the reference-data view"
            )
    return data


def _write_ferebus_manifest(staging_dir: Path, payload: Dict[str, Any]) -> Path:
    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    path = ferebus_manifest_path(staging)
    atomic_write_json(path, payload)
    return path


def _alf_to_ferebus(alf: Any, atom: str) -> List[int]:
    try:
        raw = [int(x) for x in alf]
    except Exception as exc:
        raise ValueError("ALF for atom " + atom + " is not iterable/int-like") from exc
    if len(raw) != 3:
        raise ValueError("ALF for atom " + atom + " must contain exactly three indexes")
    if any(x < 0 for x in raw):
        raise ValueError("ALF for atom " + atom + " contains a negative zero-indexed entry")
    return [x + 1 for x in raw]


def _write_pyferebus_job_details(
    path: Path,
    *,
    system: str,
    atoms: Sequence[str],
    properties: Sequence[str],
    alf_by_atom: Dict[str, Sequence[int]],
    stats_by_prop_atom: Dict[Tuple[str, str], Dict[str, float]],
) -> Path:
    lines = [
        "system_name " + str(system),
        "natoms " + str(len(atoms)),
        "atoms " + " ".join(str(a) for a in atoms),
        "props " + " ".join(str(p) for p in properties),
    ]
    for atom in atoms:
        lines.append(str(atom) + " " + " ".join(str(int(x)) for x in alf_by_atom[atom]))
    stat_keys = ("min", "max", "range", "mean", "median", "std", "cv")
    for prop in properties:
        for atom in atoms:
            stats = stats_by_prop_atom.get((str(prop), str(atom)), {})
            missing = [k for k in stat_keys if k not in stats]
            if missing:
                raise ValueError(
                    "missing target-property stats for "
                    + str(prop)
                    + "-"
                    + str(atom)
                    + ": "
                    + repr(missing)
                )
            lines.append(
                str(prop)
                + "-"
                + str(atom)
                + " "
                + " ".join(str(float(stats[k])) for k in stat_keys)
            )
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def stage_gaussian_inputs(
    campaign_dir,
    config,
    phase_name,
    iteration,
    sample_xyz,
    *,
    partition_override: Optional[str] = None,
) -> Tuple[Path, int]:
    """Write one POINT_<k>.pointdir/input.gjf per frame in sample_xyz, plus
    POINTS.txt. Returns (staging_dir, n_points)."""
    from ichor.core.files.gaussian.gjf import GJF

    frames = _load_frames(sample_xyz)
    phase_b_records: List[Dict[str, Any]] = []
    allocation_records: List[Dict[str, Any]] = []
    allocation_manifest_hash: Optional[str] = None
    initial_seed_frame_ids: List[Optional[int]] = []
    initial_seed_selection_origins: List[str] = []
    initial_provenance_context: Optional[Dict[str, str]] = None
    replacement_context: Optional[str] = None
    is_replacement = str(phase_name) in {
        "INITIAL_REPLACEMENT_GAUSSIAN",
        "REPLACEMENT_GAUSSIAN",
    }
    if is_replacement:
        from ..replacement_sampling import read_replacement_sample

        replacement_manifest = read_replacement_sample(
            Path(sample_xyz).parent,
            verify_allocation=True,
        )
        replacement_context = str(replacement_manifest["context"])
        allocation_records = list(replacement_manifest["records"])
        allocation_manifest_hash = str(
            replacement_manifest["point_allocation_sha256"]
        )
        if len(allocation_records) != len(frames):
            raise ValueError(
                "replacement allocation record count does not match sample"
            )
        if replacement_context == "active":
            phase_b_records = list(allocation_records)
        elif replacement_context == "bootstrap":
            from .state import read_state, DEFAULT_STATE_FILENAME
            from ..acquisition.trajectory_pool import TrajectoryPool

            current_state = read_state(
                Path(campaign_dir) / ".DATA" / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME
            )
            pool = TrajectoryPool.load(campaign_dir)
            initial_provenance_context = {
                "campaign_uid": str(current_state.campaign_uid),
                "trajectory_sha256": str(pool.sha256),
            }
            initial_seed_frame_ids = [int(record["frame_id"]) for record in allocation_records]
            initial_seed_selection_origins = ["phase_a_replacement"] * len(allocation_records)
        else:
            raise ValueError("replacement sample context is invalid")
    if str(phase_name) == "GAUSSIAN":
        from ..handoff_manifests import read_phase_b_selection_manifest

        phase_b_manifest = read_phase_b_selection_manifest(
            Path(sample_xyz).parent,
            expected_iteration=int(iteration),
        )
        phase_b_records = list(phase_b_manifest.get("final", []))
        if len(phase_b_records) != len(frames):
            raise ValueError(
                "Phase B selection manifest final count "
                + str(len(phase_b_records))
                + " != sample frame count "
                + str(len(frames))
            )
        allocation_records = list(phase_b_records)
        from ..point_allocation import allocation_manifest_sha256

        allocation_manifest_hash = allocation_manifest_sha256(
            phase_b_manifest["point_allocation"]["manifest"]
        )
    if str(phase_name) == "INITIAL_GAUSSIAN":
        try:
            from .state import read_state, DEFAULT_STATE_FILENAME
            from ..acquisition.trajectory_pool import TrajectoryPool
            from ..handoff_manifests import read_phase_a_sample_manifest

            phase_a_manifest = read_phase_a_sample_manifest(Path(sample_xyz).parent)
            selected = phase_a_manifest.get("selected_indices")
            if isinstance(selected, list):
                if len(selected) != len(frames):
                    raise ValueError(
                        "Phase A selected index count "
                        + str(len(selected))
                        + " != initial sample frame count "
                        + str(len(frames))
                    )
                initial_seed_frame_ids = [
                    int(value) if value is not None else None
                    for value in selected
                ]
                initial_seed_selection_origins = [
                    "bootstrap_anchor" if value is None else "phase_a_polus"
                    for value in selected
                ]
                current_state = read_state(
                    Path(campaign_dir) / ".DATA" / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME
                )
                pool = TrajectoryPool.load(campaign_dir)
                initial_provenance_context = {
                    "campaign_uid": str(current_state.campaign_uid),
                    "trajectory_sha256": str(pool.sha256),
                }
                allocation_records = list(
                    phase_a_manifest["point_allocation"]["primary"]
                )
                if len(allocation_records) != len(frames):
                    raise ValueError(
                        "Phase A point-allocation record count does not match sample"
                    )
                from ..point_allocation import allocation_manifest_sha256

                allocation_manifest_hash = allocation_manifest_sha256(
                    phase_a_manifest["point_allocation"]["manifest"]
                )
        except FileNotFoundError:
            initial_seed_frame_ids = []
            initial_seed_selection_origins = []
            initial_provenance_context = None
        except Exception as exc:
            raise ValueError(
                "failed to load Phase A provenance context for initial Gaussian staging: "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc
    provenance_context = None
    if phase_b_records:
        try:
            from .state import read_state, DEFAULT_STATE_FILENAME
            from ..acquisition.trajectory_pool import TrajectoryPool

            current_state = read_state(
                Path(campaign_dir) / ".DATA" / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME
            )
            pool = TrajectoryPool.load(campaign_dir)
            provenance_context = {
                "campaign_uid": str(current_state.campaign_uid),
                "trajectory_sha256": str(pool.sha256),
            }
        except Exception as exc:
            raise ValueError(
                "failed to load provenance validation context for Phase B Gaussian staging: "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc
    g = config.gaussian
    staging = (
        Path(sample_xyz).parent
        if is_replacement
        else bucket_dir(campaign_dir, phase_name, iteration)
    )
    preserve_existing_layout = False
    if staging.exists():
        try:
            preserve_existing_layout = (
                len(_points_file_names(staging)) == len(frames)
            )
        except Exception:
            preserve_existing_layout = False
    # Keep an existing same-size layout so partial array recovery can reuse
    # completed Gaussian outputs.  If the layout is malformed or the task
    # count changed, clear it before staging fresh inputs.
    if staging.exists() and not preserve_existing_layout:
        _checked_rmtree(
            staging,
            campaign_dir=Path(campaign_dir),
            allowed_roots=[Path(campaign_dir) / ".DATA" / "STAGING"],
        )
    staging.mkdir(parents=True, exist_ok=True)

    keywords = ["nosymm", "output=wfn", "force", "geom=notest"]
    if str(g.extra_keywords).strip():
        keywords += str(g.extra_keywords).split()

    gaussian_resources = None
    if str(config.resources.gaussian_memory_mode_for()) == "link0":
        gaussian_resources = resolve_phase_resources(
            phase_name=str(phase_name),
            config=config,
            partition=str(partition_override or config.resources.partition_for(str(phase_name))),
            campaign_dir=campaign_dir,
            iteration=int(iteration),
            array_size=len(frames),
        )
        validate_gaussian_link0_memory(config, gaussian_resources)

    pointdirs: List[Path] = []
    for k, atoms in enumerate(frames):
        point_index = (
            int(allocation_records[k].get("pointdir_index", k))
            if is_replacement and allocation_records
            else int(k)
        )
        pd = staging / ("POINT_" + str(point_index).zfill(4) + ".pointdir")
        pd.mkdir(parents=True, exist_ok=True)
        gjf = GJF(
            pd / "input.gjf",
            method=str(g.method),
            basis_set=str(g.basis_set),
            keywords=list(keywords),
            charge=int(g.charge),
            spin_multiplicity=int(g.spin_multiplicity),
            atoms=atoms,
        )
        if gaussian_resources is not None:
            gjf.set_nproc(int(gaussian_resources.cpus_per_task))
            gjf.set_mem(str(config.resources.gaussian_link0_mem_for()))
        gjf.write()
        if (
            initial_provenance_context is not None
            and k < len(initial_seed_frame_ids)
        ):
            from ..versioning.provenance import write_seed_provenance

            write_seed_provenance(
                pd,
                campaign_uid=str(initial_provenance_context["campaign_uid"]),
                iteration=int(iteration),
                trajectory_sha256=str(initial_provenance_context["trajectory_sha256"]),
                seed_frame_id=initial_seed_frame_ids[k],
                seed_selection_origin=(
                    initial_seed_selection_origins[k]
                    if k < len(initial_seed_selection_origins)
                    else "phase_a_polus"
                ),
                seed_variance_at_selection=None,
                subspace_neighbour_frame_ids=[],
                subspace_dimension=0,
                subspace_eigenvalues=[],
                mode_weighting_policy="phase_a_diversity",
            )
        if phase_b_records:
            from ..versioning.provenance import (
                PROVENANCE_FILENAME,
                enrich_with_phase_b,
                validate_provenance,
            )

            src_prov = Path(str(phase_b_records[k].get("provenance_json", "")))
            if not src_prov.is_file():
                raise FileNotFoundError(
                    "Phase B provenance missing for staged Gaussian point "
                    + str(k)
                    + ": "
                    + str(src_prov)
                )
            if is_replacement:
                enrich_with_phase_b(
                    src_prov.parent,
                    selected_after_fps=True,
                    diversity_rank=phase_b_records[k].get("reserve_rank"),
                    descriptor_used="replacement_reserve",
                    candidate_id=str(phase_b_records[k]["candidate_id"]),
                    reserve_candidate=True,
                )
            validate_provenance(
                src_prov.parent,
                campaign_uid=str(provenance_context["campaign_uid"]),
                iteration=int(iteration),
                trajectory_sha256=str(provenance_context["trajectory_sha256"]),
                seed_frame_id=phase_b_records[k].get("seed_frame_id"),
                require_phase_b_selected=True,
            )
            shutil.copy2(str(src_prov), str(pd / PROVENANCE_FILENAME))
        if allocation_records:
            from ..versioning.provenance import enrich_with_point_allocation

            allocation_record = allocation_records[k]
            enrich_with_point_allocation(
                pd,
                candidate_id=str(allocation_record["candidate_id"]),
                context=(
                    replacement_context
                    if is_replacement
                    else ("bootstrap" if str(phase_name) == "INITIAL_GAUSSIAN" else "active")
                ),
                slot_id=int(allocation_record["slot_id"]),
                split=str(allocation_record["split"]),
                replacement_round=(
                    int(allocation_record.get("round", 0)) if is_replacement else 0
                ),
                allocation_manifest_sha256=allocation_manifest_hash,
            )
        pointdirs.append(pd)

    write_points_file(staging, pointdirs)
    return staging, len(pointdirs)


def stage_aimall_inputs(
    campaign_dir,
    config,
    phase_name,
    iteration,
    *,
    partition_override: Optional[str] = None,
    staging_override: Optional[Path] = None,
    expected_gaussian_phase: Optional[str] = None,
) -> Tuple[Path, int]:
    """AIMAll runs on the .wfn files Gaussian produced in the same bucket. The
    pointdirs already exist; rewrite POINTS.txt over the Gaussian-accepted
    pointdirs so the array only indexes ready points. Returns (dir, n_points)."""
    staging = (
        Path(staging_override)
        if staging_override is not None
        else bucket_dir(campaign_dir, phase_name, iteration)
    )
    expected_phase = str(
        expected_gaussian_phase
        or ("INITIAL_GAUSSIAN" if phase_name.startswith("INITIAL_") else "GAUSSIAN")
    )
    pointdirs, _manifest = read_quantum_acceptance_manifest(
        staging,
        expected_phase=expected_phase,
        expected_iteration=int(iteration),
        require_nonempty=False,
        require_points_file_membership=True,
    )
    aimall_resources = resolve_phase_resources(
        phase_name=str(phase_name),
        config=config,
        partition=str(partition_override or config.resources.partition_for(str(phase_name))),
        campaign_dir=campaign_dir,
        iteration=int(iteration),
        array_size=len(pointdirs),
    )
    aimall_cpus = int(aimall_resources.cpus_per_task)
    raw_naat = getattr(config.aimall, "naat", "auto")
    for pointdir in pointdirs:
        if not (pointdir / "input.wfn").is_file():
            raise FileNotFoundError(
                "Gaussian-accepted pointdir is missing input.wfn: " + str(pointdir)
            )
        try:
            atom_count = len(PointDirectory(pointdir).atoms)
        except Exception as exc:
            raise ValueError(
                "failed to count atoms for AIMAll pointdir: " + str(pointdir)
            ) from exc
        if atom_count <= 0:
            raise ValueError("AIMAll pointdir has no atoms: " + str(pointdir))
        if isinstance(raw_naat, str) and raw_naat.strip().lower() == "auto":
            resolved_naat = resolve_aimall_naat(config, aimall_cpus, atom_count)
        else:
            resolved_naat = int(raw_naat)
            if resolved_naat < 1 or resolved_naat > int(aimall_cpus):
                raise ValueError(
                    "aimall.naat must be in [1, resolved resources.aimall.cpus_per_task]"
                )
        atomic_write_json(
            pointdir / AIMALL_TASK_METADATA,
            {
                "schema_version": AIMALL_TASK_METADATA_SCHEMA_VERSION,
                "atom_count": int(atom_count),
                "nproc": int(aimall_cpus),
                "naat": int(resolved_naat),
                "resource_resolution": aimall_resources.journal_payload(
                    phase_name=str(phase_name)
                ),
            },
        )
    write_points_file(staging, pointdirs)
    return staging, len(pointdirs)


def record_allocation_quantum_results(
    campaign_dir: Path,
    *,
    context: str,
    iteration: int,
    staging_dir: Path,
    gaussian_phase: str,
    aimall_phase: str,
) -> Dict[str, Any]:
    """Join Gaussian and AIMAll outcomes into one exact allocation update."""
    from ..point_allocation import (
        pending_attempts,
        point_allocation_path,
        read_point_allocation,
        record_quantum_results,
    )
    from ..versioning.provenance import read_provenance, validate_provenance

    campaign = Path(campaign_dir)
    staging = Path(staging_dir)
    allocation_path = point_allocation_path(
        campaign,
        context=str(context),
        iteration=int(iteration),
    )
    allocation = read_point_allocation(allocation_path)
    pending = pending_attempts(allocation)
    gaussian_accepted, gaussian_manifest = read_quantum_acceptance_manifest(
        staging,
        expected_phase=str(gaussian_phase),
        expected_iteration=int(iteration),
        require_nonempty=False,
        require_points_file_membership=True,
    )
    pending_ids = {str(record["candidate_id"]) for record in pending}
    submitted_names = [Path(path).name for path in gaussian_accepted]
    submitted_names.extend(
        str(record.get("pointdir"))
        for record in list(gaussian_manifest.get("rejected") or [])
        if isinstance(record, dict)
    )
    if len(submitted_names) != len(set(submitted_names)):
        raise ValueError("Gaussian allocation handoff contains duplicate pointdirs")
    if len(submitted_names) != len(pending_ids):
        raise ValueError(
            "quantum staging task count does not match pending point allocation: "
            + str(len(submitted_names))
            + " staged, "
            + str(len(pending_ids))
            + " pending"
        )
    try:
        aimall_accepted, aimall_manifest = read_quantum_acceptance_manifest(
            staging,
            expected_phase=str(aimall_phase),
            expected_iteration=int(iteration),
            require_nonempty=False,
            require_points_file_membership=False,
        )
    except FileNotFoundError:
        if gaussian_accepted:
            raise
        aimall_accepted = []
        aimall_manifest = {"rejected": []}

    gaussian_accepted_names = {Path(path).name for path in gaussian_accepted}
    aimall_accepted_names = {Path(path).name for path in aimall_accepted}
    gaussian_rejected = {
        str(record.get("pointdir")): str(record.get("reason") or "gaussian_rejected")
        for record in list(gaussian_manifest.get("rejected") or [])
        if isinstance(record, dict)
    }
    aimall_rejected = {
        str(record.get("pointdir")): str(record.get("reason") or "aimall_rejected")
        for record in list(aimall_manifest.get("rejected") or [])
        if isinstance(record, dict)
    }
    results: List[Dict[str, Any]] = []
    observed_ids: set[str] = set()
    pending_by_id = {str(record["candidate_id"]): record for record in pending}
    for name in submitted_names:
        pointdir = staging / name
        provenance = read_provenance(pointdir)
        allocation_provenance = provenance.get("point_allocation")
        if not isinstance(allocation_provenance, dict):
            raise ValueError("staged pointdir lacks allocation provenance: " + name)
        candidate_id = str(allocation_provenance.get("candidate_id") or "")
        if candidate_id not in pending_by_id or candidate_id in observed_ids:
            raise ValueError(
                "staged pointdir candidate does not match pending allocation: " + name
            )
        attempt = pending_by_id[candidate_id]
        validate_provenance(
            pointdir,
            allocation_candidate_id=candidate_id,
            allocation_context=str(context),
            allocation_slot_id=int(attempt["slot_id"]),
            allocation_split=str(attempt["split"]),
        )
        observed_ids.add(candidate_id)
        if name in aimall_accepted_names:
            accepted = True
            reason = None
        elif name in gaussian_rejected:
            accepted = False
            reason = gaussian_rejected[name]
        elif name in aimall_rejected:
            accepted = False
            reason = aimall_rejected[name]
        elif name in gaussian_accepted_names:
            accepted = False
            reason = "aimall_result_missing"
        else:
            accepted = False
            reason = "gaussian_result_missing"
        results.append(
            {
                "candidate_id": candidate_id,
                "accepted": bool(accepted),
                "pointdir": str(pointdir.resolve()),
                "reason": reason,
                "quality_manifest": str(
                    (staging / "quantum_quality.json").resolve(strict=False)
                ),
            }
        )
    if observed_ids != pending_ids:
        raise ValueError("quantum staging does not cover every pending allocation candidate")
    return record_quantum_results(
        allocation_path,
        results,
        expected_generation=int(allocation.get("generation", 0)),
    )


def accepted_allocation_pointdirs(
    campaign_dir: Path,
    *,
    context: str,
    iteration: int,
) -> Tuple[List[Path], Dict[str, Any]]:
    """Resolve and verify the exact accepted slot set for a commit."""
    from ..point_allocation import (
        accepted_attempts,
        point_allocation_path,
        read_point_allocation,
    )
    from ..versioning.provenance import validate_provenance

    campaign = Path(campaign_dir).resolve(strict=False)
    allocation_path = point_allocation_path(
        campaign,
        context=str(context),
        iteration=int(iteration),
    )
    allocation = read_point_allocation(allocation_path)
    if not bool((allocation.get("summary") or {}).get("complete", False)):
        raise ValueError("point allocation is incomplete: " + str(allocation_path))
    attempts = sorted(accepted_attempts(allocation), key=lambda row: int(row["slot_id"]))
    target_total = int(allocation["targets"]["total"])
    if len(attempts) != target_total:
        raise ValueError("complete point allocation has the wrong accepted count")
    staging_root = (campaign / ".DATA" / "STAGING").resolve(strict=False)
    pointdirs: List[Path] = []
    seen: set[Path] = set()
    for attempt in attempts:
        pointdir = Path(str(attempt.get("pointdir") or ""))
        resolved = pointdir.resolve(strict=False)
        if resolved in seen:
            raise ValueError("point allocation reuses an accepted pointdir")
        if staging_root not in resolved.parents:
            raise ValueError(
                "accepted allocation pointdir is outside daemon staging: "
                + str(pointdir)
            )
        if not resolved.is_dir() or resolved.is_symlink():
            raise FileNotFoundError(
                "accepted allocation pointdir is missing or symlinked: "
                + str(pointdir)
            )
        validate_provenance(
            resolved,
            allocation_candidate_id=str(attempt["candidate_id"]),
            allocation_context=str(context),
            allocation_slot_id=int(attempt["slot_id"]),
            allocation_split=str(attempt["split"]),
        )
        seen.add(resolved)
        pointdirs.append(resolved)
    return pointdirs, allocation


def verify_committed_allocation_snapshot(
    campaign_dir: Path,
    *,
    reference_data_version: int,
    context: str,
    iteration: int,
) -> Dict[str, Any]:
    """Prove that a committed delta reproduces one completed allocation."""
    from ..point_allocation import accepted_attempts, point_allocation_path, read_point_allocation
    from ..versioning.reference_data import ReferenceDataVersioning

    campaign = Path(campaign_dir)
    version = int(reference_data_version)
    versioning = ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA")
    source_path = point_allocation_path(
        campaign,
        context=str(context),
        iteration=int(iteration),
    )
    source = read_point_allocation(source_path)
    if not bool((source.get("summary") or {}).get("complete", False)):
        raise ValueError("committed allocation source is incomplete: " + str(source_path))
    committed_dir = versioning.iteration_path(version)
    snapshot_path = committed_dir / (
        "POINT_ALLOCATION.version-" + str(version).zfill(4) + ".json"
    )
    if not snapshot_path.is_file() or snapshot_path.is_symlink():
        raise FileNotFoundError(
            "committed reference-data version lacks its allocation snapshot: "
            + str(snapshot_path)
        )
    view = versioning.resolve(version, verification="deep")
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            "committed allocation snapshot is unreadable: " + str(snapshot_path)
        ) from exc
    if snapshot != source:
        raise ValueError("committed allocation snapshot does not match the source")
    expected = {
        str(record["candidate_id"]): (int(record["slot_id"]), str(record["split"]))
        for record in accepted_attempts(source)
    }
    observed = {
        entry.candidate_id: (entry.slot_id, entry.split)
        for entry in view.entries
        if entry.introduced_in_version == version
    }
    if observed != expected:
        raise ValueError(
            "committed reference-data delta does not reproduce allocation provenance"
        )
    return source


def commit_reference_data_delta(
    campaign_dir: Path,
    *,
    reference_data_version: int,
    context: str,
    iteration: int,
) -> Tuple[Any, List[str], bool]:
    """Commit exactly one immutable QM reference-data delta."""
    from ..point_allocation import (
        accepted_attempts,
        allocation_manifest_sha256,
        point_allocation_path,
    )
    from ..versioning.manifest import sha256_file
    from ..versioning.provenance import PROVENANCE_FILENAME
    from ..versioning.reference_data import (
        POINTDIR_NAME_WIDTH,
        REFERENCE_DATA_VERSION_FILENAME,
        ReferenceDataEntry,
        ReferenceDataVersioning,
        build_reference_data_version_payload,
        hash_pointdir_tree as hash_reference_pointdir_tree,
        seal_reference_data_version,
    )

    campaign = Path(campaign_dir)
    version = int(reference_data_version)
    versioning = ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA")
    allocation_path = point_allocation_path(
        campaign,
        context=str(context),
        iteration=int(iteration),
    )
    committed_versions = versioning.list_committed_versions()
    if version in committed_versions:
        verify_committed_allocation_snapshot(
            campaign,
            reference_data_version=version,
            context=str(context),
            iteration=int(iteration),
        )
        view = versioning.resolve(version, verification="deep")
        seal_reference_data_version(versioning.iteration_path(version))
        newest_version = max(committed_versions)
        if newest_version != version:
            versioning.resolve(newest_version, verification="deep")
        versioning.ensure_current(newest_version)
        names = [
            entry.pointdir_name
            for entry in view.entries
            if entry.introduced_in_version == version
        ]
        return view, names, False
    if not allocation_path.is_file():
        raise FileNotFoundError(
            "point-allocation manifest missing: " + str(allocation_path)
        )
    accepted_pointdirs, allocation = accepted_allocation_pointdirs(
        campaign,
        context=str(context),
        iteration=int(iteration),
    )
    attempts = sorted(accepted_attempts(allocation), key=lambda row: int(row["slot_id"]))
    if len(attempts) != len(accepted_pointdirs):
        raise ValueError("reference-data allocation/pointdir cardinality mismatch")
    parent_view = None
    if version > 0:
        parent_view = versioning.resolve(version - 1, verification="metadata")
    elif versioning.list_committed_versions():
        raise ValueError("reference-data bootstrap is not the first commit")
    existing_ids = {
        entry.candidate_id for entry in (parent_view.entries if parent_view else ())
    }
    if any(str(attempt["candidate_id"]) in existing_ids for attempt in attempts):
        raise ValueError("reference-data delta reuses a committed candidate ID")

    versioning.recover_dangling_staging()
    staging = versioning.stage(source_version=None, target_version=version)
    first_ordinal = len(parent_view.entries) if parent_view is not None else 0
    added_entries: List[ReferenceDataEntry] = []
    added_names: List[str] = []
    for offset, (source, attempt) in enumerate(zip(accepted_pointdirs, attempts)):
        ordinal = first_ordinal + offset
        name = "POINT_" + str(ordinal).zfill(POINTDIR_NAME_WIDTH) + ".pointdir"
        destination = staging / name
        _copytree_no_symlinks(source, destination)
        lock = destination / ".provenance.lock"
        if lock.exists():
            lock.unlink()
        provenance_path = destination / PROVENANCE_FILENAME
        if not provenance_path.is_file() or provenance_path.is_symlink():
            raise ValueError("committed reference point lacks provenance: " + str(destination))
        added_entries.append(
            ReferenceDataEntry(
                global_ordinal=int(ordinal),
                introduced_in_version=version,
                pointdir_name=name,
                pointdir_path=destination.resolve(),
                candidate_id=str(attempt["candidate_id"]),
                slot_id=int(attempt["slot_id"]),
                split=str(attempt["split"]),
                replacement_round=int(attempt.get("round", 0)),
                pointdir_tree_sha256=hash_reference_pointdir_tree(destination),
                provenance_sha256=sha256_file(provenance_path),
            )
        )
        added_names.append(name)

    allocation_relative = allocation_path.resolve().relative_to(campaign.resolve()).as_posix()
    payload = build_reference_data_version_payload(
        campaign_uid=str(allocation["campaign_uid"]),
        version=version,
        source_context=str(context),
        source_iteration=int(iteration),
        parent_view=parent_view,
        point_allocation_manifest=allocation_relative,
        point_allocation_sha256=allocation_manifest_sha256(allocation_path),
        added_entries=added_entries,
    )
    allocation_history = allocation_path.parent / ".point_allocation_history"
    if allocation_history.is_dir():
        _copytree_no_symlinks(
            allocation_history,
            staging / ".point_allocation_history",
        )
    atomic_write_json(staging / "POINT_ALLOCATION.json", allocation)
    atomic_write_json(
        staging / ("POINT_ALLOCATION.version-" + str(version).zfill(4) + ".json"),
        allocation,
    )
    atomic_write_json(staging / REFERENCE_DATA_VERSION_FILENAME, payload)
    versioning.commit(version)
    view = versioning.resolve(version, verification="deep")
    seal_reference_data_version(versioning.iteration_path(version))
    versioning.update_current(version)
    return view, added_names, True


def commit_initial_reference_data(campaign_dir) -> bool:
    """Commit the complete bootstrap allocation as reference-data version 0."""
    allocation_path = (
        Path(campaign_dir)
        / "3_DIVERSITY_SAMPLING"
        / "initial"
        / "POINT_ALLOCATION.json"
    )
    _, _, created = commit_reference_data_delta(
        Path(campaign_dir),
        reference_data_version=0,
        context="bootstrap",
        iteration=0,
    )
    return bool(created)


def stage_ferebus_inputs(
    campaign_dir,
    config,
    reference_data_version,
    is_initial=False,
) -> Tuple[Path, int]:
    """Stage pyferebus/FEREBUS inputs and return (staging_dir, n_tasks).

    pyferebus owns the FEREBUS folder/config/command/slurm contract. The daemon stages the flat
    property directories that pyferebus expects, plus the pyferebus job-details file and a daemon
    manifest used for strict postprocess validation.
    """
    from ichor.core.files import PointDirectory, PointsDirectory
    from ichor.core.calculators import calculate_alf_atom_sequence
    from ..versioning.reference_data import ReferenceDataVersioning

    campaign = Path(campaign_dir)
    # the initial bootstrap has no APPEND phase ahead of it, so QM_REFERENCE_DATA/iteration-0 may not
    # exist yet when INITIAL_FEREBUS stages. build it from the initial quantum staging so the
    # export has something to read. idempotent; and we only create the dir here -- the
    # reference_data_version bump stays in postprocess so a restart mid-flight reconciles cleanly.
    if is_initial:
        reference_data_version = 0
        commit_initial_reference_data(campaign)
    version = int(reference_data_version)
    reference_data = ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA")
    view = reference_data.resolve(version, verification="deep")
    if not view.entries:
        raise ValueError("committed QM reference data contains no pointdirs")

    staging = campaign / "6_TRAINED_MODELS" / "iteration-staging"
    # iteration-staging is ONE shared scratch dir reused every iteration, so wipe it first.
    # otherwise last iteration's *_train.csv / *.model lying around get globbed back in and we
    # either split stale data or re-commit an old model as a fresh version (both silent + nasty).
    if staging.exists():
        _checked_rmtree(
            staging,
            campaign_dir=campaign,
            allowed_roots=[campaign / "6_TRAINED_MODELS"],
        )
    staging.mkdir(parents=True, exist_ok=True)

    # PointsDirectory is list-backed, so the daemon can supply the authoritative
    # cumulative order without materialising a duplicate directory tree.
    pd = PointsDirectory(campaign / "QM_REFERENCE_DATA", needs_parsing=False)
    pd.path = campaign / "QM_REFERENCE_DATA"
    for entry in view.entries:
        pd.append(PointDirectory(entry.pointdir_path))
    pointdir_names = [entry.pointdir_name for entry in view.entries]
    observed_order = [Path(getattr(point, "path", point)).name for point in pd]
    if observed_order != pointdir_names:
        raise ValueError("FEREBUS PointsDirectory row order differs from reference-data view")
    pointdir_identities = {
        entry.pointdir_name: entry.provenance_sha256
        for entry in view.entries
    }
    forced_ferebus_splits = {
        entry.pointdir_name: entry.split for entry in view.entries
    }
    # system ALF defines the per-atom local frame the features are built in.
    system_alf = pd.alf_dict(calculate_alf_atom_sequence)
    f = config.ferebus
    properties = [str(p) for p in getattr(f, "properties", ["iqa"])]
    if not properties:
        raise ValueError("ferebus.properties must contain at least one property")

    # write one <atom>_train.csv per atom containing every configured target property.
    cwd = os.getcwd()
    try:
        os.chdir(staging)
        pd.features_with_properties_to_csv(
            system_alf,
            str_to_append_to_fname="_train.csv",
            property_types=properties,
        )
    finally:
        os.chdir(cwd)
    feature_csvs = sorted(staging.glob("*_train.csv"))
    if not feature_csvs:
        raise ValueError("PointsDirectory export produced no *_train.csv files for FEREBUS")

    system = str(getattr(config.campaign, "system_name", "SYSTEM"))
    from . import ferebus_dataset as _fds
    from .ferebus_split_ledger import ensure_split_assignments
    from ..point_allocation import (
        allocation_manifest_sha256,
        allocation_targets,
        point_allocation_path,
        read_point_allocation,
    )

    allocation_context = "bootstrap" if version == 0 else "active"
    allocation_iteration = 0 if allocation_context == "bootstrap" else version - 1
    allocation_path = point_allocation_path(
        campaign,
        context=allocation_context,
        iteration=allocation_iteration,
    )
    allocation_payload = read_point_allocation(allocation_path)
    if not bool((allocation_payload.get("summary") or {}).get("complete", False)):
        raise ValueError(
            "FEREBUS staging requires a complete point allocation: "
            + str(allocation_path)
        )
    allocation_hash = allocation_manifest_sha256(allocation_path)
    expected_new_counts = allocation_targets(config, allocation_context)
    expected_new_counts.pop("total", None)

    split_ledger = ensure_split_assignments(
        campaign,
        pointdir_names,
        reference_data_version=version,
        reference_data_view_sha256=str(view.cumulative_view_sha256),
        expected_new_counts=expected_new_counts,
        pointdir_identity=pointdir_identities,
        forced_splits=forced_ferebus_splits,
        allocation_manifest_sha256=allocation_hash,
    )
    ledger_row_ids = dict(split_ledger["row_ids"])
    atom_labels = []
    alf_by_atom: Dict[str, List[int]] = {}
    split_counts: Dict[str, Dict[str, int]] = {}
    row_ids_by_atom: Dict[str, Dict[str, List[int]]] = {}
    stats_by_prop_atom: Dict[Tuple[str, str], Dict[str, float]] = {}
    prop_dirs = {prop: staging / prop for prop in properties}
    for prop_dir in prop_dirs.values():
        prop_dir.mkdir(parents=True, exist_ok=True)
    degenerate_property_stats: List[Dict[str, Any]] = []
    for atom_csv in feature_csvs:
        # filename is "<atom>_train.csv"; recover the atom label.
        atom = atom_csv.name[:-len("_train.csv")]
        validate_safe_path_token("FEREBUS atom label", atom)
        if atom not in system_alf:
            raise ValueError("system ALF is missing atom " + atom)
        row_count = _fds._row_count(atom_csv)
        if int(row_count) != len(pointdir_names):
            raise ValueError(
                "FEREBUS CSV row count for "
                + str(atom_csv)
                + " is "
                + str(row_count)
                + " but training pointdir count is "
                + str(len(pointdir_names))
            )
        alf_by_atom[atom] = _alf_to_ferebus(system_alf[atom], atom)
        split = _fds.split_atom_csv_to_property_dirs(
            atom_csv,
            prop_dirs,
            system,
            atom,
            properties,
            row_ids=ledger_row_ids,
        )
        counts = dict(split["counts"])
        atom_labels.append(atom)
        split_counts[atom] = counts
        row_ids_by_atom[atom] = dict(split["row_ids"])
        for prop in properties:
            train_csv = prop_dirs[prop] / (system + "_" + atom + "_TRAINING_SET.csv")
            stats = _fds.prop_stats(train_csv, prop)
            if not stats:
                raise ValueError(
                    "could not compute target-property stats for "
                    + prop
                    + "-"
                    + atom
                    + " from "
                    + str(train_csv)
                )
            stats_by_prop_atom[(prop, atom)] = stats
            if bool(stats.get("degenerate_property_stats", False)):
                degenerate_property_stats.append({
                    "property": str(prop),
                    "atom": str(atom),
                    "std": float(stats["std"]),
                    "range": float(stats["range"]),
                })
    n_atoms = len(atom_labels)
    atoms_file = staging / "ATOMS.txt"
    atoms_file.write_text(
        chr(10).join(atom_labels) + (chr(10) if atom_labels else ""),
        encoding="utf-8",
        newline="\n",
    )
    props_file = staging / "PROPERTIES.txt"
    props_file.write_text(
        chr(10).join(properties) + chr(10),
        encoding="utf-8",
        newline="\n",
    )

    job_details = _write_pyferebus_job_details(
        staging / FEREBUS_JOB_DETAILS,
        system=system,
        atoms=atom_labels,
        properties=properties,
        alf_by_atom=alf_by_atom,
        stats_by_prop_atom=stats_by_prop_atom,
    )

    tasks: List[Dict[str, Any]] = []
    task_index = 1
    for prop in properties:
        for atom in atom_labels:
            output_dir = staging / prop / atom
            input_dir = output_dir / "datasets"
            config_path = output_dir / "ferebus.config"
            alf_cli = "_".join(str(x) for x in alf_by_atom[atom])
            training_csv = input_dir / (system + "_" + atom + "_TRAINING_SET.csv")
            int_csv = input_dir / (system + "_" + atom + "_INT_VALIDATION_SET.csv")
            ext_csv = input_dir / (system + "_" + atom + "_EXT_VALIDATION_SET.csv")
            model_path = output_dir / (system + "_" + prop + "_" + atom + ".model")
            command = (
                "-c "
                + str(config_path)
                + "  -I "
                + str(input_dir)
                + " -O "
                + str(output_dir)
                + " -P "
                + prop
                + " -A "
                + atom
                + " -ALF "
                + alf_cli
            )
            tasks.append(
                {
                    "task_index": int(task_index),
                    "property": prop,
                    "atom": atom,
                    "alf_1_indexed": [int(x) for x in alf_by_atom[atom]],
                    "alf_cli": alf_cli,
                    "property_dir": str((staging / prop).resolve()),
                    "output_dir": str(output_dir.resolve()),
                    "input_dir": str(input_dir.resolve()),
                    "config_path": str(config_path.resolve()),
                    "training_csv": str(training_csv.resolve()),
                    "int_validation_csv": str(int_csv.resolve()),
                    "ext_validation_csv": str(ext_csv.resolve()),
                    "expected_model_path": str(model_path.resolve()),
                    "command": command,
                    "row_counts": dict(split_counts[atom]),
                    "row_ids": dict(row_ids_by_atom[atom]),
                    "degenerate_property_stats": bool(
                        stats_by_prop_atom.get((prop, atom), {}).get(
                            "degenerate_property_stats", False
                        )
                    ),
                }
            )
            task_index += 1

    _write_ferebus_manifest(
        staging,
        {
            "schema_version": FEREBUS_TASK_SCHEMA_VERSION,
            "system": system,
            "reference_data_version": version,
            "reference_data_head_manifest_sha256": str(view.head_manifest_sha256),
            "reference_data_view_sha256": str(view.cumulative_view_sha256),
            "n_reference_points": int(len(view.entries)),
            "pointdir_row_order": list(pointdir_names),
            "properties": properties,
            "atoms": atom_labels,
            "n_atoms": int(n_atoms),
            "n_tasks": int(len(tasks)),
            "degenerate_property_stats": list(degenerate_property_stats),
            "job_details": str(job_details.resolve()),
            "split_ledger": {
                "path": str(split_ledger["path"]),
                "counts": dict(split_ledger["counts"]),
                "version_allocation": dict(split_ledger["version_allocation"]),
                "allocation_policy": str(split_ledger["allocation_policy"]),
                "allocation_manifest": str(allocation_path.resolve()),
                "allocation_manifest_sha256": str(allocation_hash),
                "forced_splits": dict(forced_ferebus_splits),
            },
            "tasks": tasks,
        },
    )
    return staging, len(tasks)
