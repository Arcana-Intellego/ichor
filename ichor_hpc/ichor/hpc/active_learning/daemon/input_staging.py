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

from ..strict_json import strict_json as json
import csv
import os  # stage_ferebus_inputs cd's into the staging dir to export csvs; this was missing and only bit on a live run
import re
import shutil
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ichor.core.atoms import Atoms
from ichor.core.files import PointDirectory, WFN
from ichor.core.files.xyz import Trajectory

from .resource_solver import (
    resolve_phase_resources,
    wfn_primitive_count,
)
from .filesystem import campaign_owned_path
from .state import atomic_write_json, atomic_write_text
from ..config import normalise_gaussian_route_keywords
from ..layout import (
    COMMITTED_VERSION_NAME_WIDTH,
    TRAINED_MODELS_DIRNAME,
    parse_staging_pointdir_name,
    staging_pointdir_name,
    staging_phase_dir,
    staging_root,
    trained_models_dir,
)
from ..versioning.manifest import sha256_file


QUANTUM_ACCEPTANCE_MANIFEST = "accepted_pointdirs.json"
QUANTUM_ACCEPTANCE_SCHEMA_VERSION = 2
AIMALL_TASK_METADATA = "AIMALL_TASK.json"
AIMALL_TASK_METADATA_SCHEMA_VERSION = 2
WFN_METHOD_RECEIPT = "WFN_METHOD_RECEIPT.json"
WFN_METHOD_RECEIPT_SCHEMA_VERSION = 1
FEREBUS_TASK_MANIFEST = "FEREBUS_TASKS.json"
FEREBUS_TASK_SCHEMA_VERSION = 5
FEREBUS_ROW_IDENTITIES = "FEREBUS_ROW_IDENTITIES.json"
FEREBUS_ROW_IDENTITIES_SCHEMA_VERSION = 1
FEREBUS_SPLIT_SNAPSHOT = "FEREBUS_SPLIT_ASSIGNMENTS.json"
FEREBUS_JOB_DETAILS = "job-details"
SAFE_PATH_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def bucket_dir(campaign_dir, phase_name: str, iteration: int) -> Path:
    return staging_phase_dir(campaign_dir, phase_name, int(iteration))


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
    atomic_write_text(points_file, body + ("\n" if body else ""))
    return points_file


def rewrite_wfn_for_aimall(
    wfn_path: Path,
    *,
    method: str,
    phase_name: str,
    iteration: int,
    task_index: int,
    source_acceptance_sha256: str,
) -> Tuple[Path, Dict[str, Any]]:
    """Atomically inject the campaign method into a Gaussian WFN."""
    from .quantum_quality import canonicalise_aimall_method

    wfn = Path(wfn_path)
    if wfn.is_symlink() or not wfn.is_file():
        raise FileNotFoundError("AIMAll WFN is missing or symlinked: " + str(wfn))
    canonical_method = canonicalise_aimall_method(method)
    before_sha256 = sha256_file(wfn)
    parsed = WFN(wfn)
    parsed.read()
    parsed.method = canonical_method
    rendered = parsed.render(wfn)
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("WFN method rewrite produced empty output: " + str(wfn))
    atomic_write_text(wfn, rendered)

    header = wfn.read_text(encoding="utf-8").splitlines()[1].split()
    observed_method = (
        "HF"
        if header[-1].upper() == "NUCLEI"
        else canonicalise_aimall_method(header[-1])
    )
    if observed_method != canonical_method:
        raise ValueError(
            "rewritten WFN method mismatch: expected "
            + canonical_method
            + ", observed "
            + observed_method
        )
    after_sha256 = sha256_file(wfn)
    payload = {
        "schema_version": WFN_METHOD_RECEIPT_SCHEMA_VERSION,
        "phase": str(phase_name),
        "iteration": int(iteration),
        "task_index": int(task_index),
        "pointdir": str(wfn.parent.name),
        "method": canonical_method,
        "source_gaussian_acceptance_sha256": str(source_acceptance_sha256),
        "wfn": {
            "path": wfn.name,
            "before_sha256": before_sha256,
            "after_sha256": after_sha256,
            "size_bytes": int(wfn.stat().st_size),
        },
    }
    receipt_path = wfn.parent / WFN_METHOD_RECEIPT
    atomic_write_json(receipt_path, payload)
    return receipt_path, payload


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


def _copy_atomic_checked_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError("refusing non-regular evidence file: " + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != sha256_file(source):
        temporary.unlink(missing_ok=True)
        raise ValueError("evidence file copy verification failed: " + str(source))
    os.replace(temporary, destination)


def _validate_pointdir_basename(name: str) -> str:
    text = str(name).strip()
    if Path(text).name != text:
        raise ValueError("unsafe pointdir name in manifest: " + repr(name))
    try:
        parse_staging_pointdir_name(text)
    except ValueError as exc:
        raise ValueError("unsafe pointdir name in manifest: " + repr(name)) from exc
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
    if not isinstance(phase_name, str) or not phase_name:
        raise ValueError("quantum acceptance phase must be a non-empty string")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("quantum acceptance iteration must be a non-negative integer")
    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    accepted_names = [_validate_pointdir_basename(_pointdir_name(p)) for p in accepted]
    rejected_payload = []
    for name, reason in rejected:
        reason_text = str(reason).strip()
        if not reason_text:
            raise ValueError("quantum rejection reason must be non-empty")
        rejected_payload.append(
            {
                "pointdir": _validate_pointdir_basename(str(name)),
                "reason": reason_text,
            }
        )
    all_names = accepted_names + [record["pointdir"] for record in rejected_payload]
    if len(all_names) != len(set(all_names)):
        raise ValueError("quantum acceptance contains duplicate dispositions")
    payload: Dict[str, Any] = {
        "schema_version": QUANTUM_ACCEPTANCE_SCHEMA_VERSION,
        "phase": phase_name,
        "iteration": iteration,
        "accepted_pointdirs": accepted_names,
        "rejected": rejected_payload,
        "n_total": int(len(accepted_names) + len(rejected_payload)),
    }
    phase_path = quantum_acceptance_manifest_path(staging, phase_name=phase_name)
    atomic_write_json(phase_path, payload)
    # Retain the latest-phase convenience copy for user tooling. Consumers
    # always read the immutable phase-specific handoff.
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
    path = phase_path
    if not path.is_file():
        raise FileNotFoundError("quantum acceptance manifest missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("quantum acceptance manifest unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise ValueError("quantum acceptance manifest must be a JSON object: " + str(path))
    if (
        not isinstance(data.get("schema_version"), int)
        or isinstance(data.get("schema_version"), bool)
        or data["schema_version"] != QUANTUM_ACCEPTANCE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported quantum acceptance manifest schema: " + str(path))
    if data.get("phase") != expected_phase:
        raise ValueError(
            "quantum acceptance manifest phase mismatch: expected "
            + expected_phase + " got " + str(data.get("phase"))
        )
    iteration = data.get("iteration")
    if not isinstance(iteration, int) or isinstance(iteration, bool):
        raise ValueError("quantum acceptance manifest iteration is not an integer")
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
    if not isinstance(n_total, int) or isinstance(n_total, bool):
        raise ValueError("quantum acceptance manifest n_total is not an integer")
    if n_total != len(accepted) + len(rejected):
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
    normalised_rejected = []
    for record in rejected:
        if not isinstance(record, dict):
            raise ValueError("quantum rejection record must be an object")
        if set(record) != {"pointdir", "reason"}:
            raise ValueError("quantum rejection record has unknown or missing fields")
        name = _validate_pointdir_basename(record.get("pointdir"))
        reason = record.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("quantum rejection reason must be a non-empty string")
        if name in seen:
            raise ValueError("duplicate or contradictory pointdir disposition: " + name)
        if points_names is not None and name not in points_names:
            raise ValueError("rejected pointdir is not present in POINTS.txt: " + name)
        seen.add(name)
        normalised_rejected.append({"pointdir": name, "reason": reason.strip()})
    if points_names is not None and seen != points_names:
        missing = sorted(points_names - seen)
        raise ValueError(
            "quantum acceptance does not cover every POINTS.txt task: "
            + repr(missing[:8])
        )
    if require_nonempty and not resolved:
        raise ValueError("quantum acceptance manifest accepted_pointdirs is empty: " + str(path))
    out = dict(data)
    out["rejected"] = normalised_rejected
    out["rejections_by_pointdir"] = {
        record["pointdir"]: record["reason"] for record in normalised_rejected
    }
    return resolved, out


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


def _exact_ferebus_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(label + " must be an exact JSON integer")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(label + " must be >= " + str(minimum))
    return parsed


def ferebus_relative_path(staging_dir: Path, path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(staging_dir).resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("FEREBUS task path escapes staging: " + str(path)) from exc


def resolve_ferebus_task_path(
    staging_dir: Path,
    raw_path: Any,
    label: str,
) -> Path:
    text = str(raw_path or "")
    if not text or "\\" in text:
        raise ValueError(label + " must be a non-empty relative POSIX path")
    relative = Path(*text.split("/"))
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(label + " is not a safe relative path: " + repr(text))
    root = Path(staging_dir).resolve(strict=False)
    path = root / relative
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ValueError(label + " escapes FEREBUS staging: " + repr(text)) from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(label + " contains a symlink: " + str(current))
    return path


def read_ferebus_manifest(
    staging_dir: Path,
    *,
    verify_dataset_files: bool = True,
) -> Dict[str, Any]:
    path = ferebus_manifest_path(Path(staging_dir))
    if not path.is_file():
        raise FileNotFoundError("FEREBUS task manifest missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("FEREBUS task manifest unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise ValueError("FEREBUS task manifest must be a JSON object: " + str(path))
    if _exact_ferebus_int(data.get("schema_version"), "FEREBUS task schema") != FEREBUS_TASK_SCHEMA_VERSION:
        raise ValueError("unsupported FEREBUS task manifest schema: " + str(path))
    campaign_uid = str(data.get("campaign_uid") or "")
    system = str(data.get("system") or "")
    if not campaign_uid or not system:
        raise ValueError("FEREBUS task manifest campaign/system identity is invalid")
    validate_safe_path_token("FEREBUS system", system)
    try:
        reference_data_version = _exact_ferebus_int(
            data["reference_data_version"], "reference_data_version"
        )
        n_reference_points = _exact_ferebus_int(
            data["n_reference_points"], "n_reference_points", minimum=1
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("FEREBUS task manifest reference-data binding is invalid") from exc
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
    row_identity_binding = data.get("row_identity_snapshot")
    if not isinstance(row_identity_binding, dict):
        raise ValueError("FEREBUS row-identity snapshot binding is invalid")
    row_identity_path = resolve_ferebus_task_path(
        staging_dir,
        row_identity_binding.get("path"),
        "row_identity_snapshot.path",
    )
    if str(row_identity_binding.get("path")) != FEREBUS_ROW_IDENTITIES:
        raise ValueError("FEREBUS row-identity snapshot path is invalid")
    row_identity_payload = None
    if verify_dataset_files:
        if not row_identity_path.is_file() or row_identity_path.is_symlink():
            raise ValueError("FEREBUS row-identity snapshot is missing")
        if _exact_ferebus_int(
            row_identity_binding.get("size"),
            "row-identity snapshot size",
        ) != int(row_identity_path.stat().st_size):
            raise ValueError("FEREBUS row-identity snapshot size mismatch")
        if str(row_identity_binding.get("sha256") or "") != sha256_file(row_identity_path):
            raise ValueError("FEREBUS row-identity snapshot SHA-256 mismatch")
        row_identity_payload = json.loads(row_identity_path.read_text(encoding="utf-8"))
        if not isinstance(row_identity_payload, dict) or _exact_ferebus_int(
            row_identity_payload.get("schema_version"),
            "row-identity schema",
        ) != FEREBUS_ROW_IDENTITIES_SCHEMA_VERSION:
            raise ValueError("FEREBUS row-identity snapshot schema is invalid")
        source_rows = row_identity_payload.get("source_rows")
        if not isinstance(source_rows, list) or [
            str(record.get("pointdir_name") or "")
            for record in source_rows
            if isinstance(record, dict)
        ] != row_order:
            raise ValueError("FEREBUS row identities do not match pointdir row order")
        from ..versioning.reference_data import canonical_json_sha256

        if str(row_identity_payload.get("source_rows_sha256") or "") != canonical_json_sha256(source_rows):
            raise ValueError("FEREBUS source-row identity digest mismatch")
        if str(row_identity_binding.get("source_rows_sha256") or "") != str(
            row_identity_payload.get("source_rows_sha256") or ""
        ):
            raise ValueError("FEREBUS row-identity binding digest mismatch")
    kernel_contract = data.get("kernel_contract")
    if not isinstance(kernel_contract, dict):
        raise ValueError("FEREBUS kernel contract is invalid")
    from ..ferebus_prior import backend_kernel_token

    family = kernel_contract.get("family")
    if kernel_contract != {
        "family": family,
        "backend_token": backend_kernel_token(family),
        "loss": "huber",
        "constant_noise": True,
        "full_ard": True,
        "feature_scaling": True,
        "property_scaling": False,
        "kernel_prefactor_mode": 2,
    }:
        raise ValueError("FEREBUS kernel contract contains unsupported settings")
    split_binding = data.get("split_ledger")
    if not isinstance(split_binding, dict):
        raise ValueError("FEREBUS split-ledger binding is invalid")
    split_snapshot_path = resolve_ferebus_task_path(
        staging_dir,
        split_binding.get("path"),
        "split_ledger.path",
    )
    if str(split_binding.get("path")) != FEREBUS_SPLIT_SNAPSHOT:
        raise ValueError("FEREBUS split-ledger snapshot path is invalid")
    if verify_dataset_files:
        if not split_snapshot_path.is_file() or split_snapshot_path.is_symlink():
            raise ValueError("FEREBUS split-ledger snapshot is missing")
        if _exact_ferebus_int(
            split_binding.get("size"), "split-ledger snapshot size"
        ) != int(split_snapshot_path.stat().st_size):
            raise ValueError("FEREBUS split-ledger snapshot size mismatch")
        if str(split_binding.get("sha256") or "") != sha256_file(split_snapshot_path):
            raise ValueError("FEREBUS split-ledger snapshot SHA-256 mismatch")
        split_payload = json.loads(split_snapshot_path.read_text(encoding="utf-8"))
        from .ferebus_split_ledger import FEREBUS_SPLIT_LEDGER_SCHEMA_VERSION

        if not isinstance(split_payload, dict) or _exact_ferebus_int(
            split_payload.get("schema_version"), "split-ledger schema"
        ) != FEREBUS_SPLIT_LEDGER_SCHEMA_VERSION:
            raise ValueError("FEREBUS split-ledger snapshot schema is invalid")
        assignments = split_payload.get("assignments")
        if not isinstance(assignments, dict) or set(assignments) != set(row_order):
            raise ValueError("FEREBUS split-ledger snapshot coverage is invalid")
        for pointdir in row_order:
            record = assignments.get(pointdir)
            if not isinstance(record, dict) or record.get("split") not in {
                "train", "int_val", "ext_val"
            }:
                raise ValueError("FEREBUS split-ledger assignment is invalid")
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("FEREBUS task manifest has no tasks: " + str(path))
    properties = data.get("properties")
    atoms = data.get("atoms")
    if not isinstance(properties, list) or not properties:
        raise ValueError("FEREBUS task manifest properties are invalid")
    if not isinstance(atoms, list) or not atoms:
        raise ValueError("FEREBUS task manifest atoms are invalid")
    property_tokens = [str(value) for value in properties]
    atom_tokens = [str(value) for value in atoms]
    for token in property_tokens:
        validate_safe_path_token("FEREBUS property", token)
    for token in atom_tokens:
        validate_safe_path_token("FEREBUS atom label", token)
    if len({token.casefold() for token in property_tokens}) != len(property_tokens):
        raise ValueError("FEREBUS properties contain a case-insensitive collision")
    if len({token.casefold() for token in atom_tokens}) != len(atom_tokens):
        raise ValueError("FEREBUS atom labels contain a case-insensitive collision")
    try:
        from ..ferebus_prior import contract_from_payload

        prior_contract = contract_from_payload(data.get("prior_mean_contract"))
        if prior_contract.strategy == "physical_atomic_iqa":
            for atom in atom_tokens:
                prior_contract.expected_mean_ha("iqa", atom)
    except Exception as exc:
        raise ValueError(
            "FEREBUS task manifest physical-prior contract is invalid: " + str(exc)
        ) from exc
    expected_keys = [
        (prop, atom) for prop in property_tokens for atom in atom_tokens
    ]
    model_bootstrap = data.get("model_bootstrap")
    if model_bootstrap is None:
        manifest_historical_training_rows = 0
    elif isinstance(model_bootstrap, dict):
        manifest_historical_training_rows = _exact_ferebus_int(
            model_bootstrap.get("historical_training_rows"),
            "model-bootstrap historical_training_rows",
            minimum=1,
        )
        if manifest_historical_training_rows <= 0:
            raise ValueError(
                "FEREBUS model-bootstrap training-row count is invalid"
            )
    else:
        raise ValueError("FEREBUS model_bootstrap record is invalid")
    observed_keys = []
    for expected_index, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            raise ValueError("FEREBUS task manifest contains a non-object task")
        prop = str(task.get("property") or "")
        atom = str(task.get("atom") or "")
        observed_keys.append((prop, atom))
        if _exact_ferebus_int(task.get("task_index"), "FEREBUS task index", minimum=1) != expected_index:
            raise ValueError("FEREBUS task indexes are not contiguous")
        task_prior = task.get("prior_mean")
        expected_prior_fields = {
            "contract_sha256",
            "strategy",
            "mean_type",
            "level_of_theory",
            "physical_prior_scale",
            "units",
            "expected_mean_ha",
            "training_dataset_sha256",
            "feature_scaling",
            "property_scaling",
        }
        if (
            not isinstance(task_prior, dict)
            or set(task_prior) != expected_prior_fields
            or str(task_prior.get("contract_sha256") or "")
            != prior_contract.contract_sha256
            or task_prior.get("strategy") != prior_contract.strategy
            or task_prior.get("mean_type") != prior_contract.mean_type
            or task_prior.get("level_of_theory") != prior_contract.level_of_theory
            or task_prior.get("units") != "ha"
            or task_prior.get("feature_scaling") is not True
            or task_prior.get("property_scaling") is not False
        ):
            raise ValueError(
                "FEREBUS task prior contract mismatch for "
                + prop
                + "/"
                + atom
            )
        try:
            recorded_scale = float(task_prior["physical_prior_scale"])
            recorded_mean = float(task_prior["expected_mean_ha"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "FEREBUS task prior contains non-numeric values for "
                + prop
                + "/"
                + atom
            ) from exc
        if (
            isinstance(task_prior["physical_prior_scale"], bool)
            or isinstance(task_prior["expected_mean_ha"], bool)
            or not np.isfinite(recorded_scale)
            or recorded_scale != float(prior_contract.physical_prior_scale)
            or not np.isfinite(recorded_mean)
        ):
            raise ValueError(
                "FEREBUS task prior numeric contract mismatch for "
                + prop
                + "/"
                + atom
            )
        if prior_contract.strategy in {"zero", "physical_atomic_iqa"} and not np.isclose(
            recorded_mean,
            prior_contract.expected_mean_ha(prop, atom),
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError(
                "FEREBUS task prior mean disagrees with its semantic contract for "
                + prop
                + "/"
                + atom
            )
        generated_config = task.get("generated_config")
        if generated_config is not None:
            if not isinstance(generated_config, dict):
                raise ValueError("FEREBUS generated_config record is invalid")
            if str(generated_config.get("path") or "") != str(task.get("config_path") or ""):
                raise ValueError("FEREBUS generated_config path mismatch")
            if str(generated_config.get("prior_mean_contract_sha256") or "") != (
                prior_contract.contract_sha256
            ):
                raise ValueError("FEREBUS generated_config prior hash mismatch")
            parsed = generated_config.get("parsed_contract")
            expected_parsed = {
                "mean_type": int(prior_contract.mean_type),
                "level_of_theory": (
                    prior_contract.level_of_theory or "not_applicable"
                ),
                "iqa_deviation_factor": float(prior_contract.iqa_deviation_factor),
                "scaling": bool(
                    prior_contract.feature_scaling or prior_contract.property_scaling
                ),
                "scale_feats": bool(prior_contract.feature_scaling),
                "scale_prop": bool(prior_contract.property_scaling),
            }
            if parsed != expected_parsed:
                raise ValueError("FEREBUS generated_config parsed contract mismatch")
            if verify_dataset_files:
                config_path = resolve_ferebus_task_path(
                    staging_dir,
                    generated_config.get("path"),
                    "generated_config.path",
                )
                if config_path.is_symlink() or not config_path.is_file():
                    raise ValueError("FEREBUS generated config file is missing")
                size = generated_config.get("size")
                if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                    raise ValueError("FEREBUS generated config size is invalid")
                digest = generated_config.get("sha256")
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ValueError("FEREBUS generated config SHA-256 is invalid")
                if int(config_path.stat().st_size) != size:
                    raise ValueError("FEREBUS generated config size mismatch")
                if sha256_file(config_path) != digest:
                    raise ValueError("FEREBUS generated config SHA-256 mismatch")
                from ..ferebus_prior import validate_ferebus_config_contract

                observed_parsed = validate_ferebus_config_contract(
                    config_path,
                    prior_contract,
                )
                if observed_parsed != expected_parsed or observed_parsed != parsed:
                    raise ValueError("FEREBUS generated config content contract mismatch")
        expected_task_dir = prop + "/" + atom
        expected_input_dir = expected_task_dir + "/datasets"
        expected_paths = {
            "property_dir": prop,
            "output_dir": expected_task_dir,
            "input_dir": expected_input_dir,
            "config_path": expected_task_dir + "/ferebus.config",
            "training_csv": (
                expected_input_dir + "/" + system + "_" + atom + "_TRAINING_SET.csv"
            ),
            "int_validation_csv": (
                expected_input_dir
                + "/"
                + system
                + "_"
                + atom
                + "_INT_VALIDATION_SET.csv"
            ),
            "ext_validation_csv": (
                expected_input_dir
                + "/"
                + system
                + "_"
                + atom
                + "_EXT_VALIDATION_SET.csv"
            ),
            "expected_model_path": (
                expected_task_dir
                + "/"
                + system
                + "_"
                + prop
                + "_"
                + atom
                + ".model"
            ),
        }
        for key in (
            "property_dir",
            "output_dir",
            "input_dir",
            "config_path",
            "training_csv",
            "int_validation_csv",
            "ext_validation_csv",
            "expected_model_path",
        ):
            resolved = resolve_ferebus_task_path(staging_dir, task.get(key), key)
            relative = ferebus_relative_path(staging_dir, resolved)
            if relative != expected_paths[key]:
                raise ValueError("FEREBUS " + key + " does not match its task")
        expected_alf_cli = "_".join(
            str(int(value)) for value in task.get("alf_1_indexed", [])
        )
        command_args = task.get("command_args")
        expected_command_args = [
            "-c",
            expected_paths["config_path"],
            "-I",
            expected_paths["input_dir"],
            "-O",
            expected_paths["output_dir"],
            "-P",
            prop,
            "-A",
            atom,
            "-ALF",
            expected_alf_cli,
        ]
        if command_args != expected_command_args:
            raise ValueError("FEREBUS command_args do not match the task contract")
        counts = task.get("row_counts")
        if not isinstance(counts, dict):
            raise ValueError("FEREBUS task manifest row_counts is invalid")
        historical_training_rows = _exact_ferebus_int(
            task.get("historical_training_rows", 0),
            "FEREBUS task historical_training_rows",
        )
        if historical_training_rows != manifest_historical_training_rows:
            raise ValueError(
                "FEREBUS task historical training-row count disagrees with the manifest"
            )
        historical_ids = [
            _exact_ferebus_int(value, "historical training-row ID")
            for value in task.get("historical_training_row_ids", [])
        ]
        if historical_ids != list(range(historical_training_rows)):
            raise ValueError(
                "FEREBUS task historical training-row IDs are invalid"
            )
        try:
            task_total = sum(
                _exact_ferebus_int(
                    counts[split], "FEREBUS " + split + " row count"
                )
                for split in ("train", "int_val", "ext_val")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("FEREBUS task manifest row_counts is invalid") from exc
        if task_total != n_reference_points + historical_training_rows:
            raise ValueError(
                "FEREBUS task row count does not match the reference-data view "
                "plus its historical model baseline"
            )
        row_ids = task.get("row_ids")
        if not isinstance(row_ids, dict):
            raise ValueError("FEREBUS task manifest row_ids is invalid")
        try:
            split_rows = {
                split: [
                    _exact_ferebus_int(value, "FEREBUS " + split + " row ID")
                    for value in row_ids[split]
                ]
                for split in ("train", "int_val", "ext_val")
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("FEREBUS task manifest row_ids is invalid") from exc
        for split, indexes in split_rows.items():
            expected_index_count = int(counts[split]) - (
                historical_training_rows if split == "train" else 0
            )
            if expected_index_count < 0 or len(indexes) != expected_index_count:
                raise ValueError(
                    "FEREBUS task row_ids/count mismatch for " + split
                )
        flattened_rows = [
            index
            for split in ("train", "int_val", "ext_val")
            for index in split_rows[split]
        ]
        if sorted(flattened_rows) != list(range(n_reference_points)):
            raise ValueError(
                "FEREBUS task row_ids do not partition the reference-data rows"
            )
        datasets = task.get("datasets")
        if not isinstance(datasets, dict) or set(datasets) != {
            "train",
            "int_val",
            "ext_val",
        }:
            raise ValueError("FEREBUS task dataset identities are invalid")
        dataset_path_fields = {
            "train": "training_csv",
            "int_val": "int_validation_csv",
            "ext_val": "ext_validation_csv",
        }
        resolved_dataset_paths: Dict[str, Path] = {}
        for split, path_field in dataset_path_fields.items():
            record = datasets.get(split)
            if not isinstance(record, dict):
                raise ValueError("FEREBUS " + split + " dataset identity is invalid")
            expected_path = expected_paths[path_field]
            if str(record.get("path") or "") != expected_path:
                raise ValueError("FEREBUS " + split + " dataset path mismatch")
            dataset_size = _exact_ferebus_int(
                record.get("size"), "FEREBUS " + split + " dataset size"
            )
            dataset_rows = _exact_ferebus_int(
                record.get("rows"), "FEREBUS " + split + " dataset rows"
            )
            dataset_sha = str(record.get("sha256") or "")
            if dataset_size < 0 or dataset_rows != int(counts[split]):
                raise ValueError("FEREBUS " + split + " dataset identity is invalid")
            if len(dataset_sha) != 64 or any(
                character not in "0123456789abcdef" for character in dataset_sha
            ):
                raise ValueError("FEREBUS " + split + " dataset SHA-256 is invalid")
            if split == "train" and str(
                task_prior.get("training_dataset_sha256") or ""
            ) != dataset_sha:
                raise ValueError(
                    "FEREBUS task prior is not bound to its training dataset"
                )
            if row_identity_payload is not None:
                split_identity = row_identity_payload.get("splits", {}).get(split)
                if not isinstance(split_identity, dict):
                    raise ValueError("FEREBUS split row identity is missing")
                if (
                    str(record.get("row_identity_sha256") or "")
                    != str(split_identity.get("row_identity_sha256") or "")
                    or _exact_ferebus_int(
                        record.get("row_identity_count"),
                        "FEREBUS row identity count",
                    )
                    != dataset_rows
                    or _exact_ferebus_int(
                        split_identity.get("n_rows"),
                        "FEREBUS split identity row count",
                    )
                    != dataset_rows
                ):
                    raise ValueError("FEREBUS dataset row-identity binding mismatch")
            if verify_dataset_files:
                dataset_path = resolve_ferebus_task_path(
                    staging_dir,
                    expected_path,
                    path_field,
                )
                if not dataset_path.is_file():
                    dataset_path = resolve_ferebus_task_path(
                        staging_dir,
                        prop + "/" + Path(expected_path).name,
                        path_field + "_pre_pyferebus",
                    )
                if not dataset_path.is_file() or dataset_path.is_symlink():
                    raise ValueError(
                        "FEREBUS " + split + " dataset file is missing"
                    )
                if int(dataset_path.stat().st_size) != dataset_size:
                    raise ValueError("FEREBUS " + split + " dataset size mismatch")
                if sha256_file(dataset_path) != dataset_sha:
                    raise ValueError("FEREBUS " + split + " dataset SHA-256 mismatch")
                from . import ferebus_dataset as _fds

                validation = _fds.validate_ferebus_csv(dataset_path, prop)
                if validation["rows"] != dataset_rows:
                    raise ValueError("FEREBUS dataset validated row count mismatch")
                resolved_dataset_paths[split] = dataset_path
        if verify_dataset_files:
            training_values = _fds.read_property_values(
                resolved_dataset_paths["train"], prop
            )
            expected_prior = prior_contract.task_payload(
                prop,
                atom,
                training_values=training_values,
                training_dataset_sha256=str(datasets["train"]["sha256"]),
            )
            if task_prior != expected_prior:
                raise ValueError(
                    "FEREBUS task prior mean is not bound to its training data for "
                    + prop
                    + "/"
                    + atom
                )
    if observed_keys != expected_keys:
        raise ValueError("FEREBUS tasks do not match the property/atom product")
    if _exact_ferebus_int(data.get("n_tasks"), "FEREBUS n_tasks", minimum=1) != len(expected_keys):
        raise ValueError("FEREBUS task manifest n_tasks is invalid")
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


def _require_native_ferebus_alf(system_alf: Mapping[str, Any]) -> None:
    """Fail before CSV generation when the native three-index ALF ABI cannot apply."""
    if not isinstance(system_alf, Mapping) or len(system_alf) < 3:
        raise ValueError(
            "FEREBUS training requires at least three atoms because the native "
            "ALF ABI requires three indexes; diatomic training is unsupported"
        )


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
    campaign_uid: Optional[str] = None,
) -> Tuple[Path, int]:
    """Write one POINT_<k>.pointdir/input.gjf per frame in sample_xyz, plus
    POINTS.txt. Returns (staging_dir, n_points)."""
    from ichor.core.files.gaussian.gjf import GJF

    frames = _load_frames(sample_xyz)
    supplied_campaign_uid = str(campaign_uid or "").strip()

    def expected_campaign_uid() -> str:
        if supplied_campaign_uid:
            return supplied_campaign_uid
        from .state import DEFAULT_STATE_FILENAME, read_state

        current_state = read_state(
            Path(campaign_dir)
            / ".DATA"
            / "ACTIVE_LEARNING"
            / DEFAULT_STATE_FILENAME
        )
        value = str(current_state.campaign_uid).strip()
        if not value:
            raise ValueError("campaign state has no campaign_uid")
        return value

    phase_b_records: List[Dict[str, Any]] = []
    allocation_records: List[Dict[str, Any]] = []
    allocation_assignment_hash: Optional[str] = None
    initial_seed_frame_ids: List[Optional[int]] = []
    initial_seed_selection_origins: List[str] = []
    initial_provenance_context: Optional[Dict[str, str]] = None
    replacement_context: Optional[str] = None
    is_replacement = str(phase_name) in {
        "INITIAL_REPLACEMENT_GAUSSIAN",
        "REPLACEMENT_GAUSSIAN",
    }
    if is_replacement:
        from ..replacement_sampling import read_replacement_sample_strict

        replacement_context = (
            "bootstrap"
            if str(phase_name) == "INITIAL_REPLACEMENT_GAUSSIAN"
            else "active"
        )
        try:
            replacement_round = int(Path(sample_xyz).parent.name.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError("replacement sample is outside a numbered round directory") from exc
        replacement_manifest = read_replacement_sample_strict(
            campaign_dir,
            context=replacement_context,
            iteration=(0 if replacement_context == "bootstrap" else int(iteration)),
            replacement_round=replacement_round,
        )
        allocation_records = list(replacement_manifest["records"])
        from ..point_allocation import read_point_allocation

        replacement_allocation = read_point_allocation(
            replacement_manifest["point_allocation_manifest"]
        )
        allocation_assignment_hash = str(
            replacement_allocation["slot_assignment_sha256"]
        )
        if len(allocation_records) != len(frames):
            raise ValueError(
                "replacement allocation record count does not match sample"
            )
        if replacement_context == "active":
            phase_b_records = list(allocation_records)
        elif replacement_context == "bootstrap":
            from ..acquisition.trajectory_pool import TrajectoryPool

            pool = TrajectoryPool.load(campaign_dir)
            initial_provenance_context = {
                "campaign_uid": expected_campaign_uid(),
                "trajectory_sha256": str(pool.sha256),
            }
            initial_seed_frame_ids = [int(record["frame_id"]) for record in allocation_records]
            initial_seed_selection_origins = ["phase_a_replacement"] * len(allocation_records)
        else:
            raise ValueError("replacement sample context is invalid")
    if str(phase_name) == "GAUSSIAN":
        from ..handoff_manifests import read_phase_b_selection_manifest

        phase_b_manifest = read_phase_b_selection_manifest(
            Path(sample_xyz).parent.parent,
            expected_iteration=int(iteration),
            expected_campaign_uid=expected_campaign_uid(),
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
        allocation_assignment_hash = str(
            phase_b_manifest["point_allocation"]["slot_assignment_sha256"]
        )
    if str(phase_name) == "INITIAL_GAUSSIAN":
        try:
            from ..acquisition.trajectory_pool import TrajectoryPool
            from ..handoff_manifests import read_phase_a_sample_manifest

            phase_a_manifest = read_phase_a_sample_manifest(
                Path(sample_xyz).parent,
                expected_campaign_uid=expected_campaign_uid(),
            )
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
                    "custom_bootstrap" if value is None else "phase_a_diversity"
                    for value in selected
                ]
                pool = TrajectoryPool.load(campaign_dir)
                initial_provenance_context = {
                    "campaign_uid": expected_campaign_uid(),
                    "trajectory_sha256": str(pool.sha256),
                }
                allocation_records = list(
                    phase_a_manifest["point_allocation"]["primary"]
                )
                if len(allocation_records) != len(frames):
                    raise ValueError(
                        "Phase A point-allocation record count does not match sample"
                    )
                from ..point_allocation import read_point_allocation

                bootstrap_allocation = read_point_allocation(
                    phase_a_manifest["point_allocation"]["manifest"]
                )
                allocation_assignment_hash = str(
                    bootstrap_allocation["slot_assignment_sha256"]
                )
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
            from ..acquisition.trajectory_pool import TrajectoryPool

            pool = TrajectoryPool.load(campaign_dir)
            provenance_context = {
                "campaign_uid": expected_campaign_uid(),
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
    campaign_owned_path(campaign_dir, staging)
    point_indexes = [
        (
            int(allocation_records[index].get("pointdir_index", index))
            if is_replacement and allocation_records
            else int(index)
        )
        for index in range(len(frames))
    ]
    expected_pointdir_names = [
        staging_pointdir_name(index) for index in point_indexes
    ]
    if len(expected_pointdir_names) != len(set(expected_pointdir_names)):
        raise ValueError("Gaussian staging pointdir identities contain duplicates")
    preserve_existing_layout = False
    if staging.exists():
        _reject_symlink_tree(staging)
        try:
            listed_names = _points_file_names(staging)
            actual_names = {
                child.name
                for child in staging.iterdir()
                if child.name.endswith(".pointdir")
            }
            preserve_existing_layout = (
                listed_names == expected_pointdir_names
                and actual_names == set(expected_pointdir_names)
                and all((staging / name).is_dir() for name in expected_pointdir_names)
            )
        except Exception:
            preserve_existing_layout = False
    # Keep only an exact task layout. Individual task products survive only
    # while their rendered GJF bytes remain unchanged.
    if staging.exists() and not preserve_existing_layout:
        _checked_rmtree(
            staging,
            campaign_dir=Path(campaign_dir),
            allowed_roots=[staging_root(campaign_dir)],
        )
    staging.mkdir(parents=True, exist_ok=True)

    keywords = ["nosymm", "output=wfn", "force", "geom=notest"]
    keywords += normalise_gaussian_route_keywords(g.extra_route_keywords)

    pointdirs: List[Path] = []
    for k, atoms in enumerate(frames):
        pd = staging / expected_pointdir_names[k]
        campaign_owned_path(campaign_dir, pd)
        if pd.is_symlink():
            raise ValueError("refusing symlinked Gaussian pointdir: " + str(pd))
        gjf = GJF(
            pd / "input.gjf",
            method=str(g.method),
            basis_set=str(g.basis_set),
            keywords=list(keywords),
            charge=int(g.charge),
            spin_multiplicity=int(g.spin_multiplicity),
            atoms=atoms,
        )
        rendered_gjf = gjf.render()
        existing_gjf = pd / "input.gjf"
        if pd.exists():
            if not pd.is_dir():
                raise ValueError("Gaussian pointdir path is not a directory: " + str(pd))
            unchanged = False
            if existing_gjf.is_file() and not existing_gjf.is_symlink():
                try:
                    unchanged = existing_gjf.read_text(encoding="utf-8") == rendered_gjf
                except OSError:
                    unchanged = False
            if not unchanged:
                _checked_rmtree(
                    pd,
                    campaign_dir=Path(campaign_dir),
                    allowed_roots=[staging],
                )
        pd.mkdir(parents=True, exist_ok=True)
        if not existing_gjf.exists():
            atomic_write_text(existing_gjf, rendered_gjf)
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
                    else "phase_a_diversity"
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
                seed_id=int(phase_b_records[k]["seed_id"]),
                seed_uid=str(phase_b_records[k]["seed_uid"]),
                array_task_id_zero_based=int(
                    phase_b_records[k]["array_task_id"]
                ),
                require_phase_b_selected=True,
                allocation_split=(
                    None if is_replacement else str(phase_b_records[k]["split"])
                ),
                allocation_slot_id=(
                    None if is_replacement else int(phase_b_records[k]["slot_id"])
                ),
                allocation_candidate_id=(
                    None
                    if is_replacement
                    else str(phase_b_records[k]["candidate_id"])
                ),
                allocation_context=(None if is_replacement else "active"),
                allocation_slot_assignment_sha256=(
                    None if is_replacement else allocation_assignment_hash
                ),
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
                allocation_slot_assignment_sha256=allocation_assignment_hash,
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
    if not pointdirs:
        write_points_file(staging, [])
        return staging, 0
    gaussian_task_names = _points_file_names(staging)
    acceptance_sha256 = sha256_file(staging / QUANTUM_ACCEPTANCE_MANIFEST)
    dimensions = []
    for task_index, pointdir in enumerate(pointdirs):
        wfn = pointdir / "input.wfn"
        if wfn.is_symlink() or not wfn.is_file():
            raise FileNotFoundError(
                "Gaussian-accepted pointdir is missing input.wfn: " + str(pointdir)
            )
        try:
            expected_atoms = list(PointDirectory(pointdir).atoms)
            atom_count = len(expected_atoms)
        except Exception as exc:
            raise ValueError(
                "failed to count atoms for AIMAll pointdir: " + str(pointdir)
            ) from exc
        if atom_count <= 0:
            raise ValueError("AIMAll pointdir has no atoms: " + str(pointdir))
        expected_atom_names = [str(atom.name) for atom in expected_atoms]
        if len(expected_atom_names) != len(set(expected_atom_names)):
            raise ValueError("AIMAll pointdir geometry has duplicate atom identities")
        try:
            gaussian_logical_task_id = gaussian_task_names.index(pointdir.name)
        except ValueError as exc:
            raise ValueError("Gaussian acceptance point is absent from POINTS.txt") from exc
        from .quantum_task_receipts import (
            GAUSSIAN_TASK_RECEIPT,
            read_quantum_task_receipt,
        )

        read_quantum_task_receipt(
            pointdir,
            phase_name=expected_phase,
            iteration=int(iteration),
            logical_task_id=int(gaussian_logical_task_id),
        )
        gaussian_receipt_path = pointdir / GAUSSIAN_TASK_RECEIPT
        receipt_path, receipt = rewrite_wfn_for_aimall(
            wfn,
            method=str(config.gaussian.method),
            phase_name=str(phase_name),
            iteration=int(iteration),
            task_index=int(task_index),
            source_acceptance_sha256=acceptance_sha256,
        )
        dimensions.append(
            (
                pointdir,
                atom_count,
                wfn_primitive_count(wfn),
                receipt_path,
                receipt,
                expected_atom_names,
                gaussian_logical_task_id,
                gaussian_receipt_path,
            )
        )
    write_points_file(staging, pointdirs)
    aimall_resources = resolve_phase_resources(
        phase_name=str(phase_name),
        config=config,
        partition=str(partition_override or config.resources.partition_for(str(phase_name))),
        campaign_dir=campaign_dir,
        iteration=int(iteration),
        array_size=len(pointdirs),
        staging_dir=staging,
        require_evidence=True,
    )
    aimall_cpus = int(aimall_resources.cpus_per_task)
    raw_naat = getattr(config.aimall, "naat", "auto")
    snapshotted_naat = list(
        aimall_resources.extra.get("aimall_task_naat") or []
    )
    if len(snapshotted_naat) != len(dimensions):
        raise ValueError(
            "AIMAll resource resolution does not cover every staged task"
        )
    for task_index, (
        pointdir,
        atom_count,
        primitive_count,
        receipt_path,
        receipt,
        expected_atom_names,
        gaussian_logical_task_id,
        gaussian_receipt_path,
    ) in enumerate(dimensions):
        if isinstance(raw_naat, str) and raw_naat.strip().lower() == "auto":
            resolved_naat = int(snapshotted_naat[task_index])
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
                "pointdir": pointdir.name,
                "task_index": int(task_index),
                "gaussian_logical_task_id": int(gaussian_logical_task_id),
                "atom_count": int(atom_count),
                "expected_atom_names": list(expected_atom_names),
                "primitive_count": int(primitive_count),
                "nproc": int(aimall_cpus),
                "naat": int(resolved_naat),
                "electronic_method": str(receipt["method"]),
                "wfn_sha256": str(receipt["wfn"]["after_sha256"]),
                "wfn_method_receipt": {
                    "path": receipt_path.name,
                    "sha256": sha256_file(receipt_path),
                },
                "gaussian_task_receipt": {
                    "path": gaussian_receipt_path.name,
                    "sha256": sha256_file(gaussian_receipt_path),
                },
                "gjf_sha256": sha256_file(pointdir / "input.gjf"),
                "resource_resolution": aimall_resources.journal_payload(
                    phase_name=str(phase_name)
                ),
            },
        )
    return staging, len(pointdirs)


def record_allocation_quantum_results(
    campaign_dir: Path,
    *,
    context: str,
    iteration: int,
    staging_dir: Path,
    gaussian_phase: str,
    aimall_phase: str,
    expected_method: Optional[str] = None,
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
    attempts_by_id = {
        str(attempt["candidate_id"]): {
            **attempt,
            "slot_id": int(slot["slot_id"]),
            "split": str(slot["split"]),
        }
        for slot in allocation["slots"]
        for attempt in list(slot.get("attempts") or [])
    }
    submitted_names = [Path(path).name for path in gaussian_accepted]
    submitted_names.extend(
        str(record.get("pointdir"))
        for record in list(gaussian_manifest.get("rejected") or [])
        if isinstance(record, dict)
    )
    if len(submitted_names) != len(set(submitted_names)):
        raise ValueError("Gaussian allocation handoff contains duplicate pointdirs")
    if pending_ids and len(submitted_names) != len(pending_ids):
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
    aimall_task_names = aimall_accepted_names | set(aimall_rejected)
    if aimall_task_names != gaussian_accepted_names:
        raise ValueError(
            "AIMAll acceptance does not cover exactly the Gaussian-accepted tasks"
        )
    quality_path = staging / "quantum_quality.json"
    if aimall_task_names:
        from .quantum_quality import read_quantum_quality_manifest

        quality_payload = read_quantum_quality_manifest(
            staging,
            expected_phase=str(aimall_phase),
            expected_iteration=int(iteration),
            expected_pointdirs=sorted(aimall_task_names),
            expected_method=expected_method,
        )
        quality_accepted = {
            str(record["pointdir"])
            for record in quality_payload["records"]
            if bool(record["accepted"])
        }
        if quality_accepted != aimall_accepted_names:
            raise ValueError(
                "quantum-quality acceptance does not match AIMAll acceptance"
            )
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
        if candidate_id not in attempts_by_id or candidate_id in observed_ids:
            raise ValueError(
                "staged pointdir candidate does not match point allocation: " + name
            )
        attempt = attempts_by_id[candidate_id]
        if pending_ids and candidate_id not in pending_by_id:
            raise ValueError("staged pointdir does not belong to the pending allocation round: " + name)
        validate_provenance(
            pointdir,
            campaign_uid=str(allocation["campaign_uid"]),
            iteration=int(iteration),
            allocation_candidate_id=candidate_id,
            allocation_context=str(context),
            allocation_slot_id=int(attempt["slot_id"]),
            allocation_split=str(attempt["split"]),
            allocation_slot_assignment_sha256=str(
                allocation["slot_assignment_sha256"]
            ),
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
        result = {
            "candidate_id": candidate_id,
            "accepted": bool(accepted),
            "pointdir": str(pointdir.resolve()),
            "reason": reason,
        }
        if name in aimall_task_names:
            result["quality_manifest"] = str(
                (staging / "quantum_quality.json")
                .resolve(strict=False)
                .relative_to(campaign.resolve())
                .as_posix()
            )
        if accepted:
            from .quantum_acceptance_receipts import (
                QUANTUM_ACCEPTANCE_RECEIPT,
                read_quantum_acceptance_receipt,
            )

            acceptance_receipt = read_quantum_acceptance_receipt(
                campaign,
                pointdir,
                expected_phase=str(aimall_phase),
                expected_iteration=int(iteration),
                expected_candidate_id=candidate_id,
                expected_assignment_sha256=str(allocation["slot_assignment_sha256"]),
            )
            receipt_path = pointdir / QUANTUM_ACCEPTANCE_RECEIPT
            result["quantum_acceptance_receipt"] = str(
                receipt_path.resolve().relative_to(campaign.resolve()).as_posix()
            )
            result["quantum_acceptance_receipt_sha256"] = sha256_file(receipt_path)
            result["accepted_pointdir_content_sha256"] = str(
                acceptance_receipt["content_sha256"]
            )
        results.append(result)
    if pending_ids and observed_ids != pending_ids:
        raise ValueError("quantum staging does not cover every pending allocation candidate")
    if not pending_ids and not observed_ids:
        raise ValueError("quantum result replay contains no allocation candidates")
    acceptance_paths = [
        quantum_acceptance_manifest_path(staging, phase_name=str(gaussian_phase)),
    ]
    aimall_path = quantum_acceptance_manifest_path(staging, phase_name=str(aimall_phase))
    if aimall_path.is_file():
        acceptance_paths.append(aimall_path)
    source_identity = {
        "campaign_uid": str(allocation["campaign_uid"]),
        "context": str(context),
        "iteration": int(iteration),
        "staging": staging.resolve().relative_to(campaign.resolve()).as_posix(),
        "gaussian_phase": str(gaussian_phase),
        "aimall_phase": str(aimall_phase),
        "manifest_sha256": [sha256_file(path) for path in acceptance_paths],
        "quality_sha256": sha256_file(quality_path) if quality_path.is_file() else None,
    }
    batch_identity = hashlib.sha256(
        json.dumps(source_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result_identity = [
        {
            "candidate_id": str(record["candidate_id"]),
            "accepted": bool(record["accepted"]),
            "pointdir": str(record["pointdir"]),
            "reason": record.get("reason"),
            "quantum_acceptance_receipt_sha256": record.get(
                "quantum_acceptance_receipt_sha256"
            ),
            "accepted_pointdir_content_sha256": record.get(
                "accepted_pointdir_content_sha256"
            ),
        }
        for record in sorted(results, key=lambda item: str(item["candidate_id"]))
    ]
    result_fingerprint = hashlib.sha256(
        json.dumps(result_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return record_quantum_results(
        allocation_path,
        results,
        expected_generation=(
            int(allocation.get("generation", 0)) if pending_ids else None
        ),
        batch_identity=batch_identity,
        result_fingerprint=result_fingerprint,
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
    canonical_staging_root = staging_root(campaign).resolve(strict=False)
    pointdirs: List[Path] = []
    seen: set[Path] = set()
    for attempt in attempts:
        pointdir = Path(str(attempt.get("pointdir") or ""))
        resolved = pointdir.resolve(strict=False)
        if resolved in seen:
            raise ValueError("point allocation reuses an accepted pointdir")
        if canonical_staging_root not in resolved.parents:
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
            campaign_uid=str(allocation["campaign_uid"]),
            iteration=int(iteration),
            allocation_candidate_id=str(attempt["candidate_id"]),
            allocation_context=str(context),
            allocation_slot_id=int(attempt["slot_id"]),
            allocation_split=str(attempt["split"]),
            allocation_slot_assignment_sha256=str(
                allocation["slot_assignment_sha256"]
            ),
        )
        from .quantum_acceptance_receipts import (
            QUANTUM_ACCEPTANCE_RECEIPT,
            read_quantum_acceptance_receipt,
        )

        receipt = read_quantum_acceptance_receipt(
            campaign,
            resolved,
            expected_iteration=int(iteration),
            expected_candidate_id=str(attempt["candidate_id"]),
            expected_assignment_sha256=str(allocation["slot_assignment_sha256"]),
        )
        receipt_path = resolved / QUANTUM_ACCEPTANCE_RECEIPT
        expected_receipt_path = str(attempt.get("quantum_acceptance_receipt") or "")
        expected_receipt_sha = str(
            attempt.get("quantum_acceptance_receipt_sha256") or ""
        )
        expected_content_sha = str(
            attempt.get("accepted_pointdir_content_sha256") or ""
        )
        if (
            not expected_receipt_path
            or receipt_path.resolve()
            != (campaign / expected_receipt_path).resolve(strict=False)
            or sha256_file(receipt_path) != expected_receipt_sha
            or str(receipt["content_sha256"]) != expected_content_sha
        ):
            raise ValueError("accepted quantum receipt does not match allocation evidence")
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
        "POINT_ALLOCATION.version-"
        + str(version).zfill(COMMITTED_VERSION_NAME_WIDTH)
        + ".json"
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
    copied_by_source: Dict[str, Path] = {}
    for offset, (source, attempt) in enumerate(zip(accepted_pointdirs, attempts)):
        ordinal = first_ordinal + offset
        name = "POINT_" + str(ordinal).zfill(POINTDIR_NAME_WIDTH) + ".pointdir"
        destination = staging / name
        _copytree_no_symlinks(source, destination)
        from .quantum_acceptance_receipts import read_quantum_acceptance_receipt

        read_quantum_acceptance_receipt(
            campaign,
            destination,
            expected_iteration=int(iteration),
            expected_candidate_id=str(attempt["candidate_id"]),
            expected_assignment_sha256=str(allocation["slot_assignment_sha256"]),
            expected_source_pointdir=source.name,
        )
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
        copied_by_source[source.name] = destination

    allocation_relative = allocation_path.relative_to(campaign).as_posix()
    quality_sources: Dict[str, Path] = {}
    missing_quality = 0
    for attempt in attempts:
        raw_quality = str(attempt.get("quality_manifest") or "")
        if not raw_quality:
            missing_quality += 1
            continue
        candidate = Path(raw_quality)
        source_quality = candidate if candidate.is_absolute() else campaign / candidate
        resolved_quality = source_quality.resolve(strict=False)
        if campaign.resolve() not in resolved_quality.parents:
            raise ValueError("quantum-quality evidence is outside the campaign")
        if resolved_quality.is_symlink() or not resolved_quality.is_file():
            raise FileNotFoundError(
                "quantum-quality evidence is missing: " + str(resolved_quality)
            )
        quality_sources[str(resolved_quality)] = resolved_quality
    if missing_quality:
        raise ValueError(
            "accepted allocation lacks mandatory quantum-quality evidence for "
            + str(missing_quality)
            + " candidate(s)"
        )
    quality_evidence: List[Dict[str, Any]] = []
    committed_by_source = {
        Path(source).name: {
            "source_pointdir": Path(source).name,
            "committed_pointdir": entry.pointdir_name,
            "candidate_id": str(attempt["candidate_id"]),
        }
        for source, attempt, entry in zip(accepted_pointdirs, attempts, added_entries)
    }
    for index, source_quality in enumerate(sorted(quality_sources.values())):
        try:
            quality_payload_raw = json.loads(source_quality.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(
                "quantum-quality evidence is unreadable: " + str(source_quality)
            ) from exc
        if not isinstance(quality_payload_raw, dict):
            raise ValueError("quantum-quality evidence must be a JSON object")
        from .quantum_quality import read_quantum_quality_manifest

        quality_payload = read_quantum_quality_manifest(
            source_quality.parent,
            expected_phase=str(quality_payload_raw.get("phase") or ""),
            expected_iteration=int(quality_payload_raw.get("iteration")),
            manifest_path=source_quality,
        )
        accepted_source_names = {
            str(record["pointdir"])
            for record in quality_payload["records"]
            if bool(record["accepted"])
        }
        relevant_names = sorted(accepted_source_names & set(committed_by_source))
        if not relevant_names:
            raise ValueError(
                "quantum-quality evidence contains no committed accepted pointdir"
            )
        destination_quality = (
            staging / "quality_evidence" / ("quantum_quality_" + str(index).zfill(4) + ".json")
        )
        destination_quality.parent.mkdir(parents=True, exist_ok=True)
        committed_quality_payload = dict(quality_payload)
        committed_records = []
        for record in quality_payload["records"]:
            committed_record = dict(record)
            source_name = str(record.get("pointdir") or "")
            if source_name in committed_by_source:
                binding = committed_by_source[source_name]
                committed_record["committed_pointdir"] = str(
                    binding["committed_pointdir"]
                )
                committed_record["candidate_id"] = str(binding["candidate_id"])
            committed_records.append(committed_record)
        committed_quality_payload["records"] = committed_records
        atomic_write_json(destination_quality, committed_quality_payload)
        from .quantum_acceptance_receipts import (
            bind_quantum_acceptance_receipt_to_commit,
        )

        committed_record_by_source = {
            str(record.get("pointdir") or ""): record
            for record in committed_records
            if bool(record.get("accepted"))
        }
        published_quality = (
            versioning.iteration_path(version)
            / destination_quality.relative_to(staging)
        )
        for source_name in relevant_names:
            bind_quantum_acceptance_receipt_to_commit(
                campaign,
                copied_by_source[source_name],
                source_pointdir=source_name,
                committed_pointdir=str(
                    committed_by_source[source_name]["committed_pointdir"]
                ),
                quality_manifest_source=destination_quality,
                quality_manifest_published=published_quality,
                quality_record=committed_record_by_source[source_name],
            )
        quality_evidence.append(
            {
                "path": destination_quality.relative_to(staging).as_posix(),
                "sha256": sha256_file(destination_quality),
                "source_path": source_quality.relative_to(campaign.resolve()).as_posix(),
                "source_sha256": sha256_file(source_quality),
                "phase": str(quality_payload["phase"]),
                "iteration": int(quality_payload["iteration"]),
                "pointdir_bindings": [
                    committed_by_source[name] for name in relevant_names
                ],
            }
        )
    # Receipt rebinding changes the committed pointdir tree.  Recompute every
    # content digest before publishing the immutable version manifest.
    added_entries = [
        ReferenceDataEntry(
            global_ordinal=entry.global_ordinal,
            introduced_in_version=entry.introduced_in_version,
            pointdir_name=entry.pointdir_name,
            pointdir_path=entry.pointdir_path,
            candidate_id=entry.candidate_id,
            slot_id=entry.slot_id,
            split=entry.split,
            replacement_round=entry.replacement_round,
            pointdir_tree_sha256=hash_reference_pointdir_tree(entry.pointdir_path),
            provenance_sha256=sha256_file(
                entry.pointdir_path / PROVENANCE_FILENAME
            ),
        )
        for entry in added_entries
    ]
    payload = build_reference_data_version_payload(
        campaign_uid=str(allocation["campaign_uid"]),
        version=version,
        source_context=str(context),
        source_iteration=int(iteration),
        parent_view=parent_view,
        point_allocation_manifest=allocation_relative,
        point_allocation_sha256=allocation_manifest_sha256(allocation_path),
        added_entries=added_entries,
        quantum_quality_evidence=quality_evidence,
    )
    allocation_history = allocation_path.parent / "history"
    if allocation_history.is_dir():
        _copytree_no_symlinks(
            allocation_history,
            staging / ".point_allocation_history",
        )
    atomic_write_json(staging / "POINT_ALLOCATION.json", allocation)
    atomic_write_json(
        staging
        / (
            "POINT_ALLOCATION.version-"
            + str(version).zfill(COMMITTED_VERSION_NAME_WIDTH)
            + ".json"
        ),
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
    _, _, created = commit_reference_data_delta(
        Path(campaign_dir),
        reference_data_version=0,
        context="bootstrap",
        iteration=0,
    )
    return bool(created)


def _model_bootstrap_context(campaign: Path) -> Optional[Dict[str, Any]]:
    """Return verified immutable model-bootstrap metadata, when configured."""
    from ..custom_bootstrap import (
        bootstrap_inputs_dir,
        custom_bootstrap_manifest_path,
        read_custom_bootstrap_manifest,
    )

    pointer = custom_bootstrap_manifest_path(campaign)
    if not pointer.exists():
        return None
    if pointer.is_symlink() or not pointer.is_file():
        raise ValueError("custom bootstrap manifest is not a regular file")
    payload = read_custom_bootstrap_manifest(campaign)
    model = payload.get("model")
    if not isinstance(model, dict):
        return None
    root = bootstrap_inputs_dir(campaign)
    model_manifest = root / "MODEL_BOOTSTRAP.json"
    if not model_manifest.is_file() or model_manifest.is_symlink():
        raise ValueError("model-bootstrap manifest is missing from immutable inputs")
    if int(model.get("training_count", 0)) <= 0:
        raise ValueError("model-bootstrap training count must be positive")
    return {"root": root, "manifest_path": model_manifest, "model": model}


def _load_model_bootstrap_tasks(
    context: Mapping[str, Any],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    from ichor.core.models import Model

    root = Path(context["root"])
    records = list(context["model"].get("files") or [])
    tasks: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("model-bootstrap file record must be an object")
        path = root / str(record.get("path") or "")
        if not path.is_file() or path.is_symlink():
            raise ValueError("model-bootstrap model is missing: " + str(path))
        if sha256_file(path) != str(record.get("sha256") or ""):
            raise ValueError("model-bootstrap model SHA mismatch: " + str(path))
        model = Model(path)
        key = (str(model.type), str(model.atom))
        if key in tasks:
            raise ValueError("duplicate model-bootstrap task: " + repr(key))
        tasks[key] = {"model": model, "path": path, "record": record}
    return tasks


def _model_bootstrap_system_alf(
    context: Mapping[str, Any],
    task_models: Mapping[Tuple[str, str], Mapping[str, Any]],
    properties: Sequence[str],
) -> Dict[str, Any]:
    from ichor.core.atoms import ALF

    atoms = [str(value) for value in context["model"].get("atoms", [])]
    if not atoms:
        raise ValueError("model-bootstrap atom list is empty")
    primary_property = "iqa" if "iqa" in properties else str(properties[0])
    out: Dict[str, Any] = {}
    for atom in atoms:
        task = task_models.get((primary_property, atom))
        if task is None:
            raise ValueError("model-bootstrap lacks ALF task for " + atom)
        values = [
            int(value)
            for value in getattr(task["model"], "ialf")
            if value is not None
        ]
        if len(values) == 2:
            out[atom] = ALF(values[0], values[1], None)
        elif len(values) == 3:
            out[atom] = ALF(values[0], values[1], values[2])
        else:
            raise ValueError("model-bootstrap ALF has an invalid length for " + atom)
        for prop in properties:
            other = task_models.get((str(prop), atom))
            if other is None:
                raise ValueError("model-bootstrap task is missing: " + str(prop) + "/" + atom)
            other_values = tuple(
                int(value)
                for value in getattr(other["model"], "ialf")
                if value is not None
            )
            if other_values != tuple(values):
                raise ValueError("model-bootstrap properties disagree on ALF for " + atom)
    return out


def _prepend_model_bootstrap_training_rows(
    csv_path: Path,
    *,
    atom: str,
    properties: Sequence[str],
    task_models: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> Tuple[int, str]:
    """Prepend immutable per-task model rows to one generated training CSV."""
    models = []
    for prop in properties:
        task = task_models.get((str(prop), str(atom)))
        if task is None:
            raise ValueError("model-bootstrap task is missing: " + str(prop) + "/" + str(atom))
        models.append(task["model"])
    reference_x = np.asarray(models[0].x, dtype=float)
    for prop, model in zip(properties[1:], models[1:]):
        if not np.allclose(
            np.asarray(model.x, dtype=float),
            reference_x,
            rtol=1.0e-10,
            atol=1.0e-10,
        ):
            raise ValueError(
                "model-bootstrap feature rows disagree for " + str(prop) + "/" + str(atom)
            )
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        existing = list(csv.reader(handle))
    if not existing:
        raise ValueError("generated FEREBUS training CSV is empty: " + str(csv_path))
    feature_headers = ["f" + str(index) for index in range(1, reference_x.shape[1] + 1)]
    expected_header = feature_headers + [str(prop) for prop in properties]
    if [str(value).strip() for value in existing[0]] != expected_header:
        raise ValueError(
            "generated FEREBUS CSV header cannot accept model baseline: " + str(csv_path)
        )
    baseline_rows: List[List[str]] = []
    target_arrays = [np.asarray(model.y, dtype=float).reshape(-1) for model in models]
    for row_index, feature_row in enumerate(reference_x):
        baseline_rows.append(
            [format(float(value), ".17g") for value in feature_row]
            + [format(float(values[row_index]), ".17g") for values in target_arrays]
        )
    temporary = csv_path.with_name("." + csv_path.name + ".baseline.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(expected_header)
        writer.writerows(baseline_rows)
        writer.writerows(existing[1:])
    os.replace(temporary, csv_path)
    return len(baseline_rows), sha256_file(csv_path)


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
    # The initial bootstrap has no APPEND phase ahead of it, so
    # QM_REFERENCE_DATA/iteration-000000 may not exist when INITIAL_FEREBUS
    # stages. Build it from the initial quantum staging so the
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

    staging = trained_models_dir(campaign) / "iteration-staging"
    # iteration-staging is ONE shared scratch dir reused every iteration, so wipe it first.
    # otherwise last iteration's *_train.csv / *.model lying around get globbed back in and we
    # either split stale data or re-commit an old model as a fresh version (both silent + nasty).
    if staging.exists():
        _checked_rmtree(
            staging,
            campaign_dir=campaign,
            allowed_roots=[campaign / TRAINED_MODELS_DIRNAME],
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
    pointdir_versions = {
        entry.pointdir_name: int(entry.introduced_in_version)
        for entry in view.entries
    }
    forced_ferebus_splits = {
        entry.pointdir_name: entry.split for entry in view.entries
    }
    f = config.ferebus
    properties = [str(p) for p in getattr(f, "properties", ["iqa"])]
    if not properties:
        raise ValueError("ferebus.properties must contain at least one property")
    model_bootstrap = _model_bootstrap_context(campaign)
    model_tasks = (
        {} if model_bootstrap is None
        else _load_model_bootstrap_tasks(model_bootstrap)
    )
    # Imported baselines must retain the ALFs used to encode their X rows.
    # Ordinary campaigns continue to use ICHOR's deterministic sequence ALFs.
    system_alf = (
        pd.alf_dict(calculate_alf_atom_sequence)
        if model_bootstrap is None
        else _model_bootstrap_system_alf(
            model_bootstrap,
            model_tasks,
            properties,
        )
    )
    _require_native_ferebus_alf(system_alf)

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
    validate_safe_path_token("FEREBUS system", system)
    from . import ferebus_dataset as _fds
    from .ferebus_split_ledger import ensure_split_assignments
    from ..point_allocation import (
        allocation_manifest_sha256,
        point_allocation_path,
        read_point_allocation,
    )

    allocation_context = "bootstrap" if version == 0 else "active"
    allocation_iteration = 0 if allocation_context == "bootstrap" else version
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
    expected_new_counts = {
        split: int((allocation_payload.get("targets") or {}).get(split, 0))
        for split in ("train", "int_val", "ext_val")
    }
    historical_training_rows = (
        0 if model_bootstrap is None
        else int(model_bootstrap["model"].get("training_count", 0))
    )

    split_ledger = ensure_split_assignments(
        campaign,
        pointdir_names,
        reference_data_version=version,
        reference_data_view_sha256=str(view.cumulative_view_sha256),
        expected_new_counts=expected_new_counts,
        pointdir_identity=pointdir_identities,
        pointdir_versions=pointdir_versions,
        forced_splits=forced_ferebus_splits,
        allocation_manifest_sha256=allocation_hash,
        historical_training_rows=historical_training_rows,
    )
    ledger_row_ids = dict(split_ledger["row_ids"])
    from ..versioning.reference_data import canonical_json_sha256

    source_row_identities = [
        {
            "source_row_index": int(index),
            **entry.identity_payload(),
        }
        for index, entry in enumerate(view.entries)
    ]
    historical_row_identities = [
        {
            "source": "model_bootstrap",
            "historical_row_index": int(index),
            "model_bootstrap_manifest_sha256": (
                None
                if model_bootstrap is None
                else sha256_file(model_bootstrap["manifest_path"])
            ),
        }
        for index in range(historical_training_rows)
    ]
    split_row_identities = {
        split: (
            list(historical_row_identities) if split == "train" else []
        )
        + [source_row_identities[index] for index in ledger_row_ids[split]]
        for split in ("train", "int_val", "ext_val")
    }
    row_identity_payload = {
        "schema_version": FEREBUS_ROW_IDENTITIES_SCHEMA_VERSION,
        "campaign_uid": str(view.campaign_uid),
        "reference_data_version": int(version),
        "reference_data_view_sha256": str(view.cumulative_view_sha256),
        "source_rows": source_row_identities,
        "source_rows_sha256": canonical_json_sha256(source_row_identities),
        "splits": {
            split: {
                "rows": split_row_identities[split],
                "n_rows": len(split_row_identities[split]),
                "row_identity_sha256": canonical_json_sha256(
                    split_row_identities[split]
                ),
            }
            for split in ("train", "int_val", "ext_val")
        },
    }
    row_identity_path = staging / FEREBUS_ROW_IDENTITIES
    atomic_write_json(row_identity_path, row_identity_payload)
    split_snapshot_path = staging / FEREBUS_SPLIT_SNAPSHOT
    ledger_payload = json.loads(Path(split_ledger["path"]).read_text(encoding="utf-8"))
    atomic_write_json(split_snapshot_path, ledger_payload)
    if sha256_file(split_snapshot_path) != sha256_file(split_ledger["path"]):
        raise ValueError("FEREBUS split-ledger snapshot copy verification failed")
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
            allow_empty_splits=("train",) if historical_training_rows else (),
        )
        counts = dict(split["counts"])
        if historical_training_rows:
            for prop in properties:
                train_csv = prop_dirs[prop] / (
                    system + "_" + atom + "_TRAINING_SET.csv"
                )
                baseline_count, merged_hash = _prepend_model_bootstrap_training_rows(
                    train_csv,
                    atom=atom,
                    properties=properties,
                    task_models=model_tasks,
                )
                if baseline_count != historical_training_rows:
                    raise ValueError(
                        "model-bootstrap training-row count changed while staging"
                    )
                if not merged_hash:
                    raise ValueError("model-bootstrap merged training CSV hash is empty")
            counts["train"] = int(counts["train"]) + historical_training_rows
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
    try:
        from ..ferebus_prior import (
            backend_kernel_token,
            resolve_ferebus_prior_contract,
        )

        prior_contract = resolve_ferebus_prior_contract(
            config,
            atom_labels=atom_labels,
        )
    except Exception as exc:
        raise ValueError("FEREBUS physical-prior contract is invalid: " + str(exc)) from exc
    if model_bootstrap is not None and model_bootstrap["model"].get(
        "prior_mean_contract"
    ) != prior_contract.to_dict():
        raise ValueError(
            "imported model bootstrap physical-prior contract does not match campaign"
        )
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
        validate_safe_path_token("FEREBUS property", prop)
    if len({str(prop).casefold() for prop in properties}) != len(properties):
        raise ValueError("FEREBUS properties contain a case-insensitive collision")
    if len({str(atom).casefold() for atom in atom_labels}) != len(atom_labels):
        raise ValueError("FEREBUS atom labels contain a case-insensitive collision")
    for prop in properties:
        for atom in atom_labels:
            output_dir = staging / prop / atom
            input_dir = output_dir / "datasets"
            config_path = output_dir / "ferebus.config"
            alf_cli = "_".join(str(x) for x in alf_by_atom[atom])
            training_csv = input_dir / (system + "_" + atom + "_TRAINING_SET.csv")
            int_csv = input_dir / (system + "_" + atom + "_INT_VALIDATION_SET.csv")
            ext_csv = input_dir / (system + "_" + atom + "_EXT_VALIDATION_SET.csv")
            source_training_csv = prop_dirs[prop] / training_csv.name
            source_int_csv = prop_dirs[prop] / int_csv.name
            source_ext_csv = prop_dirs[prop] / ext_csv.name
            model_path = output_dir / (system + "_" + prop + "_" + atom + ".model")
            dataset_records: Dict[str, Dict[str, Any]] = {}
            for split_name, dataset_path, source_dataset_path in (
                ("train", training_csv, source_training_csv),
                ("int_val", int_csv, source_int_csv),
                ("ext_val", ext_csv, source_ext_csv),
            ):
                identity_record = row_identity_payload["splits"][split_name]
                dataset_records[split_name] = {
                    "path": ferebus_relative_path(staging, dataset_path),
                    "size": int(source_dataset_path.stat().st_size),
                    "sha256": sha256_file(source_dataset_path),
                    "rows": int(split_counts[atom][split_name]),
                    "row_identity_sha256": str(
                        identity_record["row_identity_sha256"]
                    ),
                    "row_identity_count": int(identity_record["n_rows"]),
                }
            training_values = _fds.read_property_values(source_training_csv, prop)
            tasks.append(
                {
                    "task_index": int(task_index),
                    "property": prop,
                    "atom": atom,
                    "prior_mean": prior_contract.task_payload(
                        prop,
                        atom,
                        training_values=training_values,
                        training_dataset_sha256=dataset_records["train"]["sha256"],
                    ),
                    "alf_1_indexed": [int(x) for x in alf_by_atom[atom]],
                    "alf_cli": alf_cli,
                    "property_dir": ferebus_relative_path(staging, staging / prop),
                    "output_dir": ferebus_relative_path(staging, output_dir),
                    "input_dir": ferebus_relative_path(staging, input_dir),
                    "config_path": ferebus_relative_path(staging, config_path),
                    "training_csv": ferebus_relative_path(staging, training_csv),
                    "int_validation_csv": ferebus_relative_path(staging, int_csv),
                    "ext_validation_csv": ferebus_relative_path(staging, ext_csv),
                    "expected_model_path": ferebus_relative_path(staging, model_path),
                    "command_args": [
                        "-c", ferebus_relative_path(staging, config_path),
                        "-I", ferebus_relative_path(staging, input_dir),
                        "-O", ferebus_relative_path(staging, output_dir),
                        "-P", prop,
                        "-A", atom,
                        "-ALF", alf_cli,
                    ],
                    "row_counts": dict(split_counts[atom]),
                    "row_ids": dict(row_ids_by_atom[atom]),
                    "historical_training_rows": int(historical_training_rows),
                    "historical_training_row_ids": list(
                        range(historical_training_rows)
                    ),
                    "datasets": dataset_records,
                    "degenerate_property_stats": bool(
                        stats_by_prop_atom.get((prop, atom), {}).get(
                            "degenerate_property_stats", False
                        )
                    ),
                }
            )
            task_index += 1

    if model_bootstrap is not None:
        copied_model_manifest = staging / "MODEL_BOOTSTRAP.json"
        shutil.copy2(model_bootstrap["manifest_path"], copied_model_manifest)
        if sha256_file(copied_model_manifest) != sha256_file(
            model_bootstrap["manifest_path"]
        ):
            raise ValueError("model-bootstrap manifest copy verification failed")

    _write_ferebus_manifest(
        staging,
        {
            "schema_version": FEREBUS_TASK_SCHEMA_VERSION,
            "campaign_uid": str(view.campaign_uid),
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
            "prior_mean_contract": prior_contract.to_dict(),
            "kernel_contract": {
                "family": str(f.kernel),
                "backend_token": backend_kernel_token(f.kernel),
                "loss": "huber",
                "constant_noise": True,
                "full_ard": True,
                "feature_scaling": True,
                "property_scaling": False,
                "kernel_prefactor_mode": 2,
            },
            "row_identity_snapshot": {
                "path": FEREBUS_ROW_IDENTITIES,
                "size": int(row_identity_path.stat().st_size),
                "sha256": sha256_file(row_identity_path),
                "source_rows_sha256": str(
                    row_identity_payload["source_rows_sha256"]
                ),
            },
            "degenerate_property_stats": list(degenerate_property_stats),
            "model_bootstrap": (
                None if model_bootstrap is None else {
                    "manifest": Path(model_bootstrap["manifest_path"]).resolve().relative_to(
                        campaign.resolve()
                    ).as_posix(),
                    "manifest_sha256": sha256_file(model_bootstrap["manifest_path"]),
                    "historical_training_rows": int(historical_training_rows),
                }
            ),
            "job_details": ferebus_relative_path(staging, job_details),
            "split_ledger": {
                "path": FEREBUS_SPLIT_SNAPSHOT,
                "size": int(split_snapshot_path.stat().st_size),
                "sha256": sha256_file(split_snapshot_path),
                "source_path": Path(split_ledger["path"]).resolve().relative_to(
                    campaign.resolve()
                ).as_posix(),
                "counts": dict(split_ledger["counts"]),
                "version_allocation": dict(split_ledger["version_allocation"]),
                "allocation_policy": str(split_ledger["allocation_policy"]),
                "allocation_manifest": allocation_path.resolve().relative_to(
                    campaign.resolve()
                ).as_posix(),
                "allocation_manifest_sha256": str(allocation_hash),
                "allocation_iteration": int(allocation_iteration),
                "slot_assignment_sha256": str(
                    allocation_payload["slot_assignment_sha256"]
                ),
                "forced_splits": dict(forced_ferebus_splits),
            },
            "tasks": tasks,
        },
    )
    return staging, len(tasks)


def prepare_imported_model_bootstrap(staging_dir: Path) -> Dict[str, Any]:
    """Materialise an imported model set as completed initial FEREBUS output.

    The ordinary staging path has already generated the exact train/internal/
    external CSV contract.  This helper only installs the user-confirmed
    models and copies those datasets into the per-task layout that pyferebus
    would otherwise create.  It never retrains or rewrites a model.
    """
    staging = Path(staging_dir)
    manifest = read_ferebus_manifest(staging)
    from ..ferebus_prior import (
        contract_from_payload,
        validate_ferebus_config_contract,
    )

    prior_contract = contract_from_payload(manifest.get("prior_mean_contract"))
    model_binding = manifest.get("model_bootstrap")
    if not isinstance(model_binding, dict):
        raise ValueError("FEREBUS staging is not bound to an imported model set")
    # Derive the campaign root from the immutable manifest path rather than
    # relying on a caller-provided directory.
    manifest_path_text = str(model_binding.get("manifest") or "")
    if not manifest_path_text:
        raise ValueError("model-bootstrap manifest binding is missing")
    campaign = staging.parent.parent
    context = _model_bootstrap_context(campaign)
    if context is None:
        raise ValueError("immutable model-bootstrap inputs are missing")
    bound_manifest = Path(context["manifest_path"])
    if bound_manifest.resolve().relative_to(campaign.resolve()).as_posix() != manifest_path_text:
        raise ValueError("model-bootstrap manifest path binding mismatch")
    if sha256_file(bound_manifest) != str(model_binding.get("manifest_sha256") or ""):
        raise ValueError("model-bootstrap manifest SHA mismatch")
    model_tasks = _load_model_bootstrap_tasks(context)

    dataset_fields = (
        "training_csv",
        "int_validation_csv",
        "ext_validation_csv",
    )
    for task in manifest.get("tasks", []):
        prop = str(task["property"])
        atom = str(task["atom"])
        imported = model_tasks.get((prop, atom))
        if imported is None:
            raise ValueError("imported model task is missing: " + prop + "/" + atom)
        model_destination = resolve_ferebus_task_path(
            staging,
            task["expected_model_path"],
            "expected_model_path",
        )
        if model_destination.exists():
            raise ValueError(
                "refusing to overwrite an existing staged model: "
                + str(model_destination)
            )
        model_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(imported["path"], model_destination)
        if sha256_file(model_destination) != sha256_file(imported["path"]):
            raise ValueError("imported model copy verification failed: " + prop + "/" + atom)

        for field in dataset_fields:
            destination = resolve_ferebus_task_path(staging, task[field], field)
            source = staging / prop / destination.name
            if not source.is_file() or source.is_symlink():
                raise ValueError("staged FEREBUS dataset is missing: " + str(source))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if sha256_file(destination) != sha256_file(source):
                raise ValueError("FEREBUS dataset copy verification failed: " + str(destination))

        config_path = resolve_ferebus_task_path(
            staging,
            task["config_path"],
            "config_path",
        )
        atomic_write_text(
            config_path,
            "# Imported model bootstrap; no version-0 FEREBUS optimisation was run.\n"
            + "system = " + str(manifest["system"]) + "\n"
            + "property = " + prop + "\n"
            + "atom = " + atom + "\n"
            + "mean_type = " + str(prior_contract.mean_type) + "\n"
            + 'level_of_theory = "'
            + str(prior_contract.level_of_theory or "not_applicable")
            + '"\n'
            + "iqaDeviationFactor = "
            + repr(prior_contract.iqa_deviation_factor)
            + "\n"
            + "scaling = "
            + ("1" if prior_contract.feature_scaling else "0")
            + "\n"
            + "scale_feats = "
            + ("1" if prior_contract.feature_scaling else "0")
            + "\n"
            + "scale_prop = 0\n",
        )
        parsed_contract = validate_ferebus_config_contract(
            config_path,
            prior_contract,
        )
        task["generated_config"] = {
            "path": str(task["config_path"]),
            "size": int(config_path.stat().st_size),
            "sha256": sha256_file(config_path),
            "parsed_contract": parsed_contract,
            "prior_mean_contract_sha256": prior_contract.contract_sha256,
        }

    _write_ferebus_manifest(staging, manifest)

    from .ferebus_task_runner import write_imported_model_receipts
    from ..submit.pyferebus_wrap import _write_structured_task_map

    _write_structured_task_map(
        staging,
        executable="ferebus",
        execution_kind="imported_model_bootstrap",
        performance_required=False,
    )
    write_imported_model_receipts(staging)

    # Re-read with full dataset verification and validate the model parser
    # contract before the caller evaluates held-out quality.
    verified = read_ferebus_manifest(staging, verify_dataset_files=True)
    from .model_contract import validate_ferebus_model_contract

    validate_ferebus_model_contract(staging, committed=False)
    return verified
