"""Hash-chained manifests for bootstrap and completed sampling iterations."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from ..daemon.state import atomic_write_json, _fsync_parent_dir
from ..handoff_manifests import (
    ariadne_landing_audit_path,
    ariadne_results_path,
    phase_b_selection_path,
    read_ariadne_landing_audit,
    read_ariadne_results_manifest,
    read_phase_a_sample_manifest,
    read_phase_b_selection_manifest,
    seeds_picked_path,
)
from ..layout import (
    active_allocation_dir,
    active_iteration_dir,
    active_learning_dir,
    active_protocol_dir,
    active_seed_selection_dir,
    bootstrap_allocation_dir,
    bootstrap_dir,
    bootstrap_selection_dir,
)
from ..point_allocation import point_allocation_path, read_point_allocation
from .manifest import sha256_file
from .reference_data import ReferenceDataVersioning
from .trained_models import resolve_trained_model_set


BOOTSTRAP_MANIFEST_FILENAME = "BOOTSTRAP_MANIFEST.json"
ITERATION_MANIFEST_FILENAME = "ITERATION_MANIFEST.json"
SAMPLING_MANIFEST_SCHEMA_VERSION = 1
_LOCK_FILENAMES = frozenset({".provenance.lock", "POINT_ALLOCATION.lock"})
_BOOTSTRAP_TOP_LEVEL_DIRECTORIES = frozenset({"selection", "allocation"})
_ACTIVE_TOP_LEVEL_DIRECTORIES = frozenset({
    "protocol",
    "seed_selection",
    "ariadne",
    "phase_b",
    "allocation",
    "calibration",
})


class SamplingIterationError(RuntimeError):
    """Raised when a sampling iteration cannot be sealed or trusted."""


def bootstrap_manifest_path(campaign_dir: Path) -> Path:
    return bootstrap_dir(campaign_dir) / BOOTSTRAP_MANIFEST_FILENAME


def active_iteration_manifest_path(campaign_dir: Path, iteration: int) -> Path:
    return active_iteration_dir(campaign_dir, iteration) / ITERATION_MANIFEST_FILENAME


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _role(relative_path: str) -> str:
    if relative_path in {
        "selection/selected.xyz",
        "selection/selected_indices.dat",
        "seed_selection/seeds.xyz",
        "phase_b/selected.xyz",
        "phase_b/selected_raw.xyz",
    }:
        return "derived_cache"
    return "authoritative"


def _remove_completed_lock_files(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.name not in _LOCK_FILENAMES:
            continue
        if path.is_symlink() or not path.is_file():
            raise SamplingIterationError(
                "completed iteration contains an invalid lock entry: " + str(path)
            )
        path.unlink()


def _inventory(root: Path, manifest_name: str) -> Tuple[list, list]:
    if root.is_symlink() or not root.is_dir():
        raise SamplingIterationError("sampling root is not a regular directory: " + str(root))
    files = []
    directories = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative == manifest_name:
            continue
        if path.is_symlink():
            raise SamplingIterationError("sampling iteration contains a symlink: " + relative)
        if (
            path.name.startswith(".tmp-")
            or ".partial-" in path.name
            or path.name.endswith(".tmp")
        ):
            raise SamplingIterationError(
                "sampling iteration contains an incomplete artefact: " + relative
            )
        if path.is_dir():
            directories.append(relative)
        elif path.is_file():
            files.append({
                "path": relative,
                "size": int(path.stat().st_size),
                "sha256": sha256_file(path),
                "role": _role(relative),
            })
        else:
            raise SamplingIterationError(
                "sampling iteration contains a special filesystem entry: " + relative
            )
    return files, directories


def _required_paths(root: Path, relative_paths: Iterable[str]) -> None:
    for relative in relative_paths:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise SamplingIterationError("required sampling artefact is missing: " + relative)


def _validate_top_level(
    root: Path,
    *,
    manifest_name: str,
    allowed_directories: frozenset[str],
) -> None:
    for child in root.iterdir():
        if child.name == manifest_name:
            if child.is_symlink() or not child.is_file():
                raise SamplingIterationError(
                    "sampling manifest is not a regular file: " + str(child)
                )
            continue
        if child.name not in allowed_directories:
            raise SamplingIterationError(
                "unexpected sampling top-level artefact: " + child.name
            )
        if child.is_symlink() or not child.is_dir():
            raise SamplingIterationError(
                "sampling top-level entry is not a regular directory: "
                + child.name
            )


def _head_binding(campaign: Path, version: int) -> Dict[str, Any]:
    reference = ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA").resolve(
        int(version),
        verification="deep",
    )
    models = resolve_trained_model_set(campaign, int(version), verification="deep")
    if reference.campaign_uid != models.campaign_uid:
        raise SamplingIterationError("reference-data/model campaign UID mismatch")
    return {
        "version": int(version),
        "reference_data": {
            "head_manifest_sha256": str(reference.head_manifest_sha256),
            "cumulative_view_sha256": str(reference.cumulative_view_sha256),
            "point_count": int(len(reference.entries)),
        },
        "trained_models": {
            "head_manifest_sha256": str(models.head_manifest_sha256),
            "model_set_sha256": str(models.model_set_sha256),
            "reference_data_view_sha256": str(models.reference_data_view_sha256),
        },
    }


def _manifest_payload(
    *,
    campaign_uid: str,
    kind: str,
    iteration: int,
    parent: Optional[Mapping[str, Any]],
    input_head: Optional[Mapping[str, Any]],
    output_head: Mapping[str, Any],
    files: list,
    directories: list,
) -> Dict[str, Any]:
    inventory = {"files": files, "directories": directories}
    return {
        "schema_version": SAMPLING_MANIFEST_SCHEMA_VERSION,
        "kind": str(kind),
        "campaign_uid": str(campaign_uid),
        "iteration": int(iteration),
        "completed_at_iso": datetime.now(timezone.utc).isoformat(),
        "parent": None if parent is None else dict(parent),
        "input_head": None if input_head is None else dict(input_head),
        "output_head": dict(output_head),
        "inventory_sha256": _canonical_sha256(inventory),
        **inventory,
    }


def _verify_inventory(root: Path, payload: Mapping[str, Any], manifest_name: str) -> None:
    expected_files = payload.get("files")
    expected_directories = payload.get("directories")
    if not isinstance(expected_files, list) or not isinstance(expected_directories, list):
        raise SamplingIterationError("sampling manifest inventory is invalid")
    actual_files, actual_directories = _inventory(root, manifest_name)
    if actual_files != expected_files or actual_directories != expected_directories:
        raise SamplingIterationError("sampling iteration exact inventory mismatch")
    if str(payload.get("inventory_sha256") or "") != _canonical_sha256({
        "files": actual_files,
        "directories": actual_directories,
    }):
        raise SamplingIterationError("sampling iteration inventory SHA mismatch")


def _read_manifest(path: Path, *, kind: str, iteration: int) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SamplingIterationError(
            "sampling manifest is not a regular file: " + str(path)
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SamplingIterationError("sampling manifest is unreadable: " + str(path)) from exc
    if not isinstance(payload, dict):
        raise SamplingIterationError("sampling manifest must be a JSON object")
    try:
        schema_version = int(payload.get("schema_version", -1))
        observed_iteration = int(payload.get("iteration", -1))
    except (TypeError, ValueError) as exc:
        raise SamplingIterationError(
            "sampling manifest schema or iteration is not an integer"
        ) from exc
    if schema_version != SAMPLING_MANIFEST_SCHEMA_VERSION:
        raise SamplingIterationError("unsupported sampling manifest schema")
    if str(payload.get("kind") or "") != str(kind):
        raise SamplingIterationError("sampling manifest kind mismatch")
    if observed_iteration != int(iteration):
        raise SamplingIterationError("sampling manifest iteration mismatch")
    return payload


def _seal(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise SamplingIterationError("cannot seal a symlinked sampling artefact")
        mode = stat.S_IMODE(path.stat().st_mode)
        os.chmod(path, mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    mode = stat.S_IMODE(root.stat().st_mode)
    os.chmod(root, mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    _fsync_parent_dir(root)


def finalise_bootstrap(campaign_dir: Path, campaign_uid: str) -> Path:
    campaign = Path(campaign_dir)
    root = bootstrap_dir(campaign)
    path = bootstrap_manifest_path(campaign)
    if path.is_file():
        verify_bootstrap(campaign, expected_campaign_uid=campaign_uid)
        _seal(root)
        return path
    read_phase_a_sample_manifest(
        bootstrap_selection_dir(campaign),
        expected_campaign_uid=campaign_uid,
    )
    allocation = read_point_allocation(
        point_allocation_path(campaign, context="bootstrap", iteration=0),
        expected_campaign_uid=campaign_uid,
    )
    if not bool((allocation.get("summary") or {}).get("complete", False)):
        raise SamplingIterationError("bootstrap allocation is incomplete")
    output_head = _head_binding(campaign, 0)
    if str(output_head["reference_data"].get("head_manifest_sha256") or "") == "":
        raise SamplingIterationError("bootstrap reference-data head is empty")
    _remove_completed_lock_files(root)
    _required_paths(root, (
        "selection/SELECTION.json",
        "selection/selected.xyz",
        "selection/selected_indices.dat",
        "allocation/POINT_ALLOCATION.json",
    ))
    _validate_top_level(
        root,
        manifest_name=BOOTSTRAP_MANIFEST_FILENAME,
        allowed_directories=_BOOTSTRAP_TOP_LEVEL_DIRECTORIES,
    )
    files, directories = _inventory(root, BOOTSTRAP_MANIFEST_FILENAME)
    payload = _manifest_payload(
        campaign_uid=campaign_uid,
        kind="bootstrap",
        iteration=0,
        parent=None,
        input_head=None,
        output_head=output_head,
        files=files,
        directories=directories,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    verify_bootstrap(campaign, expected_campaign_uid=campaign_uid)
    _seal(root)
    return path


def verify_bootstrap(
    campaign_dir: Path,
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    campaign = Path(campaign_dir)
    root = bootstrap_dir(campaign)
    path = bootstrap_manifest_path(campaign)
    payload = _read_manifest(path, kind="bootstrap", iteration=0)
    if expected_campaign_uid is not None and str(payload.get("campaign_uid")) != str(
        expected_campaign_uid
    ):
        raise SamplingIterationError("bootstrap campaign UID mismatch")
    output_head = _head_binding(campaign, 0)
    if payload.get("output_head") != output_head:
        raise SamplingIterationError("bootstrap output-head binding mismatch")
    _verify_inventory(root, payload, BOOTSTRAP_MANIFEST_FILENAME)
    _validate_top_level(
        root,
        manifest_name=BOOTSTRAP_MANIFEST_FILENAME,
        allowed_directories=_BOOTSTRAP_TOP_LEVEL_DIRECTORIES,
    )
    return payload


def finalise_active_iteration(
    campaign_dir: Path,
    iteration: int,
    campaign_uid: str,
) -> Path:
    campaign = Path(campaign_dir)
    value = int(iteration)
    if value < 1:
        raise SamplingIterationError("active iteration must be >= 1")
    root = active_iteration_dir(campaign, value)
    path = active_iteration_manifest_path(campaign, value)
    if path.is_file():
        verify_active_iteration(campaign, value, expected_campaign_uid=campaign_uid)
        _seal(root)
        return path

    read_ariadne_results_manifest(root, expected_iteration=value)
    read_ariadne_landing_audit(root, expected_iteration=value)
    read_phase_b_selection_manifest(
        root,
        expected_iteration=value,
        expected_campaign_uid=campaign_uid,
    )
    allocation = read_point_allocation(
        point_allocation_path(campaign, context="active", iteration=value),
        expected_campaign_uid=campaign_uid,
    )
    if not bool((allocation.get("summary") or {}).get("complete", False)):
        raise SamplingIterationError("active point allocation is incomplete")
    input_head = _head_binding(campaign, value - 1)
    output_head = _head_binding(campaign, value)
    if str(input_head["trained_models"]["reference_data_view_sha256"]) != str(
        input_head["reference_data"]["cumulative_view_sha256"]
    ):
        raise SamplingIterationError("input model/reference-data view mismatch")
    if str(output_head["trained_models"]["reference_data_view_sha256"]) != str(
        output_head["reference_data"]["cumulative_view_sha256"]
    ):
        raise SamplingIterationError("output model/reference-data view mismatch")

    if value == 1:
        parent_path = bootstrap_manifest_path(campaign)
        verify_bootstrap(campaign, expected_campaign_uid=campaign_uid)
    else:
        parent_path = active_iteration_manifest_path(campaign, value - 1)
        verify_active_iteration(
            campaign,
            value - 1,
            expected_campaign_uid=campaign_uid,
        )
    parent = {
        "path": parent_path.resolve().relative_to(campaign.resolve()).as_posix(),
        "sha256": sha256_file(parent_path),
    }
    _remove_completed_lock_files(root)
    _required_paths(root, (
        "protocol/SAMPLING_PROTOCOL_RESOLVED.json",
        "protocol/SAMPLING_PROTOCOL_AUDIT.json",
        "protocol/SAMPLING_SCALE_MODEL.json",
        "seed_selection/SELECTION.json",
        "seed_selection/seeds.xyz",
        "ariadne/TASK_MAP.json",
        "ariadne/RESULTS.json",
        "ariadne/AUDIT.json",
        "phase_b/SELECTION.json",
        "phase_b/selected.xyz",
        "allocation/POINT_ALLOCATION.json",
        "allocation/SPLIT_RECEIPT.json",
    ))
    _validate_top_level(
        root,
        manifest_name=ITERATION_MANIFEST_FILENAME,
        allowed_directories=_ACTIVE_TOP_LEVEL_DIRECTORIES,
    )
    files, directories = _inventory(root, ITERATION_MANIFEST_FILENAME)
    payload = _manifest_payload(
        campaign_uid=campaign_uid,
        kind="active_iteration",
        iteration=value,
        parent=parent,
        input_head=input_head,
        output_head=output_head,
        files=files,
        directories=directories,
    )
    atomic_write_json(path, payload)
    verify_active_iteration(campaign, value, expected_campaign_uid=campaign_uid)
    _seal(root)
    return path


def verify_active_iteration(
    campaign_dir: Path,
    iteration: int,
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    campaign = Path(campaign_dir)
    value = int(iteration)
    root = active_iteration_dir(campaign, value)
    path = active_iteration_manifest_path(campaign, value)
    payload = _read_manifest(path, kind="active_iteration", iteration=value)
    if expected_campaign_uid is not None and str(payload.get("campaign_uid")) != str(
        expected_campaign_uid
    ):
        raise SamplingIterationError("active iteration campaign UID mismatch")
    if payload.get("input_head") != _head_binding(campaign, value - 1):
        raise SamplingIterationError("active iteration input-head binding mismatch")
    if payload.get("output_head") != _head_binding(campaign, value):
        raise SamplingIterationError("active iteration output-head binding mismatch")
    parent = payload.get("parent")
    if not isinstance(parent, dict):
        raise SamplingIterationError("active iteration parent binding is missing")
    expected_parent = (
        bootstrap_manifest_path(campaign)
        if value == 1
        else active_iteration_manifest_path(campaign, value - 1)
    )
    expected_parent_relative = expected_parent.resolve().relative_to(
        campaign.resolve()
    ).as_posix()
    if str(parent.get("path") or "") != expected_parent_relative:
        raise SamplingIterationError("active iteration parent path mismatch")
    if str(parent.get("sha256") or "") != sha256_file(expected_parent):
        raise SamplingIterationError("active iteration parent SHA mismatch")
    _verify_inventory(root, payload, ITERATION_MANIFEST_FILENAME)
    _validate_top_level(
        root,
        manifest_name=ITERATION_MANIFEST_FILENAME,
        allowed_directories=_ACTIVE_TOP_LEVEL_DIRECTORIES,
    )
    return payload


def verify_sampling_chain(
    campaign_dir: Path,
    through_iteration: int,
    *,
    expected_campaign_uid: Optional[str] = None,
) -> None:
    campaign = Path(campaign_dir)
    verify_bootstrap(campaign, expected_campaign_uid=expected_campaign_uid)
    for iteration in range(1, int(through_iteration) + 1):
        verify_active_iteration(
            campaign,
            iteration,
            expected_campaign_uid=expected_campaign_uid,
        )


__all__ = [
    "BOOTSTRAP_MANIFEST_FILENAME",
    "ITERATION_MANIFEST_FILENAME",
    "SamplingIterationError",
    "bootstrap_manifest_path",
    "active_iteration_manifest_path",
    "finalise_bootstrap",
    "verify_bootstrap",
    "finalise_active_iteration",
    "verify_active_iteration",
    "verify_sampling_chain",
]
