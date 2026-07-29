"""Immutable resource-resolution records for submitted daemon attempts.

Schema 2 binds both the scientific inputs used by the resource formula and
the Python/native implementation that will consume them.  Job-side
verification therefore proves more than the integrity of this JSON file: it
also proves that every named input and relevant implementation byte is still
the byte that was resolved before submission.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import sys
from ..strict_json import strict_json as json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

from .state import atomic_write_json
from .filesystem import campaign_owned_path
from .script_bundles import safe_component
from ..versioning.manifest import sha256_file


RESOURCE_RESOLUTION_SCHEMA_VERSION = 2
RESOURCE_FORMULA_VERSION = "2"
IMPLEMENTATION_IDENTITY_SCHEMA_VERSION = 1
ICHOR_PACKAGE_TREE_IDENTITY_KIND = "explicit_import_origins_v1"


_BACKEND_SOURCE_FILES = {
    "diversity": (
        "resource_solver.py",
        "../sampling/diversity.py",
        "../sampling/descriptors.py",
    ),
    "gaussian": (
        "resource_solver.py",
        "live_executor.py",
        "input_staging.py",
    ),
    "aimall": (
        "resource_solver.py",
        "live_executor.py",
        "input_staging.py",
    ),
    "ariadne": (
        "resource_solver.py",
        "../acquisition/ariadne_runner.py",
        "../acquisition/ariadne_local_runner.py",
        "live_executor.py",
    ),
    "ferebus": (
        "resource_solver.py",
        "../submit/pyferebus_wrap.py",
        "input_staging.py",
    ),
}


def _file_record(path: Path) -> Dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("implementation input is not a regular file: " + str(source))
    return {
        "path": str(source.resolve()),
        "size": int(source.stat().st_size),
        "sha256": sha256_file(source),
    }


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _dependency_lock_sha256() -> str:
    rows = []
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name") or "").strip()
        if name:
            rows.append((name.casefold(), str(distribution.version)))
    return _canonical_sha256(sorted(rows))


def _legacy_ichor_package_tree_sha256() -> str:
    from ..execution_identity import _tree_hash

    roots = []
    try:
        import ichor

        roots = [Path(value) for value in getattr(ichor, "__path__", [])]
    except Exception as exc:  # pragma: no cover - installation failure path
        raise ValueError("ICHOR package roots cannot be inspected") from exc
    if not roots:
        raise ValueError("ICHOR package roots cannot be inspected")
    return _tree_hash(roots)


def _ichor_package_tree_sha256() -> str:
    from ..execution_identity import ichor_package_tree_sha256

    return ichor_package_tree_sha256()


def capture_implementation_identity(
    campaign_dir: Union[str, Path],
    *,
    backend: str,
    backend_executable_path: Optional[Union[str, Path]] = None,
    require_environment_generation: bool = False,
) -> Dict[str, Any]:
    """Capture the implementation bytes that a queued attempt will execute."""
    campaign = Path(campaign_dir)
    backend_name = str(backend).strip().casefold()
    if backend_name not in _BACKEND_SOURCE_FILES:
        raise ValueError("unsupported resource backend identity: " + backend_name)
    module_root = Path(__file__).resolve().parent
    source_records = []
    for relative in _BACKEND_SOURCE_FILES[backend_name]:
        source_records.append(_file_record((module_root / relative).resolve()))

    operational = campaign / ".DATA" / "ACTIVE_LEARNING"
    current_path = operational / "environment_current.json"
    current_record = None
    generation_record = None
    generation_digest = None
    if current_path.exists() or current_path.is_symlink():
        from ..execution_identity import (
            read_active_environment_generation,
            read_execution_identity,
        )

        identity = read_execution_identity(campaign)
        active = read_active_environment_generation(
            campaign,
            expected_campaign_uid=str(identity["campaign_uid"]),
        )
        generation_path = Path(active["generation_path"])
        generation = active["generation"]
        generation_digest = str(generation["digest_sha256"])
        current_record = _file_record(current_path)
        generation_record = _file_record(generation_path)
    elif require_environment_generation:
        raise ValueError(
            "live resource resolution requires an active environment generation"
        )

    executable_record: Dict[str, Any]
    if backend_executable_path is None:
        executable_record = {"configured_path": None, "resolved_file": None}
    else:
        expanded = Path(os.path.expandvars(os.path.expanduser(str(backend_executable_path))))
        if "$" in str(expanded) or not expanded.is_absolute() or not expanded.is_file():
            executable_record = {
                "configured_path": str(backend_executable_path),
                "resolved_file": None,
            }
        else:
            executable_record = {
                "configured_path": str(backend_executable_path),
                "resolved_file": _file_record(expanded),
            }

    return {
        "schema_version": IMPLEMENTATION_IDENTITY_SCHEMA_VERSION,
        "captured_at_iso": datetime.now(timezone.utc).isoformat(),
        "backend": backend_name,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "ichor_package_tree_sha256": _ichor_package_tree_sha256(),
        "ichor_package_tree_identity_kind": ICHOR_PACKAGE_TREE_IDENTITY_KIND,
        "dependency_lock_sha256": _dependency_lock_sha256(),
        "source_files": source_records,
        "environment_current": current_record,
        "environment_generation": generation_record,
        "environment_generation_digest_sha256": generation_digest,
        "backend_executable": executable_record,
    }


def _iter_file_records(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        keys = set(value)
        if {"path", "size", "sha256"}.issubset(keys):
            yield value
            return
        for child in value.values():
            yield from _iter_file_records(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_file_records(child)


def _verify_file_record(
    record: Mapping[str, Any],
    *,
    campaign_dir: Optional[Path],
    label: str,
) -> None:
    raw_path = record.get("path")
    raw_size = record.get("size")
    raw_digest = record.get("sha256")
    if raw_path is None and raw_size is None and raw_digest is None:
        return
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(label + " file record has no path")
    if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 0:
        raise ValueError(label + " file record has an invalid size")
    if (
        not isinstance(raw_digest, str)
        or len(raw_digest) != 64
        or any(ch not in "0123456789abcdef" for ch in raw_digest)
    ):
        raise ValueError(label + " file record has an invalid SHA-256")
    path = Path(raw_path)
    if not path.is_absolute():
        raise ValueError(label + " file record path must be absolute: " + raw_path)
    if campaign_dir is not None:
        path = campaign_owned_path(campaign_dir, path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(label + " file is not a regular file: " + str(path))
    observed_size = int(path.stat().st_size)
    if observed_size != raw_size:
        raise ValueError(
            label
            + " file size mismatch for "
            + str(path)
            + ": expected "
            + str(raw_size)
            + " got "
            + str(observed_size)
        )
    observed_digest = sha256_file(path)
    if observed_digest != raw_digest:
        raise ValueError(
            label
            + " file SHA-256 mismatch for "
            + str(path)
            + ": expected "
            + raw_digest
            + " got "
            + observed_digest
        )


def verify_scientific_evidence(
    campaign_dir: Union[str, Path], evidence: Mapping[str, Any]
) -> None:
    records = list(_iter_file_records(evidence))
    for record in records:
        _verify_file_record(
            record,
            campaign_dir=Path(campaign_dir),
            label="scientific evidence",
        )


def _verify_bound_environment_generation(
    identity: Mapping[str, Any],
    *,
    campaign_dir: Path,
    expected_campaign_uid: str,
    current_package_digest: str,
) -> None:
    generation_record = identity.get("environment_generation")
    generation_digest = identity.get("environment_generation_digest_sha256")
    if not isinstance(generation_record, Mapping):
        raise ValueError(
            "resource implementation has no authenticated environment generation"
        )
    if (
        not isinstance(generation_digest, str)
        or len(generation_digest) != 64
        or any(ch not in "0123456789abcdef" for ch in generation_digest)
    ):
        raise ValueError(
            "resource implementation environment-generation digest is invalid"
        )
    raw_path = generation_record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(
            "resource implementation environment generation has no path"
        )
    path = campaign_owned_path(campaign_dir, Path(raw_path))
    expected_parent = (
        campaign_dir.resolve()
        / ".DATA"
        / "ACTIVE_LEARNING"
        / "environment_generations"
    )
    if path.parent.resolve() != expected_parent:
        raise ValueError(
            "resource implementation environment generation path is not canonical"
        )
    name = path.name
    if (
        not name.startswith("generation-")
        or not name.endswith(".json")
        or len(name) != len("generation-000000.json")
    ):
        raise ValueError(
            "resource implementation environment generation path is not canonical"
        )
    generation_text = name[len("generation-") : -len(".json")]
    if not generation_text.isdigit():
        raise ValueError(
            "resource implementation environment generation path is not canonical"
        )
    from ..execution_identity import read_environment_generation

    generation = read_environment_generation(
        campaign_dir,
        generation=int(generation_text),
        expected_campaign_uid=str(expected_campaign_uid),
    )
    if str(generation.get("digest_sha256") or "") != generation_digest:
        raise ValueError(
            "resource implementation environment-generation digest mismatch"
        )
    if (
        str(generation.get("ichor_package_tree_sha256") or "")
        != str(current_package_digest)
    ):
        raise ValueError("resource implementation ICHOR package tree has drifted")


def verify_implementation_identity(
    identity: Mapping[str, Any],
    *,
    campaign_dir: Optional[Union[str, Path]] = None,
    expected_campaign_uid: Optional[str] = None,
) -> None:
    if identity.get("schema_version") != IMPLEMENTATION_IDENTITY_SCHEMA_VERSION:
        raise ValueError("resource implementation identity has an unsupported schema")
    if str(identity.get("python_executable") or "") != str(Path(sys.executable).resolve()):
        raise ValueError("resource implementation Python executable has drifted")
    if str(identity.get("python_version") or "") != platform.python_version():
        raise ValueError("resource implementation Python version has drifted")
    if str(identity.get("dependency_lock_sha256") or "") != _dependency_lock_sha256():
        raise ValueError("resource implementation dependency environment has drifted")
    for record in _iter_file_records(identity.get("source_files") or []):
        _verify_file_record(record, campaign_dir=None, label="implementation source")
    for key in ("environment_current", "environment_generation"):
        value = identity.get(key)
        if value is not None:
            if not isinstance(value, Mapping):
                raise ValueError("resource implementation " + key + " record is invalid")
            _verify_file_record(value, campaign_dir=None, label="implementation " + key)
    backend_executable = identity.get("backend_executable")
    if not isinstance(backend_executable, Mapping):
        raise ValueError("resource backend executable identity is invalid")
    for record in _iter_file_records(backend_executable):
        _verify_file_record(record, campaign_dir=None, label="backend executable")

    recorded_digest = str(identity.get("ichor_package_tree_sha256") or "")
    current_digest = _ichor_package_tree_sha256()
    identity_kind = identity.get("ichor_package_tree_identity_kind")
    bound_generation_present = identity.get("environment_generation") is not None
    legacy_direct_match = False
    if identity_kind is not None:
        if identity_kind != ICHOR_PACKAGE_TREE_IDENTITY_KIND:
            raise ValueError(
                "resource implementation ICHOR package-tree identity kind is unsupported"
            )
        if recorded_digest != current_digest:
            raise ValueError("resource implementation ICHOR package tree has drifted")
    else:
        try:
            legacy_direct_match = (
                recorded_digest == _legacy_ichor_package_tree_sha256()
            )
        except ValueError:
            legacy_direct_match = False
        if not legacy_direct_match and not bound_generation_present:
            raise ValueError("resource implementation ICHOR package tree has drifted")

    if bound_generation_present:
        if campaign_dir is None or expected_campaign_uid is None:
            if identity_kind is None and not legacy_direct_match:
                raise ValueError(
                    "resource implementation ICHOR package tree has drifted"
                )
        else:
            _verify_bound_environment_generation(
                identity,
                campaign_dir=Path(campaign_dir).resolve(),
                expected_campaign_uid=str(expected_campaign_uid),
                current_package_digest=current_digest,
            )
    elif recorded_digest != current_digest and identity_kind is not None:
        raise ValueError("resource implementation ICHOR package tree has drifted")


def resolution_path(
    campaign_dir: Union[str, Path],
    phase_name: str,
    iteration: int,
    submission_identity: str,
) -> Path:
    path = (
        Path(campaign_dir)
        / ".DATA"
        / "ACTIVE_LEARNING"
        / "resource_resolutions"
        / safe_component(phase_name, "phase")
        / ("iteration-" + str(int(iteration)).zfill(6))
        / (safe_component(submission_identity, "submission identity") + ".json")
    )
    return campaign_owned_path(campaign_dir, path)


def resolution_payload(
    *,
    campaign_uid: str,
    phase_name: str,
    iteration: int,
    attempt_id: str,
    submission_identity: str,
    resolved: Any,
    evidence: Mapping[str, Any],
    scratch_path_template: str,
    implementation_identity: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    resources = (
        resolved.to_dict()
        if hasattr(resolved, "to_dict")
        else dict(resolved)
    )
    return {
        "schema_version": RESOURCE_RESOLUTION_SCHEMA_VERSION,
        "formula_version": RESOURCE_FORMULA_VERSION,
        "created_at_iso": datetime.now(timezone.utc).isoformat(),
        "campaign_uid": str(campaign_uid),
        "phase": str(phase_name),
        "iteration": int(iteration),
        "attempt_id": str(attempt_id),
        "submission_identity": str(submission_identity),
        "resources": resources,
        "evidence": dict(evidence),
        "implementation_identity": dict(implementation_identity or {}),
        "scratch_path_template": str(scratch_path_template),
    }


def write_resolution(
    campaign_dir: Union[str, Path],
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    proposed = dict(payload)
    implementation_identity = proposed.get("implementation_identity")
    if not isinstance(implementation_identity, dict) or not implementation_identity:
        resources = proposed.get("resources")
        backend = resources.get("backend") if isinstance(resources, dict) else None
        if not isinstance(backend, str) or not backend:
            raise ValueError("resource resolution has no backend implementation identity")
        proposed["implementation_identity"] = capture_implementation_identity(
            campaign_dir,
            backend=backend,
            require_environment_generation=False,
        )
    path = resolution_path(
        campaign_dir,
        str(proposed["phase"]),
        int(proposed["iteration"]),
        str(proposed["submission_identity"]),
    )
    if path.exists() or path.is_symlink():
        existing = read_resolution(path)
        existing_comparable = dict(existing)
        proposed_comparable = dict(proposed)
        existing_comparable.pop("created_at_iso", None)
        proposed_comparable.pop("created_at_iso", None)
        existing_identity = existing_comparable.get("implementation_identity")
        proposed_identity = proposed_comparable.get("implementation_identity")
        if isinstance(existing_identity, dict):
            existing_identity = dict(existing_identity)
            existing_identity.pop("captured_at_iso", None)
            existing_comparable["implementation_identity"] = existing_identity
        if isinstance(proposed_identity, dict):
            proposed_identity = dict(proposed_identity)
            proposed_identity.pop("captured_at_iso", None)
            proposed_comparable["implementation_identity"] = proposed_identity
        if existing_comparable != proposed_comparable:
            existing_without_identity = dict(existing_comparable)
            proposed_without_identity = dict(proposed_comparable)
            existing_without_identity.pop("implementation_identity", None)
            proposed_without_identity.pop("implementation_identity", None)
            legacy_identity = existing.get("implementation_identity")
            legacy_replay = bool(
                existing_without_identity == proposed_without_identity
                and isinstance(legacy_identity, Mapping)
                and legacy_identity.get("ichor_package_tree_identity_kind")
                is None
            )
            if not legacy_replay:
                raise ValueError(
                    "resource resolution already exists with different content"
                )
            verify_implementation_identity(
                legacy_identity,
                campaign_dir=Path(campaign_dir).resolve(),
                expected_campaign_uid=str(existing["campaign_uid"]),
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, proposed)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "formula_version": str(proposed["formula_version"]),
    }


def read_resolution(path: Union[str, Path]) -> Dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("resource resolution is not a regular file: " + str(source))
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("resource resolution is unreadable: " + str(source)) from exc
    if not isinstance(payload, dict):
        raise ValueError("resource resolution must be a JSON object")
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("resource-resolution schema_version is malformed")
    if schema_version != RESOURCE_RESOLUTION_SCHEMA_VERSION:
        raise ValueError("unsupported resource-resolution schema")
    required_text = (
        "formula_version",
        "campaign_uid",
        "phase",
        "attempt_id",
        "submission_identity",
        "scratch_path_template",
    )
    if any(
        not isinstance(payload.get(key), str) or not str(payload.get(key))
        for key in required_text
    ):
        raise ValueError("resource resolution ownership fields are incomplete")
    iteration = payload.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int):
        raise ValueError("resource-resolution iteration is malformed")
    if iteration < 0:
        raise ValueError("resource-resolution iteration must be >= 0")
    if not isinstance(payload.get("resources"), dict):
        raise ValueError("resource resolution resources must be an object")
    if not isinstance(payload.get("evidence"), dict):
        raise ValueError("resource resolution evidence must be an object")
    if not isinstance(payload.get("implementation_identity"), dict):
        raise ValueError("resource resolution implementation_identity must be an object")
    if str(payload.get("formula_version")) != RESOURCE_FORMULA_VERSION:
        raise ValueError("resource resolution formula version is unsupported")
    return payload


def verify_resolution(
    path: Union[str, Path],
    expected_sha256: str,
    *,
    campaign_dir: Optional[Union[str, Path]] = None,
    verify_bound_inputs: bool = True,
) -> Dict[str, Any]:
    source = Path(path)
    observed = sha256_file(source)
    if observed != str(expected_sha256):
        raise ValueError(
            "resource resolution SHA-256 mismatch: expected "
            + str(expected_sha256)
            + " got "
            + observed
        )
    payload = read_resolution(source)
    if verify_bound_inputs:
        if campaign_dir is not None:
            campaign = Path(campaign_dir)
        else:
            campaign = None
            resolved_source = source.resolve()
            for parent in resolved_source.parents:
                if parent.name == ".DATA":
                    campaign = parent.parent
                    break
            if campaign is None:
                raise ValueError(
                    "cannot infer campaign root from resource-resolution path"
                )
        verify_scientific_evidence(campaign, payload["evidence"])
        verify_implementation_identity(
            payload["implementation_identity"],
            campaign_dir=campaign,
            expected_campaign_uid=str(payload["campaign_uid"]),
        )
    return payload


__all__ = [
    "IMPLEMENTATION_IDENTITY_SCHEMA_VERSION",
    "ICHOR_PACKAGE_TREE_IDENTITY_KIND",
    "RESOURCE_FORMULA_VERSION",
    "RESOURCE_RESOLUTION_SCHEMA_VERSION",
    "capture_implementation_identity",
    "read_resolution",
    "resolution_path",
    "resolution_payload",
    "verify_implementation_identity",
    "verify_resolution",
    "verify_scientific_evidence",
    "write_resolution",
]
