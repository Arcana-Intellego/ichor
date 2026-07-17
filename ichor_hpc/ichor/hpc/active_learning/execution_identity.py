"""Immutable campaign execution mode and environment generation records."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
from .strict_json import strict_json as json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from .config import CampaignConfig
from .daemon.config_lock import config_fingerprint
from .daemon.state import (
    CampaignPhase,
    atomic_write_json,
    read_state,
    write_state,
)
from .daemon.filesystem import campaign_owned_path, operational_path


EXECUTION_IDENTITY_SCHEMA_VERSION = 1
ENVIRONMENT_GENERATION_SCHEMA_VERSION = 1
ENVIRONMENT_CURRENT_SCHEMA_VERSION = 1
VALID_EXECUTION_MODES = frozenset({"live", "dry_run"})

_ENVIRONMENT_FINGERPRINT_KEYS = (
    "python_executable",
    "python_version",
    "ichor_git",
    "ichor_package_tree_sha256",
    "dependencies",
    "pyferebus",
    "ariadne",
    "ferebus_executable",
    "machine_profile",
    "loaded_modules",
    "native_library_paths",
    "campaign_schema_version",
)
_ENVIRONMENT_GENERATION_KEYS = frozenset({
    "schema_version",
    "generation",
    "campaign_uid",
    "created_at_iso",
    "host",
    "operator",
    "python_executable",
    "python_version",
    "ichor_git",
    "ichor_package_tree_sha256",
    "dependencies",
    "pyferebus",
    "ariadne",
    "ferebus_executable",
    "machine_profile",
    "loaded_modules",
    "native_library_paths",
    "campaign_schema_version",
    "campaign_config_sha256",
    "config_lock_sha256",
    "campaign_dir",
    "environment_fingerprint_sha256",
    "digest_sha256",
})
_EXECUTION_IDENTITY_KEYS = frozenset({
    "schema_version",
    "campaign_uid",
    "mode",
    "campaign_random_seed",
    "campaign_schema_version",
    "initial_config_sha256",
    "environment_generation",
    "environment_generation_digest_sha256",
    "created_at_iso",
    "digest_sha256",
})


class ExecutionIdentityError(ValueError):
    """Raised when launch mode or environment identity is inconsistent."""


def execution_identity_path(campaign_dir: Union[str, Path]) -> Path:
    return operational_path(campaign_dir, "execution_identity.json")


def environment_generations_dir(campaign_dir: Union[str, Path]) -> Path:
    return operational_path(campaign_dir, "environment_generations")


def environment_current_path(campaign_dir: Union[str, Path]) -> Path:
    return operational_path(campaign_dir, "environment_current.json")


def _canonical_digest(payload: Dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("digest_sha256", None)
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _environment_fingerprint_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only execution-affecting fields from a generation record.

    Campaign configuration is governed by the config-lock contract.  Host,
    operator, timestamps, generation numbers and paths are provenance rather
    than execution identity, so they must not make a login-node restart look
    like software drift.
    """
    return {key: payload.get(key) for key in _ENVIRONMENT_FINGERPRINT_KEYS}


def _environment_fingerprint(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        _environment_fingerprint_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExecutionIdentityError(label + " must be a lowercase SHA-256 digest")
    return value


def _exact_non_negative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ExecutionIdentityError(label + " must be a non-negative integer")
    return int(value)


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    observed = set(payload)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ExecutionIdentityError(label + " fields are invalid: " + "; ".join(details))


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExecutionIdentityError(label + " must be a non-empty string")
    return value


def _validate_timestamp(value: Any, label: str) -> str:
    text = _non_empty_string(value, label)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ExecutionIdentityError(label + " must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ExecutionIdentityError(label + " must include a timezone")
    return text


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file() or path.is_symlink():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(roots: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    resolved_roots = sorted(
        (path.resolve() for path in roots),
        key=lambda item: str(item),
    )
    for root_index, root in enumerate(resolved_roots):
        if not root.is_dir():
            continue
        root_identity = (
            str(root_index)
            + ":"
            + root.parent.name
            + "/"
            + root.name
        )
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root).as_posix()
            digest.update(root_identity.encode("utf-8"))
            digest.update(b"\0")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _git_identity(repo_root: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"commit": None, "tracked_tree_clean": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        result = {
            "commit": commit,
            "tracked_tree_clean": not bool(status.strip()),
        }
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _distribution_version(name: str) -> Optional[str]:
    candidates = [name, name.replace("_", "-"), name.replace("-", "_")]
    for candidate in candidates:
        try:
            return str(importlib.metadata.version(candidate))
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _module_identity(name: str) -> Dict[str, Any]:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        spec = None
    path = None if spec is None else spec.origin
    file_path = None if path in (None, "built-in") else Path(str(path)).resolve()
    package_roots = []
    if spec is not None and spec.submodule_search_locations:
        package_roots = [Path(value).resolve() for value in spec.submodule_search_locations]
    return {
        "module": name,
        "path": None if file_path is None else str(file_path),
        "sha256": None if file_path is None else _sha256_file(file_path),
        "package_tree_sha256": (
            None if not package_roots else _tree_hash(package_roots)
        ),
        "version": _distribution_version(name),
    }


def _json_safe_probe(value: Any) -> Any:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)
    return value


def _ariadne_identity() -> Dict[str, Any]:
    identity = _module_identity("ariadne")
    identity["import_ok"] = False
    identity["abi_probe"] = None
    identity["abi_probe_error"] = None
    try:
        module = importlib.import_module("ariadne")
        identity["import_ok"] = True
        from .acquisition.ariadne_abi import probe_ariadne_module

        identity["abi_probe"] = _json_safe_probe(probe_ariadne_module(module))
    except Exception as exc:
        identity["abi_probe_error"] = type(exc).__name__ + ": " + str(exc)
    return identity


def _active_profile_identity() -> Dict[str, Any]:
    machine = os.environ.get("ICHOR_MACHINE")
    profile = None
    try:
        global_variables = importlib.import_module("ichor.hpc.global_variables")
        machine = getattr(global_variables, "MACHINE", None) or machine
        config = getattr(global_variables, "ICHOR_CONFIG", None)
        if isinstance(config, dict) and machine in config:
            profile = config[machine]
    except Exception:
        profile = None
    digest = None
    if isinstance(profile, dict):
        digest = hashlib.sha256(
            json.dumps(
                profile,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    software = profile.get("software", {}) if isinstance(profile, dict) else {}
    python_modules = (
        software.get("python", {}).get("modules", [])
        if isinstance(software.get("python", {}), dict)
        else []
    )
    ariadne_modules = (
        software.get("ariadne_runtime", {}).get("modules", [])
        if isinstance(software.get("ariadne_runtime", {}), dict)
        else []
    )
    return {
        "name": machine,
        "digest_sha256": digest,
        "module_sequence": {
            "purge_first": True,
            "python_modules": list(python_modules or []),
            "ariadne_runtime_modules": list(ariadne_modules or []),
        },
    }


def _installed_dependencies() -> list[Dict[str, str]]:
    rows = []
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name") or "").strip()
        if name:
            rows.append({"name": name, "version": str(distribution.version)})
    return sorted(rows, key=lambda row: (row["name"].lower(), row["version"]))


def _profile_ferebus_executable() -> Optional[str]:
    try:
        from .daemon.cluster_profile import expanded_profile_value

        value = expanded_profile_value(
            "software",
            "ferebus",
            "executable_path",
            default=None,
        )
    except Exception:
        value = None
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _configured_ferebus_identity() -> Dict[str, Any]:
    configured = os.environ.get("FEREBUS_PATH") or _profile_ferebus_executable()
    resolved = Path(configured).expanduser() if configured else None
    if resolved is None:
        located = shutil.which("ferebus")
        resolved = Path(located) if located else None
    return {
        "path": None if resolved is None else str(resolved.resolve()),
        "sha256": None if resolved is None else _sha256_file(resolved.resolve()),
    }


def capture_environment_generation(
    campaign_dir: Union[str, Path],
    *,
    campaign_uid: str,
    config: CampaignConfig,
    generation: int = 0,
) -> Dict[str, Any]:
    """Capture the exact Python/package/native identity for one generation."""
    campaign = Path(campaign_dir).resolve()
    repo_root = Path(__file__).resolve().parents[4]
    package_roots = [
        repo_root / "ichor_core" / "ichor",
        repo_root / "ichor_hpc" / "ichor",
        repo_root / "ichor_cli" / "ichor",
    ]
    payload: Dict[str, Any] = {
        "schema_version": ENVIRONMENT_GENERATION_SCHEMA_VERSION,
        "generation": int(generation),
        "campaign_uid": str(campaign_uid),
        "created_at_iso": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "operator": os.environ.get("USER") or os.environ.get("USERNAME") or "",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "ichor_git": _git_identity(repo_root),
        "ichor_package_tree_sha256": _tree_hash(package_roots),
        "dependencies": _installed_dependencies(),
        "pyferebus": _module_identity("pyferebus"),
        "ariadne": _ariadne_identity(),
        "ferebus_executable": _configured_ferebus_identity(),
        "machine_profile": _active_profile_identity(),
        "loaded_modules": [
            value for value in os.environ.get("LOADEDMODULES", "").split(":") if value
        ],
        "native_library_paths": {
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            "LIBRARY_PATH": os.environ.get("LIBRARY_PATH", ""),
        },
        "campaign_schema_version": int(config.schema_version),
        "campaign_config_sha256": config_fingerprint(config.to_dict()),
        "config_lock_sha256": _sha256_file(
            campaign / ".DATA" / "ACTIVE_LEARNING" / "config_lock.json"
        ),
        "campaign_dir": str(campaign),
    }
    payload["environment_fingerprint_sha256"] = _environment_fingerprint(payload)
    payload["digest_sha256"] = _canonical_digest(payload)
    return payload


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExecutionIdentityError(label + " is not a regular file: " + str(path))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExecutionIdentityError(label + " is unreadable: " + str(path)) from exc
    if not isinstance(value, dict):
        raise ExecutionIdentityError(label + " must contain a JSON object")
    return value


def _validate_generation_payload(
    payload: Dict[str, Any],
    *,
    expected_generation: int,
    expected_campaign_uid: str,
    path: Path,
) -> Dict[str, Any]:
    _require_exact_keys(
        payload,
        _ENVIRONMENT_GENERATION_KEYS,
        "environment generation",
    )
    if payload.get("schema_version") != ENVIRONMENT_GENERATION_SCHEMA_VERSION:
        raise ExecutionIdentityError(
            "unsupported environment generation schema: " + str(path)
        )
    generation = _exact_non_negative_int(
        payload.get("generation"), "environment generation"
    )
    if generation != int(expected_generation):
        raise ExecutionIdentityError("environment generation number mismatch")
    campaign_uid = _non_empty_string(
        payload.get("campaign_uid"),
        "environment generation campaign UID",
    )
    if campaign_uid != str(expected_campaign_uid):
        raise ExecutionIdentityError("environment generation campaign UID mismatch")
    _validate_timestamp(
        payload.get("created_at_iso"),
        "environment generation creation time",
    )
    for key in ("host", "operator"):
        if not isinstance(payload.get(key), str):
            raise ExecutionIdentityError(
                "environment generation " + key + " must be a string"
            )
    python_executable = Path(
        _non_empty_string(
            payload.get("python_executable"),
            "environment generation Python executable",
        )
    )
    if not python_executable.is_absolute():
        raise ExecutionIdentityError(
            "environment generation Python executable must be absolute"
        )
    _non_empty_string(
        payload.get("python_version"),
        "environment generation Python version",
    )
    _sha256_digest(
        payload.get("ichor_package_tree_sha256"),
        "ICHOR package-tree digest",
    )
    for key in (
        "ichor_git",
        "pyferebus",
        "ariadne",
        "ferebus_executable",
        "machine_profile",
    ):
        if not isinstance(payload.get(key), Mapping):
            raise ExecutionIdentityError(
                "environment generation " + key + " must be an object"
            )
    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, list) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("name"), str)
        or not row["name"]
        or not isinstance(row.get("version"), str)
        for row in dependencies
    ):
        raise ExecutionIdentityError(
            "environment generation dependencies must contain name/version objects"
        )
    loaded_modules = payload.get("loaded_modules")
    if not isinstance(loaded_modules, list) or any(
        not isinstance(module, str) or not module for module in loaded_modules
    ):
        raise ExecutionIdentityError(
            "environment generation loaded_modules must be a string list"
        )
    native_paths = payload.get("native_library_paths")
    if not isinstance(native_paths, Mapping) or set(native_paths) != {
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
    } or any(not isinstance(value, str) for value in native_paths.values()):
        raise ExecutionIdentityError(
            "environment generation native-library paths are invalid"
        )
    if _exact_non_negative_int(
        payload.get("campaign_schema_version"),
        "environment generation campaign schema",
    ) < 1:
        raise ExecutionIdentityError(
            "environment generation campaign schema must be >= 1"
        )
    _sha256_digest(
        payload.get("campaign_config_sha256"),
        "environment generation campaign-config digest",
    )
    config_lock_digest = payload.get("config_lock_sha256")
    if config_lock_digest is not None:
        _sha256_digest(config_lock_digest, "environment generation config-lock digest")
    campaign_dir = Path(
        _non_empty_string(
            payload.get("campaign_dir"),
            "environment generation campaign directory",
        )
    )
    if not campaign_dir.is_absolute():
        raise ExecutionIdentityError(
            "environment generation campaign directory must be absolute"
        )
    recorded_fingerprint = _sha256_digest(
        payload.get("environment_fingerprint_sha256"),
        "environment fingerprint",
    )
    if recorded_fingerprint != _environment_fingerprint(payload):
        raise ExecutionIdentityError("environment generation fingerprint mismatch")
    recorded_digest = _sha256_digest(
        payload.get("digest_sha256"), "environment generation digest"
    )
    if recorded_digest != _canonical_digest(payload):
        raise ExecutionIdentityError("environment generation digest mismatch")
    return payload


def read_active_environment_generation(
    campaign_dir: Union[str, Path],
    *,
    expected_campaign_uid: str,
) -> Dict[str, Any]:
    """Read and fully validate the active generation and its pointer."""
    campaign = Path(campaign_dir).resolve()
    current_path = environment_current_path(campaign)
    current = _read_json_object(current_path, "active environment pointer")
    _require_exact_keys(
        current,
        frozenset({
            "schema_version",
            "generation",
            "generation_path",
            "generation_digest_sha256",
        }),
        "active environment pointer",
    )
    if current.get("schema_version") != ENVIRONMENT_CURRENT_SCHEMA_VERSION:
        raise ExecutionIdentityError("unsupported active environment pointer schema")
    generation = _exact_non_negative_int(
        current.get("generation"), "active environment generation"
    )
    expected_relative = (
        Path(".DATA")
        / "ACTIVE_LEARNING"
        / "environment_generations"
        / ("generation-" + str(generation).zfill(6) + ".json")
    )
    recorded_relative = current.get("generation_path")
    if not isinstance(recorded_relative, str) or not recorded_relative:
        raise ExecutionIdentityError("active environment generation path is missing")
    if Path(recorded_relative) != expected_relative:
        raise ExecutionIdentityError("active environment generation path is not canonical")
    generation_path = campaign_owned_path(campaign, campaign / recorded_relative)
    payload = _validate_generation_payload(
        _read_json_object(generation_path, "environment generation"),
        expected_generation=generation,
        expected_campaign_uid=str(expected_campaign_uid),
        path=generation_path,
    )
    pointer_digest = _sha256_digest(
        current.get("generation_digest_sha256"),
        "active environment generation digest",
    )
    if pointer_digest != payload["digest_sha256"]:
        raise ExecutionIdentityError("active environment pointer digest mismatch")
    return {
        "pointer": current,
        "generation": payload,
        "generation_path": generation_path,
    }


def environment_status(
    campaign_dir: Union[str, Path],
    *,
    campaign_uid: str,
    config: CampaignConfig,
) -> Dict[str, Any]:
    """Compare the current process and native backends with the active generation."""
    active = read_active_environment_generation(
        campaign_dir,
        expected_campaign_uid=str(campaign_uid),
    )
    generation = active["generation"]
    observed = capture_environment_generation(
        campaign_dir,
        campaign_uid=str(campaign_uid),
        config=config,
        generation=int(generation["generation"]),
    )
    expected_fields = _environment_fingerprint_payload(generation)
    observed_fields = _environment_fingerprint_payload(observed)
    changed_fields: List[str] = [
        key
        for key in _ENVIRONMENT_FINGERPRINT_KEYS
        if expected_fields.get(key) != observed_fields.get(key)
    ]
    matches = (
        generation["environment_fingerprint_sha256"]
        == observed["environment_fingerprint_sha256"]
    )
    return {
        "schema_version": 1,
        "campaign_uid": str(campaign_uid),
        "generation": int(generation["generation"]),
        "generation_digest_sha256": str(generation["digest_sha256"]),
        "expected_environment_fingerprint_sha256": str(
            generation["environment_fingerprint_sha256"]
        ),
        "observed_environment_fingerprint_sha256": str(
            observed["environment_fingerprint_sha256"]
        ),
        "matches": bool(matches),
        "changed_fields": changed_fields,
        "active_generation_path": str(active["generation_path"]),
    }


def assert_environment_unchanged(
    campaign_dir: Union[str, Path],
    *,
    campaign_uid: str,
    config: CampaignConfig,
) -> Dict[str, Any]:
    status = environment_status(
        campaign_dir,
        campaign_uid=str(campaign_uid),
        config=config,
    )
    if not bool(status["matches"]):
        fields = ", ".join(status["changed_fields"]) or "unknown fields"
        raise ExecutionIdentityError(
            "active execution environment has drifted: " + fields
        )
    return status


def read_execution_identity(
    campaign_dir: Union[str, Path],
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    path = execution_identity_path(campaign_dir)
    payload = _read_json_object(path, "execution identity")
    _require_exact_keys(payload, _EXECUTION_IDENTITY_KEYS, "execution identity")
    if payload.get("schema_version") != EXECUTION_IDENTITY_SCHEMA_VERSION:
        raise ExecutionIdentityError("unsupported execution identity schema")
    if str(payload.get("digest_sha256") or "") != _canonical_digest(payload):
        raise ExecutionIdentityError("execution identity digest mismatch")
    campaign_uid = _non_empty_string(
        payload.get("campaign_uid"),
        "execution identity campaign UID",
    )
    if expected_campaign_uid is not None and campaign_uid != str(expected_campaign_uid):
        raise ExecutionIdentityError("execution identity campaign UID mismatch")
    if str(payload.get("mode") or "") not in VALID_EXECUTION_MODES:
        raise ExecutionIdentityError("execution identity mode is invalid")
    _exact_non_negative_int(
        payload.get("campaign_random_seed"),
        "execution identity random seed",
    )
    if _exact_non_negative_int(
        payload.get("campaign_schema_version"),
        "execution identity campaign schema",
    ) < 1:
        raise ExecutionIdentityError("execution identity campaign schema must be >= 1")
    _sha256_digest(
        payload.get("initial_config_sha256"),
        "execution identity initial-config digest",
    )
    if _exact_non_negative_int(
        payload.get("environment_generation"),
        "execution identity initial environment generation",
    ) != 0:
        raise ExecutionIdentityError(
            "execution identity initial environment generation must be zero"
        )
    initial_generation_digest = _sha256_digest(
        payload.get("environment_generation_digest_sha256"),
        "execution identity initial environment digest",
    )
    _validate_timestamp(
        payload.get("created_at_iso"),
        "execution identity creation time",
    )
    initial_generation_path = environment_generations_dir(campaign_dir) / (
        "generation-000000.json"
    )
    initial_generation = _validate_generation_payload(
        _read_json_object(initial_generation_path, "initial environment generation"),
        expected_generation=0,
        expected_campaign_uid=campaign_uid,
        path=initial_generation_path,
    )
    if initial_generation_digest != initial_generation["digest_sha256"]:
        raise ExecutionIdentityError(
            "execution identity initial environment digest mismatch"
        )
    return payload


def ensure_execution_identity(
    campaign_dir: Union[str, Path],
    *,
    campaign_uid: str,
    config: CampaignConfig,
    requested_mode: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    """Create the first identity or require an exact mode match thereafter."""
    campaign = Path(campaign_dir).resolve()
    path = execution_identity_path(campaign)
    if path.is_file() or path.is_symlink():
        payload = read_execution_identity(
            campaign,
            expected_campaign_uid=str(campaign_uid),
        )
        stored_mode = str(payload.get("mode") or "")
        if payload.get("campaign_random_seed") != int(
            config.campaign.reproducibility_seed
        ):
            raise ExecutionIdentityError(
                "campaign.reproducibility_seed differs from the immutable "
                "execution identity"
            )
        if payload.get("campaign_schema_version") != int(config.schema_version):
            raise ExecutionIdentityError(
                "campaign schema differs from the immutable execution identity"
            )
        if requested_mode is not None and requested_mode != stored_mode:
            raise ExecutionIdentityError(
                "campaign execution mode is permanently bound to "
                + stored_mode
                + "; requested "
                + requested_mode
            )
        assert_environment_unchanged(
            campaign,
            campaign_uid=str(campaign_uid),
            config=config,
        )
        return stored_mode, payload

    if requested_mode not in VALID_EXECUTION_MODES:
        raise ExecutionIdentityError(
            "the first start requires --mode live or --mode dry_run"
        )
    generations = environment_generations_dir(campaign)
    generations.mkdir(parents=True, exist_ok=True)
    generation = capture_environment_generation(
        campaign,
        campaign_uid=str(campaign_uid),
        config=config,
        generation=0,
    )
    generation_path = generations / "generation-000000.json"
    atomic_write_json(generation_path, generation)
    current = {
        "schema_version": ENVIRONMENT_CURRENT_SCHEMA_VERSION,
        "generation": 0,
        "generation_path": str(generation_path.relative_to(campaign)),
        "generation_digest_sha256": str(generation["digest_sha256"]),
    }
    atomic_write_json(environment_current_path(campaign), current)
    payload = {
        "schema_version": EXECUTION_IDENTITY_SCHEMA_VERSION,
        "campaign_uid": str(campaign_uid),
        "mode": str(requested_mode),
        "campaign_random_seed": int(config.campaign.reproducibility_seed),
        "campaign_schema_version": int(config.schema_version),
        "initial_config_sha256": config_fingerprint(config.to_dict()),
        "environment_generation": 0,
        "environment_generation_digest_sha256": str(generation["digest_sha256"]),
        "created_at_iso": datetime.now(timezone.utc).isoformat(),
    }
    payload["digest_sha256"] = _canonical_digest(payload)
    atomic_write_json(path, payload)
    return str(requested_mode), payload


def rebind_environment(
    campaign_dir: Union[str, Path],
    *,
    config: CampaignConfig,
    live_preflight_ok: bool = False,
    scheduler_ownership_clear: bool = False,
) -> Dict[str, Any]:
    """Bind an idle campaign to the current verified execution environment.

    Generation publication is ordered so interruption remains fail-closed:
    write the immutable generation, clear derived reference scales in state,
    then advance the current pointer.  Repeating the command after any partial
    attempt is safe.
    """
    campaign = Path(campaign_dir).resolve()
    state = read_state(operational_path(campaign, "state.json"))
    identity = read_execution_identity(
        campaign,
        expected_campaign_uid=str(state.campaign_uid),
    )
    if identity["mode"] == "live" and not bool(live_preflight_ok):
        raise ExecutionIdentityError(
            "live environment rebind requires a successful backend preflight"
        )
    if state.phase not in {CampaignPhase.SEED_SELECT, CampaignPhase.DONE}:
        raise ExecutionIdentityError(
            "environment rebind requires an idle SEED_SELECT or DONE boundary"
        )
    if any(value is not None for value in state.pending_jobs.values()):
        raise ExecutionIdentityError(
            "environment rebind is blocked by pending scheduler ownership"
        )
    if not bool(scheduler_ownership_clear):
        raise ExecutionIdentityError(
            "environment rebind requires a conclusive scheduler ownership check"
        )

    from .daemon.artifact_contracts import verify_state_referenced_artifacts
    from .daemon.config_lock import review_config_changes
    from .daemon.submission_intent import ACTIVE_STATUSES, inventory_intents

    review = review_config_changes(
        campaign,
        config,
        state,
        initialise_missing=False,
    )
    if review.changed:
        raise ExecutionIdentityError(
            "campaign configuration differs from its lock; reconcile it before rebind"
        )
    inventory = inventory_intents(
        campaign,
        expected_campaign_uid=str(state.campaign_uid),
    )
    if inventory["errors"]:
        raise ExecutionIdentityError(
            "environment rebind is blocked by malformed submission intents"
        )
    active_intents = [
        record
        for record in inventory["records"]
        if str(record.get("status")) in ACTIVE_STATUSES
    ]
    if active_intents:
        raise ExecutionIdentityError(
            "environment rebind is blocked by active submission intents"
        )
    verify_state_referenced_artifacts(campaign, state, strict_models=True)

    active = read_active_environment_generation(
        campaign,
        expected_campaign_uid=str(state.campaign_uid),
    )
    active_generation = active["generation"]
    candidate = capture_environment_generation(
        campaign,
        campaign_uid=str(state.campaign_uid),
        config=config,
        generation=int(active_generation["generation"]) + 1,
    )
    if (
        candidate["environment_fingerprint_sha256"]
        == active_generation["environment_fingerprint_sha256"]
    ):
        return {
            "schema_version": 1,
            "changed": False,
            "generation": int(active_generation["generation"]),
            "generation_digest_sha256": str(active_generation["digest_sha256"]),
            "message": "active environment already matches the current process",
        }

    generation_number = int(candidate["generation"])
    generations_root = environment_generations_dir(campaign)
    generations_root.mkdir(parents=True, exist_ok=True)
    while True:
        generation_path = generations_root / (
            "generation-" + str(generation_number).zfill(6) + ".json"
        )
        if not generation_path.exists() and not generation_path.is_symlink():
            atomic_write_json(generation_path, candidate)
            break
        existing = _validate_generation_payload(
            _read_json_object(generation_path, "environment generation"),
            expected_generation=generation_number,
            expected_campaign_uid=str(state.campaign_uid),
            path=generation_path,
        )
        if (
            existing["environment_fingerprint_sha256"]
            == candidate["environment_fingerprint_sha256"]
        ):
            candidate = existing
            break
        generation_number += 1
        candidate = capture_environment_generation(
            campaign,
            campaign_uid=str(state.campaign_uid),
            config=config,
            generation=generation_number,
        )

    state.reference_scales = None
    state.reference_scales_iteration = -1
    state.reference_scales_models_version = -1
    state.reference_scales_model_manifest_sha256 = None
    from .daemon.ferebus_row_cache import (
        clear_row_caches,
        ensure_cumulative_row_caches,
    )

    clear_row_caches(campaign)
    if int(state.reference_data_version) >= 0:
        from .versioning.reference_data import ReferenceDataVersioning

        reference_view = ReferenceDataVersioning(
            campaign / "QM_REFERENCE_DATA"
        ).resolve(int(state.reference_data_version), verification="metadata")
        ensure_cumulative_row_caches(campaign, reference_view)
    write_state(operational_path(campaign, "state.json"), state)

    current = {
        "schema_version": ENVIRONMENT_CURRENT_SCHEMA_VERSION,
        "generation": generation_number,
        "generation_path": str(generation_path.relative_to(campaign)),
        "generation_digest_sha256": str(candidate["digest_sha256"]),
    }
    atomic_write_json(environment_current_path(campaign), current)
    try:
        from .daemon.journal import append_event

        append_event(
            operational_path(campaign, "journal.ndjson"),
            "environment_rebound",
            max_bytes=int(config.runtime.journal_max_bytes),
            retained_files=int(config.runtime.journal_retained_files),
            lock_timeout_seconds=int(config.runtime.ledger_lock_timeout_seconds),
            previous_generation=int(active_generation["generation"]),
            generation=generation_number,
            previous_generation_digest_sha256=str(active_generation["digest_sha256"]),
            generation_digest_sha256=str(candidate["digest_sha256"]),
            changed_fields=[
                key
                for key in _ENVIRONMENT_FINGERPRINT_KEYS
                if active_generation.get(key) != candidate.get(key)
            ],
        )
    except Exception:
        pass
    return {
        "schema_version": 1,
        "changed": True,
        "previous_generation": int(active_generation["generation"]),
        "generation": generation_number,
        "generation_digest_sha256": str(candidate["digest_sha256"]),
        "generation_path": str(generation_path),
    }


__all__ = [
    "ENVIRONMENT_GENERATION_SCHEMA_VERSION",
    "EXECUTION_IDENTITY_SCHEMA_VERSION",
    "ExecutionIdentityError",
    "VALID_EXECUTION_MODES",
    "assert_environment_unchanged",
    "capture_environment_generation",
    "environment_status",
    "ensure_execution_identity",
    "environment_current_path",
    "environment_generations_dir",
    "execution_identity_path",
    "read_active_environment_generation",
    "read_execution_identity",
    "rebind_environment",
]
