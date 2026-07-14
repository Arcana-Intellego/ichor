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
from typing import Any, Dict, Iterable, Optional, Tuple, Union

from .config import CampaignConfig
from .daemon.config_lock import config_fingerprint
from .daemon.state import atomic_write_json
from .daemon.filesystem import operational_path


EXECUTION_IDENTITY_SCHEMA_VERSION = 1
ENVIRONMENT_GENERATION_SCHEMA_VERSION = 1
ENVIRONMENT_CURRENT_SCHEMA_VERSION = 1
VALID_EXECUTION_MODES = frozenset({"live", "dry_run"})


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
    for root in sorted((path.resolve() for path in roots), key=lambda item: str(item)):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root).as_posix()
            digest.update(str(root.name).encode("utf-8"))
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
        probe = getattr(module, "get_abi_info_py", None)
        if callable(probe):
            identity["abi_probe"] = _json_safe_probe(probe())
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


def _configured_ferebus_identity() -> Dict[str, Any]:
    configured = os.environ.get("FEREBUS_PATH")
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
        payload = _read_json_object(path, "execution identity")
        if payload.get("schema_version") != EXECUTION_IDENTITY_SCHEMA_VERSION:
            raise ExecutionIdentityError("unsupported execution identity schema")
        if str(payload.get("campaign_uid") or "") != str(campaign_uid):
            raise ExecutionIdentityError("execution identity campaign UID mismatch")
        if str(payload.get("digest_sha256") or "") != _canonical_digest(payload):
            raise ExecutionIdentityError("execution identity digest mismatch")
        stored_mode = str(payload.get("mode") or "")
        if stored_mode not in VALID_EXECUTION_MODES:
            raise ExecutionIdentityError("execution identity mode is invalid")
        if payload.get("campaign_random_seed") != int(config.campaign.random_seed):
            raise ExecutionIdentityError(
                "campaign.random_seed differs from the immutable execution identity"
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
        "campaign_random_seed": int(config.campaign.random_seed),
        "campaign_schema_version": int(config.schema_version),
        "initial_config_sha256": config_fingerprint(config.to_dict()),
        "environment_generation": 0,
        "environment_generation_digest_sha256": str(generation["digest_sha256"]),
        "created_at_iso": datetime.now(timezone.utc).isoformat(),
    }
    payload["digest_sha256"] = _canonical_digest(payload)
    atomic_write_json(path, payload)
    return str(requested_mode), payload


__all__ = [
    "ENVIRONMENT_GENERATION_SCHEMA_VERSION",
    "EXECUTION_IDENTITY_SCHEMA_VERSION",
    "ExecutionIdentityError",
    "VALID_EXECUTION_MODES",
    "capture_environment_generation",
    "ensure_execution_identity",
    "environment_current_path",
    "environment_generations_dir",
    "execution_identity_path",
]
