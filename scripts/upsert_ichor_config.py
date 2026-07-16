"""Atomically install one canonical ICHOR machine profile for a user."""

from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml


def _load_mapping(path: Path, *, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(label + " is missing or symlinked: " + str(path))
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(label + " must contain a YAML mapping")
    return payload


def _fsync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(time.time_ns()))
    _write_exclusive(temporary, payload)
    try:
        os.replace(temporary, path)
        _fsync_parent(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def upsert_profile(
    *,
    destination: Path,
    canonical_config: Path,
    machine: str,
    python_path: str,
    python_library_path: str,
    aimall_path: str,
    ferebus_path: str,
    plumed_kernel: str,
    plumed_library_path: str,
) -> Optional[Path]:
    """Replace one active profile and preserve every unrelated profile."""
    canonical = _load_mapping(canonical_config, label="canonical ICHOR config")
    raw_profile = canonical.get(machine)
    if not isinstance(raw_profile, dict):
        raise ValueError(
            str(canonical_config) + " does not define profile " + repr(machine)
        )

    destination = Path(destination)
    data: Dict[str, Any] = {}
    backup: Optional[Path] = None
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink():
            raise ValueError("refusing symlinked ICHOR config: " + str(destination))
        data = _load_mapping(destination, label="user ICHOR config")
        backup = destination.with_name(
            destination.name + ".bak." + str(time.time_ns())
        )
        _write_exclusive(backup, destination.read_bytes())
        _fsync_parent(backup)

    profile = copy.deepcopy(raw_profile)
    software = profile.setdefault("software", {})
    if not isinstance(software, dict):
        raise ValueError("canonical profile software block must be a mapping")
    python = software.setdefault("python", {})
    if not isinstance(python, dict):
        raise ValueError("canonical profile software.python must be a mapping")
    python["python_path"] = str(python_path)
    if machine == "csf3":
        python["library_path"] = str(python_library_path)
    software.setdefault("aimall", {})["executable_path"] = str(aimall_path)
    software.setdefault("ferebus", {})["executable_path"] = str(ferebus_path)
    plumed = software.setdefault("plumed", {})
    plumed["kernel_path"] = str(plumed_kernel)
    plumed["library_path"] = str(plumed_library_path)

    # Replacement, rather than recursive merge, removes retired active-profile
    # entries while preserving unrelated machine profiles verbatim.
    data[str(machine)] = profile
    payload = yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=False,
    ).encode("utf-8")
    _atomic_write(destination, payload)
    return backup


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--canonical-config", required=True, type=Path)
    parser.add_argument("--machine", required=True, choices=("csf3", "csf4"))
    parser.add_argument("--python-path", required=True)
    parser.add_argument("--python-library-path", required=True)
    parser.add_argument("--aimall-path", required=True)
    parser.add_argument("--ferebus-path", required=True)
    parser.add_argument("--plumed-kernel", required=True)
    parser.add_argument("--plumed-library-path", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    backup = upsert_profile(
        destination=args.destination,
        canonical_config=args.canonical_config,
        machine=args.machine,
        python_path=args.python_path,
        python_library_path=args.python_library_path,
        aimall_path=args.aimall_path,
        ferebus_path=args.ferebus_path,
        plumed_kernel=args.plumed_kernel,
        plumed_library_path=args.plumed_library_path,
    )
    if backup is not None:
        print("Backed up existing config to " + str(backup))
    print("Updated " + str(args.destination) + " profile " + args.machine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
