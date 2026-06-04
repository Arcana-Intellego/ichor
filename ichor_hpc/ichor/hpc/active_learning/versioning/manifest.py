"""SHA256 manifest of a directory tree.

The training-set commit protocol writes a `.manifest.json` next to the
iteration directory that pins the SHA256 of every regular file inside it.
The daemon validates the manifest at startup and before any FEREBUS
submission, so a partially-committed iteration is detected as a hash
mismatch rather than silently consumed.

Symlinks and special files are excluded; only regular files are hashed.
The manifest itself (the `.manifest.json` file) is excluded so the
manifest can sit inside the directory it describes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union


__all__ = [
    "MANIFEST_FILENAME",
    "ManifestMismatchError",
    "compute_directory_manifest",
    "read_manifest",
    "unmanifested_directories",
    "write_manifest",
    "verify_manifest",
    "sha256_file",
]


MANIFEST_FILENAME = ".manifest.json"
_BLOCK_SIZE = 1 << 20   # 1 MiB; fits comfortably in L2 cache while limiting syscalls.


class ManifestMismatchError(RuntimeError):
    """Raised when verify_manifest detects content drift."""


def sha256_file(path: Union[str, Path]) -> str:
    """Streamed SHA-256 of one regular file. Memory use is bounded by
    _BLOCK_SIZE regardless of file size."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_BLOCK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _is_transient(name: str) -> bool:
    """transient junk that can be present mid-commit but is not committed data and may vanish by the
    time verify runs -- chiefly NFS silly-rename files (.nfsXXXX, left when the daemon deletes a file
    while it is still open during a copy), plus editor swaps and tmp/lock leftovers. matched by
    PATTERN, not by exact name, because the .nfs suffix is random. (A55)"""
    n = name.lower()
    return n.startswith(".nfs") or n.endswith((".swp", ".swo", ".swx", ".tmp", ".lock"))


def _iter_files(root: Path, *, exclude: Iterable[str]) -> Iterable[Path]:
    excluded = {str(Path(root) / name) for name in exclude}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        if str(p) in excluded:
            continue
        if p.name == MANIFEST_FILENAME:
            continue
        if _is_transient(p.name):
            # hashing one of these into the manifest would make a later verify_committed report it
            # "missing" (it has since vanished) and trip the strict-verify crash forever, on a file
            # that was never real committed data.
            continue
        yield p


def compute_directory_manifest(
    root: Union[str, Path],
    *,
    exclude: Iterable[str] = (),
) -> Dict[str, str]:
    """Compute {relative_path: sha256_hexdigest} for every regular file under
    `root`. Keys use forward slashes (POSIX) for cross-platform stability.

    `exclude` is an iterable of paths (relative to `root`) to skip. The
    manifest file itself (`.manifest.json`) is always excluded.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError("not a directory: " + str(root_path))
    manifest: Dict[str, str] = {}
    for f in _iter_files(root_path, exclude=exclude):
        rel = f.relative_to(root_path).as_posix()
        manifest[rel] = sha256_file(f)
    return manifest


def read_manifest(root: Union[str, Path]) -> Dict[str, str]:
    """Load the manifest sitting at `root/.manifest.json`."""
    p = Path(root) / MANIFEST_FILENAME
    if not p.exists():
        raise FileNotFoundError("no manifest at " + str(p))
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        # a corrupt manifest used to let a raw JSONDecodeError fly straight out of commit() /
        # verify_committed and take the daemon down with an ugly traceback. wrap it in our own
        # mismatch error (the class verify_manifest already raises) so callers handle it one way. (A4)
        raise ManifestMismatchError(
            "manifest at " + str(p) + " is not valid json: " + str(exc)
        ) from exc
    if not isinstance(data, dict):
        raise ValueError("manifest must be a JSON object at " + str(p))
    return {str(k): str(v) for k, v in data.items()}


def write_manifest(root: Union[str, Path], manifest: Dict[str, str]) -> Path:
    """Write `manifest` atomically to `root/.manifest.json`.

    Uses the atomic-write helper from :mod:`..daemon.state` so a kill mid-write
    leaves either no manifest or a complete one.
    """
    from ..daemon.state import atomic_write_json

    p = Path(root) / MANIFEST_FILENAME
    atomic_write_json(p, manifest)
    return p


def _manifest_covers_directory(manifest: Dict[str, str], rel_dir: str) -> bool:
    prefix = str(Path(rel_dir).as_posix()).rstrip("/") + "/"
    return any(str(key).startswith(prefix) for key in manifest)


def unmanifested_directories(
    root: Union[str, Path],
    *,
    manifest: Optional[Dict[str, str]] = None,
    pattern: str = "*",
) -> List[str]:
    """Return directory paths present on disk but not represented by manifest entries.

    This deliberately does not change :func:`verify_manifest`: generic manifests pin the
    committed files, while daemon-owned training commits sometimes need a stricter check that
    extra POINT_*.pointdir directories have not been manually dropped alongside the commit.
    """
    root_path = Path(root)
    if manifest is None:
        manifest = read_manifest(root_path)
    extras: List[str] = []
    for candidate in sorted(root_path.rglob(pattern)):
        if not candidate.is_dir() or candidate.is_symlink():
            continue
        if _is_transient(candidate.name):
            continue
        rel = candidate.relative_to(root_path).as_posix()
        if not _manifest_covers_directory(manifest, rel):
            extras.append(rel)
    return extras


def verify_manifest(
    root: Union[str, Path],
    *,
    manifest: Optional[Dict[str, str]] = None,
    strict: bool = True,
) -> Tuple[List[str], List[str]]:
    """Check that every file recorded in the manifest still hashes to its
    recorded SHA-256. Returns (missing, mismatched) lists.

    When `strict=True` (the default) and either list is non-empty, raises
    ManifestMismatchError. When `strict=False` the lists are returned without
    raising, useful for the reconcile path that wants to display diff.

    Files present on disk but absent from the manifest are silently ignored
    (those are unrelated to the manifest contract; manifest pins WHAT WAS
    COMMITTED, not what may live alongside).
    """
    root_path = Path(root)
    if manifest is None:
        manifest = read_manifest(root_path)
    missing: List[str] = []
    mismatched: List[str] = []
    for rel, expected_sha in manifest.items():
        candidate = root_path / rel
        if not candidate.is_file():
            missing.append(rel)
            continue
        actual = sha256_file(candidate)
        if actual != expected_sha:
            mismatched.append(rel)
    if strict and (missing or mismatched):
        raise ManifestMismatchError(
            "manifest verification failed for " + str(root_path) + ": "
            + str(len(missing)) + " missing, " + str(len(mismatched)) + " mismatched"
        )
    return missing, mismatched
