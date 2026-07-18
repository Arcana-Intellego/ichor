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
from ..strict_json import strict_json as json
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union


__all__ = [
    "MANIFEST_FILENAME",
    "ManifestMismatchError",
    "compute_directory_manifest",
    "read_manifest",
    "fsync_regular_files",
    "unmanifested_directories",
    "write_manifest",
    "verify_manifest",
    "sha256_file",
]


MANIFEST_FILENAME = ".manifest.json"
_BLOCK_SIZE = 1 << 20   # 1 MiB; fits comfortably in L2 cache while limiting syscalls.


class ManifestMismatchError(RuntimeError):
    """Raised when verify_manifest detects content drift."""


def _canonical_manifest_entry(key: object, digest: object) -> Tuple[str, str]:
    if not isinstance(key, str) or not key:
        raise ManifestMismatchError("manifest path must be a non-empty string")
    if "\\" in key:
        raise ManifestMismatchError("manifest path must use POSIX separators: " + repr(key))
    path = PurePosixPath(key)
    windows_path = PureWindowsPath(key)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or key.startswith("//")
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != key
    ):
        raise ManifestMismatchError("manifest path is noncanonical: " + repr(key))
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ManifestMismatchError(
            "manifest digest must be a lowercase SHA-256 for " + repr(key)
        )
    return key, digest


def _validated_manifest(manifest: object) -> Dict[str, str]:
    if not isinstance(manifest, dict):
        raise ManifestMismatchError("manifest must be a JSON object")
    validated: Dict[str, str] = {}
    for raw_key, raw_digest in manifest.items():
        key, digest = _canonical_manifest_entry(raw_key, raw_digest)
        if key in validated:
            raise ManifestMismatchError("duplicate canonical manifest path: " + key)
        validated[key] = digest
    return validated


def _manifest_file(root: Path, relative: str) -> Path:
    root_absolute = Path(os.path.abspath(os.fspath(root)))
    candidate = root_absolute.joinpath(*PurePosixPath(relative).parts)
    try:
        candidate.relative_to(root_absolute)
    except ValueError as exc:
        raise ManifestMismatchError("manifest path escapes its root: " + relative) from exc
    current = root_absolute
    if current.is_symlink():
        raise ManifestMismatchError("manifest root is a symlink: " + str(root))
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ManifestMismatchError("manifest path contains a symlink: " + relative)
    return candidate


def sha256_file(
    path: Union[str, Path],
    *,
    progress_callback: Optional[Callable[[Path, int], None]] = None,
) -> str:
    """Streamed SHA-256 of one regular file. Memory use is bounded by
    _BLOCK_SIZE regardless of file size."""
    h = hashlib.sha256()
    source = Path(path)
    with open(source, "rb") as f:
        while True:
            chunk = f.read(_BLOCK_SIZE)
            if not chunk:
                break
            h.update(chunk)
            if progress_callback is not None:
                progress_callback(source, len(chunk))
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


def fsync_regular_files(
    root: Union[str, Path],
    *,
    exclude: Iterable[str] = (),
) -> None:
    """Durably synchronise every regular file that will enter a manifest."""
    from ..daemon.state import _fsync_file_descriptor

    root_path = Path(root)
    for f in _iter_files(root_path, exclude=exclude):
        mode = "rb+" if platform.system() == "Windows" else "rb"
        with open(f, mode) as handle:
            _fsync_file_descriptor(handle.fileno())


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
    try:
        return _validated_manifest(data)
    except ManifestMismatchError as exc:
        raise ManifestMismatchError("invalid manifest at " + str(p) + ": " + str(exc)) from exc


def write_manifest(root: Union[str, Path], manifest: Dict[str, str]) -> Path:
    """Write `manifest` atomically to `root/.manifest.json`.

    Uses the atomic-write helper from :mod:`..daemon.state` so a kill mid-write
    leaves either no manifest or a complete one.
    """
    from ..daemon.state import atomic_write_json

    p = Path(root) / MANIFEST_FILENAME
    atomic_write_json(p, _validated_manifest(manifest))
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
    exact: bool = False,
    digest_file: Optional[Callable[[Path, bool], str]] = None,
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
    else:
        manifest = _validated_manifest(manifest)
    missing: List[str] = []
    mismatched: List[str] = []
    for rel, expected_sha in manifest.items():
        candidate = _manifest_file(root_path, rel)
        if not candidate.is_file():
            missing.append(rel)
            continue
        actual = (
            digest_file(candidate, False)
            if digest_file is not None
            else sha256_file(candidate)
        )
        if actual != expected_sha:
            mismatched.append(rel)
    unexpected: List[str] = []
    if exact:
        actual_keys = set(compute_directory_manifest(root_path))
        unexpected = sorted(actual_keys - set(manifest))
        mismatched.extend("unexpected:" + value for value in unexpected)
    if strict and (missing or mismatched):
        raise ManifestMismatchError(
            "manifest verification failed for " + str(root_path) + ": "
            + str(len(missing)) + " missing, "
            + str(len(mismatched) - len(unexpected)) + " mismatched, "
            + str(len(unexpected)) + " unexpected"
        )
    return missing, mismatched
