"""Bounded, prevalidated publication for one FEREBUS model snapshot."""

from __future__ import annotations

import os
import stat as stat_module
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Tuple

from ..versioning.manifest import (
    MANIFEST_FILENAME,
    read_manifest,
    sha256_file,
    write_manifest,
)
from .state import _fsync_parent_dir


class FerebusSnapshotPublicationError(ValueError):
    """Raised when a staged model snapshot changes before publication."""


@dataclass(frozen=True)
class FerebusSnapshotFile:
    relative_path: str
    path: Path
    stat_identity: Tuple[int, int, int, int, int]
    sha256: str

    @property
    def size(self) -> int:
        return int(self.stat_identity[2])


@dataclass(frozen=True)
class FerebusSnapshotDirectory:
    relative_path: str
    path: Path
    stat_identity: Tuple[int, int]


@dataclass(frozen=True)
class FerebusSnapshotPublicationContext:
    staging_root: Path
    files: Tuple[FerebusSnapshotFile, ...]
    directories: Tuple[FerebusSnapshotDirectory, ...]
    manifest: Mapping[str, str]

    @property
    def file_by_path(self) -> Mapping[str, FerebusSnapshotFile]:
        return {entry.relative_path: entry for entry in self.files}


def _file_identity(value: os.stat_result) -> Tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _directory_identity(value: os.stat_result) -> Tuple[int, int]:
    return (int(value.st_dev), int(value.st_ino))


def _scan_tree(
    root: Path,
    *,
    digest_for_identity: Optional[
        Callable[[Path, Tuple[int, int, int, int, int]], Optional[str]]
    ] = None,
    include_manifest: bool = False,
) -> Tuple[
    Tuple[FerebusSnapshotFile, ...],
    Tuple[FerebusSnapshotDirectory, ...],
]:
    root = Path(root)
    root_stat = root.lstat()
    if not stat_module.S_ISDIR(root_stat.st_mode):
        raise FerebusSnapshotPublicationError(
            "FEREBUS snapshot root is missing or unsafe"
        )
    files: Dict[str, FerebusSnapshotFile] = {}
    directories: Dict[str, FerebusSnapshotDirectory] = {
        ".": FerebusSnapshotDirectory(
            relative_path=".",
            path=root,
            stat_identity=_directory_identity(root_stat),
        )
    }

    def visit(directory: Path, relative_parent: str) -> None:
        try:
            with os.scandir(directory) as scanner:
                entries = sorted(scanner, key=lambda item: item.name)
        except OSError as exc:
            raise FerebusSnapshotPublicationError(
                "FEREBUS snapshot inventory is unreadable: " + str(directory)
            ) from exc
        for entry in entries:
            relative = (
                entry.name
                if relative_parent == "."
                else relative_parent + "/" + entry.name
            )
            value = entry.stat(follow_symlinks=False)
            mode = value.st_mode
            path = directory / entry.name
            if stat_module.S_ISLNK(mode):
                raise FerebusSnapshotPublicationError(
                    "FEREBUS snapshot contains a symlink: " + relative
                )
            if stat_module.S_ISDIR(mode):
                directories[relative] = FerebusSnapshotDirectory(
                    relative_path=relative,
                    path=path,
                    stat_identity=_directory_identity(value),
                )
                visit(path, relative)
                continue
            if not stat_module.S_ISREG(mode):
                raise FerebusSnapshotPublicationError(
                    "FEREBUS snapshot contains a special file: " + relative
                )
            if relative == MANIFEST_FILENAME and not include_manifest:
                continue
            identity = _file_identity(value)
            digest = (
                None
                if digest_for_identity is None
                else digest_for_identity(path, identity)
            )
            files[relative] = FerebusSnapshotFile(
                relative_path=relative,
                path=path,
                stat_identity=identity,
                sha256="" if digest is None else str(digest),
            )
    visit(root, ".")
    return (
        tuple(files[key] for key in sorted(files)),
        tuple(directories[key] for key in sorted(directories)),
    )


def capture_ferebus_snapshot_publication(
    staging_root: Path,
    *,
    digest_for_identity: Callable[
        [Path, Tuple[int, int, int, int, int]], Optional[str]
    ],
) -> FerebusSnapshotPublicationContext:
    root = Path(staging_root)
    if (root / MANIFEST_FILENAME).exists() or (root / MANIFEST_FILENAME).is_symlink():
        raise FerebusSnapshotPublicationError(
            "FEREBUS snapshot already contains a directory manifest"
        )
    files, directories = _scan_tree(
        root,
        digest_for_identity=digest_for_identity,
    )
    missing = [entry.relative_path for entry in files if len(entry.sha256) != 64]
    if missing:
        raise FerebusSnapshotPublicationError(
            "FEREBUS snapshot has no authenticated digest: " + missing[0]
        )
    manifest = {entry.relative_path: entry.sha256 for entry in files}
    return FerebusSnapshotPublicationContext(
        staging_root=root,
        files=files,
        directories=directories,
        manifest=manifest,
    )


def assert_ferebus_snapshot_unchanged(
    context: FerebusSnapshotPublicationContext,
) -> None:
    files, directories = _scan_tree(context.staging_root)
    expected_files = {
        entry.relative_path: entry.stat_identity for entry in context.files
    }
    observed_files = {
        entry.relative_path: entry.stat_identity for entry in files
    }
    expected_directories = {
        entry.relative_path: entry.stat_identity for entry in context.directories
    }
    observed_directories = {
        entry.relative_path: entry.stat_identity for entry in directories
    }
    if observed_files != expected_files:
        raise FerebusSnapshotPublicationError(
            "FEREBUS snapshot file inventory changed before publication"
        )
    for entry in context.files:
        if (os.name == "nt" or entry.stat_identity[1] == 0) and sha256_file(
            entry.path
        ) != entry.sha256:
            raise FerebusSnapshotPublicationError(
                "FEREBUS snapshot file content changed before publication: "
                + entry.relative_path
            )
    if observed_directories != expected_directories:
        raise FerebusSnapshotPublicationError(
            "FEREBUS snapshot directory identity changed before publication"
        )
    if read_manifest(context.staging_root) != dict(context.manifest):
        raise FerebusSnapshotPublicationError(
            "FEREBUS snapshot directory manifest changed before publication"
        )


def publish_ferebus_snapshot_manifest(
    context: FerebusSnapshotPublicationContext,
) -> Path:
    path = write_manifest(context.staging_root, dict(context.manifest))
    if read_manifest(context.staging_root) != dict(context.manifest):
        raise FerebusSnapshotPublicationError(
            "FEREBUS snapshot directory manifest publication failed"
        )
    return path


def synchronise_ferebus_snapshot_directories(
    context: FerebusSnapshotPublicationContext,
) -> int:
    if os.name == "nt":
        return 0
    count = 0
    for entry in sorted(
        context.directories,
        key=lambda item: item.relative_path.count("/"),
        reverse=True,
    ):
        _fsync_parent_dir(entry.path / ".ichor-directory-entry")
        count += 1
    return count


def commit_prevalidated_ferebus_snapshot(
    context: FerebusSnapshotPublicationContext,
    target_root: Path,
) -> None:
    target = Path(target_root)
    if target.exists() or target.is_symlink():
        raise FileExistsError("FEREBUS model commit target already exists: " + str(target))
    if int(context.staging_root.lstat().st_dev) != int(target.parent.lstat().st_dev):
        raise FerebusSnapshotPublicationError(
            "FEREBUS staging and target are on different filesystems"
        )
    assert_ferebus_snapshot_unchanged(context)
    os.replace(context.staging_root, target)
    _fsync_parent_dir(target)


def assert_committed_ferebus_snapshot(
    context: FerebusSnapshotPublicationContext,
    committed_root: Path,
) -> bool:
    """Bind staged bytes to their committed paths; return whether hashes are reusable."""
    files, directories = _scan_tree(Path(committed_root))
    observed_files = {entry.relative_path: entry for entry in files}
    observed_directories = {entry.relative_path: entry for entry in directories}
    expected_file_paths = {entry.relative_path for entry in context.files}
    if set(observed_files) != expected_file_paths:
        raise FerebusSnapshotPublicationError(
            "committed FEREBUS snapshot file inventory changed across publication"
        )
    expected_directory_paths = {
        entry.relative_path for entry in context.directories
    }
    if set(observed_directories) != expected_directory_paths:
        raise FerebusSnapshotPublicationError(
            "committed FEREBUS snapshot directory inventory changed across publication"
        )
    reusable = os.name != "nt"
    for before in context.files:
        after = observed_files[before.relative_path]
        if before.stat_identity[:4] != after.stat_identity[:4]:
            raise FerebusSnapshotPublicationError(
                "committed FEREBUS file changed across publication: "
                + before.relative_path
            )
        if before.stat_identity[1] == 0:
            reusable = False
    for before in context.directories:
        after = observed_directories[before.relative_path]
        if before.stat_identity != after.stat_identity:
            raise FerebusSnapshotPublicationError(
                "committed FEREBUS directory changed across publication: "
                + before.relative_path
            )
    if read_manifest(committed_root) != dict(context.manifest):
        raise FerebusSnapshotPublicationError(
            "committed FEREBUS directory manifest changed across publication"
        )
    return reusable


__all__ = [
    "FerebusSnapshotFile",
    "FerebusSnapshotPublicationContext",
    "FerebusSnapshotPublicationError",
    "assert_committed_ferebus_snapshot",
    "assert_ferebus_snapshot_unchanged",
    "capture_ferebus_snapshot_publication",
    "commit_prevalidated_ferebus_snapshot",
    "publish_ferebus_snapshot_manifest",
    "synchronise_ferebus_snapshot_directories",
]
