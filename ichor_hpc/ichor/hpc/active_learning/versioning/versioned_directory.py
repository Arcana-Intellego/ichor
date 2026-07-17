"""Atomic versioned-directory protocol used by daemon-owned artefacts."""

from __future__ import annotations

import os
import platform
import re
import shutil
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from .manifest import (
    compute_directory_manifest,
    fsync_regular_files,
    read_manifest,
    verify_manifest,
    write_manifest,
)
from ..layout import COMMITTED_VERSION_NAME_WIDTH


DEFAULT_PREFIX = "iteration"
STAGING_SUFFIX = ".staging"
CURRENT_LINK_NAME = "current"
_CURRENT_POINTER_FALLBACK = ".current.pointer"
_VERSION_RE = re.compile(r"^(?P<prefix>[a-zA-Z0-9_-]+)-(?P<version>\d+)$")


@dataclass(frozen=True)
class VersionedDirectory:
    parent: Path
    prefix: str = DEFAULT_PREFIX
    name_width: int = COMMITTED_VERSION_NAME_WIDTH

    def _version(self, version: int) -> int:
        if isinstance(version, bool):
            raise ValueError("version must be a non-negative integer")
        if isinstance(version, float) and not version.is_integer():
            raise ValueError("version must be a non-negative integer")
        try:
            parsed = int(version)
        except (TypeError, ValueError) as exc:
            raise ValueError("version must be a non-negative integer") from exc
        if parsed < 0:
            raise ValueError("version must be a non-negative integer")
        return parsed

    def iteration_name(self, version: int) -> str:
        parsed = self._version(version)
        return f"{self.prefix}-{parsed:0{self.name_width}d}"

    def iteration_path(self, version: int) -> Path:
        return Path(self.parent) / self.iteration_name(version)

    def staging_name(self, version: int) -> str:
        return self.iteration_name(version) + STAGING_SUFFIX

    def staging_path(self, version: int) -> Path:
        return Path(self.parent) / self.staging_name(version)

    def current_link_path(self) -> Path:
        return Path(self.parent) / CURRENT_LINK_NAME

    def _pointer_path(self) -> Path:
        return Path(self.parent) / _CURRENT_POINTER_FALLBACK

    def list_committed_versions(self) -> List[int]:
        result: List[int] = []
        if not self.parent.exists():
            return result
        for child in self.parent.iterdir():
            if child.name.endswith(STAGING_SUFFIX):
                continue
            match = _VERSION_RE.match(child.name)
            if match is None or match.group("prefix") != self.prefix:
                continue
            if child.is_symlink() or not child.is_dir():
                raise ValueError(
                    "version entry is not a regular directory: " + str(child)
                )
            version = int(match.group("version"))
            expected = self.iteration_name(version)
            if child.name != expected:
                raise ValueError(
                    "non-canonical version directory name: "
                    + child.name
                    + "; expected "
                    + expected
                )
            result.append(version)
        if len(result) != len(set(result)):
            raise ValueError("duplicate numeric versions under " + str(self.parent))
        return sorted(result)

    def list_dangling_staging(self) -> List[Path]:
        if not self.parent.exists():
            return []
        result: List[Path] = []
        for child in self.parent.iterdir():
            if not child.name.endswith(STAGING_SUFFIX):
                continue
            base_name = child.name[: -len(STAGING_SUFFIX)]
            match = _VERSION_RE.match(base_name)
            if match is None or match.group("prefix") != self.prefix:
                continue
            if child.is_symlink() or not child.is_dir():
                raise ValueError(
                    "version staging entry is not a regular directory: "
                    + str(child)
                )
            version = int(match.group("version"))
            expected = self.staging_name(version)
            if child.name != expected:
                raise ValueError(
                    "non-canonical version staging name: "
                    + child.name
                    + "; expected "
                    + expected
                )
            result.append(child)
        return sorted(result)

    def current_version(self) -> Optional[int]:
        link = self.current_link_path()
        target_name: Optional[str] = None
        if link.is_symlink():
            try:
                target_name = os.readlink(str(link))
            except OSError:
                target_name = None
        elif link.exists():
            raise ValueError("current path is not a symlink")
        if target_name is None:
            pointer = self._pointer_path()
            if pointer.is_symlink():
                raise ValueError("current pointer fallback must not be a symlink")
            if pointer.is_file():
                target_name = pointer.read_text(encoding="utf-8").strip()
        if not target_name:
            return None
        if Path(target_name).name != target_name:
            raise ValueError("current pointer target must be a version basename")
        match = _VERSION_RE.match(Path(target_name).name)
        if match is None or match.group("prefix") != self.prefix:
            return None
        version = int(match.group("version"))
        if Path(target_name).name != self.iteration_name(version):
            raise ValueError("current pointer uses a non-canonical version name")
        return version if self.iteration_path(version).is_dir() else None

    def stage(self, source_version: Optional[int], target_version: int) -> Path:
        """Create staging, optionally copying a generic predecessor.

        QM reference data never supplies ``source_version``; its specialised
        commit path stores deltas only. The copy option remains for other
        versioned daemon artefacts.
        """
        self.parent.mkdir(parents=True, exist_ok=True)
        staging = self.staging_path(target_version)
        if staging.exists():
            raise FileExistsError("staging directory already exists: " + str(staging))
        if source_version is None:
            staging.mkdir(parents=True)
        else:
            source = self.iteration_path(source_version)
            if not source.is_dir():
                raise FileNotFoundError("source iteration not found: " + str(source))
            shutil.copytree(str(source), str(staging))
        return staging

    def commit(
        self,
        target_version: int,
        *,
        exclude_from_manifest: tuple = (),
    ) -> Dict[str, str]:
        staging = self.staging_path(target_version)
        target = self.iteration_path(target_version)
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_dir():
                raise ValueError(
                    "commit target is not a regular directory: " + str(target)
                )
            if staging.exists() or staging.is_symlink():
                if staging.is_symlink() or not staging.is_dir():
                    raise ValueError(
                        "stale staging path is not a regular directory: "
                        + str(staging)
                    )
                shutil.rmtree(str(staging))
                if staging.exists() or staging.is_symlink():
                    raise OSError("stale staging cleanup did not complete: " + str(staging))
            return read_manifest(target)
        if not staging.is_dir():
            raise FileNotFoundError("no staging directory to commit: " + str(staging))
        for lock in staging.rglob(".provenance.lock"):
            try:
                lock.unlink()
            except OSError:
                pass
        fsync_regular_files(staging, exclude=exclude_from_manifest)
        manifest = compute_directory_manifest(staging, exclude=exclude_from_manifest)
        write_manifest(staging, manifest)
        os.replace(str(staging), str(target))
        from ..daemon.state import _fsync_parent_dir

        _fsync_parent_dir(target)
        return manifest

    def commit_prehashed(
        self,
        target_version: int,
        *,
        manifest: Mapping[str, str],
        expected_sizes: Mapping[str, int],
    ) -> Dict[str, str]:
        """Publish a staging tree using trusted producer hashes.

        This path deliberately validates exact paths, regular-file types and
        sizes without reading payload bytes. It is reserved for sealed quantum
        pointdirs whose hashes were frozen by acceptance receipts.
        """
        staging = self.staging_path(target_version)
        target = self.iteration_path(target_version)
        if target.exists() or target.is_symlink():
            raise FileExistsError("prehashed commit target already exists: " + str(target))
        if not staging.is_dir() or staging.is_symlink():
            raise FileNotFoundError("no regular staging directory to commit: " + str(staging))
        expected = {str(key): str(value) for key, value in manifest.items()}
        sizes = {str(key): int(value) for key, value in expected_sizes.items()}
        if set(expected) != set(sizes):
            raise ValueError("prehashed manifest paths and size paths differ")
        actual_sizes: Dict[str, int] = {}
        for path in sorted(staging.rglob("*")):
            if path.is_symlink():
                raise ValueError("prehashed staging contains a symlink: " + str(path))
            relative_path = path.relative_to(staging)
            path_stat = path.stat()
            inside_sealed_pointdir = bool(
                relative_path.parts
                and relative_path.parts[0].endswith(".pointdir")
            )
            if inside_sealed_pointdir and stat.S_IMODE(path_stat.st_mode) & (
                stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
            ):
                raise ValueError(
                    "prehashed pointdir entry is writable: " + str(path)
                )
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError("prehashed staging contains a special file: " + str(path))
            if path.name == ".manifest.json":
                continue
            relative = relative_path.as_posix()
            actual_sizes[relative] = int(path_stat.st_size)
        if actual_sizes != sizes:
            missing = sorted(set(sizes) - set(actual_sizes))
            unexpected = sorted(set(actual_sizes) - set(sizes))
            wrong_sizes = sorted(
                key
                for key in set(actual_sizes) & set(sizes)
                if actual_sizes[key] != sizes[key]
            )
            raise ValueError(
                "prehashed staging inventory mismatch: missing="
                + repr(missing[:5])
                + " unexpected="
                + repr(unexpected[:5])
                + " wrong_sizes="
                + repr(wrong_sizes[:5])
            )
        write_manifest(staging, expected)
        os.replace(str(staging), str(target))
        from ..daemon.state import _fsync_parent_dir

        _fsync_parent_dir(target)
        return dict(expected)

    def update_current(self, target_version: int) -> None:
        target = self.iteration_path(target_version)
        if not target.is_dir():
            raise FileNotFoundError("cannot point current at missing " + str(target))
        link = self.current_link_path()
        if platform.system() == "Windows":
            from ..daemon.state import atomic_write_text

            if link.is_symlink():
                link.unlink()
            elif link.exists():
                raise ValueError("current path is not a replaceable symlink")
            atomic_write_text(self._pointer_path(), target.name + "\n")
            return
        tmp_link = link.with_name(
            CURRENT_LINK_NAME + ".tmp." + str(os.getpid()) + "." + uuid.uuid4().hex
        )
        try:
            os.symlink(target.name, str(tmp_link))
        except OSError:
            from ..daemon.state import atomic_write_text

            atomic_write_text(self._pointer_path(), target.name + "\n")
            return
        os.replace(str(tmp_link), str(link))
        from ..daemon.state import _fsync_parent_dir

        pointer = self._pointer_path()
        if pointer.is_file() or pointer.is_symlink():
            pointer.unlink()
        _fsync_parent_dir(link)

    def ensure_current(self, target_version: int) -> bool:
        if self.current_version() == int(target_version):
            return False
        self.update_current(target_version)
        return True

    def recover_dangling_staging(self, *, delete: bool = True) -> List[Path]:
        dangling = self.list_dangling_staging()
        if delete:
            for path in dangling:
                trash = path.with_name(path.name + ".discard." + uuid.uuid4().hex)
                try:
                    os.replace(str(path), str(trash))
                except OSError:
                    continue
                shutil.rmtree(str(trash), ignore_errors=True)
        return dangling

    def verify_committed(self, version: int) -> None:
        target = self.iteration_path(version)
        if not target.is_dir():
            raise FileNotFoundError(str(target))
        verify_manifest(target)


__all__ = [
    "DEFAULT_PREFIX",
    "STAGING_SUFFIX",
    "CURRENT_LINK_NAME",
    "VersionedDirectory",
]
