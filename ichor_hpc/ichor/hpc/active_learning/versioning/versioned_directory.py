"""Atomic versioned-directory protocol used by daemon-owned artefacts."""

from __future__ import annotations

import os
import platform
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .manifest import (
    compute_directory_manifest,
    fsync_regular_files,
    read_manifest,
    verify_manifest,
    write_manifest,
)


DEFAULT_PREFIX = "iteration"
STAGING_SUFFIX = ".staging"
CURRENT_LINK_NAME = "current"
_CURRENT_POINTER_FALLBACK = ".current.pointer"
_VERSION_RE = re.compile(r"^(?P<prefix>[a-zA-Z0-9_-]+)-(?P<version>\d+)$")


@dataclass(frozen=True)
class VersionedDirectory:
    parent: Path
    prefix: str = DEFAULT_PREFIX
    name_width: int = 4

    def iteration_name(self, version: int) -> str:
        return f"{self.prefix}-{int(version):0{self.name_width}d}"

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
            if not child.is_dir() or child.name.endswith(STAGING_SUFFIX):
                continue
            match = _VERSION_RE.match(child.name)
            if match is None or match.group("prefix") != self.prefix:
                continue
            result.append(int(match.group("version")))
        return sorted(result)

    def list_dangling_staging(self) -> List[Path]:
        if not self.parent.exists():
            return []
        return sorted(
            child
            for child in self.parent.iterdir()
            if child.is_dir() and child.name.endswith(STAGING_SUFFIX)
        )

    def current_version(self) -> Optional[int]:
        link = self.current_link_path()
        target_name: Optional[str] = None
        if link.is_symlink():
            try:
                target_name = os.readlink(str(link))
            except OSError:
                target_name = None
        if target_name is None:
            pointer = self._pointer_path()
            if pointer.is_file():
                target_name = pointer.read_text(encoding="utf-8").strip()
        if not target_name:
            return None
        match = _VERSION_RE.match(Path(target_name).name)
        if match is None or match.group("prefix") != self.prefix:
            return None
        version = int(match.group("version"))
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
        if target.is_dir():
            if staging.is_dir():
                shutil.rmtree(str(staging), ignore_errors=True)
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

    def update_current(self, target_version: int) -> None:
        target = self.iteration_path(target_version)
        if not target.is_dir():
            raise FileNotFoundError("cannot point current at missing " + str(target))
        link = self.current_link_path()
        if platform.system() == "Windows":
            from ..daemon.state import atomic_write_text

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
