"""Versioned training-set directory protocol.

Layout on disk:

    parent/
        iteration-0000/    iteration-0001/    ...   iteration-0017/
        iteration-0018.staging/                  (transient; gone after commit)
        current  -> iteration-0017               (atomic symlink)
        .current.pointer                         (Windows fallback only)

Commit sequence (atomic at every step):

    1. stage(source=NNNN, target=NNNN+1)
         copytree iteration-0000..NNNN/ -> iteration-NNNN+1.staging/
       Killed here: only .staging exists -> recover_dangling cleans it up
       on the next daemon start.

    2. (caller writes new content into iteration-NNNN+1.staging/)

    3. commit(target=NNNN+1)
         compute_directory_manifest(staging) -> manifest dict
         write_manifest(staging, manifest)               -- atomic JSON write
         os.replace(staging, iteration-NNNN+1)           -- POSIX-atomic rename
       Killed between manifest-write and rename: staging dir still exists; the
       atomic rename either happens or it does not. There is no partial state.

    4. update_current(target=NNNN+1)
         write .current.tmp.<uuid> symlink -> iteration-NNNN+1
         os.replace(.current.tmp.<uuid>, current)         -- atomic symlink swap
       On Windows where symlinks need elevated privileges, fall back to writing
       a .current.pointer file containing the target directory name; readers
       must call read_current_version which understands both forms.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from .manifest import (
    MANIFEST_FILENAME,
    ManifestMismatchError,
    compute_directory_manifest,
    fsync_regular_files,
    read_manifest,
    unmanifested_directories,
    verify_manifest,
    write_manifest,
)


__all__ = [
    "DEFAULT_PREFIX",
    "STAGING_SUFFIX",
    "CURRENT_LINK_NAME",
    "TrainingSetVersioning",
]


DEFAULT_PREFIX = "iteration"
STAGING_SUFFIX = ".staging"
CURRENT_LINK_NAME = "current"
_CURRENT_POINTER_FALLBACK = ".current.pointer"
_VERSION_RE = re.compile(r"^(?P<prefix>[a-zA-Z0-9_-]+)-(?P<version>\d+)$")


@dataclass(frozen=True)
class TrainingSetVersioning:
    """Stateless helper for versioned iteration directories.

    The class itself holds only ``parent`` and ``prefix``; all operations are
    methods that walk the live filesystem. This keeps the helper safe to
    re-instantiate on every daemon tick without any caching to invalidate.
    """

    parent: Path
    prefix: str = DEFAULT_PREFIX
    name_width: int = 4

    # --- path helpers ---------------------------------------------------

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

    # --- discovery ------------------------------------------------------

    def list_committed_versions(self) -> List[int]:
        """Return the sorted list of committed iteration versions present
        on disk. Staging directories (with the ``.staging`` suffix) are
        excluded; they are not committed yet."""
        result: List[int] = []
        if not self.parent.exists():
            return result
        for child in self.parent.iterdir():
            if not child.is_dir():
                continue
            if child.name.endswith(STAGING_SUFFIX):
                continue
            m = _VERSION_RE.match(child.name)
            if m is None or m.group("prefix") != self.prefix:
                continue
            try:
                result.append(int(m.group("version")))
            except ValueError:
                continue
        return sorted(result)

    def list_dangling_staging(self) -> List[Path]:
        """All ``*.staging`` directories under parent (typically left behind
        by a daemon that died mid-stage). The caller decides whether to
        remove them (default behaviour of :meth:`recover_dangling_staging`)."""
        if not self.parent.exists():
            return []
        return sorted(
            child for child in self.parent.iterdir()
            if child.is_dir() and child.name.endswith(STAGING_SUFFIX)
        )

    def current_version(self) -> Optional[int]:
        """Resolve `current` -> committed version, or None if no pointer.

        Tries the POSIX symlink first; falls back to the
        ``.current.pointer`` text file. Returns None when neither exists or
        the target is missing / not parseable.
        """
        link = self.current_link_path()
        target_name: Optional[str] = None
        if link.is_symlink():
            try:
                target_name = os.readlink(str(link))
            except OSError:
                target_name = None
        elif link.exists() and link.is_dir():
            target_name = link.name
        if target_name is None:
            pointer = self._pointer_path()
            if pointer.exists():
                target_name = pointer.read_text(encoding="utf-8").strip()
        if not target_name:
            return None
        # target_name might be either bare ("iteration-0017") or relative
        # path; we only care about the base name.
        base = Path(target_name).name
        m = _VERSION_RE.match(base)
        if m is None or m.group("prefix") != self.prefix:
            return None
        try:
            version = int(m.group("version"))
        except ValueError:
            return None
        # a dangling pointer (target pruned, or a half-finished crash) still parses to a number whose
        # directory is gone. a function called current_version must not hand back a version with no
        # backing dir, so confirm it exists before trusting it. (A54)
        if not self.iteration_path(version).is_dir():
            return None
        return version

    # --- mutating operations -------------------------------------------

    def stage(self, source_version: Optional[int], target_version: int) -> Path:
        """Create the staging directory for `target_version`.

        If `source_version` is not None, copy that committed iteration's
        contents into the staging dir with a genuine byte copy (not a hardlink:
        the append flow rewrites some carried-over point files in place, so
        shared inodes would corrupt the previous committed iteration). If
        `source_version` is None, create an empty staging dir (used for
        iteration 0).

        Raises FileExistsError if the staging dir already exists; the caller
        must call :meth:`recover_dangling_staging` first.
        """
        if not self.parent.exists():
            self.parent.mkdir(parents=True, exist_ok=True)
        staging = self.staging_path(target_version)
        if staging.exists():
            raise FileExistsError(
                "staging dir already exists at " + str(staging)
                + "; call recover_dangling_staging() first"
            )
        if source_version is None:
            staging.mkdir(parents=True)
            return staging
        source = self.iteration_path(source_version)
        if not source.is_dir():
            raise FileNotFoundError(
                "source iteration not found: " + str(source)
            )
        # genuine byte copy, deliberately not a hardlink: the append flow
        # rewrites some carried-over point files in place, so sharing inodes
        # with the previous committed iteration would corrupt it (the dry-run
        # smoke test demonstrates exactly that). disk growth is better tackled
        # by pruning old iterations than by aliasing live ones.
        shutil.copytree(str(source), str(staging))
        return staging

    def commit(
        self,
        target_version: int,
        *,
        exclude_from_manifest: tuple = (),
    ) -> Dict[str, str]:
        """Finalise a staging directory by writing its manifest and
        atomically renaming it to the canonical iteration directory.

        Returns the manifest dict so the caller can store the SHA256 hash
        in the campaign journal for cross-iteration provenance.

        Raises FileNotFoundError if the staging dir is missing. If the target
        iteration already exists (a re-run after a crash) the commit is a no-op
        and returns the stored manifest.
        """
        staging = self.staging_path(target_version)
        target = self.iteration_path(target_version)
        if target.is_dir():
            # already committed -- most likely a crash after the rename but
            # before the caller recorded it. re-running should be a no-op, not
            # a crash, so read the stored manifest back and hand it over, and
            # bin any half-built staging dir left lying around.
            from .manifest import read_manifest
            existing = read_manifest(target)
            if staging.is_dir():
                shutil.rmtree(str(staging), ignore_errors=True)
            return existing
        if not staging.is_dir():
            raise FileNotFoundError(
                "no staging dir to commit at " + str(staging)
            )
        # strip stray pointdir lock files so they neither land in the committed
        # manifest nor trip a later strict verify.
        for lock in staging.rglob(".provenance.lock"):
            try:
                lock.unlink()
            except OSError:
                pass
        fsync_regular_files(staging, exclude=exclude_from_manifest)
        manifest = compute_directory_manifest(staging, exclude=exclude_from_manifest)
        write_manifest(staging, manifest)
        os.replace(str(staging), str(target))
        # Best-effort parent fsync; safe on Linux, no-op on Windows.
        from ..daemon.state import _fsync_parent_dir
        _fsync_parent_dir(target)
        return manifest

    def update_current(self, target_version: int) -> None:
        """Atomically point the `current` symlink at iteration `target_version`.

        On POSIX: creates a uniquely-named tmp symlink, then ``os.replace``s
        it onto ``current``. On Windows (where symlink creation often requires
        elevation), falls back to writing the target name into a small
        `.current.pointer` text file via :func:`atomic_write_text`.
        """
        target = self.iteration_path(target_version)
        if not target.is_dir():
            raise FileNotFoundError(
                "cannot point current at missing iteration " + str(target)
            )
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
            # Symlink creation refused (e.g. unprivileged user on Windows).
            from ..daemon.state import atomic_write_text
            atomic_write_text(self._pointer_path(), target.name + "\n")
            return
        os.replace(str(tmp_link), str(link))
        from ..daemon.state import _fsync_parent_dir
        _fsync_parent_dir(link)

    def ensure_current(self, target_version: int) -> bool:
        """Idempotently ensure ``current`` points at ``target_version``.

        Returns True when the pointer needed repair, False when it was already
        correct. This is used on crash-retry paths where the committed version
        directory exists but the daemon died before ``update_current`` ran.
        """
        current = self.current_version()
        if current == int(target_version):
            return False
        self.update_current(target_version)
        return True

    def recover_dangling_staging(self, *, delete: bool = True) -> List[Path]:
        """Find (and optionally delete) any leftover ``*.staging`` directories.

        Called by the daemon on startup -- a staging dir present without a
        matching committed iteration means the previous run died between
        :meth:`stage` and :meth:`commit`; the contents are unreliable and
        we discard them rather than promote them.

        Returns the list of paths that were dangling (regardless of whether
        delete=True or False) so the caller can log them.
        """
        dangling = self.list_dangling_staging()
        if delete:
            for p in dangling:
                # rename out of the way first (atomic) so a partial rmtree can't
                # leave a stump at the staging path and wedge the next stage().
                trash = p.with_name(p.name + ".discard." + uuid.uuid4().hex)
                try:
                    os.replace(str(p), str(trash))
                except OSError:
                    # could not move it aside. do NOT then rmtree in place -- a partial in-place
                    # delete leaves exactly the stump the rename was meant to avoid, wedging the
                    # next stage(). leave the dir whole and let the next start retry the rename. (A5)
                    continue
                shutil.rmtree(str(trash), ignore_errors=True)
        return dangling

    def verify_committed(self, version: int) -> None:
        """Re-validate the manifest of a committed iteration. Raises
        ManifestMismatchError on drift; used at daemon startup to detect
        post-commit tampering or partial transfers."""
        target = self.iteration_path(version)
        if not target.is_dir():
            raise FileNotFoundError(str(target))
        verify_manifest(target)

    def verify_committed_training_inputs(self, version: int) -> None:
        """Re-validate a committed training iteration as a daemon input.

        Generic manifests intentionally ignore extra files. FEREBUS export cannot: a manually
        copied ``POINT_*.pointdir`` would be discovered by ``PointsDirectory`` and silently train
        on data that was never committed. Reject such pointdirs before any consumer can glob them.
        """
        target = self.iteration_path(version)
        if not target.is_dir():
            raise FileNotFoundError(str(target))
        manifest = read_manifest(target)
        verify_manifest(target, manifest=manifest)
        extras = unmanifested_directories(
            target,
            manifest=manifest,
            pattern="POINT_*.pointdir",
        )
        if extras:
            raise ManifestMismatchError(
                "unmanifested committed training pointdir(s) in "
                + str(target)
                + ": "
                + ", ".join(extras[:5])
            )
