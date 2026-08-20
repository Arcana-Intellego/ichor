"""Strict classification of canonical ARIADNE seed-tree entries."""
from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from .layout import (
    ariadne_seeds_dir,
    parse_seed_directory_name,
    seed_directory_name,
)


_TRANSACTION_RESIDUE_PREFIX = "."
_TRANSACTION_RESIDUE_MARKER = ".partial-task-"
_TRANSACTION_RESIDUE_PID_MARKER = "-pid-"


class AriadneSeedTreeError(ValueError):
    """Raised when the ARIADNE seed tree is structurally unsafe."""


class AriadneTransactionResidueError(AriadneSeedTreeError):
    """Raised when authenticated transaction residue still needs retention."""


@dataclass(frozen=True)
class AriadneTransactionResidue:
    path: Path
    name: str
    seed_id: int
    array_task_id: int
    pid: int
    total_bytes: int
    n_files: int
    n_directories: int
    tree_sha256: str
    root_identity: Tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class AriadneSeedTreeContext:
    iteration_dir: Path
    seeds_root: Path
    canonical_seed_names: Tuple[str, ...]
    residues: Tuple[AriadneTransactionResidue, ...]
    snapshot_sha256: str

    def assert_unchanged(self, task_map: Mapping[str, Any]) -> None:
        current = classify_ariadne_seed_tree(self.iteration_dir, task_map)
        if current.snapshot_sha256 != self.snapshot_sha256:
            raise AriadneSeedTreeError(
                "ARIADNE seed tree changed while transaction residue was being handled"
            )


def _integer(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AriadneSeedTreeError(label + " is invalid")
    return int(value)


def _canonical_decimal(value: str, label: str, *, minimum: int) -> int:
    if not value or not value.isascii() or not value.isdecimal():
        raise AriadneSeedTreeError(label + " is invalid")
    parsed = int(value)
    if parsed < minimum or str(parsed) != value:
        raise AriadneSeedTreeError(label + " is noncanonical")
    return parsed


def parse_ariadne_transaction_residue_name(name: str) -> Tuple[int, int, int]:
    """Return ``(seed_id, array_task_id, pid)`` for an exact residue name."""
    value = str(name)
    if not value.startswith(_TRANSACTION_RESIDUE_PREFIX):
        raise AriadneSeedTreeError(
            "invalid ARIADNE transaction residue name: " + repr(value)
        )
    body = value[1:]
    seed_name, marker, tail = body.partition(_TRANSACTION_RESIDUE_MARKER)
    if marker != _TRANSACTION_RESIDUE_MARKER:
        raise AriadneSeedTreeError(
            "invalid ARIADNE transaction residue name: " + repr(value)
        )
    task_text, pid_marker, pid_text = tail.partition(
        _TRANSACTION_RESIDUE_PID_MARKER
    )
    if pid_marker != _TRANSACTION_RESIDUE_PID_MARKER:
        raise AriadneSeedTreeError(
            "invalid ARIADNE transaction residue name: " + repr(value)
        )
    try:
        seed_id = parse_seed_directory_name(seed_name)
    except ValueError as exc:
        raise AriadneSeedTreeError(str(exc)) from exc
    task_id = _canonical_decimal(
        task_text,
        "ARIADNE transaction residue task ID",
        minimum=0,
    )
    pid = _canonical_decimal(
        pid_text,
        "ARIADNE transaction residue PID",
        minimum=1,
    )
    return seed_id, task_id, pid


def _stat_identity(info: os.stat_result) -> Tuple[int, int, int, int, int, int]:
    return (
        int(info.st_mode),
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _hash_regular_file(path: Path, before: os.stat_result) -> Tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
        after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise AriadneSeedTreeError(
            "ARIADNE transaction residue file is unreadable: " + str(path)
        ) from exc
    if _stat_identity(before) != _stat_identity(after) or size != int(before.st_size):
        raise AriadneSeedTreeError(
            "ARIADNE transaction residue changed while it was inventoried: "
            + str(path)
        )
    return size, digest.hexdigest()


def _inventory_residue_tree(
    path: Path,
    *,
    seed_id: int,
    array_task_id: int,
    pid: int,
) -> AriadneTransactionResidue:
    try:
        root_before = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise AriadneSeedTreeError(
            "ARIADNE transaction residue is unreadable: " + str(path)
        ) from exc
    if not stat.S_ISDIR(root_before.st_mode):
        raise AriadneSeedTreeError(
            "ARIADNE transaction residue is not a regular directory: " + str(path)
        )

    records = []
    total_bytes = 0
    n_files = 0
    n_directories = 1
    pending = [(path, Path("."))]
    while pending:
        directory, relative_directory = pending.pop()
        try:
            directory_before = directory.stat(follow_symlinks=False)
            with os.scandir(directory) as scan:
                entries = sorted(scan, key=lambda item: item.name)
        except OSError as exc:
            raise AriadneSeedTreeError(
                "ARIADNE transaction residue directory is unreadable: "
                + str(directory)
            ) from exc
        for entry in entries:
            child = directory / entry.name
            relative = relative_directory / entry.name
            try:
                info = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise AriadneSeedTreeError(
                    "ARIADNE transaction residue entry is unreadable: " + str(child)
                ) from exc
            if stat.S_ISLNK(info.st_mode):
                raise AriadneSeedTreeError(
                    "ARIADNE transaction residue contains a symlink: " + str(child)
                )
            if stat.S_ISDIR(info.st_mode):
                n_directories += 1
                records.append(("d", relative.as_posix(), _stat_identity(info)))
                pending.append((child, relative))
                continue
            if stat.S_ISREG(info.st_mode):
                size, digest = _hash_regular_file(child, info)
                total_bytes += size
                n_files += 1
                records.append(
                    ("f", relative.as_posix(), _stat_identity(info), digest)
                )
                continue
            raise AriadneSeedTreeError(
                "ARIADNE transaction residue contains a special file: " + str(child)
            )
        try:
            directory_after = directory.stat(follow_symlinks=False)
        except OSError as exc:
            raise AriadneSeedTreeError(
                "ARIADNE transaction residue directory disappeared: "
                + str(directory)
            ) from exc
        if _stat_identity(directory_before) != _stat_identity(directory_after):
            raise AriadneSeedTreeError(
                "ARIADNE transaction residue changed while it was inventoried: "
                + str(directory)
            )

    try:
        root_after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise AriadneSeedTreeError(
            "ARIADNE transaction residue disappeared: " + str(path)
        ) from exc
    root_identity = _stat_identity(root_after)
    if _stat_identity(root_before) != root_identity:
        raise AriadneSeedTreeError(
            "ARIADNE transaction residue changed while it was inventoried: " + str(path)
        )
    tree = hashlib.sha256()
    for record in sorted(records, key=lambda item: (item[1], item[0])):
        tree.update(repr(record).encode("utf-8"))
        tree.update(b"\n")
    return AriadneTransactionResidue(
        path=path,
        name=path.name,
        seed_id=int(seed_id),
        array_task_id=int(array_task_id),
        pid=int(pid),
        total_bytes=int(total_bytes),
        n_files=int(n_files),
        n_directories=int(n_directories),
        tree_sha256=tree.hexdigest(),
        root_identity=root_identity,
    )


def _task_map_index(
    task_map: Mapping[str, Any],
) -> Tuple[Mapping[int, Mapping[str, Any]], Mapping[str, Mapping[str, Any]]]:
    raw_tasks = task_map.get("tasks")
    if not isinstance(raw_tasks, Sequence) or isinstance(raw_tasks, (str, bytes)):
        raise AriadneSeedTreeError("ARIADNE task-map tasks are invalid")
    tasks_by_id = {}
    tasks_by_name = {}
    declared_n_tasks = _integer(
        task_map.get("n_tasks"),
        "ARIADNE task count",
        minimum=1,
    )
    if declared_n_tasks != len(raw_tasks):
        raise AriadneSeedTreeError("ARIADNE task-map count mismatch")
    for raw in raw_tasks:
        if not isinstance(raw, Mapping):
            raise AriadneSeedTreeError("ARIADNE task-map record is invalid")
        task_id = _integer(raw.get("array_task_id"), "ARIADNE task ID", minimum=0)
        seed_id = _integer(raw.get("seed_id"), "ARIADNE seed ID", minimum=1)
        seed_path = str(raw.get("seed_directory") or "")
        seed_name = seed_directory_name(seed_id)
        if seed_path != "ariadne/seeds/" + seed_name:
            raise AriadneSeedTreeError(
                "ARIADNE task-map seed directory is noncanonical"
            )
        if parse_seed_directory_name(seed_name) != seed_id:
            raise AriadneSeedTreeError("ARIADNE task-map seed directory is invalid")
        if task_id in tasks_by_id or seed_name in tasks_by_name:
            raise AriadneSeedTreeError("ARIADNE task-map seed identity is duplicated")
        tasks_by_id[task_id] = raw
        tasks_by_name[seed_name] = raw
    if set(tasks_by_id) != set(range(len(tasks_by_id))):
        raise AriadneSeedTreeError("ARIADNE task IDs are not contiguous")
    return tasks_by_id, tasks_by_name


def classify_ariadne_seed_tree(
    iteration_dir: Path,
    task_map: Mapping[str, Any],
) -> AriadneSeedTreeContext:
    """Classify canonical outputs and authenticated transaction residue."""
    iter_dir = Path(iteration_dir)
    seeds_root = ariadne_seeds_dir(iter_dir)
    tasks_by_id, tasks_by_name = _task_map_index(task_map)
    for path in (iter_dir, seeds_root.parent, seeds_root):
        if not path.exists() and not path.is_symlink():
            continue
        try:
            info = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise AriadneSeedTreeError(
                "ARIADNE seed-tree path is unreadable: " + str(path)
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise AriadneSeedTreeError(
                "ARIADNE seed-tree path contains a symlink: " + str(path)
            )
        if not stat.S_ISDIR(info.st_mode):
            raise AriadneSeedTreeError(
                "ARIADNE seed-tree parent is not a directory: " + str(path)
            )
    if not seeds_root.exists() and not seeds_root.is_symlink():
        empty = hashlib.sha256(b"missing\n").hexdigest()
        return AriadneSeedTreeContext(
            iteration_dir=iter_dir,
            seeds_root=seeds_root,
            canonical_seed_names=(),
            residues=(),
            snapshot_sha256=empty,
        )
    try:
        root_before = seeds_root.stat(follow_symlinks=False)
    except OSError as exc:
        raise AriadneSeedTreeError(
            "ARIADNE seeds root is unreadable: " + str(seeds_root)
        ) from exc
    if not stat.S_ISDIR(root_before.st_mode):
        raise AriadneSeedTreeError("ARIADNE seeds root is not a regular directory")

    canonical_names = []
    residues = []
    try:
        with os.scandir(seeds_root) as scan:
            entries = sorted(scan, key=lambda item: item.name)
    except OSError as exc:
        raise AriadneSeedTreeError(
            "ARIADNE seeds root is unreadable: " + str(seeds_root)
        ) from exc
    for entry in entries:
        child = seeds_root / entry.name
        try:
            info = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise AriadneSeedTreeError(
                "ARIADNE seed entry is unreadable: " + str(child)
            ) from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise AriadneSeedTreeError(
                "unexpected non-directory ARIADNE seed entry: " + entry.name
            )
        if entry.name.startswith("."):
            seed_id, task_id, pid = parse_ariadne_transaction_residue_name(
                entry.name
            )
            task = tasks_by_id.get(task_id)
            if task is None or int(task["seed_id"]) != seed_id:
                raise AriadneSeedTreeError(
                    "ARIADNE transaction residue does not match TASK_MAP: "
                    + entry.name
                )
            residues.append(
                _inventory_residue_tree(
                    child,
                    seed_id=seed_id,
                    array_task_id=task_id,
                    pid=pid,
                )
            )
            continue
        try:
            parse_seed_directory_name(entry.name)
        except ValueError as exc:
            raise AriadneSeedTreeError(str(exc)) from exc
        if entry.name not in tasks_by_name:
            raise AriadneSeedTreeError(
                "ARIADNE seed directory is outside TASK_MAP: " + entry.name
            )
        canonical_names.append(entry.name)

    try:
        root_after = seeds_root.stat(follow_symlinks=False)
    except OSError as exc:
        raise AriadneSeedTreeError("ARIADNE seeds root disappeared") from exc
    if _stat_identity(root_before) != _stat_identity(root_after):
        raise AriadneSeedTreeError(
            "ARIADNE seeds root changed while it was inventoried"
        )
    snapshot = hashlib.sha256()
    snapshot.update(repr(_stat_identity(root_after)).encode("ascii"))
    snapshot.update(b"\n")
    for name in canonical_names:
        snapshot.update(("canonical\0" + name + "\n").encode("utf-8"))
    for residue in residues:
        snapshot.update(
            (
                "residue\0"
                + residue.name
                + "\0"
                + residue.tree_sha256
                + "\n"
            ).encode("utf-8")
        )
    return AriadneSeedTreeContext(
        iteration_dir=iter_dir,
        seeds_root=seeds_root,
        canonical_seed_names=tuple(canonical_names),
        residues=tuple(residues),
        snapshot_sha256=snapshot.hexdigest(),
    )


def require_clean_ariadne_seed_tree(
    iteration_dir: Path,
    task_map: Mapping[str, Any],
) -> AriadneSeedTreeContext:
    context = classify_ariadne_seed_tree(iteration_dir, task_map)
    if context.residues:
        names = ", ".join(residue.name for residue in context.residues)
        raise AriadneTransactionResidueError(
            "recoverable ARIADNE transaction residue requires quarantine: " + names
        )
    return context


__all__ = [
    "AriadneSeedTreeContext",
    "AriadneSeedTreeError",
    "AriadneTransactionResidue",
    "AriadneTransactionResidueError",
    "classify_ariadne_seed_tree",
    "parse_ariadne_transaction_residue_name",
    "require_clean_ariadne_seed_tree",
]
