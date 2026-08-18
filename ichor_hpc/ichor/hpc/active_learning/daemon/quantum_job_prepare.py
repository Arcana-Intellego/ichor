"""Prepare one staged quantum task without permitting stale output reuse."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

from ..strict_json import load_path
from ..layout import parse_staging_pointdir_name
from .filesystem import campaign_owned_path


_GAUSSIAN_STALE_FILES = frozenset(
    {
        "AIMALL_COMPLETION_RECEIPT.json",
        "AIMALL_TASK.json",
        "GAUSSIAN_TASK_RECEIPT.json",
        "QUANTUM_ACCEPTANCE_RECEIPT.json",
        "WFN_METHOD_RECEIPT.json",
    }
)
_AIMALL_STALE_FILES = frozenset(
    {
        "AIMALL_COMPLETION_RECEIPT.json",
        "QUANTUM_ACCEPTANCE_RECEIPT.json",
    }
)
_AIMALL_STALE_PATTERNS = (
    "*.aim",
    "*.agp",
    "*.agpviz",
    "*.extout",
    "*.int",
    "*.mgp",
    "*.mgpviz",
    "*.sum",
    "*.sumviz",
)


def _reject_symlink_tree(root: Path) -> None:
    if root.is_symlink():
        raise ValueError("quantum task pointdir is symlinked: " + str(root))
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("quantum task tree contains a symlink: " + str(path))


def _remove_files(paths: Iterable[Path]) -> None:
    for path in sorted(set(paths), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise ValueError("stale quantum output is not a regular file: " + str(path))
        path.unlink()
        if path.exists():
            raise OSError("failed to remove stale quantum output: " + str(path))


def prepare_quantum_task(
    campaign_dir: Path,
    pointdir: Path,
    *,
    backend: str,
) -> None:
    """Remove outputs that could otherwise be mistaken for this attempt's work."""
    campaign = Path(campaign_dir)
    root = campaign_owned_path(campaign, pointdir)
    parse_staging_pointdir_name(root.name)
    if not root.is_dir():
        raise FileNotFoundError("quantum task pointdir is missing: " + str(root))
    _reject_symlink_tree(root)

    selected_backend = str(backend).strip().lower()
    if selected_backend not in {"gaussian", "aimall"}:
        raise ValueError("quantum task backend must be gaussian or aimall")

    atomic_directories = [
        child
        for child in root.iterdir()
        if child.is_dir() and child.name.endswith("_atomicfiles")
    ]
    for directory in sorted(atomic_directories, key=lambda item: item.name):
        shutil.rmtree(directory, ignore_errors=False)
        if directory.exists():
            raise OSError(
                "failed to remove stale AIMAll output directory: " + str(directory)
            )

    stale_names = (
        _GAUSSIAN_STALE_FILES
        if selected_backend == "gaussian"
        else _AIMALL_STALE_FILES
    )
    stale_files = [root / name for name in stale_names if (root / name).exists()]
    if selected_backend == "gaussian":
        stale_files.extend(root.glob("*.gau"))
        stale_files.extend(root.glob("*.gaussianoutput"))
        stale_files.extend(root.glob("*.wfn"))
    for pattern in _AIMALL_STALE_PATTERNS:
        stale_files.extend(root.glob(pattern))
    _remove_files(stale_files)
    if selected_backend == "aimall":
        task_path = root / "AIMALL_TASK.json"
        if task_path.is_file() and not task_path.is_symlink():
            task = load_path(task_path)
            binding = task.get("ferebus_row_shard") if isinstance(task, dict) else None
            if isinstance(binding, dict):
                relative = Path(str(binding.get("directory") or ""))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("FEREBUS row-shard directory escapes the campaign")
                shard = campaign_owned_path(campaign, campaign / relative)
                if shard.exists():
                    if shard.is_symlink() or not shard.is_dir():
                        raise ValueError("stale FEREBUS row shard is not a regular directory")
                    shutil.rmtree(shard)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--pointdir", required=True)
    parser.add_argument("--backend", required=True, choices=("gaussian", "aimall"))
    args = parser.parse_args(argv)
    prepare_quantum_task(
        Path(args.campaign_dir),
        Path(args.pointdir),
        backend=str(args.backend),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through scheduler scripts
    raise SystemExit(main())


__all__ = ["prepare_quantum_task"]
