"""Resolve user-supplied campaign input paths consistently."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union


def resolve_campaign_input_path(
    campaign_dir: Union[str, Path],
    configured_path: Union[str, Path],
) -> Path:
    text = os.path.expandvars(str(configured_path)).strip()
    if not text:
        raise ValueError("campaign input path must be non-empty")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path(campaign_dir) / path
    return Path(os.path.abspath(os.fspath(path)))


def reject_operator_input_symlinks(path: Union[str, Path]) -> Path:
    """Reject a source file or parent directory supplied through a symlink."""
    from .daemon.filesystem import reject_symlink_components

    return reject_symlink_components(path)


__all__ = ["reject_operator_input_symlinks", "resolve_campaign_input_path"]
