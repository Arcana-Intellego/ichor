"""Campaign-owned path and symlink invariants for daemon mutation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union


def lexical_absolute_path(path: Union[str, Path]) -> Path:
    """Return an absolute lexical path without following symlinks."""
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _portable_resolved_path(path: Path) -> Path:
    """Normalise Windows extended paths before containment comparison."""
    text = os.path.normcase(os.path.normpath(os.fspath(path.resolve(strict=False))))
    if os.name == "nt":
        if text.startswith("\\\\?\\UNC\\"):
            text = "\\\\" + text[8:]
        elif text.startswith("\\\\?\\"):
            text = text[4:]
    return Path(text)


def reject_symlink_components(
    path: Union[str, Path],
    *,
    start: Union[str, Path, None] = None,
) -> Path:
    """Reject every existing symlink component between ``start`` and ``path``."""
    target = lexical_absolute_path(path)
    boundary = None if start is None else lexical_absolute_path(start)
    if boundary is not None:
        try:
            relative = target.relative_to(boundary)
        except ValueError as exc:
            raise ValueError("path is outside the inspection boundary: " + str(path)) from exc
        components = [boundary]
        current = boundary
        for part in relative.parts:
            current = current / part
            components.append(current)
    else:
        components = []
        current = target
        while True:
            components.append(current)
            if current == current.parent:
                break
            current = current.parent
        components.reverse()

    for index, component in enumerate(components):
        if component.is_symlink():
            raise ValueError("path contains a symlink: " + str(component))
        if component.exists() and index < len(components) - 1 and not component.is_dir():
            raise ValueError("path ancestor is not a directory: " + str(component))
    return target


def campaign_owned_path(
    campaign_dir: Union[str, Path],
    path: Union[str, Path],
) -> Path:
    """Validate lexical and resolved containment without following owned links."""
    campaign = lexical_absolute_path(campaign_dir)
    raw_candidate = Path(path).expanduser()
    if raw_candidate.is_absolute():
        candidate = lexical_absolute_path(raw_candidate)
    else:
        candidate = lexical_absolute_path(campaign / raw_candidate)
    try:
        candidate.relative_to(campaign)
    except ValueError as exc:
        raise ValueError("daemon-owned path escapes campaign root: " + str(path)) from exc

    reject_symlink_components(candidate, start=campaign)
    resolved_campaign = _portable_resolved_path(campaign)
    resolved_candidate = _portable_resolved_path(candidate)
    try:
        resolved_candidate.relative_to(resolved_campaign)
    except ValueError as exc:
        raise ValueError("daemon-owned path escapes campaign root: " + str(path)) from exc
    return candidate


def operational_data_dir(campaign_dir: Union[str, Path]) -> Path:
    return campaign_owned_path(
        campaign_dir,
        Path(".DATA") / "ACTIVE_LEARNING",
    )


def operational_path(campaign_dir: Union[str, Path], *parts: str) -> Path:
    return campaign_owned_path(
        campaign_dir,
        operational_data_dir(campaign_dir).joinpath(*parts),
    )


__all__ = [
    "campaign_owned_path",
    "lexical_absolute_path",
    "operational_data_dir",
    "operational_path",
    "reject_symlink_components",
]
