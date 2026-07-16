"""Shared active-learning campaign selection helpers.

The menu starts with the global path set to the process cwd. Treating that
implicit default as a campaign path is unsafe because daemon commands can then
operate on whichever directory happened to launch ``ichor-cli``. This module is
the single guard for menu dispatchers that need a real selected campaign.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import ichor.cli.global_menu_variables


class CampaignSelectionError(RuntimeError):
    """Raised when a menu action is attempted without a usable campaign."""


_INITIAL_DEFAULT_PATH = Path("").absolute()
_explicit_selection = False


def _current_global_campaign_path() -> Path:
    return Path(
        ichor.cli.global_menu_variables.SELECTED_ACTIVE_LEARNING_CAMPAIGN_DIRECTORY_PATH
    ).expanduser().absolute()


def is_explicit_campaign_selection() -> bool:
    return _explicit_selection


def is_campaign_directory(path: str | Path) -> bool:
    candidate = Path(path).expanduser().absolute()
    return (
        candidate.exists()
        and candidate.is_dir()
        and (candidate / "campaign.yaml").is_file()
    )


def adopt_initial_campaign_dir_if_valid() -> Optional[Path]:
    """Make a valid launch-cwd campaign an explicit menu selection.

    The top-level menu may inherit the process cwd through the legacy global.
    That path is safe to adopt only when it already contains campaign.yaml.
    First-run campaigns do not need .DATA/ACTIVE_LEARNING yet.
    """
    if _explicit_selection:
        return selected_campaign_dir_or_none()
    candidate = _current_global_campaign_path()
    if not is_campaign_directory(candidate):
        return None
    return set_selected_campaign_dir(candidate)


def set_selected_campaign_dir(path: str | Path) -> Path:
    """Record a user-selected campaign path and mirror the legacy global."""
    global _explicit_selection
    selected = Path(path).expanduser().absolute()
    ichor.cli.global_menu_variables.SELECTED_ACTIVE_LEARNING_CAMPAIGN_DIRECTORY_PATH = selected
    _explicit_selection = True
    return selected


def selected_campaign_dir() -> Path:
    """Return the selected campaign directory or raise a fail-closed error."""
    path = _current_global_campaign_path()
    if not _explicit_selection and path == _INITIAL_DEFAULT_PATH:
        raise CampaignSelectionError(
            "No active-learning campaign directory has been selected. "
            "Use 'Select / switch campaign directory' first."
        )
    if not path.exists():
        raise CampaignSelectionError("Campaign directory does not exist: " + str(path))
    if not path.is_dir():
        raise CampaignSelectionError("Campaign path is not a directory: " + str(path))
    return path


def selected_campaign_dir_or_none() -> Optional[Path]:
    try:
        return selected_campaign_dir()
    except CampaignSelectionError:
        return None


def campaign_dir_ns(**kwargs) -> argparse.Namespace:
    """Build an argparse Namespace rooted at the selected campaign directory."""
    return argparse.Namespace(campaign_dir=str(selected_campaign_dir()), **kwargs)


def print_campaign_selection_error(exc: CampaignSelectionError) -> None:
    print(str(exc))
