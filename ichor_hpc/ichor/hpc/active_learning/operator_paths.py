"""Resolve operator-supplied campaign input paths consistently."""
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
    return path.resolve(strict=False)


__all__ = ["resolve_campaign_input_path"]
