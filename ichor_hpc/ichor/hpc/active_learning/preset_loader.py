"""Preset loader for ichor-al-daemon.

Three YAML files live under 'presets/':

  - balanced.yaml             (matches CampaignConfig dataclass defaults)
  - spectroscopy_focused.yaml (low-frequency / mode-frequency emphasis)
  - thermodynamics_focused.yaml (force-dominant for MD stability)

The loader reads a preset by name, deep-merges it against an existing
campaign.yaml payload (campaign.yaml wins on every explicit key), and
returns the merged dict. The CLI's --preset NAME flag uses this to
produce the effective CampaignConfig.

Important: every preset is CURRENTLY UNCALIBRATED. The values are
best-effort starting points based on heuristics; 
calibration against real water-tetramer benchmark
data lands still needed once the live executor produces end-to-end iteration output.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple


__all__ = [
    "PRESETS_DIR",
    "available_presets",
    "load_preset",
    "deep_merge",
    "apply_preset",
    "PresetError",
]


PRESETS_DIR = Path(__file__).resolve().parent / "presets"


class PresetError(ValueError):
    """Raised when a preset name is unknown or the YAML cannot be parsed."""


def available_presets() -> List[str]:
    """Return sorted preset names (file stems) found under presets/."""
    if not PRESETS_DIR.is_dir():
        return []
    return sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))


def _preset_path(name: str) -> Path:
    return PRESETS_DIR / (str(name) + ".yaml")


def load_preset(name: str) -> Dict[str, Any]:
    """Read the named preset YAML and return the parsed payload.

    Raises PresetError on missing file or bad YAML.
    """
    import yaml

    p = _preset_path(name)
    if not p.is_file():
        raise PresetError(
            "unknown preset " + repr(name) + "; available: "
            + repr(available_presets())
        )
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise PresetError(
            "preset " + repr(name) + " failed to parse: " + str(exc)
        ) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise PresetError(
            "preset " + repr(name) + " must be a YAML mapping at the top level"
        )
    return data


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-merge 'overlay' onto 'base'. Both must be plain dict-shaped.

    Rules:
      * If both 'base[k]' and 'overlay[k]' are dicts, recurse.
      * Otherwise overlay wins.

    The campaign.yaml semantic is: presets are the BASE (defaults to overlay
    onto the dataclass-default config); operator's campaign.yaml is the
    OVERLAY (their explicit edits beat the preset's choices).
    """
    result: Dict[str, Any] = copy.deepcopy(dict(base))
    for k, v in overlay.items():
        if (
            k in result
            and isinstance(result[k], Mapping)
            and isinstance(v, Mapping)
        ):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def apply_preset(preset_name: str, campaign_payload: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load 'preset_name' and overlay 'campaign_payload' on top.

    Returns '(effective_payload, preset_payload)'. The preset payload is
    returned alongside so the CLI can journal a 'preset_loaded' event
    with the preset's contents for traceability.
    """
    preset = load_preset(preset_name)
    return deep_merge(preset, campaign_payload), preset
