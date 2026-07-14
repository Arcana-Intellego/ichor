"""Round-trip campaign.yaml helpers for operator-owned campaign files."""
from __future__ import annotations

from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence
from uuid import uuid4

from .config import CampaignConfig, ConfigValidationError
from .daemon.state import atomic_write_text


class CampaignYamlError(ValueError):
    """Raised when campaign.yaml cannot be initialised or patched safely."""


def _yaml():
    try:
        from ruamel.yaml import YAML
    except ImportError as exc:  # pragma: no cover - exercised in deployment
        raise CampaignYamlError(
            "ruamel.yaml is required for comment-preserving campaign.yaml "
            "updates. Reinstall ichor-hpc in the active environment."
        ) from exc
    yaml = YAML(typ="rt")
    yaml.allow_duplicate_keys = False
    yaml.preserve_quotes = True
    yaml.default_flow_style = False
    return yaml


def _commented_map():
    try:
        from ruamel.yaml.comments import CommentedMap
    except ImportError as exc:  # pragma: no cover - exercised in deployment
        raise CampaignYamlError(
            "ruamel.yaml is required for comment-preserving campaign.yaml "
            "updates. Reinstall ichor-hpc in the active environment."
        ) from exc
    return CommentedMap()


def _to_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_plain(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_to_plain(child) for child in value]
    if isinstance(value, tuple):
        return [_to_plain(child) for child in value]
    return value


def load_template_yaml():
    """Load the packaged commented campaign template."""
    with (Path(__file__).with_name("templates") / "campaign.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        data = _yaml().load(handle)
    if data is None:
        data = _commented_map()
    if not isinstance(data, MutableMapping):
        raise CampaignYamlError("packaged campaign template is not a mapping")
    return data


def load_campaign_yaml(path: str | Path):
    """Load an existing campaign.yaml as a round-trip YAML mapping."""
    p = Path(path)
    if not p.is_file():
        return _commented_map()
    with open(p, "r", encoding="utf-8") as handle:
        data = _yaml().load(handle)
    if data is None:
        data = _commented_map()
    if not isinstance(data, MutableMapping):
        raise CampaignYamlError("campaign.yaml must be a mapping: " + str(p))
    return data


def merge_template_with_user(user_payload):
    """Return the template with operator-provided values overlaid."""
    merged = load_template_yaml()
    if user_payload is None:
        return merged
    if not isinstance(user_payload, MutableMapping):
        raise CampaignYamlError("campaign.yaml must be a mapping")
    _overlay_user_values(merged, user_payload)
    return merged


def _overlay_user_values(target, user_payload) -> None:
    for key, user_value in user_payload.items():
        if (
            key in target
            and isinstance(target[key], MutableMapping)
            and isinstance(user_value, Mapping)
        ):
            _overlay_user_values(target[key], user_value)
        else:
            target[key] = deepcopy(user_value)


def _dump_yaml_to_text(payload) -> str:
    stream = StringIO()
    _yaml().dump(payload, stream)
    text = stream.getvalue()
    if not text.endswith("\n"):
        text += "\n"
    return text


def write_roundtrip_yaml(path: str | Path, payload) -> None:
    atomic_write_text(path, _dump_yaml_to_text(payload))


def initialise_campaign_yaml(campaign_dir: str | Path) -> CampaignConfig:
    """Create or populate campaign.yaml from the packaged template.

    Existing operator values override template values. Comments and key order
    are preserved where possible by ruamel.yaml's round-trip loader.
    """
    campaign, config, text = prepare_campaign_yaml(campaign_dir)
    campaign.mkdir(parents=True, exist_ok=True)
    atomic_write_text(campaign / "campaign.yaml", text)
    return config


def prepare_campaign_yaml(
    campaign_dir: str | Path,
) -> tuple[Path, CampaignConfig, str]:
    """Validate the merged campaign YAML without changing campaign files."""
    campaign = Path(campaign_dir).expanduser().resolve()
    target = campaign / "campaign.yaml"
    user_payload = load_campaign_yaml(target)
    merged = merge_template_with_user(user_payload)
    config = CampaignConfig.from_dict(_to_plain(merged))
    return campaign, config, _dump_yaml_to_text(merged)


def set_path(payload, path: str, value: Any) -> None:
    parts = path.split(".")
    if not parts or any(not part for part in parts):
        raise CampaignYamlError("invalid campaign.yaml path: " + repr(path))
    cursor = payload
    for part in parts[:-1]:
        if part not in cursor or cursor[part] is None:
            cursor[part] = _commented_map()
        if not isinstance(cursor[part], MutableMapping):
            raise CampaignYamlError(
                "cannot set nested campaign.yaml path through non-mapping: "
                + path
            )
        cursor = cursor[part]
    cursor[parts[-1]] = deepcopy(value)


def patch_campaign_yaml_fields(
    target: str | Path,
    config: CampaignConfig,
    dirty_paths: Sequence[str],
) -> CampaignConfig:
    """Patch only changed config fields into an existing campaign.yaml."""
    p = Path(target)
    base = load_campaign_yaml(p)
    if not p.is_file() or not base:
        base = merge_template_with_user(base)
    config_dict = config.to_dict()
    for path in dirty_paths:
        if path == "campaign.yaml":
            continue
        set_path(base, path, _get_plain_path(config_dict, path))
    validated = CampaignConfig.from_dict(_to_plain(base))
    return validated_with_write(p, base, validated)


def validated_with_write(path: Path, payload, validated: CampaignConfig) -> CampaignConfig:
    temp = path.with_name("." + path.name + ".verify." + uuid4().hex + ".tmp")
    write_roundtrip_yaml(temp, payload)
    try:
        reloaded = CampaignConfig.from_yaml(temp)
    except Exception as exc:
        try:
            temp.unlink(missing_ok=True)
        except Exception:
            pass
        raise CampaignYamlError("campaign.yaml reload failed before write: " + str(exc)) from exc
    if reloaded.to_dict() != validated.to_dict():
        try:
            temp.unlink(missing_ok=True)
        except Exception:
            pass
        raise CampaignYamlError("campaign.yaml reloads to a different config")
    text = temp.read_text(encoding="utf-8")
    try:
        temp.unlink(missing_ok=True)
    except Exception:
        pass
    atomic_write_text(path, text)
    return reloaded


def _get_plain_path(payload: Mapping[str, Any], path: str) -> Any:
    cursor: Any = payload
    for part in path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            raise CampaignYamlError("config path not found: " + path)
        cursor = cursor[part]
    return cursor
