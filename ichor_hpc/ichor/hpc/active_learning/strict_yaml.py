"""Strict YAML loading for authoritative campaign configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import yaml


class StrictYamlError(ValueError):
    """Raised when YAML is ambiguous or cannot be parsed safely."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    mapping = {}
    first_marks = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise StrictYamlError(
                "YAML mapping key is not a scalar at line "
                + str(key_node.start_mark.line + 1)
            ) from exc
        if duplicate:
            first = first_marks[key]
            raise StrictYamlError(
                "duplicate YAML key "
                + repr(key)
                + " at line "
                + str(key_node.start_mark.line + 1)
                + ", column "
                + str(key_node.start_mark.column + 1)
                + "; first defined at line "
                + str(first.line + 1)
                + ", column "
                + str(first.column + 1)
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
        first_marks[key] = key_node.start_mark
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_yaml_strict(path: Union[str, Path]) -> Any:
    """Load YAML while rejecting duplicate keys at every mapping depth."""
    candidate = Path(path)
    try:
        with candidate.open("r", encoding="utf-8") as handle:
            return yaml.load(handle, Loader=_UniqueKeySafeLoader)
    except StrictYamlError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise StrictYamlError(
            "could not read strict YAML from " + str(candidate) + ": " + str(exc)
        ) from exc


__all__ = ["StrictYamlError", "load_yaml_strict"]
