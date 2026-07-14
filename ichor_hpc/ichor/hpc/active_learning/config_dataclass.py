"""Generic nested block parser for dataclass-shaped YAML configs.

Used by 'CampaignConfig.from_dict' to recursively parse nested
mapping blocks against a dataclass tree. Unknown keys raise with a
path-qualified message so users see e.g.
'acquisition.weights.lambda_for: unknown' rather than
'lambda_for: unknown'.

Why not a third-party validator?
- pydantic is an alternative but would be overkill for the schema size and would add a dependency.
- dataclasses are already the canonical type for every config struct in
  the project. Walking 'fields(cls)' keeps a single source of truth.

The parser handles every type shape used in CampaignConfig:
- Primitive: int, float, str, bool.
- Optional[primitive]: None passes through; otherwise primitive rules.
- Nested dataclass: recursive call.
- Typed lists and mappings: container and element types are checked.
- General unions: each declared branch is tried without scalar coercion.
"""
from __future__ import annotations

import dataclasses
import types
import typing
from typing import Any, Dict, Mapping, Type, TypeVar


__all__ = ["parse_dataclass_block", "DataclassParseError"]


T = TypeVar("T")


class DataclassParseError(ValueError):
    """Raised on unknown keys, type mismatches, or any other parser
    failure. Carries a path-qualified message so operators can spot the
    unsupported YAML line quickly."""


def _is_optional(tp) -> bool:
    """Return True if 'tp' is 'Optional[X]' i.e. 'Union[X, None]'."""
    origin = typing.get_origin(tp)
    if origin is typing.Union:
        return type(None) in typing.get_args(tp)
    #Python 3.10+ `X | None` syntax produces a types.UnionType.
    try:
        import types
        if isinstance(tp, types.UnionType):
            return type(None) in typing.get_args(tp)
    except Exception:
        pass
    return False


def _strip_optional(tp):
    """Return the non-None branch of an Optional[X], or 'tp' unchanged."""
    if not _is_optional(tp):
        return tp
    args = [a for a in typing.get_args(tp) if a is not type(None)]
    return args[0] if len(args) == 1 else tp


def _coerce_primitive(value, target, path: str):
    """Validate a YAML scalar against the exact dataclass field type.

    bool is checked BEFORE int because bool is a subclass of int in Python
    and a naive isinstance(value, int) would let 'False' masquerade as an
    int field.
    """
    if target is bool:
        if isinstance(value, bool):
            return value
        #don't accept 0/1 silently -- that hides typos in YAML.
        raise DataclassParseError(
            path + ": expected bool, got " + type(value).__name__
        )
    if target is int:
        if isinstance(value, bool):
            raise DataclassParseError(
                path + ": expected int, got bool"
            )
        if isinstance(value, int):
            return value
        raise DataclassParseError(
            path + ": expected int, got " + type(value).__name__
        )
    if target is float:
        if isinstance(value, bool):
            raise DataclassParseError(
                path + ": expected float, got bool"
            )
        if isinstance(value, (int, float)):
            return float(value)
        raise DataclassParseError(
            path + ": expected float, got " + type(value).__name__
        )
    if target is str:
        if isinstance(value, str):
            return value
        raise DataclassParseError(
            path + ": expected str, got " + type(value).__name__
        )
    # Unknown leaf annotations are passed through for their owning block's
    # semantic validator.
    return value


def _parse_value(value: Any, target_type: Any, path: str) -> Any:
    """Parse one value without coercing between distinct YAML scalar types."""
    optional = _is_optional(target_type)
    if value is None:
        if optional:
            return None
        raise DataclassParseError(path + ": null is not allowed")

    inner = _strip_optional(target_type)
    if dataclasses.is_dataclass(inner):
        return parse_dataclass_block(inner, value, path=path)

    origin = typing.get_origin(inner)
    args = typing.get_args(inner)
    if origin in (list, typing.List):
        if not isinstance(value, list):
            raise DataclassParseError(
                path + ": expected list, got " + type(value).__name__
            )
        element_type = args[0] if args else Any
        return [
            _parse_value(item, element_type, path + "[" + str(index) + "]")
            for index, item in enumerate(value)
        ]
    if origin in (dict, typing.Dict, Mapping, typing.Mapping):
        if not isinstance(value, Mapping):
            raise DataclassParseError(
                path + ": expected mapping, got " + type(value).__name__
            )
        key_type, value_type = args if len(args) == 2 else (Any, Any)
        parsed: Dict[Any, Any] = {}
        for key, item in value.items():
            parsed_key = key if key_type is Any else _parse_value(
                key, key_type, path + ".<key>"
            )
            parsed[parsed_key] = (
                item
                if value_type is Any
                else _parse_value(item, value_type, path + "." + str(key))
            )
        return parsed
    if origin in (typing.Union, types.UnionType):
        errors = []
        for branch in args:
            if branch is type(None):
                continue
            try:
                return _parse_value(value, branch, path)
            except DataclassParseError as exc:
                errors.append(str(exc))
        raise DataclassParseError(
            path
            + ": value does not match any allowed type ("
            + "; ".join(errors)
            + ")"
        )
    if inner is Any:
        return value
    return _coerce_primitive(value, inner, path)


def parse_dataclass_block(cls: Type[T], data: Any, *, path: str = "") -> T:
    """Build an instance of dataclass 'cls' from 'data'.

    Parameters
    ----------
    cls
        A dataclass type. Fields with dataclass types are recursed into.
    data
        A mapping (dict-like). Anything else raises DataclassParseError.
    path
        Dotted path prefix for error messages. The top-level call leaves
        this empty; recursive calls pass e.g. "acquisition.weights".

    Returns
    -------
    cls
        Fully-constructed dataclass instance. Defaults are honoured for
        any keys absent from 'data'.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError("parse_dataclass_block expects a dataclass type")
    prefix = path + "." if path else ""

    if not isinstance(data, Mapping):
        raise DataclassParseError(
            (path or "<top>") + ": expected mapping, got "
            + type(data).__name__
        )

    type_hints = typing.get_type_hints(cls)
    known_fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data.keys()) - set(known_fields.keys())
    if unknown:
        raise DataclassParseError(
            (path or "<top>") + ": unknown keys "
            + repr(sorted(unknown))
        )

    kwargs: Dict[str, Any] = {}
    for name, field in known_fields.items():
        if name not in data:
            continue
        value = data[name]
        target_type = type_hints.get(name, field.type)
        kwargs[name] = _parse_value(value, target_type, prefix + name)

    return cls(**kwargs)
