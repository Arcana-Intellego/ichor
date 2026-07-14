"""Strict JSON decoding for authoritative active-learning artefacts."""
from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any, IO, Optional, Union


class StrictJSONDecodeError(_json.JSONDecodeError):
    """Raised when JSON is ambiguous or uses non-standard constants."""


class _DuplicateKey(ValueError):
    def __init__(self, key: str) -> None:
        self.key = str(key)
        super().__init__(self.key)


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(str(key))
        result[key] = value
    return result


def _reject_non_finite(value: str) -> None:
    raise StrictJSONDecodeError(
        "non-standard JSON constant " + repr(str(value)),
        str(value),
        0,
    )


def loads(
    document: Union[str, bytes, bytearray],
    *,
    source: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Any:
    """Decode unambiguous RFC-style JSON and reject duplicate keys."""
    if "object_pairs_hook" in kwargs:
        raise TypeError("strict JSON does not permit an object_pairs_hook override")
    if "parse_constant" in kwargs:
        raise TypeError("strict JSON does not permit a parse_constant override")
    try:
        return _json.loads(
            document,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_non_finite,
            **kwargs,
        )
    except _DuplicateKey as exc:
        text = (
            document.decode("utf-8", errors="replace")
            if isinstance(document, (bytes, bytearray))
            else str(document)
        )
        prefix = "" if source is None else str(source) + ": "
        raise StrictJSONDecodeError(
            prefix + "duplicate object key " + repr(exc.key),
            text,
            0,
        ) from exc
    except _json.JSONDecodeError as exc:
        if source is None or str(source) in exc.msg:
            raise
        raise StrictJSONDecodeError(
            str(source) + ": " + exc.msg,
            exc.doc,
            exc.pos,
        ) from exc


def load(handle: IO[str], **kwargs: Any) -> Any:
    """Read and strictly decode JSON from a text file object."""
    return loads(handle.read(), source=getattr(handle, "name", None), **kwargs)


def load_path(path: Union[str, Path]) -> Any:
    """Read one UTF-8 JSON file with path-aware strict diagnostics."""
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        return load(handle)


class _StrictJSONFacade:
    """Drop-in facade retaining standard encoding APIs with strict decoding."""

    JSONDecodeError = _json.JSONDecodeError
    dump = staticmethod(_json.dump)
    dumps = staticmethod(_json.dumps)
    load = staticmethod(load)
    loads = staticmethod(loads)


strict_json = _StrictJSONFacade()


__all__ = [
    "StrictJSONDecodeError",
    "load",
    "load_path",
    "loads",
    "strict_json",
]
