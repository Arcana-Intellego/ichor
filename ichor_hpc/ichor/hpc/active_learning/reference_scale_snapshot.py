"""Immutable per-iteration reference-scale snapshots for ARIADNE tasks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .strict_json import strict_json as json
from .daemon.state import atomic_write_json


REFERENCE_SCALE_SNAPSHOT_SCHEMA_VERSION = 1
REFERENCE_SCALE_SNAPSHOT_FILENAME = "reference_scales.json"


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: Any, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ValueError(label + " must be a lowercase SHA-256 digest")
    return text


def build_reference_scale_snapshot(
    *,
    iteration: int,
    source_iteration: int,
    models_version: int,
    model_set_manifest_sha256: str,
    values: Mapping[str, Any],
) -> dict:
    from .daemon.model_contract import validate_reference_scales

    if not isinstance(values, Mapping):
        raise ValueError("reference-scale values must be an object")
    checked_values = validate_reference_scales(dict(values))
    payload = {
        "schema_version": REFERENCE_SCALE_SNAPSHOT_SCHEMA_VERSION,
        "iteration": int(iteration),
        "source_iteration": int(source_iteration),
        "models_version": int(models_version),
        "model_set_manifest_sha256": _digest(
            model_set_manifest_sha256, "reference-scale model manifest SHA"
        ),
        "values": checked_values,
        "values_sha256": _canonical_sha256(checked_values),
    }
    if payload["iteration"] < 0 or payload["source_iteration"] < 0:
        raise ValueError("reference-scale iterations must be non-negative")
    if payload["source_iteration"] > payload["iteration"]:
        raise ValueError("reference-scale source iteration is in the future")
    if payload["models_version"] < 0:
        raise ValueError("reference-scale model version must be non-negative")
    return payload


def validate_reference_scale_snapshot(
    payload: Mapping[str, Any],
    *,
    expected_iteration: int | None = None,
) -> dict:
    if not isinstance(payload, Mapping):
        raise ValueError("reference-scale snapshot must be an object")
    if payload.get("schema_version") != REFERENCE_SCALE_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported reference-scale snapshot schema")
    for field in ("iteration", "source_iteration", "models_version"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("reference-scale " + field + " must be an integer")
    checked = build_reference_scale_snapshot(
        iteration=int(payload["iteration"]),
        source_iteration=int(payload["source_iteration"]),
        models_version=int(payload["models_version"]),
        model_set_manifest_sha256=_digest(
            payload.get("model_set_manifest_sha256"),
            "reference-scale model manifest SHA",
        ),
        values=payload.get("values"),
    )
    if payload.get("values_sha256") != checked["values_sha256"]:
        raise ValueError("reference-scale values SHA mismatch")
    if expected_iteration is not None and checked["iteration"] != int(
        expected_iteration
    ):
        raise ValueError("reference-scale iteration mismatch")
    return checked


def write_reference_scale_snapshot(path: Any, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    checked = validate_reference_scale_snapshot(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        existing = read_reference_scale_snapshot(
            target,
            expected_iteration=int(checked["iteration"]),
        )
        if existing != checked:
            raise ValueError(
                "reference-scale snapshot already exists with different content"
            )
        return target
    atomic_write_json(target, checked)
    return target


def read_reference_scale_snapshot(
    path: Any, *, expected_iteration: int | None = None
) -> dict:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError("reference-scale snapshot is missing or symlinked")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("reference-scale snapshot is unreadable") from exc
    return validate_reference_scale_snapshot(
        payload, expected_iteration=expected_iteration
    )


__all__ = [
    "REFERENCE_SCALE_SNAPSHOT_FILENAME",
    "REFERENCE_SCALE_SNAPSHOT_SCHEMA_VERSION",
    "build_reference_scale_snapshot",
    "read_reference_scale_snapshot",
    "validate_reference_scale_snapshot",
    "write_reference_scale_snapshot",
]
