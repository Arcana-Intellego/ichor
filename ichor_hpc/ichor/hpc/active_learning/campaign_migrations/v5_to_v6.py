"""Schema-v5 to schema-v6 campaign migration."""
from __future__ import annotations

import copy
from typing import Any, Dict


def _positive_int(value: Any, default: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return int(default)
    return int(out)


def migrate_v5_to_v6(data: Dict[str, Any]) -> Dict[str, Any]:
    migrated = copy.deepcopy(data)
    migrated["schema_version"] = 6

    has_initial_train = "initial_train_size" in migrated
    has_initial_val = "initial_val_size" in migrated
    initial_train = _positive_int(migrated.pop("initial_train_size", None), 250)
    initial_val = _positive_int(migrated.pop("initial_val_size", None), 50)
    initial_labelled_size = (
        initial_train + initial_val
        if has_initial_train or has_initial_val
        else 12
    )
    bootstrap = migrated.get("bootstrap")
    if not isinstance(bootstrap, dict):
        bootstrap = {}
    bootstrap.setdefault("initial_labelled_size", initial_labelled_size)
    migrated["bootstrap"] = bootstrap

    batch_sizing = migrated.pop("batch_sizing", None)
    floor = 5 if isinstance(batch_sizing, dict) else 4
    if isinstance(batch_sizing, dict):
        floor = _positive_int(batch_sizing.get("floor"), floor)
    active_batch = migrated.get("active_batch")
    if not isinstance(active_batch, dict):
        active_batch = {}
    active_batch.setdefault("final_batch_size", floor)
    migrated["active_batch"] = active_batch

    return migrated


__all__ = ["migrate_v5_to_v6"]
