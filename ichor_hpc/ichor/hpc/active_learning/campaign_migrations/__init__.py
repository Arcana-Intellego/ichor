"""Campaign schema migration registry.

Campaign files are upgraded at the configuration-loading boundary so the
daemon, CLI, and resource solver only need to understand the current canonical
schema shape.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Mapping

from .v2_to_v3 import migrate_v2_to_v3
from .v3_to_v4 import migrate_v3_to_v4
from .v4_to_v5 import migrate_v4_to_v5
from .v5_to_v6 import migrate_v5_to_v6
from .v7_to_v8 import migrate_v7_to_v8


CURRENT_SCHEMA_VERSION = 8


class CampaignMigrationError(ValueError):
    """Raised when a campaign payload cannot be migrated safely."""


MIGRATORS: Dict[int, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    2: migrate_v2_to_v3,
    3: migrate_v3_to_v4,
    4: migrate_v4_to_v5,
    5: migrate_v5_to_v6,
    7: migrate_v7_to_v8,
}


def migrate_campaign_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deep-copied payload upgraded to the current schema."""
    if not isinstance(payload, Mapping):
        raise CampaignMigrationError("campaign.yaml must be a mapping at the top level")
    data: Dict[str, Any] = copy.deepcopy(dict(payload))
    try:
        version = int(data.get("schema_version", -1))
    except (TypeError, ValueError) as exc:
        raise CampaignMigrationError("campaign.yaml schema_version must be an integer") from exc
    if version > CURRENT_SCHEMA_VERSION:
        raise CampaignMigrationError(
            "campaign.yaml schema_version "
            + str(version)
            + " is newer than supported schema "
            + str(CURRENT_SCHEMA_VERSION)
        )
    while version < CURRENT_SCHEMA_VERSION:
        migrator = MIGRATORS.get(version)
        if migrator is None:
            raise CampaignMigrationError(
                "no campaign.yaml migrator registered for schema_version "
                + str(version)
            )
        data = migrator(data)
        try:
            version = int(data.get("schema_version", -1))
        except (TypeError, ValueError) as exc:
            raise CampaignMigrationError(
                "campaign migrator produced a non-integer schema_version"
            ) from exc
    if version != CURRENT_SCHEMA_VERSION:
        raise CampaignMigrationError(
            "campaign migration ended at schema_version "
            + str(version)
            + " instead of "
            + str(CURRENT_SCHEMA_VERSION)
        )
    return data


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "CampaignMigrationError",
    "MIGRATORS",
    "migrate_campaign_payload",
]
