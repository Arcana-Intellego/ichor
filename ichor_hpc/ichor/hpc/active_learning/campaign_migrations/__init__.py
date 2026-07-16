"""Clean-break campaign schema version gate."""
from __future__ import annotations

import copy
from typing import Any, Dict, Mapping


CURRENT_SCHEMA_VERSION = 14


class CampaignMigrationError(ValueError):
    """Raised when a campaign payload cannot be migrated safely."""


MIGRATORS: Dict[int, Any] = {}


def migrate_campaign_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deep-copied payload upgraded to the current schema."""
    if not isinstance(payload, Mapping):
        raise CampaignMigrationError("campaign.yaml must be a mapping at the top level")
    data: Dict[str, Any] = copy.deepcopy(dict(payload))
    version = data.get("schema_version", -1)
    if isinstance(version, bool) or not isinstance(version, int):
        raise CampaignMigrationError("campaign.yaml schema_version must be an integer")
    if version != CURRENT_SCHEMA_VERSION:
        raise CampaignMigrationError(
            "campaign.yaml schema_version "
            + str(version)
            + " is unsupported; this release requires schema_version "
            + str(CURRENT_SCHEMA_VERSION)
            + "; schema 14 is a clean break and old campaigns must be "
            + "initialised again"
        )
    return data


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "CampaignMigrationError",
    "MIGRATORS",
    "migrate_campaign_payload",
]
