"""Campaign schema version gate.

Schema 11 is an intentional clean break in the bootstrap configuration.  Pool
and bootstrap inputs now use fixed campaign-relative names and are admitted by
``ichor-al-daemon init`` only after inspection and operator confirmation.
Older files are rejected rather than silently rewritten.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Mapping


CURRENT_SCHEMA_VERSION = 11


class CampaignMigrationError(ValueError):
    """Raised when a campaign payload cannot be migrated safely."""


MIGRATORS: Dict[int, Any] = {}


def migrate_campaign_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deep-copied payload upgraded to the current schema."""
    if not isinstance(payload, Mapping):
        raise CampaignMigrationError("campaign.yaml must be a mapping at the top level")
    data: Dict[str, Any] = copy.deepcopy(dict(payload))
    try:
        version = int(data.get("schema_version", -1))
    except (TypeError, ValueError) as exc:
        raise CampaignMigrationError("campaign.yaml schema_version must be an integer") from exc
    if version != CURRENT_SCHEMA_VERSION:
        raise CampaignMigrationError(
            "campaign.yaml schema_version "
            + str(version)
            + " is unsupported; this release requires schema_version "
            + str(CURRENT_SCHEMA_VERSION)
            + " with campaign.custom_bootstrap and fixed campaign-relative "
            + "pool/bootstrap inputs"
        )
    return data


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "CampaignMigrationError",
    "MIGRATORS",
    "migrate_campaign_payload",
]
