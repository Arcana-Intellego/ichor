"""Campaign schema version gate.

Schema 10 is an intentional clean break in the operator configuration.  The
trajectory source, anchor source, and sampling aggressiveness now belong to
the campaign block.  Older files are rejected rather than silently rewritten.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Mapping


CURRENT_SCHEMA_VERSION = 10


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
            + " with source_path, anchor_path, and sampling_aggressiveness "
            + "under the campaign block"
        )
    return data


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "CampaignMigrationError",
    "MIGRATORS",
    "migrate_campaign_payload",
]
