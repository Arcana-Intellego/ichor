"""Immutable scientific contract for exact diversity selection."""
from __future__ import annotations

from typing import Any, Dict, Mapping


DIVERSITY_SELECTOR_VERSION = 1
DIVERSITY_SELECTOR_CONTRACT: Dict[str, Any] = {
    "algorithm": "exact_greedy_maximin",
    "version": DIVERSITY_SELECTOR_VERSION,
    "initial_geometry": "metric_medoid",
    "second_geometry": "farthest_from_medoid",
    "subsequent_geometry": "maximum_minimum_distance",
    "tie_break": "lowest_source_index",
    "distance_dtype": "float64",
}


def diversity_selector_contract() -> Dict[str, Any]:
    """Return an independent serialisable copy of the frozen contract."""
    return dict(DIVERSITY_SELECTOR_CONTRACT)


def selector_contract_matches(value: Any) -> bool:
    return isinstance(value, Mapping) and dict(value) == DIVERSITY_SELECTOR_CONTRACT


__all__ = [
    "DIVERSITY_SELECTOR_CONTRACT",
    "DIVERSITY_SELECTOR_VERSION",
    "diversity_selector_contract",
    "selector_contract_matches",
]
