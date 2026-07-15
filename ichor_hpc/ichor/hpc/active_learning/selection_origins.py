"""Canonical seed-selection origin values shared by all handoff layers."""

SEED_SELECTION_ORIGINS = frozenset(
    {
        "bulk",
        "variance",
        "d_optimal",
        "d_optimal_backfill",
    }
)

SEED_SELECTION_READER_ORIGINS = SEED_SELECTION_ORIGINS | {"unknown"}


__all__ = ["SEED_SELECTION_ORIGINS", "SEED_SELECTION_READER_ORIGINS"]
