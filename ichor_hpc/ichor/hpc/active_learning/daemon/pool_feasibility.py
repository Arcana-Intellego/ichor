"""Trajectory-pool feasibility checks for active-learning campaigns."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


class PoolFeasibilityError(ValueError):
    """Raised when pool.xyz cannot support the configured protocol."""


@dataclass(frozen=True)
class PoolFeasibility:
    pool_n_frames: int
    bootstrap_total_size: int
    bootstrap_pool_frame_count: int
    bootstrap_custom_count: int
    bootstrap_model_training_count: int
    excluded_pool_frame_count: int
    max_iterations: int
    n_seeds_per_iteration: int
    batch_total_size: int
    skip_training_seeds: bool
    required_pool_frames: int
    expression: str

    @property
    def reserve_after_bootstrap(self) -> int:
        return (
            int(self.pool_n_frames)
            - int(self.excluded_pool_frame_count)
            - int(self.bootstrap_pool_frame_count)
        )

    @property
    def ok(self) -> bool:
        return int(self.pool_n_frames) >= int(self.required_pool_frames)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool_n_frames": int(self.pool_n_frames),
            "bootstrap_total_size": int(self.bootstrap_total_size),
            "bootstrap_pool_frame_count": int(self.bootstrap_pool_frame_count),
            "bootstrap_custom_count": int(self.bootstrap_custom_count),
            "bootstrap_model_training_count": int(
                self.bootstrap_model_training_count
            ),
            "excluded_pool_frame_count": int(self.excluded_pool_frame_count),
            "max_iterations": int(self.max_iterations),
            "n_seeds_per_iteration": int(self.n_seeds_per_iteration),
            "batch_total_size": int(self.batch_total_size),
            "configured_seed_surplus": int(
                self.n_seeds_per_iteration - self.batch_total_size
            ),
            "skip_training_seeds": bool(self.skip_training_seeds),
            "required_pool_frames": int(self.required_pool_frames),
            "reserve_after_bootstrap": int(self.reserve_after_bootstrap),
            "expression": str(self.expression),
            "ok": bool(self.ok),
        }


def _pool_frame_count(campaign_dir: str | Path) -> int:
    from ..acquisition.trajectory_pool import TrajectoryPool

    return int(TrajectoryPool.load(campaign_dir).n_frames())


def evaluate_pool_feasibility(
    campaign_dir: str | Path,
    config: Any,
) -> PoolFeasibility:
    from ..custom_bootstrap import (
        custom_bootstrap_manifest_path,
        read_custom_bootstrap_manifest,
    )

    pool_n = _pool_frame_count(campaign_dir)
    bootstrap_manifest_path = custom_bootstrap_manifest_path(campaign_dir)
    if bootstrap_manifest_path.is_file() and not bootstrap_manifest_path.is_symlink():
        manifest = read_custom_bootstrap_manifest(campaign_dir)
    else:
        if bool(config.campaign.custom_bootstrap):
            raise FileNotFoundError(
                "confirmed custom bootstrap manifest is missing: "
                + str(bootstrap_manifest_path)
            )
        configured = {
            "train": int(config.point_allocation.bootstrap_training_size),
            "int_val": int(
                config.point_allocation.bootstrap_internal_validation_size
            ),
            "ext_val": int(
                config.point_allocation.bootstrap_external_validation_size
            ),
        }
        manifest = {
            "effective_qm_targets": configured,
            "polus_deficits": configured,
            "supplied_counts": {split: 0 for split in configured},
            "excluded_pool_frame_ids": [],
            "model": None,
        }
    return evaluate_pool_feasibility_manifest(pool_n, config, manifest)


def evaluate_pool_feasibility_manifest(
    pool_n: int,
    config: Any,
    manifest: Any,
) -> PoolFeasibility:
    """Evaluate one confirmed or in-memory bootstrap manifest payload."""
    if not isinstance(manifest, dict):
        raise PoolFeasibilityError("bootstrap feasibility manifest must be an object")
    effective_targets = dict(manifest.get("effective_qm_targets") or {})
    bootstrap_n = sum(
        int(effective_targets.get(split, 0))
        for split in ("train", "int_val", "ext_val")
    )
    bootstrap_pool_n = sum(
        int(value) for value in dict(manifest.get("polus_deficits") or {}).values()
    )
    custom_n = sum(
        int(value) for value in dict(manifest.get("supplied_counts") or {}).values()
    )
    model_n = int((manifest.get("model") or {}).get("training_count", 0))
    excluded_n = len(set(int(value) for value in manifest.get("excluded_pool_frame_ids", [])))
    max_iterations = int(config.campaign.max_iterations)
    n_seeds = int(config.seed_selection.n_seeds_per_iteration)
    batch_total = int(config.point_allocation.batch_total_size)
    skip_training = bool(config.anti_overlap.skip_training_seeds)
    if skip_training:
        required = excluded_n + bootstrap_pool_n + max_iterations * n_seeds
        expression = (
            "excluded custom/model pool matches + bootstrap POLUS top-ups + "
            "max_iterations * seed_selection.n_seeds_per_iteration = "
            + str(excluded_n)
            + " + "
            + str(bootstrap_pool_n)
            + " + "
            + str(max_iterations)
            + " * "
            + str(n_seeds)
            + " = "
            + str(required)
        )
    else:
        required = excluded_n + bootstrap_pool_n
        expression = (
            "excluded custom/model pool matches + bootstrap POLUS top-ups = "
            + str(excluded_n)
            + " + "
            + str(bootstrap_pool_n)
            + " (anti_overlap.skip_training_seeds=false)"
        )
    return PoolFeasibility(
        pool_n_frames=pool_n,
        bootstrap_total_size=bootstrap_n,
        bootstrap_pool_frame_count=bootstrap_pool_n,
        bootstrap_custom_count=custom_n,
        bootstrap_model_training_count=model_n,
        excluded_pool_frame_count=excluded_n,
        max_iterations=max_iterations,
        n_seeds_per_iteration=n_seeds,
        batch_total_size=batch_total,
        skip_training_seeds=skip_training,
        required_pool_frames=required,
        expression=expression,
    )


def format_pool_feasibility_failure(result: PoolFeasibility) -> str:
    return (
        "pool_feasibility_failed: pool_n_frames="
        + str(int(result.pool_n_frames))
        + ", required_pool_frames="
        + str(int(result.required_pool_frames))
        + "; "
        + str(result.expression)
    )


def require_pool_feasibility(
    campaign_dir: str | Path,
    config: Any,
) -> PoolFeasibility:
    result = evaluate_pool_feasibility(campaign_dir, config)
    if not result.ok:
        raise PoolFeasibilityError(format_pool_feasibility_failure(result))
    return result


def try_evaluate_pool_feasibility(
    campaign_dir: str | Path,
    config: Any,
) -> tuple[Optional[PoolFeasibility], Optional[str]]:
    try:
        return evaluate_pool_feasibility(campaign_dir, config), None
    except Exception as exc:
        return None, type(exc).__name__ + ": " + str(exc)


__all__ = [
    "PoolFeasibility",
    "PoolFeasibilityError",
    "evaluate_pool_feasibility",
    "evaluate_pool_feasibility_manifest",
    "format_pool_feasibility_failure",
    "require_pool_feasibility",
    "try_evaluate_pool_feasibility",
]
