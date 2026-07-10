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
    bootstrap_anchor_count: int
    max_iterations: int
    n_seeds_per_iteration: int
    batch_total_size: int
    skip_training_seeds: bool
    required_pool_frames: int
    expression: str

    @property
    def reserve_after_bootstrap(self) -> int:
        return int(self.pool_n_frames) - int(self.bootstrap_pool_frame_count)

    @property
    def ok(self) -> bool:
        return int(self.pool_n_frames) >= int(self.required_pool_frames)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool_n_frames": int(self.pool_n_frames),
            "bootstrap_total_size": int(self.bootstrap_total_size),
            "bootstrap_pool_frame_count": int(self.bootstrap_pool_frame_count),
            "bootstrap_anchor_count": int(self.bootstrap_anchor_count),
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
    from ..acquisition.trajectory_pool import TrajectoryPool
    from ..bootstrap_anchor import plan_bootstrap_anchors

    pool_frames = None
    if bool(getattr(config.point_allocation, "anchor", False)):
        pool = TrajectoryPool.load(campaign_dir)
        pool_frames = pool.to_atoms_list()
        pool_n = len(pool_frames)
    else:
        pool_n = _pool_frame_count(campaign_dir)
    bootstrap_n = int(config.point_allocation.bootstrap_total_size)
    anchor_plan, _anchor_frames = plan_bootstrap_anchors(
        campaign_dir,
        config,
        pool_frames=pool_frames,
    )
    bootstrap_pool_n = int(anchor_plan.pool_total_needed)
    anchor_n = int(anchor_plan.n_anchor)
    max_iterations = int(config.campaign.max_iterations)
    n_seeds = int(config.seed_selection.n_seeds_per_iteration)
    batch_total = int(config.point_allocation.batch_total_size)
    skip_training = bool(config.anti_overlap.skip_training_seeds)
    if skip_training:
        required = bootstrap_pool_n + max_iterations * n_seeds
        expression = (
            "bootstrap pool frames after anchors + "
            "max_iterations * seed_selection.n_seeds_per_iteration = "
            + str(bootstrap_pool_n)
            + " + "
            + str(max_iterations)
            + " * "
            + str(n_seeds)
            + " = "
            + str(required)
        )
    else:
        required = bootstrap_pool_n
        expression = (
            "bootstrap pool frames after anchors = "
            + str(bootstrap_pool_n)
            + " (anti_overlap.skip_training_seeds=false)"
        )
    return PoolFeasibility(
        pool_n_frames=pool_n,
        bootstrap_total_size=bootstrap_n,
        bootstrap_pool_frame_count=bootstrap_pool_n,
        bootstrap_anchor_count=anchor_n,
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
    "format_pool_feasibility_failure",
    "require_pool_feasibility",
    "try_evaluate_pool_feasibility",
]
