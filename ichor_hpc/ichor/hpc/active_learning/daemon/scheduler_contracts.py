"""Shared scheduler task-count contracts for daemon and reconcile."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional, Union

from ..layout import active_iteration_dir, staging_phase_dir, trained_models_dir
from .state import CampaignPhase


def _count_nonempty_lines(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        count = sum(1 for line in handle if line.strip())
    return count if count > 0 else None


def infer_expected_tasks_from_artifacts(
    campaign_dir: Union[str, Path],
    *,
    phase: Union[str, CampaignPhase],
    iteration: int,
    replacement_round: int = 0,
    on_error: Optional[Callable[[Exception], None]] = None,
) -> Optional[int]:
    campaign = Path(campaign_dir)
    phase_name = phase.value if isinstance(phase, CampaignPhase) else str(phase)
    if phase_name in {"PHASE_A_POLUS", "PHASE_B_POLUS"}:
        return 1
    try:
        if phase_name in {
            "INITIAL_REPLACEMENT_GAUSSIAN",
            "INITIAL_REPLACEMENT_AIMALL",
            "REPLACEMENT_GAUSSIAN",
            "REPLACEMENT_AIMALL",
        }:
            from ..replacement_sampling import (
                read_replacement_sample_strict,
                replacement_round_dir,
            )

            context = "bootstrap" if phase_name.startswith("INITIAL_") else "active"
            effective_iteration = 0 if context == "bootstrap" else int(iteration)
            manifest = read_replacement_sample_strict(
                campaign,
                context=context,
                iteration=effective_iteration,
                replacement_round=int(replacement_round),
            )
            round_dir = replacement_round_dir(
                campaign,
                context=context,
                iteration=effective_iteration,
                replacement_round=int(replacement_round),
            )
            staged_count = _count_nonempty_lines(round_dir / "POINTS.txt")
            manifest_count = int(manifest.get("n_candidates", 0))
            if staged_count != manifest_count or manifest_count <= 0:
                raise ValueError(
                    "replacement POINTS.txt count does not match its strict sample manifest"
                )
            return manifest_count
        if phase_name in {"INITIAL_GAUSSIAN", "INITIAL_AIMALL"}:
            return _count_nonempty_lines(
                staging_phase_dir(campaign, phase_name, int(iteration))
                / "POINTS.txt"
            )
        if phase_name in {"GAUSSIAN", "AIMALL"}:
            return _count_nonempty_lines(
                staging_phase_dir(campaign, phase_name, int(iteration))
                / "POINTS.txt"
            )
        if phase_name in {"INITIAL_FEREBUS", "FEREBUS"}:
            manifest = trained_models_dir(campaign) / "iteration-staging" / "FEREBUS_TASKS.json"
            if manifest.is_file():
                data = json.loads(manifest.read_text(encoding="utf-8"))
                raw = data.get("n_tasks")
                if raw is not None:
                    value = int(raw)
                    if value <= 0:
                        raise ValueError("FEREBUS task manifest contains no model commands")
                else:
                    tasks = data.get("tasks")
                    if not isinstance(tasks, list) or not tasks:
                        raise ValueError("FEREBUS task manifest contains no model commands")
            else:
                return None
            # pyferebus executes all validated model commands serially inside
            # one submitted batch script.  The manifest count is therefore a
            # scientific command count, not a Slurm array task count.
            return 1
        if phase_name == "ARIADNE_ARRAY":
            from ..seed_identity import read_ariadne_task_map

            task_map = read_ariadne_task_map(
                active_iteration_dir(campaign, int(iteration)),
                expected_iteration=int(iteration),
            )
            return int(task_map["n_tasks"])
    except Exception as exc:
        if on_error is not None:
            on_error(exc)
        return None
    return None


__all__ = ["infer_expected_tasks_from_artifacts"]
