"""Shared scheduler task-count contracts for daemon and reconcile."""
from __future__ import annotations

from ..strict_json import strict_json as json
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
    if phase_name in {"PHASE_A_DIVERSITY", "PHASE_B_DIVERSITY"}:
        return 1
    try:
        if phase_name in {
            "INITIAL_GAUSSIAN",
            "INITIAL_AIMALL",
            "INITIAL_REPLACEMENT_GAUSSIAN",
            "INITIAL_REPLACEMENT_AIMALL",
            "GAUSSIAN",
            "AIMALL",
            "REPLACEMENT_GAUSSIAN",
            "REPLACEMENT_AIMALL",
        }:
            from .quantum_task_contracts import (
                AIMALL_PHASES,
                gaussian_phase_for_aimall,
                quantum_task_contract,
            )

            contract = quantum_task_contract(
                campaign,
                phase_name,
                int(iteration),
                replacement_round=int(replacement_round),
                validate_points_file=False,
            )
            points_file = contract.staging_dir / "POINTS.txt"
            if points_file.exists() or points_file.is_symlink():
                from .input_staging import _points_file_names

                listed = tuple(_points_file_names(contract.staging_dir))
                permitted = {contract.pointdir_names}
                if phase_name in AIMALL_PHASES:
                    gaussian = quantum_task_contract(
                        campaign,
                        gaussian_phase_for_aimall(phase_name),
                        int(iteration),
                        replacement_round=int(replacement_round),
                        validate_points_file=False,
                    )
                    permitted.add(gaussian.pointdir_names)
                if listed not in permitted:
                    raise ValueError(
                        phase_name
                        + " POINTS.txt does not match its authoritative task set"
                    )
            return contract.logical_total if contract.logical_total > 0 else None
        if phase_name in {"INITIAL_FEREBUS", "FEREBUS"}:
            manifest = trained_models_dir(campaign) / "iteration-staging" / "FEREBUS_TASKS.json"
            if manifest.is_file():
                data = json.loads(manifest.read_text(encoding="utf-8"))
                raw = data.get("n_tasks")
                if raw is not None:
                    if isinstance(raw, bool) or not isinstance(raw, int):
                        raise ValueError(
                            "FEREBUS task manifest n_tasks must be an exact integer"
                        )
                    value = raw
                    if value <= 0:
                        raise ValueError("FEREBUS task manifest contains no model commands")
                else:
                    tasks = data.get("tasks")
                    if not isinstance(tasks, list) or not tasks:
                        raise ValueError("FEREBUS task manifest contains no model commands")
            else:
                return None
            return value if raw is not None else len(tasks)
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
