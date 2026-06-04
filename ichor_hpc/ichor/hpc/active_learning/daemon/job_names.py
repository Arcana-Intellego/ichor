"""Shared SLURM job-name formatting for live daemon phases."""
from __future__ import annotations

from typing import Optional


def live_job_name(campaign_uid: Optional[str], phase_name: str, iteration: int) -> str:
    tag = (str(campaign_uid)[:8] + "-") if campaign_uid else ""
    return tag + str(phase_name) + "-" + str(int(iteration))
