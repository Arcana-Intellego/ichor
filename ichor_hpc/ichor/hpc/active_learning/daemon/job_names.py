"""Shared SLURM job-name formatting for live daemon phases."""
from __future__ import annotations

import hashlib
import re
from typing import Optional


_JOB_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_MAX_SLURM_JOB_NAME_LEN = 128


def live_job_name(campaign_uid: Optional[str], phase_name: str, iteration: int) -> str:
    tag = (str(campaign_uid)[:12] + "-") if campaign_uid else ""
    value = tag + str(phase_name) + "-" + str(int(iteration))
    if not _JOB_NAME_RE.fullmatch(value):
        raise ValueError("unsafe Slurm job name: " + repr(value))
    if len(value) <= _MAX_SLURM_JOB_NAME_LEN:
        return value
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    keep = _MAX_SLURM_JOB_NAME_LEN - len(digest) - 1
    return value[:keep] + "-" + digest
