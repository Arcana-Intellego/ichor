from __future__ import annotations

from pathlib import Path

from ichor.hpc.active_learning.daemon import array_recovery
from ichor.hpc.active_learning.daemon.live_executor import build_sbatch_script
from ichor.hpc.active_learning.config import CampaignConfig


def _write_points(campaign: Path, phase: str, iteration: int, n: int) -> None:
    bucket = "initial" if phase.startswith("INITIAL_") else "iter_" + str(iteration)
    staging = campaign / ".DATA" / "STAGING" / bucket
    staging.mkdir(parents=True)
    pointdirs = []
    for idx in range(n):
        pointdir = staging / ("POINT_" + str(idx).zfill(4) + ".pointdir")
        pointdir.mkdir()
        pointdirs.append(pointdir)
    (staging / "POINTS.txt").write_text(
        "\n".join(str(path.resolve()) for path in pointdirs) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def test_prepare_retry_submission_writes_only_incomplete_task_ids(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    phase = "INITIAL_GAUSSIAN"
    _write_points(campaign, phase, 0, 4)

    def fake_validate(_campaign, _phase, _iteration, task_id):
        return (int(task_id) in {0, 2}, "" if int(task_id) in {0, 2} else "missing", "out")

    monkeypatch.setattr(array_recovery, "_validate_task", fake_validate)

    payload = array_recovery.prepare_retry_submission(campaign, phase, 0)

    assert payload["logical_total"] == 4
    assert payload["n_reuse"] == 2
    assert payload["retry_task_ids"] == [1, 3]
    retry_file = Path(payload["retry_task_file"])
    assert retry_file.read_text(encoding="utf-8").splitlines() == ["1", "3"]


def test_force_resubmit_array_ignores_reusable_outputs(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    phase = "GAUSSIAN"
    _write_points(campaign, phase, 7, 3)

    monkeypatch.setattr(
        array_recovery,
        "_validate_task",
        lambda _campaign, _phase, _iteration, task_id: (True, "", "out"),
    )

    payload = array_recovery.prepare_retry_submission(
        campaign,
        phase,
        7,
        force_resubmit=True,
    )

    assert payload["force_resubmit"] is True
    assert payload["n_reuse"] == 0
    assert payload["retry_task_ids"] == [0, 1, 2]


def test_sbatch_retry_map_uses_logical_task_id_for_ariadne(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    config = CampaignConfig()
    retry_map = campaign / ".DATA" / "ACTIVE_LEARNING" / "RETRY_TASKS.ARIADNE_ARRAY.0002.txt"
    retry_map.parent.mkdir(parents=True)
    retry_map.write_text("4\n9\n", encoding="utf-8", newline="\n")

    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.live_executor._configured_scheduler",
        lambda: "slurm",
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.live_executor._configured_max_array_task_id",
        lambda: None,
    )

    script = build_sbatch_script(
        phase_name="ARIADNE_ARRAY",
        iteration=2,
        campaign_dir=campaign,
        config=config,
        array_size=2,
        array_task_map=retry_map,
    )

    assert "ICHOR_LOGICAL_ARRAY_TASK_ID" in script
    assert "--seed-index $ICHOR_LOGICAL_ARRAY_TASK_ID" in script
    assert "#SBATCH --array=0-1" in script
