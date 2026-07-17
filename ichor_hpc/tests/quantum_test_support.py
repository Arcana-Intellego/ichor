"""Shared strict quantum-evidence fixtures for active-learning tests."""

from __future__ import annotations

import json
from pathlib import Path

from ichor.core.common.constants import multipole_names
from ichor.hpc.active_learning.daemon.quantum_acceptance_receipts import (
    write_quantum_acceptance_receipt,
)
from ichor.hpc.active_learning.daemon.quantum_quality import (
    write_quantum_quality_manifest,
)
from ichor.hpc.active_learning.daemon.submission_intent import (
    mark_submitted,
    write_pre_submit_intent,
)
from ichor.hpc.active_learning.versioning.manifest import sha256_file


def prepare_dry_submitted_phase(executor, state, phase) -> int:
    """Stage a dry backend phase with the same intent order as the daemon."""
    phase_name = phase.value if hasattr(phase, "value") else str(phase)
    iteration = int(getattr(state, "iteration", 0))
    expected_tasks = int(
        executor._prepare_dry_submission(state, phase_name)
    )
    write_pre_submit_intent(
        executor.campaign_dir,
        campaign_uid=str(getattr(state, "campaign_uid")),
        phase_name=phase_name,
        iteration=iteration,
        replacement_round=int(getattr(state, "replacement_round", 0)),
        expected_tasks=expected_tasks,
        scheduler_identity_kind="synthetic",
    )
    mark_submitted(
        executor.campaign_dir,
        phase_name,
        iteration,
        "DRYRUN-" + phase_name + "-" + str(iteration),
        expected_tasks=expected_tasks,
    )
    return expected_tasks


def synthetic_quantum_quality_record(pointdir_name: str) -> dict:
    atom_iqa = {"O1": -0.6, "H2": -0.2, "H3": -0.2}
    return {
        "pointdir": str(pointdir_name),
        "accepted": True,
        "reasons": [],
        "atom_count": 3,
        "expected_atom_names": list(atom_iqa),
        "n_int": 3,
        "sum_iqa_ha": -1.0,
        "wfn_total_energy_ha": -1.0,
        "wfn_virial_ratio": 2.0,
        "iqa_energy_recovery_error_ha": 0.0,
        "max_abs_integration_error": 0.0,
        "per_atom": [
            {
                "atom": atom_name,
                "int_file": atom_name.lower() + ".int",
                "dft_model": "B3LYP",
                "canonical_dft_model": "B3LYP",
                "iqa_ha": iqa_ha,
                "integration_error": 0.0,
                "multipoles": {name: 0.0 for name in multipole_names},
                "reasons": [],
            }
            for atom_name, iqa_ha in atom_iqa.items()
        ],
    }


def attach_synthetic_quantum_acceptance(
    campaign: Path,
    pointdir: Path,
    *,
    phase_name: str,
    iteration: int,
    quality_manifest: Path,
    quality_record: dict,
) -> dict:
    root = Path(pointdir)
    for filename in (
        "input.gjf",
        "input.gau",
        "AIMALL_TASK.json",
        "GAUSSIAN_TASK_RECEIPT.json",
        "WFN_METHOD_RECEIPT.json",
        "AIMALL_COMPLETION_RECEIPT.json",
    ):
        path = root / filename
        if not path.exists():
            path.write_text("fixture\n", encoding="utf-8")
    from ichor.hpc.active_learning.daemon.dry_run_executor import (
        DryRunPhaseExecutor,
    )
    from ichor.core.atoms import Atom, Atoms

    DryRunPhaseExecutor._write_dry_quantum_wfn(
        root / "input.wfn",
        method="B3LYP",
        atoms=Atoms(
            [
                Atom("O", 0.0, 0.0, 0.0),
                Atom("H", 0.95, 0.0, 0.0),
                Atom("H", -0.24, 0.92, 0.0),
            ]
        ),
        total_energy_ha=float(quality_record["wfn_total_energy_ha"]),
    )
    atomic_dir = root / "input_atomicfiles"
    atomic_dir.mkdir(exist_ok=True)
    for atom_record in quality_record["per_atom"]:
        DryRunPhaseExecutor._write_dry_aimall_int(
            atomic_dir / str(atom_record["int_file"]),
            atom_name=str(atom_record["atom"]),
            method="B3LYP",
            iqa_ha=float(atom_record["iqa_ha"]),
            multipole_names=multipole_names,
        )
    # Schema-9 reference commits require the immutable feature contract that
    # production freezes while staging INITIAL_AIMALL.  Older focused fixtures
    # bypass that staging path, so create the same contract here before sealing
    # the accepted pointdir.
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.ferebus_row_cache import (
        ensure_feature_contract,
        produce_task_row_shard,
    )

    config_path = Path(campaign) / "campaign.yaml"
    config = (
        CampaignConfig.from_yaml(config_path)
        if config_path.is_file()
        else CampaignConfig()
    )
    try:
        from ichor.core.files import GJF

        _ = GJF(root / "input.gjf").atoms
    except Exception:
        (root / "input.gjf").write_text(
            "# B3LYP/aug-cc-pVTZ output=wfn\n\n"
            "Synthetic quantum fixture\n\n"
            "0 1\n"
            "O 0.000000 0.000000 0.000000\n"
            "H 0.950000 0.000000 0.000000\n"
            "H -0.240000 0.920000 0.000000\n\n"
            "input.wfn\n\n",
            encoding="utf-8",
            newline="\n",
        )
    contract = ensure_feature_contract(campaign, config, root)
    task_path = root / "AIMALL_TASK.json"
    try:
        task_payload = json.loads(task_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        task_payload = None
    if not isinstance(task_payload, dict) or not isinstance(
        task_payload.get("ferebus_row_shard"), dict
    ):
        shard_dir = (
            root.parent
            / ".ferebus-row-shards"
            / root.name
        )
        task_payload = {
            "schema_version": 2,
            "ferebus_feature_contract": {
                "path": (
                    Path(campaign)
                    / ".DATA"
                    / "ACTIVE_LEARNING"
                    / "FEREBUS_FEATURE_CONTRACT.json"
                ).resolve().relative_to(Path(campaign).resolve()).as_posix(),
                "contract_sha256": str(contract["contract_sha256"]),
            },
            "ferebus_row_shard": {
                "directory": shard_dir.resolve()
                .relative_to(Path(campaign).resolve())
                .as_posix(),
            },
        }
        task_path.write_text(
            json.dumps(task_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    produce_task_row_shard(campaign, root)
    receipt_path = write_quantum_acceptance_receipt(
        campaign,
        root,
        phase_name=phase_name,
        iteration=int(iteration),
        quality_manifest=quality_manifest,
        quality_record=quality_record,
    )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    return {
        "quantum_acceptance_receipt": receipt_path.resolve()
        .relative_to(Path(campaign).resolve())
        .as_posix(),
        "quantum_acceptance_receipt_sha256": sha256_file(receipt_path),
        "accepted_pointdir_content_sha256": str(payload["content_sha256"]),
    }


def attach_synthetic_quantum_batch(
    campaign: Path,
    results: list[dict],
    *,
    phase_name: str,
    iteration: int,
) -> Path:
    """Attach one strict synthetic quality manifest and receipts to test results."""
    if not results:
        raise ValueError("synthetic quantum batch must not be empty")
    pointdirs = [Path(str(result["pointdir"])) for result in results]
    staging_roots = {pointdir.parent.resolve() for pointdir in pointdirs}
    if len(staging_roots) != 1:
        raise ValueError("synthetic quantum batch pointdirs must share one staging root")
    records = []
    for result, pointdir in zip(results, pointdirs):
        if result.get("accepted") is True:
            record = synthetic_quantum_quality_record(pointdir.name)
        else:
            record = {
                "pointdir": pointdir.name,
                "accepted": False,
                "reasons": [str(result.get("reason") or "synthetic_rejection")],
                "atom_count": None,
                "expected_atom_names": [],
                "n_int": 0,
                "per_atom": [],
            }
        records.append(record)
    quality_path = write_quantum_quality_manifest(
        pointdirs[0].parent,
        phase_name=str(phase_name),
        iteration=int(iteration),
        records=records,
        gates={},
    )
    for result, pointdir, record in zip(results, pointdirs, records):
        result["quality_manifest"] = str(quality_path.resolve())
        if result.get("accepted") is True:
            result.update(
                attach_synthetic_quantum_acceptance(
                    campaign,
                    pointdir,
                    phase_name=str(phase_name),
                    iteration=int(iteration),
                    quality_manifest=quality_path,
                    quality_record=record,
                )
            )
    return quality_path


__all__ = [
    "attach_synthetic_quantum_acceptance",
    "attach_synthetic_quantum_batch",
    "prepare_dry_submitted_phase",
    "synthetic_quantum_quality_record",
]
