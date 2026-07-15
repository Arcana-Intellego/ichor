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
from ichor.hpc.active_learning.versioning.manifest import sha256_file


def synthetic_quantum_quality_record(pointdir_name: str) -> dict:
    return {
        "pointdir": str(pointdir_name),
        "accepted": True,
        "reasons": [],
        "atom_count": 1,
        "expected_atom_names": ["H1"],
        "n_int": 1,
        "sum_iqa_ha": -0.5,
        "wfn_total_energy_ha": -0.5,
        "wfn_virial_ratio": 2.0,
        "iqa_energy_recovery_error_ha": 0.0,
        "max_abs_integration_error": 0.0,
        "per_atom": [
            {
                "atom": "H1",
                "int_file": "h1.int",
                "canonical_dft_model": "B3LYP",
                "iqa_ha": -0.5,
                "integration_error": 0.0,
                "multipoles": {name: 0.0 for name in multipole_names},
                "reasons": [],
            }
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
        "input.wfn",
        "input.gau",
        "AIMALL_TASK.json",
        "GAUSSIAN_TASK_RECEIPT.json",
        "WFN_METHOD_RECEIPT.json",
        "AIMALL_COMPLETION_RECEIPT.json",
    ):
        path = root / filename
        if not path.exists():
            path.write_text("fixture\n", encoding="utf-8")
    atomic_dir = root / "input_atomicfiles"
    atomic_dir.mkdir(exist_ok=True)
    int_path = atomic_dir / "h1.int"
    if not int_path.exists():
        int_path.write_text("fixture\n", encoding="utf-8")
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
    "synthetic_quantum_quality_record",
]
