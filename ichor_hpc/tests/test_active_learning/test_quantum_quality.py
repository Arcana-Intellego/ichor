"""AIMAll quantum-quality sidecar tests."""
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

from ichor.core.files.point_directory import PointDirectory
from ichor.hpc.active_learning.config import QualityGatesConfigBlock
from ichor.hpc.active_learning.daemon.quantum_quality import (
    QUANTUM_QUALITY_MANIFEST,
    canonicalise_aimall_method,
    evaluate_aimall_pointdir,
    write_quantum_quality_manifest,
)
from ichor.hpc.active_learning.daemon.input_staging import (
    WFN_METHOD_RECEIPT,
    rewrite_wfn_for_aimall,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "live_outputs"


def _clean_fixture_pointdir():
    return PointDirectory(FIXTURES / "initial_quantum" / "POINT_0000.pointdir")


def test_quantum_quality_accepts_clean_aimall_fixture_with_finite_metrics():
    record = evaluate_aimall_pointdir(
        _clean_fixture_pointdir(),
        expected_method="B3LYP",
    )

    assert record["accepted"] is True
    assert record["reasons"] == []
    assert record["atom_count"] == 3
    assert record["n_int"] == 3
    assert math.isfinite(record["sum_iqa_ha"])
    assert math.isfinite(record["wfn_total_energy_ha"])
    assert math.isfinite(record["iqa_energy_recovery_error_ha"])
    assert math.isfinite(record["max_abs_integration_error"])
    assert len(record["per_atom"]) == 3
    assert record["expected_dft_model"] == "B3LYP"
    assert record["observed_dft_models"] == ["B3LYP"]


def test_aimall_method_canonicalisation_accepts_reported_spin_forms():
    assert canonicalise_aimall_method("Restricted B3LYP") == "B3LYP"
    assert canonicalise_aimall_method("Unrestricted M06-2X") == "M062X"
    assert canonicalise_aimall_method("RHF") == "HF"


def test_quantum_quality_rejects_wrong_reported_method(tmp_path):
    pointdir = SimpleNamespace(
        path=tmp_path / "WRONG.pointdir",
        atoms=[object()],
        ints=SimpleNamespace(
            ints=[
                SimpleNamespace(
                    atom_name="O1",
                    dft_model="Restricted HF",
                    iqa=-75.0,
                    integration_error=0.0,
                )
            ]
        ),
        wfn=SimpleNamespace(total_energy=-75.0),
    )
    record = evaluate_aimall_pointdir(pointdir, expected_method="B3LYP")
    assert record["accepted"] is False
    assert "dft_model_mismatch" in record["reasons"]


def test_wfn_method_rewrite_is_hash_bound_and_parseable(tmp_path):
    source = next(
        (FIXTURES / "initial_quantum" / "POINT_0000.pointdir").glob("*.wfn")
    )
    pointdir = tmp_path / "POINT_0000.pointdir"
    pointdir.mkdir()
    wfn = pointdir / "input.wfn"
    shutil.copy2(source, wfn)

    receipt_path, receipt = rewrite_wfn_for_aimall(
        wfn,
        method="B3LYP",
        phase_name="INITIAL_AIMALL",
        iteration=0,
        task_index=0,
        source_acceptance_sha256="a" * 64,
    )

    assert receipt_path.name == WFN_METHOD_RECEIPT
    assert receipt["method"] == "B3LYP"
    assert receipt["wfn"]["before_sha256"] != receipt["wfn"]["after_sha256"]
    assert wfn.read_text(encoding="utf-8").splitlines()[1].endswith("B3LYP")


def test_quantum_quality_rejects_missing_int_file(tmp_path):
    src = FIXTURES / "initial_quantum" / "POINT_0000.pointdir"
    dst = tmp_path / "POINT_0000.pointdir"
    shutil.copytree(src, dst)
    first_int = next(dst.glob("*_atomicfiles/*.int"))
    first_int.unlink()

    record = evaluate_aimall_pointdir(PointDirectory(dst))

    assert record["accepted"] is False
    assert any(reason.startswith("aimall_partial_") for reason in record["reasons"])


def test_quantum_quality_rejects_unreadable_geometry_when_required(tmp_path):
    class _BrokenAtoms:
        def __len__(self):
            raise RuntimeError("geometry parse failed")

    pointdir = SimpleNamespace(
        path=tmp_path / "BROKEN.pointdir",
        atoms=_BrokenAtoms(),
        ints=SimpleNamespace(ints=[]),
        wfn=None,
    )

    record = evaluate_aimall_pointdir(pointdir)

    assert record["accepted"] is False
    assert "aimall_geometry_unreadable" in record["reasons"]
    assert "no_int_files" in record["reasons"]


def test_quantum_quality_rejects_nonfinite_iqa_and_integration_error(tmp_path):
    pointdir = SimpleNamespace(
        path=tmp_path / "BAD.pointdir",
        atoms=[object()],
        ints=SimpleNamespace(ints=[
            SimpleNamespace(atom_name="O1", iqa=float("nan"), integration_error=float("inf")),
        ]),
        wfn=SimpleNamespace(total_energy=-75.0),
    )

    record = evaluate_aimall_pointdir(pointdir)

    assert record["accepted"] is False
    assert "iqa_missing_or_nonfinite" in record["reasons"]
    assert "integration_error_missing_or_nonfinite" in record["reasons"]


def test_quantum_quality_optional_thresholds_only_apply_when_configured():
    pdir = _clean_fixture_pointdir()
    default_record = evaluate_aimall_pointdir(pdir, QualityGatesConfigBlock())
    assert default_record["accepted"] is True

    gates = QualityGatesConfigBlock(max_abs_integration_error=1.0e-10)
    threshold_record = evaluate_aimall_pointdir(pdir, gates)
    assert threshold_record["accepted"] is False
    assert "integration_error_threshold_exceeded" in threshold_record["reasons"]


def test_write_quantum_quality_manifest_summarises_records(tmp_path):
    records = [
        {"pointdir": "POINT_0000.pointdir", "accepted": True, "reasons": []},
        {"pointdir": "POINT_0001.pointdir", "accepted": False, "reasons": ["bad"]},
    ]

    path = write_quantum_quality_manifest(
        tmp_path,
        phase_name="INITIAL_AIMALL",
        iteration=0,
        records=records,
        gates=QualityGatesConfigBlock(),
    )

    assert path.name == QUANTUM_QUALITY_MANIFEST
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["n_total"] == 2
    assert payload["n_accepted"] == 1
    assert payload["n_rejected"] == 1
