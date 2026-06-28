"""LiveBackendsPhaseExecutor -- the configured Slurm deployment target.

Inherits the file system layout management from :class: DryRunPhaseExecutor
but overrides the SBATCH phase submission to:

    1. Write a real submission script with SLURM directives + module loads +
       the actual backend invocation (Gaussian / AIMAll / FEREBUS / ARIADNE).
    2. Shell out to "sbatch --parsable" via subprocess.run and capture the
       returned JobID.
    3. On terminal sacct outcome, parse the real output files into the
       canonical ICHOR data structures (PointsDirectory, Models, etc.).

The per-phase output parsers below are real: they read the Gaussian /
AIMAll / FEREBUS / ARIADNE / POLUS results into the canonical ICHOR data
structures.

The wiring is verified via the smoke tests ('pytest -m live'), which
skip cleanly when the required binaries are absent. 
On configured Slurm clusters they exercise sbatch + sacct against tiny one-shot
jobs that take seconds, not hours.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..config import (
    CampaignConfig,
    VALID_AIMALL_BOAQ_VALUES,
    VALID_AIMALL_IASMESH_VALUES,
)
from ..versioning.provenance import (
    PROVENANCE_FILENAME,
    append_to_index,
    ensure_index,
    enrich_with_anti_overlap,
    enrich_with_ariadne,
    enrich_with_error_calibration_input,
    enrich_with_phase_b,
    validate_provenance,
    write_seed_provenance,
)
from .dry_run_executor import (
    DryRunPhaseExecutor,
    anti_overlap_whitened_distance_bounds,
)
from .phase_executor import (
    BackendSubmissionError,
    FailureAction,
    INLINE_PHASES,
    PhaseResult,
    SBATCH_PHASES,
)
from .resource_solver import (
    ResolvedPhaseResources,
    gaussian_mdef_gb,
    resolve_phase_resources,
    validate_gaussian_link0_memory,
)
from .preflight import BackendAvailability, check_backends, missing_backend_message
from .cluster_profile import (
    active_machine,
    expanded_profile_value,
    profile_value,
)
from .state import CampaignPhase, atomic_write_json
from .job_names import live_job_name


__all__ = [
    "LiveBackendsPhaseExecutor",
    "LiveBackendNotAvailableError",
    "build_sbatch_script",
    "live_job_name",
    "make_live_job_finder",
    "make_live_job_accounting_finder",
    "make_live_job_liveness_checker",
    "LIVE_POSTPROCESS_IMPLEMENTED",
]

DEFAULT_DAEMON_PYTHON_MODULES: List[str] = [
    "python/3.11.3-gcccore-12.3.0",
]

DEFAULT_DAEMON_ARIADNE_RUNTIME_MODULES: List[str] = [
    "compilers/oneapi/2024.2.0",
    "compiler-rt tbb compiler",
    "mkl/2024.2",
]

DEFAULT_DAEMON_RUNTIME_MODULES: List[str] = (
    DEFAULT_DAEMON_PYTHON_MODULES + DEFAULT_DAEMON_ARIADNE_RUNTIME_MODULES
)

_MODULE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]*(?: [A-Za-z0-9][A-Za-z0-9_.:/+-]*)*$")
_SHEBANG_RE = re.compile(r"^#![A-Za-z0-9_./ -]+$")
_SHELL_PATH_FRAGMENT_RE = re.compile(r"^[A-Za-z0-9_./${}:+-]+$")


def _iteration_active_learning_dir(campaign_dir: Path, iteration: int) -> Path:
    return (
        Path(campaign_dir)
        / "7_ACTIVE_LEARNING"
        / ("iteration-" + str(int(iteration)).zfill(4))
    )


def clean_stale_ariadne_seed_outputs(campaign_dir, iteration: int) -> List[str]:
    """Remove stale per-seed ARIADNE result files before a retry submission.

    Only daemon-owned output files are removed. Seed directories and iteration
    manifests remain intact, so a retry can still use the same seed list while
    early Fortran/Python aborts cannot leave misleading old result.json payloads
    behind.
    """
    iter_dir = _iteration_active_learning_dir(Path(campaign_dir), int(iteration))
    pool_dir = iter_dir / "pool"
    if not pool_dir.exists():
        return []
    if pool_dir.is_symlink() or not pool_dir.is_dir():
        raise BackendSubmissionError(
            "refusing to clean ARIADNE pool that is not a real directory: "
            + str(pool_dir)
        )

    iter_root = iter_dir.resolve()
    removed: List[str] = []
    for seed_dir in sorted(pool_dir.glob("seed_*")):
        if seed_dir.is_symlink():
            raise BackendSubmissionError(
                "refusing to clean symlinked ARIADNE seed directory: "
                + str(seed_dir)
            )
        if not seed_dir.is_dir():
            continue
        candidates = [seed_dir / "result.json", seed_dir / "ARIADNE_TRACE.jsonl"]
        candidates.extend(sorted(seed_dir.glob("result.json.*.tmp")))
        for target in candidates:
            if not target.exists() and not target.is_symlink():
                continue
            resolved = target.resolve()
            try:
                resolved.relative_to(iter_root)
            except ValueError as exc:
                raise BackendSubmissionError(
                    "refusing to clean ARIADNE output outside iteration dir: "
                    + str(target)
                ) from exc
            if target.is_symlink() or not target.is_file():
                raise BackendSubmissionError(
                    "refusing to remove non-regular ARIADNE output: "
                    + str(target)
                )
            target.unlink()
            removed.append(str(target))
    return removed

#  SBATCH-phase postprocess refusal guard.
#
#
# Each parser registers itself by adding its phase name
# to this frozenset. Until then, postprocess() raises NotImplementedError
# with a hint pointing user at --dry-run or --mock-ariadne.
LIVE_POSTPROCESS_IMPLEMENTED: frozenset = frozenset({
    "INITIAL_GAUSSIAN", "GAUSSIAN",
    "INITIAL_AIMALL", "AIMALL",
    "INITIAL_FEREBUS", "FEREBUS",
    "ARIADNE_ARRAY",
    "PHASE_A_POLUS", "PHASE_B_POLUS",
})


#Validator functions for live parser pointdir inspection.
#Each takes a PointDirectory and returns (ok: bool, reason: str). The
#reason is a short tag like "scf_nonconvergence" or "missing_wfn" used
#in the quantum_output_rejected journal event.


def validate_gaussian_completed(pdir) -> tuple:
    """Return (True, "") if the pointdir contains a complete Gaussian output;
    (False, reason_tag) otherwise. Validates: .gaussianoutput present,
    contains "Normal termination", .wfn present and parseable."""
    #1. Gaussian output present + Normal termination.
    gau = getattr(pdir, "gaussian_output", None)
    if gau is None or not getattr(gau, "path", None):
        return False, "missing_gaussian_output"
    try:
        gau_text = Path(gau.path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False, "gaussian_output_unreadable"
    last_normal = gau_text.rfind("Normal termination")
    last_error = gau_text.rfind("Error termination")
    if last_normal < 0 or last_error > last_normal:
        return False, "scf_nonconvergence_or_crash"
    #2. WFN present + parseable.
    wfn = getattr(pdir, "wfn", None)
    if wfn is None or not getattr(wfn, "path", None):
        return False, "missing_wfn"
    try:
        #trigger the WFN lazy parse via total_energy; raises on malformed file.
        _ = wfn.total_energy
    except Exception:
        return False, "wfn_parse_failure"
    gjf = getattr(pdir, "gjf", None)
    if gjf is not None and getattr(gjf, "path", None):
        try:
            import numpy as _np

            gjf_atoms = list(gjf.atoms)
            wfn_atoms = list(wfn.atoms.to_angstroms())
            if len(gjf_atoms) != len(wfn_atoms):
                return False, "wfn_gjf_atom_count_mismatch"
            gjf_types = [str(a.type).capitalize() for a in gjf_atoms]
            wfn_types = [str(a.type).capitalize() for a in wfn_atoms]
            if gjf_types != wfn_types:
                return False, "wfn_gjf_atom_order_mismatch"
            gjf_coords = _np.asarray([a.coordinates for a in gjf_atoms], dtype=float)
            wfn_coords = _np.asarray([a.coordinates for a in wfn_atoms], dtype=float)
            if not _np.all(_np.isfinite(gjf_coords)) or not _np.all(_np.isfinite(wfn_coords)):
                return False, "wfn_gjf_geometry_nonfinite"
            if float(_np.max(_np.abs(gjf_coords - wfn_coords))) > 1.0e-4:
                return False, "wfn_gjf_geometry_mismatch"
        except Exception:
            return False, "wfn_gjf_geometry_check_failed"
    return True, ""


def validate_aimall_completed(pdir) -> tuple:
    """Return (True, "") if the pointdir contains a complete AIMAll output;
    (False, reason_tag) otherwise. Validates: *_atomicfiles/ directory
    exists; each .int file parses without raising."""
    ints = getattr(pdir, "ints", None)
    if ints is None or not getattr(ints, "path", None):
        return False, "missing_atomicfiles_dir"
    int_path = Path(ints.path)
    if not int_path.is_dir():
        return False, "atomicfiles_not_a_dir"
    #IntDirectory auto discovers .int files; iterate + trigger parse on each.
    try:
        n_int = 0
        for int_file in ints.ints:
         #int_file is an Int instance; touching net_charge triggers the
         # lazy parse and raises on malformed input.
            _ = int_file.net_charge
            n_int += 1
    except Exception:
        return False, "int_parse_failure"
    if n_int == 0:
        return False, "no_int_files"
    # one .int per atom. a partial AIMAll (crashed after a few atoms, or one atom whose integration
    # failed) leaves fewer .int than there are atoms -- the old n_int>=1 check waved that through and
    # then the FEREBUS feature export later choked on the point missing IQA for some atoms (A36). so
    # demand one per atom and reject unreadable geometry rather than accepting a point we cannot
    # count.
    try:
        n_atoms = len(pdir.atoms)
    except Exception:
        return False, "aimall_geometry_unreadable"
    if n_int != n_atoms:
        return False, "aimall_partial_" + str(n_int) + "_of_" + str(n_atoms) + "_int"
    try:
        for int_file in ints.ints:
            iqa = getattr(int_file, "iqa")
            integration_error = getattr(int_file, "integration_error")
            if iqa is None or not math.isfinite(float(iqa)):
                return False, "iqa_missing_or_nonfinite"
            if integration_error is None or not math.isfinite(float(integration_error)):
                return False, "integration_error_missing_or_nonfinite"
    except Exception:
        return False, "aimall_quality_parse_failure"
    return True, ""


def _pointdir_index(name: str) -> Optional[int]:
    if not (name.startswith("POINT_") and name.endswith(".pointdir")):
        return None
    try:
        return int(name[len("POINT_"):-len(".pointdir")])
    except ValueError:
        return None


def _next_pointdir_index(root) -> int:
    indexes = []
    for child in Path(root).glob("POINT_*.pointdir"):
        idx = _pointdir_index(child.name)
        if idx is not None:
            indexes.append(idx)
    return (max(indexes) + 1) if indexes else 0


def _ferebus_model_data_rows_ok(model_path) -> tuple:
    """spot a truncated .model. FEREBUS declares number_of_training_points N in the header then
    writes an [training_data.x] block of N rows. if it died mid-write the block is short, and the
    core reader fills the gap with uninitialised np.empty memory instead of erroring (A56) -- so
    the GP would quietly train on garbage. that reader lives in ichor_core/models which we are not
    allowed to touch, so we catch it here: parse the declared count, count the rows ourselves,
    reject anything short.
    """
    ntrain = None
    rows = None
    try:
        with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
            it = iter(f)
            for line in it:
                if "number_of_training_points" in line:
                    try:
                        ntrain = int(line.split()[1])
                    except (IndexError, ValueError):
                        return False, "model_ntrain_unparseable"
                if "[training_data.x]" in line:
                    rows = 0
                    for row in it:
                        if row.strip() == "":
                            break
                        rows += 1
                    break  # the x block is enough to spot a truncation
    except OSError:
        return False, "model_file_unreadable"
    if ntrain is None or rows is None:
        # no FEREBUS header/data block we recognise. that is NOT the A56 case (which is a header that
        # claims N rows followed by fewer than N) -- we just cannot assess truncation here, so do not
        # block on it. size>0 already passed and a genuinely corrupt model fails loudly at load.
        return True, ""
    if rows < ntrain:
        return False, "model_truncated"
    return True, ""


def _read_ferebus_model_metadata(model_path) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    try:
        with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("["):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                key = parts[0]
                if key == "name":
                    metadata["system"] = parts[1]
                elif key == "atom":
                    metadata["atom"] = parts[1]
                elif key == "property":
                    metadata["property"] = parts[1]
                elif key == "ALF":
                    try:
                        metadata["alf_1_indexed"] = [int(x) for x in parts[1:4]]
                    except ValueError:
                        metadata["alf_1_indexed"] = []
                elif key == "number_of_training_points":
                    try:
                        metadata["ntrain"] = int(parts[1])
                    except ValueError:
                        metadata["ntrain"] = None
    except OSError:
        metadata["unreadable"] = True
    return metadata


def validate_ferebus_completed(staging_dir) -> tuple:
    """Validate the exact pyferebus task manifest and expected model set."""
    staging = Path(staging_dir)
    try:
        from .model_contract import (
            ModelContractError,
            validate_ferebus_model_contract,
        )
        validate_ferebus_model_contract(staging, committed=False)
    except FileNotFoundError as exc:
        return False, "ferebus_manifest_invalid: " + type(exc).__name__ + ": " + str(exc)
    except ModelContractError as exc:
        if str(exc).startswith("ferebus_model_root_missing"):
            return False, "ferebus_staging_missing"
        return False, str(exc)
    except Exception as exc:
        return False, "ferebus_manifest_invalid: " + type(exc).__name__ + ": " + str(exc)
    return True, ""




def _high_quantile(vals, q=0.9):
    """a high quantile (default p90) of the kept-seed alphas -- our per-iteration convergence
    scalar. numpy-free + linear-interpolated. p90 keeps the worst-case/tail focus an adversarial
    loop needs WITHOUT being hostage to the single worst seed (which a bare max was -- one stuck
    seed pinned it high forever), and being a real order statistic it never returns a value no seed
    actually produced. empty -> 0.0, though callers guarantee at least one kept seed in practice."""
    if not vals:
        return 0.0
    s = sorted(float(v) for v in vals)
    if len(s) == 1:
        return s[0]
    pos = float(q) * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _ariadne_geometry_quality(result_dict: Dict[str, Any], validated: Dict[str, Any], gates: Any) -> Dict[str, Any]:
    import numpy as _np

    reasons = []
    final = _np.asarray(validated.get("final_coordinates"), dtype=float)
    metrics: Dict[str, Any] = {
        "max_displacement_ang": None,
        "min_pair_distance_ang": None,
    }
    if final.ndim != 2 or final.shape[1] != 3 or not _np.all(_np.isfinite(final)):
        reasons.append("ariadne_final_geometry_nonfinite")
        return {"accepted": False, "reasons": reasons, "metrics": metrics}
    if final.shape[0] >= 2:
        dmins = []
        for i in range(final.shape[0]):
            for j in range(i + 1, final.shape[0]):
                dmins.append(float(_np.linalg.norm(final[i] - final[j])))
        metrics["min_pair_distance_ang"] = min(dmins) if dmins else None
    initial_raw = result_dict.get("initial_coordinates")
    try:
        initial = _np.asarray(initial_raw, dtype=float)
        if initial.shape == final.shape and _np.all(_np.isfinite(initial)):
            displacements = _np.linalg.norm(final - initial, axis=1)
            metrics["max_displacement_ang"] = float(_np.max(displacements))
    except Exception:
        metrics["max_displacement_ang"] = None
    max_disp = getattr(gates, "ariadne_max_displacement_ang", None)
    if (
        max_disp is not None
        and metrics["max_displacement_ang"] is not None
        and float(metrics["max_displacement_ang"]) > float(max_disp)
    ):
        reasons.append("ariadne_max_displacement_threshold_exceeded")
    min_pair = getattr(gates, "ariadne_min_pair_distance_ang", None)
    if (
        min_pair is not None
        and metrics["min_pair_distance_ang"] is not None
        and float(metrics["min_pair_distance_ang"]) < float(min_pair)
    ):
        reasons.append("ariadne_min_pair_distance_threshold_exceeded")
    return {"accepted": not reasons, "reasons": reasons, "metrics": metrics}


def _ariadne_landing_audit_summary(seed_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "accepted": 0,
        "salvaged": 0,
        "backtracked": 0,
        "rejected": 0,
        "handoff_accepted": 0,
        "handoff_rejected": 0,
        "rejection_reasons": {},
        "handoff_rejection_reasons": {},
        "policies": {},
    }
    unique_records: Dict[str, Dict[str, Any]] = {}
    for pos, rec in enumerate(seed_records):
        key = str(rec.get("seed_index", "__pos_" + str(pos)))
        unique_records[key] = rec
    for rec in unique_records.values():
        safety = rec.get("landing_safety")
        if not isinstance(safety, dict):
            summary["rejected"] += 1
            reasons = [str(rec.get("reason", "missing_landing_safety"))]
            policy = "missing_landing_safety"
        else:
            accepted = bool(safety.get("accepted", False))
            policy = str(safety.get("policy", "unknown"))
            reasons = [str(r) for r in safety.get("reasons", [])]
            if accepted:
                summary["accepted"] += 1
            else:
                summary["rejected"] += 1
            if policy == "salvaged_iterate":
                summary["salvaged"] += 1
            if policy == "backtracked":
                summary["backtracked"] += 1
        policies = summary["policies"]
        policies[policy] = int(policies.get(policy, 0)) + 1
        if reasons:
            bucket = summary["rejection_reasons"]
            for reason in reasons:
                bucket[reason] = int(bucket.get(reason, 0)) + 1
        handoff_accepted = rec.get("handoff_accepted")
        if handoff_accepted is True:
            summary["handoff_accepted"] += 1
        elif handoff_accepted is False:
            summary["handoff_rejected"] += 1
            reason = str(
                rec.get(
                    "handoff_rejection_reason",
                    rec.get("reason", "handoff_rejected"),
                )
            )
            bucket = summary["handoff_rejection_reasons"]
            bucket[reason] = int(bucket.get(reason, 0)) + 1
    return summary


def _ariadne_optional_diagnostic_warnings(result_dict: Dict[str, Any]) -> List[str]:
    """Return warnings for optional ARIADNE diagnostics.

    These fields are telemetry, not handoff contract. Reconcile must keep old
    result.json files readable and must not reject an otherwise safe landing
    because an optional scale diagnostic was malformed.
    """
    diagnostics = result_dict.get("optimiser_diagnostics")
    if not isinstance(diagnostics, dict):
        return []
    numeric_fields = (
        "trqn_objective_scale",
        "trqn_target_initial_grad_norm",
        "trqn_initial_raw_grad_norm",
        "trqn_initial_scaled_grad_norm",
        "trqn_retry_objective_scale",
        "trqn_retry_target_initial_grad_norm",
        "trqn_retry_raw_grad_norm",
        "trqn_retry_scaled_grad_norm",
    )
    warnings: List[str] = []
    for field in numeric_fields:
        if field not in diagnostics or diagnostics.get(field) is None:
            continue
        try:
            value = float(diagnostics.get(field))
        except (TypeError, ValueError):
            warnings.append(field + "_not_numeric")
            continue
        if not math.isfinite(value):
            warnings.append(field + "_not_finite")
    mode = diagnostics.get("trqn_scale_mode")
    if mode is not None and str(mode) not in {
        "not_applicable",
        "off",
        "fixed",
        "adaptive_initial_gradient",
    }:
        warnings.append("trqn_scale_mode_unknown")
    return warnings


class LiveBackendNotAvailableError(RuntimeError):
    """Raised when --live is requested but a required backend is missing."""


@dataclass
class LiveBackendsPhaseExecutor(DryRunPhaseExecutor):
    """Production PhaseExecutor.

    Compared with :class: DryRunPhaseExecutor:

    * "submit_or_run" for SBATCH phases writes a real sbatch script
      (with module loads + backend invocation) and submits it via
      "sbatch --parsable".
    * "postprocess" for SBATCH phases parses real output files (Gaussian
      .log + .wfn, AIMAll .int, FEREBUS .model) into ICHOR data
      structures and runs the atomic append pipeline.

    Inline phases (SEED_SELECT / SPLIT / APPEND / STOP_CHECK) are inherited
    from the dry-run executor since they do not change.
    """

    sbatch_runner: Callable[..., Any] = subprocess.run
    walltime_hours: Optional[int] = None
    partition: Optional[str] = None
    backend_check: bool = True
    strict_committed_artifact_verification: bool = True

    def __post_init__(self) -> None:
        DryRunPhaseExecutor.__post_init__(self)
        if self.backend_check:
            avail = check_backends()
            if not avail.profile:
                raise LiveBackendNotAvailableError(missing_backend_message(avail))
            if not avail.sbatch:
                raise LiveBackendNotAvailableError(
                    "sbatch is not on PATH; LiveBackendsPhaseExecutor cannot "
                    "submit jobs. Use --dry-run or --mock-ariadne off-cluster."
                )

    def handle_failure(self, state, phase, observations) -> FailureAction:
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        if phase_name in ("ARIADNE_ARRAY", "PHASE_B_POLUS"):
            return FailureAction.HALT
        return super().handle_failure(state, phase, observations)

    # --- SBATCH submission ---------------------------------------------

    def _locate_sample_xyz(self, phase_name, iteration):
        camp = Path(self.campaign_dir)
        if phase_name == "INITIAL_GAUSSIAN":
            from ..handoff_manifests import read_phase_a_sample_manifest

            outdir = camp / "3_DIVERSITY_SAMPLING" / "initial"
            try:
                manifest = read_phase_a_sample_manifest(outdir)
            except Exception as exc:
                raise BackendSubmissionError(
                    "phase_a_sample_manifest_invalid: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ) from exc
            return Path(str(manifest["sample_xyz"]))
        iter_dir = camp / "7_ACTIVE_LEARNING" / ("iteration-" + str(int(iteration)).zfill(4))
        candidate = iter_dir / "phase_b_SAMPLE.xyz"
        return candidate if candidate.is_file() else None

    def _count_seeds(self, iteration):
        from ..handoff_manifests import load_seeds_picked

        iter_dir = Path(self.campaign_dir) / "7_ACTIVE_LEARNING" / ("iteration-" + str(int(iteration)).zfill(4))
        try:
            data = load_seeds_picked(iter_dir, expected_iteration=int(iteration))
        except Exception as exc:
            raise BackendSubmissionError("seeds_picked.json unreadable: " + str(exc))
        return int(data.get("n_picked", 0))

    def _seed_dir_for_record(self, iteration: int, seed_record: Dict[str, Any]) -> Path:
        seed_index = int(seed_record["seed_index"])
        return (
            Path(self.campaign_dir)
            / "7_ACTIVE_LEARNING"
            / ("iteration-" + str(int(iteration)).zfill(4))
            / "pool"
            / ("seed_" + str(seed_index).zfill(4))
        )

    @staticmethod
    def _seed_record_frame_id(seed_record: Dict[str, Any]) -> Optional[int]:
        raw = seed_record.get("frame_id")
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise BackendSubmissionError(
                "ARIADNE seed record frame_id is not an integer: " + repr(raw)
            ) from exc

    @staticmethod
    def _seed_record_variance(seed_record: Dict[str, Any]) -> Optional[float]:
        raw = seed_record.get("variance_at_selection")
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise BackendSubmissionError(
                "ARIADNE seed record variance_at_selection is not numeric: "
                + repr(raw)
            ) from exc
        return value if math.isfinite(value) else None

    @staticmethod
    def _seed_record_subspace_payload(seed_record: Dict[str, Any]) -> tuple:
        neighbours = (
            seed_record.get("subspace_neighbour_frame_ids")
            if isinstance(seed_record.get("subspace_neighbour_frame_ids"), list)
            else seed_record.get("neighbour_frame_ids")
        )
        if not isinstance(neighbours, list):
            neighbours = []
        eigenvalues = seed_record.get("subspace_eigenvalues")
        if not isinstance(eigenvalues, list):
            eigenvalues = []
        raw_dimension = seed_record.get("subspace_dimension")
        if raw_dimension is None:
            raw_dimension = len(eigenvalues) if eigenvalues else 0
        try:
            dimension = int(raw_dimension)
        except (TypeError, ValueError) as exc:
            raise BackendSubmissionError(
                "ARIADNE seed record subspace_dimension is not an integer: "
                + repr(raw_dimension)
            ) from exc
        return neighbours, dimension, eigenvalues

    def _ensure_ariadne_seed_provenance(
        self,
        state,
        picked: Dict[str, Any],
        seed_record: Dict[str, Any],
    ) -> tuple:
        """Ensure one live ARIADNE seed directory has its provenance sidecar.

        Dry-run has always written this before enriching ARIADNE/Phase-B
        blocks. Live mode must do the same because downstream Gaussian staging
        validates and copies the sidecar into the labelled pointdir.
        """
        iteration = int(state.iteration)
        seed_dir = self._seed_dir_for_record(iteration, seed_record)
        prov_path = seed_dir / PROVENANCE_FILENAME
        seed_frame_id = self._seed_record_frame_id(seed_record)
        trajectory_sha = str(picked.get("trajectory_sha256", "") or "")
        trajectory_sha_for_validation = trajectory_sha if trajectory_sha else None
        if prov_path.is_file():
            try:
                validate_provenance(
                    seed_dir,
                    campaign_uid=str(getattr(state, "campaign_uid", "")),
                    iteration=iteration,
                    trajectory_sha256=trajectory_sha_for_validation,
                    seed_frame_id=seed_frame_id,
                )
            except Exception as exc:
                raise BackendSubmissionError(
                    "ARIADNE seed provenance invalid for "
                    + seed_dir.name
                    + ": "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ) from exc
            return prov_path, False

        neighbours, dimension, eigenvalues = self._seed_record_subspace_payload(seed_record)
        try:
            write_seed_provenance(
                seed_dir,
                campaign_uid=str(getattr(state, "campaign_uid", "")),
                iteration=iteration,
                trajectory_sha256=trajectory_sha,
                seed_frame_id=seed_frame_id,
                seed_selection_origin=str(seed_record.get("selection_origin", "unknown")),
                seed_variance_at_selection=self._seed_record_variance(seed_record),
                subspace_neighbour_frame_ids=neighbours,
                subspace_dimension=dimension,
                subspace_eigenvalues=eigenvalues,
                mode_weighting_policy=self._mode_weighting_policy_or_default(),
            )
        except Exception as exc:
            raise BackendSubmissionError(
                "ARIADNE seed provenance write failed for "
                + seed_dir.name
                + ": "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc
        return prov_path, True

    def _ensure_ariadne_seed_provenance_for_iteration(self, state) -> int:
        from ..handoff_manifests import load_seeds_picked

        iteration = int(state.iteration)
        iter_dir = (
            Path(self.campaign_dir)
            / "7_ACTIVE_LEARNING"
            / ("iteration-" + str(iteration).zfill(4))
        )
        try:
            picked = load_seeds_picked(iter_dir, expected_iteration=iteration)
        except Exception as exc:
            raise BackendSubmissionError("seeds_picked.json unreadable: " + str(exc))
        created = 0
        for seed_record in list(picked.get("seed_records", [])):
            _path, was_created = self._ensure_ariadne_seed_provenance(
                state,
                picked,
                seed_record,
            )
            if was_created:
                created += 1
        if created:
            self._journal_event(
                "ariadne_seed_provenance_staged",
                iteration=iteration,
                n_created=int(created),
            )
        return int(picked.get("n_picked", 0))

    def _array_size_after_staging(self, phase_name, state):
        """Stage this phase's per-point inputs and return the SLURM array size.
        None for single-job phases (FEREBUS / POLUS) so no --array is emitted."""
        from . import input_staging as _stg
        camp = Path(self.campaign_dir)
        it = int(state.iteration)
        effective_partition = (
            str(self.partition)
            if self.partition is not None
            else str(getattr(self.config.resources, "partition", "multicore"))
        )
        if phase_name in ("INITIAL_GAUSSIAN", "GAUSSIAN"):
            sample = self._locate_sample_xyz(phase_name, it)
            if sample is None:
                raise BackendSubmissionError(
                    "no POLUS sample to stage for " + phase_name
                )
            _, n = _stg.stage_gaussian_inputs(
                camp,
                self.config,
                phase_name,
                it,
                sample,
                partition_override=effective_partition,
            )
            return n
        if phase_name in ("INITIAL_AIMALL", "AIMALL"):
            _, n = _stg.stage_aimall_inputs(
                camp,
                self.config,
                phase_name,
                it,
                partition_override=effective_partition,
            )
            return n
        if phase_name == "ARIADNE_ARRAY":
            return self._ensure_ariadne_seed_provenance_for_iteration(state)
        return None

    def _submit_ferebus_phase(self, state, phase_name: str) -> PhaseResult:
        from . import input_staging as _stg
        from ..submit.pyferebus_wrap import FerebusSubmissionError, submit_ferebus

        try:
            tv = int(getattr(state, "training_set_version", 0))
            is_initial = phase_name == "INITIAL_FEREBUS"
            state_updates: Dict[str, Any] = {}
            if not is_initial:
                from ..versioning.training_set import TrainingSetVersioning

                v_train = TrainingSetVersioning(Path(self.campaign_dir) / self.training_dir_name)
                committed = v_train.list_committed_versions()
                committed_max = max(committed) if committed else -1
                if committed_max > tv:
                    tv = committed_max
                    v_train.ensure_current(committed_max)
                    state_updates["training_set_version"] = int(committed_max)
                elif committed_max < tv:
                    raise BackendSubmissionError(
                        "state.training_set_version "
                        + str(tv)
                        + " is ahead of committed training versions "
                        + repr(committed)
                    )
            staging, n_tasks = _stg.stage_ferebus_inputs(
                self.campaign_dir,
                self.config,
                tv,
                is_initial=is_initial,
            )
            if int(n_tasks) <= 0:
                raise BackendSubmissionError("nothing to submit for " + phase_name + ": staged 0 tasks")
            f = self.config.ferebus
            resources = getattr(self.config, "resources", None)
            effective_partition = (
                str(self.partition)
                if self.partition is not None
                else str(getattr(resources, "partition", "multicore"))
            )
            resolved = resolve_phase_resources(
                phase_name=phase_name,
                config=self.config,
                partition=effective_partition,
                campaign_dir=self.campaign_dir,
                iteration=int(getattr(state, "iteration", 0)),
            )
            self._journal_event(
                "resolved_phase_resources",
                **resolved.journal_payload(phase_name=phase_name),
            )
            ferebus_path = _configured_backend_path("ferebus", "ferebus")
            allow_bare_ferebus = (
                os.environ.get("ICHOR_ALLOW_BARE_FEREBUS", "") == "1"
                or self.sbatch_runner is not subprocess.run
            )
            if ferebus_path == "ferebus" and not allow_bare_ferebus:
                raise BackendSubmissionError(
                    "live FEREBUS requires software.ferebus.executable_path "
                    "in ichor_config.yaml; set ICHOR_ALLOW_BARE_FEREBUS=1 only for "
                    "development tests"
                )
            path_to_executable = None if ferebus_path == "ferebus" else ferebus_path
            ferebus_platform = _configured_ferebus_platform()
            ferebus_manifest = _stg.read_ferebus_manifest(staging)
            expected_ferebus_tasks = int(ferebus_manifest.get("n_tasks", 0))
            expected_job_name = live_job_name(
                getattr(state, "campaign_uid", None),
                phase_name,
                int(getattr(state, "iteration", 0)),
            )
            effective_walltime = (
                int(self.walltime_hours)
                if self.walltime_hours is not None
                else int(resources.walltime_for(phase_name) if resources is not None else 24)
            )
            submission = submit_ferebus(
                staging / _stg.FEREBUS_JOB_DETAILS,
                staging,
                platform=ferebus_platform,
                walltime_hours=effective_walltime,
                ncores=max(1, int(resolved.cpus_per_task)),
                partition=str(resolved.partition),
                mem_per_cpu=str(resolved.mem_per_cpu),
                cpus_per_task=int(resolved.cpus_per_task),
                ntasks=int(resolved.ntasks),
                kernel=str(f.kernel),
                loss=str(f.loss),
                is_constant_noise=bool(f.is_constant_noise),
                nagents=int(f.nagents),
                maxiter=int(f.maxiter),
                full_ARD=bool(getattr(f, "full_ARD", True)),
                scaling=bool(getattr(f, "scaling", True)),
                overwrite_workdir=False,
                move_dataset_files=True,
                path_to_executable=path_to_executable,
                expected_tasks=expected_ferebus_tasks,
                expected_job_name=expected_job_name,
                submit_runner=self.sbatch_runner,
            )
        except BackendSubmissionError:
            raise
        except (FerebusSubmissionError, OSError, ValueError) as exc:
            raise BackendSubmissionError(
                "pyferebus submission failed for "
                + phase_name
                + ": "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc
        except Exception as exc:
            raise BackendSubmissionError(
                "pre-submit staging failed for "
                + phase_name
                + ": "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc
        self.artefact_log.append(str(submission.submission_script))
        return PhaseResult(
            is_complete=False,
            submitted_job_id=str(submission.job_id),
            expected_tasks=int(n_tasks),
            state_updates=state_updates,
        )

    def _seed_selection_posterior(self, state, training_atoms):
        """Live seed selection ranks the exploit half by the real GP posterior
        variance, so the most uncertain pool frames get attacked."""
        models_version = int(getattr(state, "models_version", -1))
        if models_version < 0:
            return super()._seed_selection_posterior(state, training_atoms)
        allow_uniform = bool(
            getattr(self.config.acquisition, "allow_uniform_posterior_fallback", False)
        )
        try:
            from pathlib import Path as _Path
            from .model_contract import smoke_total_energy_posterior
            from .artifact_contracts import verify_committed_model_version
            models_dir = (
                _Path(self.campaign_dir)
                / self.models_dir_name
                / ("iteration-" + str(models_version).zfill(4))
            )
            if not models_dir.is_dir():
                raise BackendSubmissionError(
                    "committed models directory missing for seed selection: "
                    + str(models_dir)
                )
            verify_committed_model_version(
                self.campaign_dir,
                models_version,
                models_dir_name=self.models_dir_name,
            )
            return smoke_total_energy_posterior(
                models_dir,
                property_name=str(self.config.acquisition.property_name),
                probe_frames=list(training_atoms)[:3],
            )
        except Exception as exc:
            if allow_uniform:
                self._journal_event(
                    "seed_posterior_fallback",
                    iteration=int(state.iteration),
                    error=str(exc)[:120],
                    explicit=True,
                )
                return super()._seed_selection_posterior(state, training_atoms)
            if isinstance(exc, BackendSubmissionError):
                raise
            raise BackendSubmissionError(
                "live seed selection requires loadable committed models: "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc

    def _inline_seed_select(self, state):
        """Live SEED_SELECT must never use the dry-run no-pool placeholder."""
        from ..acquisition.trajectory_pool import TrajectoryPool

        try:
            TrajectoryPool.load(self.campaign_dir)
        except Exception as exc:
            raise BackendSubmissionError(
                "live seed selection requires an imported trajectory pool: "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc
        return super()._inline_seed_select(state)

    def _inline_append(self, state):
        """Live APPEND: commit the validated quantum outputs staged at
        .DATA/STAGING/iter_<N>/POINT_* into the training set -- the real
        Gaussian/AIMAll results -- instead of the ARIADNE pool seeds the
        dry-run path commits. Live mode requires the AIMAll acceptance
        manifest; missing live staging is a halt-worthy contract failure.

        Re-commit is idempotent (a crash-retry after commit returns the
        existing version). The committed points carry whatever provenance the
        staging tree holds. APPEND commits every validated point into the one
        cumulative training set; the train / internal-val / external-val split
        is done later, per atom, at FEREBUS staging -- not here. (FEREBUS reads
        the pre-split csvs; it does not split internally.)
        """
        from . import input_staging as _stg

        v = self._versioning("training")
        committed = v.list_committed_versions()
        state_version = int(getattr(state, "training_set_version", 0))
        committed_max = max(committed) if committed else -1
        if committed_max > state_version:
            if int(committed_max) != int(state_version) + 1:
                raise BackendSubmissionError(
                    "training_set_version_gap: state="
                    + str(state_version)
                    + " committed_versions="
                    + repr(committed)
                )
            v.ensure_current(committed_max)
            self._journal_event(
                "training_set_committed",
                iteration=int(state.iteration),
                training_set_version=int(committed_max),
                n_committed_points=0,
                idempotent_skip=True,
            )
            return {"training_set_version": int(committed_max)}
        if committed_max < state_version:
            raise BackendSubmissionError(
                "training_set_version "
                + str(state_version)
                + " is ahead of committed training versions "
                + repr(committed)
            )
        if committed:
            v.ensure_current(committed_max)

        staging_root = _stg.bucket_dir(self.campaign_dir, "APPEND", int(state.iteration))
        try:
            point_dirs, _manifest = _stg.read_quantum_acceptance_manifest(
                staging_root,
                expected_phase="AIMALL",
                expected_iteration=int(state.iteration),
                require_points_file_membership=True,
            )
        except Exception as exc:
            raise BackendSubmissionError(
                "live APPEND requires valid AIMAll acceptance manifest: "
                + type(exc).__name__ + ": " + str(exc)
            ) from exc

        next_version = committed_max + 1
        v.recover_dangling_staging()
        source = committed_max if committed_max >= 0 else None
        staging = v.stage(source_version=source, target_version=next_version)

        committed_names = []
        next_point_index = _next_pointdir_index(staging)
        for pd in point_dirs:
            dest = staging / ("POINT_" + str(next_point_index).zfill(4) + ".pointdir")
            while dest.exists():
                next_point_index += 1
                dest = staging / ("POINT_" + str(next_point_index).zfill(4) + ".pointdir")
            _stg._copytree_no_symlinks(pd, dest)
            committed_names.append(dest.name)
            next_point_index += 1

        v.commit(next_version)
        v.update_current(next_version)
        ensure_index(self.campaign_dir)
        committed_iter_dir = v.iteration_path(next_version)
        for pdir_name in committed_names:
            pdir = committed_iter_dir / pdir_name
            seed_frame_id = self._read_seed_frame_id_from_pointdir(pdir)
            append_to_index(
                self.campaign_dir,
                iteration=int(next_version),
                pointdir_name=pdir_name,
                seed_frame_id=seed_frame_id,
            )
        self._journal_event(
            "training_set_committed",
            iteration=int(state.iteration),
            training_set_version=int(next_version),
            n_committed_points=len(committed_names),
            source="live_quantum_staging",
        )
        return {"training_set_version": int(next_version)}

    def submit_or_run(self, state, phase) -> PhaseResult:
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        if phase_name in ("INITIAL_FEREBUS", "FEREBUS"):
            return self._submit_ferebus_phase(state, phase_name)
        if phase_name not in SBATCH_PHASES:
            if phase_name in INLINE_PHASES:
                try:
                    return super().submit_or_run(state, phase)
                except BackendSubmissionError:
                    raise
                except Exception as exc:
                    raise BackendSubmissionError(
                        "inline phase failed before completion for " + phase_name
                        + ": " + type(exc).__name__ + ": " + str(exc)
                    ) from exc
            #defensive: a phase that is neither inline nor sbatch is a bug.
            raise RuntimeError(
                "phase " + phase_name + " classified as neither INLINE nor SBATCH"
            )
        try:
            array_size = self._array_size_after_staging(phase_name, state)
            if array_size is not None and array_size <= 0:
                raise BackendSubmissionError(
                    "nothing to submit for " + phase_name + ": staged 0 points/seeds"
                )
            if phase_name == "ARIADNE_ARRAY":
                removed = clean_stale_ariadne_seed_outputs(
                    self.campaign_dir,
                    int(getattr(state, "iteration", 0)),
                )
                if removed:
                    self._journal_event(
                        "ariadne_stale_outputs_cleaned",
                        iteration=int(getattr(state, "iteration", 0)),
                        removed=int(len(removed)),
                        sample=[str(p) for p in removed[:5]],
                    )
            script = self._write_real_script(phase_name, state, array_size)
        except BackendSubmissionError:
            raise
        except Exception as exc:
            raise BackendSubmissionError(
                "pre-submit staging failed for " + phase_name + ": "
                + type(exc).__name__ + ": " + str(exc)
            ) from exc
        result = self.sbatch_runner(
            ["sbatch", "--parsable", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        stdout = getattr(result, "stdout", "") or ""
        stderr = getattr(result, "stderr", "") or ""
        return_code = int(getattr(result, "returncode", 1))
        if return_code != 0:
            raise BackendSubmissionError(
                "sbatch failed for phase " + phase_name + ": "
                + repr(stderr) + " stdout=" + repr(stdout)
            )
        head = next((ln.strip() for ln in stdout.splitlines() if ln.strip()), "")
        job_id = head.split(";", 1)[0].strip()
        if not job_id or not job_id[0].isdigit():
            raise BackendSubmissionError(
                "sbatch returned unparsable JobID for phase " + phase_name + ": " + repr(head)
            )
        self.artefact_log.append(str(script))
        expected_tasks = int(array_size) if array_size is not None else 1
        return PhaseResult(
            is_complete=False,
            submitted_job_id=job_id,
            expected_tasks=expected_tasks,
        )

    # --- real script bodies --------------------------------------------

    def _write_real_script(self, phase_name: str, state, array_size=None) -> Path:
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        (self.scripts_dir / "OUTPUTS").mkdir(parents=True, exist_ok=True)
        (self.scripts_dir / "ERRORS").mkdir(parents=True, exist_ok=True)
        path = self.scripts_dir / (phase_name + "-" + str(state.iteration) + ".sh")
        effective_partition = (
            str(self.partition)
            if self.partition is not None
            else str(getattr(self.config.resources, "partition", "multicore"))
        )
        resolved = resolve_phase_resources(
            phase_name=phase_name,
            config=self.config,
            partition=effective_partition,
            campaign_dir=self.campaign_dir,
            iteration=int(state.iteration),
            array_size=array_size,
        )
        self._journal_event(
            "resolved_phase_resources",
            **resolved.journal_payload(phase_name=phase_name),
        )
        body = build_sbatch_script(
            phase_name=phase_name,
            iteration=state.iteration,
            campaign_dir=self.campaign_dir,
            config=self.config,
            array_size=array_size,
            walltime_hours=self.walltime_hours,
            partition=self.partition,
            campaign_uid=getattr(state, "campaign_uid", None),
            resolved_resources=resolved,
        )
        path.write_text(body, encoding="utf-8")
        try:
            os.chmod(path, 0o755)
        except OSError:
            pass
        return path

    # --- postprocess (CSF4-only implementation) -------------------------

    def _parse_staged_pointdirs(self, staging_root, *, validators, allowed_pointdir_names=None):
        """Walk staging_root for POINT_*.pointdir/ children, run
        every validator on each, return (kept, rejected).

        Parameters
        ----------
        staging_root
            Directory expected to contain POINT_NNNN.pointdir subdirectories
            (typically ".DATA/STAGING/initial/" or ".DATA/STAGING/iter_<N>/").
        validators
            Sequence of callables, each "validator(pdir) -> (ok, reason)".
            A pointdir passes only if EVERY validator returns ok=True.

        Returns
        -------
        kept : List[PointDirectory]
            Pointdirs that pass every validator.
        rejected : List[Tuple[str, str]]
            "(pointdir_name, first_failure_reason)" for every reject.
        """
        from pathlib import Path as _Path
        from ichor.core.files.point_directory import PointDirectory

        staging_root = _Path(staging_root)
        kept = []
        rejected = []
        if not staging_root.is_dir():
            return kept, rejected
        allowed = None
        if allowed_pointdir_names is not None:
            allowed = {str(name) for name in allowed_pointdir_names}
            actual = {
                child.name
                for child in staging_root.iterdir()
                if child.is_dir() and PointDirectory.check_path(child)
            }
            extras = sorted(actual - allowed)
            if extras:
                raise ValueError(
                    "staging contains unsubmitted pointdirs: " + ", ".join(extras[:5])
                )
        children = sorted(staging_root.iterdir()) if allowed is None else [
            staging_root / name for name in sorted(allowed)
        ]
        for child in children:
            if allowed is not None and not child.exists():
                rejected.append((child.name, "submitted_pointdir_missing"))
                continue
            if not (child.is_dir() and PointDirectory.check_path(child)):
                continue
            pdir = PointDirectory(child)
            failure_reason = None
            for validator in validators:
                ok, reason = validator(pdir)
                if not ok:
                    failure_reason = reason
                    break
            if failure_reason is None:
                kept.append(pdir)
            else:
                rejected.append((child.name, failure_reason))
        return kept, rejected

    def postprocess(self, state, phase, observations: Sequence[Any]) -> PhaseResult:
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        handler = self._live_postprocess_handlers().get(phase_name)
        if handler is not None:
            #handlers take (state, phase, observations) so the shared
            #handler can dispatch by phase name to the right validator.
            return handler(state, phase, observations)
        # if this is a registered SBATCH phase, refuse rather than
        #silently call super() -- the dry-run postprocess writes stub
        # artefacts that would overwrite real backend output.
        if phase_name in SBATCH_PHASES and phase_name not in LIVE_POSTPROCESS_IMPLEMENTED:
            self._journal_event(
                "live_postprocess_refused",
                phase=phase_name,
                iteration=int(getattr(state, "iteration", -1)),
            )
            raise NotImplementedError(
                "Live postprocess for " + phase_name + " is not yet implemented. "
                "Refusing to overwrite real backend output with dry-run stub "
                "artefacts. Register a real parser in LIVE_POSTPROCESS_IMPLEMENTED "
                "+ _live_postprocess_handlers, or run with --dry-run / --mock-ariadne "
            )
        # Inline equivalent bookkeeping (manifests / versioning) defers to the
        #dry-run executor so artefact handling is identical between executors.
        return super().postprocess(state, phase, observations)

    def _live_postprocess_handlers(self):
        """Return phase name -> handler dict for postprocess dispatch.

        Four ab initio phases (Gaussian + AIMAll, INITIAL + iter)
        all share _parse_quantum_postprocess. The handler reads its phase
        argument and picks the right validator set internally.
        """
        return {
            "INITIAL_GAUSSIAN": self._parse_quantum_postprocess,
            "GAUSSIAN":         self._parse_quantum_postprocess,
            "INITIAL_AIMALL":   self._parse_quantum_postprocess,
            "AIMALL":           self._parse_quantum_postprocess,
            "INITIAL_FEREBUS":  self._parse_ferebus_postprocess,
            "FEREBUS":          self._parse_ferebus_postprocess,
            "ARIADNE_ARRAY":    self._parse_ariadne_array_postprocess,
            "PHASE_A_POLUS":    self._parse_polus_postprocess,
            "PHASE_B_POLUS":    self._parse_polus_postprocess,
        }

    # --- quantum-phase parser body ----------------------------

    def _quantum_staging_path(self, state, phase_name):
        """Return the canonical staging root for the given quantum phase.

        INITIAL_GAUSSIAN / INITIAL_AIMALL share ".DATA/STAGING/initial/";
        GAUSSIAN / AIMALL share ".DATA/STAGING/iter_<N>/" (per-iteration).
        """
        from pathlib import Path as _Path
        initial = phase_name.startswith("INITIAL_")
        subdir = "initial" if initial else ("iter_" + str(int(state.iteration)))
        return _Path(self.campaign_dir) / ".DATA" / "STAGING" / subdir

    def _validators_for(self, phase_name):
        """Pick the right validator tuple for the phase. Gaussian / AIMAll
        validators are independent; iterative-phase validation only checks
        the phase-specific output not the prior phase's output (the prior
        phase already had its own postprocess call to validate)."""
        if "GAUSSIAN" in phase_name:
            return (validate_gaussian_completed,)
        if "AIMALL" in phase_name:
            return (validate_aimall_completed,)
        raise ValueError("unknown quantum phase: " + phase_name)

    def _parse_quantum_postprocess(self, state, phase, observations):
        """Shared postprocess for the four quantum phases.

        Reads the canonical staging root (via _quantum_staging_path),
        applies the per-phase validator(s) to each POINT_*.pointdir/, and:
          - journals quantum_output_rejected per rejected pointdir.
          - journals phase_succeeded_live on overall success.
          - returns PhaseResult with failure_reason set when rejection
            rate exceeds config.failure_threshold_fraction (or when no
            pointdirs are found at all).

        Does NOT commit anything to 5_TRAINING / 6_TRAINED_MODELS itself --
        that lives in the subsequent INITIAL_FEREBUS / APPEND / FEREBUS
        phases. This parser is a pure validator + observability/report hook.
        """
        from .phase_executor import PhaseResult
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        staging_root = self._quantum_staging_path(state, phase_name)
        validators = self._validators_for(phase_name)
        from . import input_staging as _stg

        if not Path(staging_root).is_dir():
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "no_pointdirs_in_staging: " + str(staging_root)
                ),
            )

        if "AIMALL" in phase_name:
            from ichor.core.files.point_directory import PointDirectory
            from .quantum_quality import (
                evaluate_aimall_pointdir,
                write_quantum_quality_manifest,
            )

            expected_phase = "INITIAL_GAUSSIAN" if phase_name.startswith("INITIAL_") else "GAUSSIAN"
            try:
                gaussian_accepted, _gaussian_manifest = _stg.read_quantum_acceptance_manifest(
                    staging_root,
                    expected_phase=expected_phase,
                    expected_iteration=int(state.iteration),
                    require_points_file_membership=True,
                )
            except Exception as exc:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "prior_gaussian_acceptance_manifest_invalid: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)[:180]
                    ),
                )
            kept = []
            rejected = []
            for candidate in gaussian_accepted:
                pdir = PointDirectory(candidate)
                failure_reason = None
                for validator in validators:
                    ok, reason = validator(pdir)
                    if not ok:
                        failure_reason = reason
                        break
                if failure_reason is None:
                    kept.append(pdir)
                else:
                    rejected.append((Path(candidate).name, failure_reason))

            quality_records = []
            quality_kept = []
            quality_rejected = []
            for pdir in kept:
                record = evaluate_aimall_pointdir(
                    pdir,
                    getattr(self.config, "quality_gates", None),
                )
                quality_records.append(record)
                if bool(record.get("accepted")):
                    quality_kept.append(pdir)
                else:
                    quality_rejected.append(
                        (
                            str(record.get("pointdir", Path(getattr(pdir, "path", pdir)).name)),
                            ";".join(record.get("reasons") or ["quantum_quality_rejected"]),
                        )
                    )
            for pdir_name, reason in rejected:
                quality_records.append(
                    {
                        "pointdir": str(pdir_name),
                        "accepted": False,
                        "reasons": [str(reason)],
                    }
                )
            kept = quality_kept
            rejected = list(rejected) + quality_rejected
            quality_path = write_quantum_quality_manifest(
                staging_root,
                phase_name=phase_name,
                iteration=int(state.iteration),
                records=quality_records,
                gates=getattr(self.config, "quality_gates", None),
            )
            self._journal_event(
                "quantum_quality_summary",
                phase=phase_name,
                iteration=int(state.iteration),
                manifest=str(quality_path),
                n_total=int(len(quality_records)),
                n_rejected=int(sum(1 for r in quality_records if not bool(r.get("accepted")))),
            )
            if phase_name == "AIMALL":
                try:
                    from .error_calibration import (
                        ERROR_CALIBRATION_AUDIT_FILENAME,
                        ERROR_CALIBRATION_MODEL_FILENAME,
                        update_from_aimall_acceptance,
                    )

                    iter_dir = self._iter_dir(state.iteration)
                    audit = update_from_aimall_acceptance(
                        campaign_dir=self.campaign_dir,
                        iter_dir=iter_dir,
                        config=self.config,
                        iteration=int(state.iteration),
                        models_version=int(getattr(state, "models_version", -1)),
                        accepted_pointdirs=kept,
                        quality_records=quality_records,
                    )
                    self.artefact_log.append(
                        str((iter_dir / ERROR_CALIBRATION_AUDIT_FILENAME).resolve())
                    )
                    self.artefact_log.append(
                        str(
                            (
                                Path(self.campaign_dir)
                                / ".DATA" / "ACTIVE_LEARNING"
                                / ERROR_CALIBRATION_MODEL_FILENAME
                            ).resolve()
                        )
                    )
                    self._journal_event(
                        "error_calibration_summary",
                        phase=phase_name,
                        iteration=int(state.iteration),
                        n_added_records=int(audit.get("n_added_records", 0)),
                        n_total_records=int(audit.get("n_total_records", 0)),
                        usable_for_acquisition=bool(
                            audit.get("usable_for_acquisition", False)
                        ),
                    )
                except Exception as exc:
                    try:
                        from .error_calibration import mark_calibration_model_stale

                        mark_calibration_model_stale(
                            self.campaign_dir,
                            reason=type(exc).__name__ + ": " + str(exc)[:240],
                            iteration=int(state.iteration),
                        )
                    except Exception:
                        pass
                    self._journal_event(
                        "error_calibration_failed",
                        phase=phase_name,
                        iteration=int(state.iteration),
                        reason=type(exc).__name__ + ": " + str(exc)[:240],
                    )
        else:
            try:
                allowed_names = _stg._points_file_names(staging_root)
                kept, rejected = self._parse_staged_pointdirs(
                    staging_root,
                    validators=validators,
                    allowed_pointdir_names=allowed_names,
                )
            except Exception as exc:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "quantum_task_membership_invalid: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)[:180]
                    ),
                )
        _stg.write_quantum_acceptance_manifest(
            staging_root,
            phase_name=phase_name,
            iteration=int(state.iteration),
            accepted=kept,
            rejected=rejected,
        )

        for pdir_name, reason in rejected:
            self._journal_event(
                "quantum_quality_rejected" if "quality" in str(reason) or "iqa_" in str(reason) or "integration_" in str(reason) else "quantum_output_rejected",
                phase=phase_name,
                iteration=int(state.iteration),
                pointdir=pdir_name,
                reason=reason,
            )

        n_total = len(kept) + len(rejected)
        if n_total == 0:
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "no_pointdirs_in_staging: " + str(staging_root)
                ),
            )
        if len(kept) == 0:
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "no_quantum_outputs_accepted: "
                    + str(len(rejected))
                    + "/"
                    + str(n_total)
                ),
            )

        rejection_rate = len(rejected) / float(n_total)
        threshold = float(self.config.failure_threshold_fraction)
        if rejection_rate > threshold:
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "too_many_rejected: " + str(len(rejected))
                    + "/" + str(n_total)
                    + " (rate=" + format(rejection_rate, ".2f")
                    + " > threshold=" + format(threshold, ".2f") + ")"
                ),
            )

        self._journal_event(
            "phase_succeeded_live",
            phase=phase_name,
            iteration=int(state.iteration),
            n_kept=int(len(kept)),
            n_rejected=int(len(rejected)),
        )
        return PhaseResult(is_complete=True, state_updates={})

    # ---helpers ---------------------------------------------

    def _models_staging_path(self):
        """Path to the FEREBUS staging directory.

        FEREBUS writes a .model file (single artefact per training run)
        to this canonical location; the parser validates and atomically
        renames it into 6_TRAINED_MODELS/iteration-NNNN/ via the
        TrainingSetVersioning helper.
        """
        from pathlib import Path as _Path
        return (
            _Path(self.campaign_dir)
            / self.models_dir_name
            / "iteration-staging"
        )

    def _initial_quantum_staging_path(self):
        """Path the initial diversity sample staging dir lives at."""
        from pathlib import Path as _Path
        return _Path(self.campaign_dir) / ".DATA" / "STAGING" / "initial"


    # --- reference scales from a real GP posterior ----------------------

    def _maybe_refresh_reference_scales(self, state) -> bool:
        """Compute per-iteration reference scales from the trained
        FEREBUS models, persist to a sidecar, update state.

        Reference scales are the five per-property anchors the adversarial
        acquisition uses to normalise its energy, force, frequency,
        anharmonicity and anharmonic-std contributions before combining
        them with the configured lambda weights. Without real scales the
        lambda weights operate on mixed-unit quantities and the result is
        nonsense -- which is exactly the dry-run synthetic stub case.

        Placement: inline at SEED_SELECT time (on the login node). costs
        roughly 480 GP evals per iteration for a 12-atom system, which is
        5-30 seconds. larger systems or higher subspace.max_subspace_dim
        push that toward a minute; if it ever becomes painful, promote
        the work to a dedicated REFERENCE_SCALES sbatch phase. for now
        the cost is small enough that the login node is the cheapest
        right place.

        Honours acquisition.references.refresh_policy exactly the same way
        the dry-run path does -- the only difference is what the cache
        gets populated with.
        """
        from ..acquisition.trajectory_pool import TrajectoryPool
        from ichor.core.adversarial.acquisition import SeedLocalAdversarialAcquisition
        from ichor.core.models import Models
        from pathlib import Path as _Path
        from .model_contract import validate_reference_scales
        from .artifact_contracts import verify_committed_model_version

        # if we have not committed any models yet (pre-INITIAL_FEREBUS),
        # there is no posterior to sample. leave state alone -- the dry
        # synthetic path will not fire either at this point, and the
        # next call after INITIAL_FEREBUS commits will populate the cache.
        models_version = int(getattr(state, "models_version", -1))
        if models_version < 0:
            return False
        allow_uniform = bool(
            getattr(self.config.acquisition, "allow_uniform_posterior_fallback", False)
        )

        # same policy decision as the dry-run path -- decide whether to
        # recompute. we duplicate the small block here rather than calling
        # the _dry method, because the dry method would clobber the real
        # scales with its synthetic dict if we let it through.
        refs = self.config.acquisition.references
        policy = refs.refresh_policy
        prev_iter = int(getattr(state, "reference_scales_iteration", -1))
        prev_scales = getattr(state, "reference_scales", None)
        should_refresh = False
        if prev_scales is None:
            should_refresh = True
        elif policy == "every_iteration":
            should_refresh = True
        elif policy == "every_n_iterations":
            period = max(1, int(refs.refresh_period))
            should_refresh = (int(state.iteration) - prev_iter) >= period
        elif policy == "never":
            should_refresh = False
        if not should_refresh:
            return False

        # load the committed models for this iteration. trying to do this
        # before the policy check would be wasted work on the no-refresh
        # branch.
        models_dir = (
            _Path(self.campaign_dir)
            / self.models_dir_name
            / ("iteration-" + str(models_version).zfill(4))
        )
        if not models_dir.is_dir():
            message = "reference scales require committed models: " + str(models_dir)
            if allow_uniform:
                self._journal_event(
                    "reference_scales_computed",
                    iteration=int(state.iteration),
                    policy=str(policy),
                    error="models_missing_uniform_fallback_enabled",
                )
                return False
            raise BackendSubmissionError(message)
        try:
            verify_committed_model_version(
                self.campaign_dir,
                models_version,
                models_dir_name=self.models_dir_name,
            )
        except Exception as exc:
            if allow_uniform:
                self._journal_event(
                    "reference_scales_computed",
                    iteration=int(state.iteration),
                    policy=str(policy),
                    error="model_contract_invalid_uniform_fallback_enabled",
                )
                return False
            raise BackendSubmissionError(
                "reference scale model contract failed: "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc

        try:
            models = Models(models_dir)
            pool = TrajectoryPool.load(_Path(self.campaign_dir))
        except Exception as exc:
            self._journal_event(
                "reference_scales_computed",
                iteration=int(state.iteration),
                policy=str(policy),
                error="load_failed: " + str(exc)[:80],
            )
            if allow_uniform:
                return False
            raise BackendSubmissionError(
                "reference scale model/pool load failed: "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc

        # pick a representative anchor geometry to fit the local subspace
        # around. the first frame in the trajectory pool is a fine choice;
        # using a seed from seeds_picked.json would be more principled but
        # requires SEED_SELECT to have already written that file which it
        # has not at the call site.
        anchor_atoms = pool.frame(0)

        acquisition_config = self.config.to_acquisition_config()

        try:
            # building the acquisition triggers _build_reference_scales as
            # a side effect (because we pass external_reference_scales=None).
            acq = SeedLocalAdversarialAcquisition(
                models=models,
                seed=anchor_atoms,
                trajectory=pool,
                config=acquisition_config,
                seed_frame_id=0,
                external_reference_scales=None,
            )
            scales = validate_reference_scales(dict(acq.reference_scales))
        except Exception as exc:
            self._journal_event(
                "reference_scales_computed",
                iteration=int(state.iteration),
                policy=str(policy),
                error="compute_failed: " + str(exc)[:80],
            )
            if allow_uniform:
                return False
            raise BackendSubmissionError(
                "reference scale computation failed: "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc

        # persist to the per-iteration sidecar that ARIADNE_ARRAY tasks
        # read at the top of their main. saves them ~480 GP evaluations
        # per seed.
        iter_dir = (
            _Path(self.campaign_dir)
            / self.al_dir_name
            / ("iteration-" + str(int(state.iteration)).zfill(4))
        )
        iter_dir.mkdir(parents=True, exist_ok=True)
        sidecar = iter_dir / "reference_scales.json"
        try:
            atomic_write_json(sidecar, scales)
        except OSError:
            # disk issue; do not crash the daemon mid-tick.
            pass

        state.reference_scales = scales
        state.reference_scales_iteration = int(state.iteration)
        self._journal_event(
            "reference_scales_computed",
            iteration=int(state.iteration),
            policy=str(policy),
            n_keys=int(len(scales)),
            models_version=models_version,
        )
        return True

    # --- FEREBUS parser body -------------------------------------------

    def _parse_ferebus_postprocess(self, state, phase, observations):
        """Parse FEREBUS output, validate the .model file, commit
        a new 6_TRAINED_MODELS/iteration-NNNN/ via TrainingSetVersioning.

        For INITIAL_FEREBUS this is iteration 0 of both 5_TRAINING and
        6_TRAINED_MODELS (the initial-quantum stage already produced the
        pointdirs that go into 5_TRAINING/iteration-0). For FEREBUS this
        is a per iteration commit of 6_TRAINED_MODELS only.

        """
        from pathlib import Path as _Path
        from .phase_executor import PhaseResult

        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        staging = self._models_staging_path()
        v_models = self._versioning("models")
        committed = v_models.list_committed_versions()
        is_initial = phase_name == "INITIAL_FEREBUS"
        if is_initial:
            expected_next = 0
        else:
            expected_next = int(getattr(state, "training_set_version", -1))
            if expected_next < 0:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "ferebus_training_version_invalid: "
                        + repr(getattr(state, "training_set_version", None))
                    ),
                )
        if expected_next in committed:
            next_version = int(expected_next)
            v_models.ensure_current(next_version)
            committed_dir = v_models.iteration_path(next_version)
            try:
                from .model_contract import validate_ferebus_model_contract
                validate_ferebus_model_contract(
                    committed_dir,
                    committed=True,
                    expected_version=next_version,
                )
            except Exception as exc:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "committed_model_contract_invalid: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)
                    ),
                )
            self._journal_event(
                "models_committed",
                phase=phase_name,
                iteration=int(state.iteration),
                models_version=int(next_version),
                idempotent_skip=True,
            )
            state_updates = {"models_version": int(next_version), "validation_set_version": int(next_version)}
            if is_initial:
                from . import input_staging as _stg
                _stg.commit_initial_training_set(self.campaign_dir)
                state_updates["training_set_version"] = 0
            return PhaseResult(is_complete=True, state_updates=state_updates)

        ok, reason = validate_ferebus_completed(staging)
        if not ok:
            self._journal_event(
                "quantum_output_rejected",
                phase=phase_name,
                iteration=int(state.iteration),
                pointdir=str(staging),
                reason=reason,
            )
            return PhaseResult(
                is_complete=True,
                failure_reason="ferebus_staging_invalid: " + reason,
            )

        try:
            from .ferebus_quality import (
                FEREBUS_QUALITY_MANIFEST,
                evaluate_ferebus_quality,
                write_ferebus_quality_manifest,
            )

            quality = evaluate_ferebus_quality(
                staging,
                getattr(self.config, "quality_gates", None),
            )
            quality_path = write_ferebus_quality_manifest(staging, quality)
            self._journal_event(
                "ferebus_quality_summary",
                phase=phase_name,
                iteration=int(state.iteration),
                manifest=str(quality_path),
                **dict(quality.get("summary") or {}),
            )
            if not bool(quality.get("accepted")):
                return PhaseResult(
                    is_complete=True,
                    failure_reason="ferebus_quality_failed: "
                    + ";".join(str(r) for r in quality.get("reasons", []))[:300],
                )
        except Exception as exc:
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "ferebus_quality_failed: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ),
            )

        v_models.recover_dangling_staging()
        next_version = int(expected_next)
        source_version = None if is_initial else (
            max(committed) if committed else None
        )
        staged = v_models.stage(
            source_version=source_version, target_version=next_version,
        )
        from . import input_staging as _stg
        manifest = _stg.read_ferebus_manifest(staging)
        for task in manifest.get("tasks", []):
            prop = str(task.get("property"))
            atom = str(task.get("atom"))
            model_file = _Path(str(task["expected_model_path"]))
            target_file = staged / model_file.name
            target_file.write_bytes(model_file.read_bytes())
            cfg_file = _Path(str(task["config_path"]))
            if cfg_file.is_file():
                cfg_target = staged / ("ferebus_" + prop + "_" + atom + ".config")
                cfg_target.write_bytes(cfg_file.read_bytes())
        for sidecar_name in (
            _stg.FEREBUS_TASK_MANIFEST,
            _stg.FEREBUS_JOB_DETAILS,
            "commands",
            "list.txt",
            "runFerebus.sh",
            "ATOMS.txt",
            "PROPERTIES.txt",
            FEREBUS_QUALITY_MANIFEST,
        ):
            sidecar = _Path(staging) / sidecar_name
            if sidecar.is_file():
                (staged / sidecar_name).write_bytes(sidecar.read_bytes())
        try:
            from .model_contract import validate_ferebus_model_contract

            validate_ferebus_model_contract(
                staged,
                committed=True,
                expected_version=next_version,
            )
        except Exception as exc:
            self._journal_event(
                "quantum_output_rejected",
                phase=phase_name,
                iteration=int(state.iteration),
                pointdir=str(staged),
                reason="staged_model_contract_invalid: " + str(exc)[:160],
            )
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "staged_model_contract_invalid: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ),
            )
        v_models.commit(next_version)
        v_models.update_current(next_version)
        committed_dir = v_models.iteration_path(next_version)
        try:
            from .model_contract import validate_ferebus_model_contract
            validate_ferebus_model_contract(
                committed_dir,
                committed=True,
                expected_version=next_version,
            )
        except Exception as exc:
            self._journal_event(
                "quantum_output_rejected",
                phase=phase_name,
                iteration=int(state.iteration),
                pointdir=str(committed_dir),
                reason="committed_model_contract_invalid: " + str(exc)[:160],
            )
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "committed_model_contract_invalid: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ),
            )

        state_updates = {"models_version": int(next_version), "validation_set_version": int(next_version)}
        if is_initial:
            # iteration-0 of the training set is normally built at INITIAL_FEREBUS staging now
            # (so the feature export has something to read). call the same helper here too -- it
            # no-ops if staging already did it, and still covers the mock/dry path that never hits
            # the live stager. only THEN advance the version; staging deliberately leaves that to
            # us so a crash between staging and here reconciles cleanly.
            _stg.commit_initial_training_set(self.campaign_dir)
            state_updates["training_set_version"] = 0

        self._journal_event(
            "models_committed",
            phase=phase_name,
            iteration=int(state.iteration),
            models_version=int(next_version),
            idempotent_skip=False,
        )
        self._journal_event(
            "phase_succeeded_live",
            phase=phase_name,
            iteration=int(state.iteration),
            models_version=int(next_version),
        )
        return PhaseResult(is_complete=True, state_updates=state_updates)


    # --- ARIADNE_ARRAY parser body -------------------------------------

    def _parse_ariadne_array_postprocess(self, state, phase, observations):
        """Validate per-seed ARIADNE results and publish ARIADNE_RESULTS.json."""
        from pathlib import Path as _Path
        import json as _json
        from ..handoff_manifests import (
            ARIADNE_RESULTS_SCHEMA_VERSION,
            acquisition_maturity_audit_payload,
            load_seeds_picked,
            validate_ariadne_result,
            write_acquisition_maturity_audit,
            write_ariadne_landing_audit,
            write_ariadne_results_manifest,
        )
        from ..acquisition.ariadne_runner import ariadne_result_usability_payload
        from .phase_executor import PhaseResult

        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        iter_dir = self._iter_dir(state.iteration)
        pool_dir = iter_dir / "pool"
        try:
            picked = load_seeds_picked(iter_dir, expected_iteration=int(state.iteration))
        except Exception as exc:
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "seeds_picked_invalid_for_ariadne: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                ),
            )

        seed_records = list(picked.get("seed_records", []))
        expected_n = int(picked.get("n_picked", len(seed_records)))
        trajectory_pool = None
        try:
            from ..acquisition.trajectory_pool import TrajectoryPool

            trajectory_pool = TrajectoryPool.load(self.campaign_dir)
        except Exception as exc:
            self._journal_event(
                "ariadne_optional_diagnostics_warning",
                phase=phase_name,
                iteration=int(state.iteration),
                seed_dir="pool",
                reason=(
                    "trajectory_pool_unavailable_for_atom_order_check: "
                    + type(exc).__name__
                    + ": "
                    + str(exc)[:160]
                ),
            )
        kept_alphas = []
        flagged_count = 0
        accepted = []
        rejected = []
        landing_audit_records = []

        if not pool_dir.is_dir():
            for seed_record in seed_records:
                seed_dir = pool_dir / ("seed_" + str(int(seed_record["seed_index"])).zfill(4))
                rejected.append({
                    "seed_index": int(seed_record["seed_index"]),
                    "seed_dir": str(seed_dir.resolve()),
                    "reason": "ariadne_pool_missing",
                })
                landing_audit_records.append({
                    "seed_index": int(seed_record["seed_index"]),
                    "seed_dir": str(seed_dir.resolve()),
                    "reason": "ariadne_pool_missing",
                    "handoff_accepted": False,
                    "handoff_rejection_reason": "ariadne_pool_missing",
                })
            write_ariadne_landing_audit(iter_dir, {
                "iteration": int(state.iteration),
                "summary": _ariadne_landing_audit_summary(landing_audit_records),
                "seeds": landing_audit_records,
            })
            write_acquisition_maturity_audit(
                iter_dir,
                acquisition_maturity_audit_payload(
                    iteration=int(state.iteration),
                    seed_records=landing_audit_records,
                ),
            )
            write_ariadne_results_manifest(iter_dir, {
                "schema_version": ARIADNE_RESULTS_SCHEMA_VERSION,
                "iteration": int(state.iteration),
                "trajectory_sha256": str(picked.get("trajectory_sha256", "")),
                "expected_n": int(expected_n),
                "n_accepted": 0,
                "n_rejected": int(len(rejected)),
                "accepted": [],
                "rejected": rejected,
            })
            return PhaseResult(
                is_complete=True,
                failure_reason="ariadne_pool_missing: " + str(pool_dir),
            )

        for seed_record in seed_records:
            seed_index = int(seed_record["seed_index"])
            seed_dir = pool_dir / ("seed_" + str(seed_index).zfill(4))
            result_path = seed_dir / "result.json"
            if not result_path.is_file():
                rejected.append({
                    "seed_index": seed_index,
                    "seed_dir": str(seed_dir.resolve()),
                    "result_json": str(result_path.resolve()),
                    "reason": "missing_result_json",
                })
                landing_audit_records.append({
                    "seed_index": seed_index,
                    "seed_dir": str(seed_dir.resolve()),
                    "result_json": str(result_path.resolve()),
                    "reason": "missing_result_json",
                    "handoff_accepted": False,
                    "handoff_rejection_reason": "missing_result_json",
                })
                self._journal_event(
                    "ariadne_task_rejected_missing_result",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    result_json=str(result_path.resolve()),
                )
                self._journal_event(
                    "quantum_output_rejected",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    pointdir=seed_dir.name,
                    reason="missing_result_json",
                )
                continue
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    result_dict = _json.load(f)
            except (OSError, ValueError):
                rejected.append({
                    "seed_index": seed_index,
                    "seed_dir": str(seed_dir.resolve()),
                    "result_json": str(result_path.resolve()),
                    "reason": "result_json_parse_failure",
                })
                landing_audit_records.append({
                    "seed_index": seed_index,
                    "seed_dir": str(seed_dir.resolve()),
                    "result_json": str(result_path.resolve()),
                    "reason": "result_json_parse_failure",
                    "handoff_accepted": False,
                    "handoff_rejection_reason": "result_json_parse_failure",
                })
                self._journal_event(
                    "ariadne_task_rejected_malformed_result",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    result_json=str(result_path.resolve()),
                    reason="result_json_parse_failure",
                )
                self._journal_event(
                    "quantum_output_rejected",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    pointdir=seed_dir.name,
                    reason="result_json_parse_failure",
                )
                continue

            try:
                seed_frame_id = seed_record.get("frame_id")
                expected_atom_types = None
                if trajectory_pool is not None and seed_frame_id is not None:
                    seed_atoms = trajectory_pool.frame(int(seed_frame_id))
                    expected_atom_types = [str(atom.type) for atom in seed_atoms]
                validated = validate_ariadne_result(
                    result_dict,
                    expected_iteration=int(state.iteration),
                    seed_record=seed_record,
                    expected_atom_types=expected_atom_types,
                    expected_trajectory_sha256=str(picked.get("trajectory_sha256", "")),
                )
            except Exception as exc:
                reason = str(exc) or type(exc).__name__
                rejected.append({
                    "seed_index": seed_index,
                    "seed_dir": str(seed_dir.resolve()),
                    "result_json": str(result_path.resolve()),
                    "reason": reason,
                })
                landing_audit_records.append({
                    "seed_index": seed_index,
                    "seed_dir": str(seed_dir.resolve()),
                    "result_json": str(result_path.resolve()),
                    "reason": reason,
                    "handoff_accepted": False,
                    "handoff_rejection_reason": reason,
                })
                self._journal_event(
                    "ariadne_task_rejected_malformed_result",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    result_json=str(result_path.resolve()),
                    reason=reason,
                )
                self._journal_event(
                    "quantum_output_rejected",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    pointdir=seed_dir.name,
                    reason=reason,
                )
                continue

            optional_diag_warnings = _ariadne_optional_diagnostic_warnings(result_dict)
            if optional_diag_warnings:
                self._journal_event(
                    "ariadne_optional_diagnostics_warning",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    result_json=str(result_path.resolve()),
                    warnings=list(optional_diag_warnings[:8]),
                    n_warnings=int(len(optional_diag_warnings)),
                )

            usability = ariadne_result_usability_payload(
                result_dict,
                allow_seed_fallback=bool(
                    getattr(
                        getattr(self.config, "adversarial_safety", None),
                        "allow_seed_fallback",
                        False,
                    )
                ),
            )

            landing_safety = result_dict.get("landing_safety")
            if not isinstance(landing_safety, dict):
                landing_safety = {
                    "accepted": True,
                    "policy": "legacy_missing_safety",
                    "selected_origin": "legacy_result",
                    "selected_candidate_index": None,
                    "reasons": [],
                    "record_only_reasons": ["legacy_missing_landing_safety"],
                    "metrics": {},
                    "raw_final": {},
                    "n_candidates_evaluated": 0,
                    "n_safe_candidates": 0,
                }
            audit_record = {
                "seed_index": seed_index,
                "seed_dir": str(seed_dir.resolve()),
                "result_json": str(result_path.resolve()),
                "landing_safety": dict(landing_safety),
                "landing_candidates": list(result_dict.get("landing_candidates") or []),
                "task_success": bool(usability.get("usable", False)),
                "task_success_reason": str(usability.get("reason", "")),
            }
            if optional_diag_warnings:
                audit_record["optional_diagnostic_warnings"] = list(
                    optional_diag_warnings
                )
            landing_audit_records.append(audit_record)
            if not bool(usability.get("usable", False)):
                reason = str(usability.get("reason", "ariadne_result_unusable"))
                audit_record["handoff_accepted"] = False
                audit_record["handoff_rejection_reason"] = reason
                rejected.append({
                    "seed_index": seed_index,
                    "seed_dir": str(seed_dir.resolve()),
                    "result_json": str(result_path.resolve()),
                    "reason": reason,
                    "landing_safety": dict(landing_safety),
                    "task_success": False,
                    "task_success_reason": reason,
                })
                self._journal_event(
                    "ariadne_task_rejected_unusable_result",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    result_json=str(result_path.resolve()),
                    return_code=int(validated["return_code"]),
                    reason=reason,
                    policy=str(landing_safety.get("policy", "unknown")),
                )
                self._journal_event(
                    "quantum_output_rejected",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    pointdir=seed_dir.name,
                    reason=reason,
                )
                continue
            if not bool(landing_safety.get("accepted", False)):
                reasons = landing_safety.get("reasons") or ["unsafe_landing"]
                reason = ";".join(str(r) for r in reasons)
                audit_record["handoff_accepted"] = False
                audit_record["handoff_rejection_reason"] = reason
                rejected.append({
                    "seed_index": seed_index,
                    "seed_dir": str(seed_dir.resolve()),
                    "result_json": str(result_path.resolve()),
                    "reason": reason,
                    "landing_safety": dict(landing_safety),
                })
                self._journal_event(
                    "ariadne_task_rejected_unsafe_landing",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    result_json=str(result_path.resolve()),
                    reason=reason,
                    policy=str(landing_safety.get("policy", "unknown")),
                )
                self._journal_event(
                    "ariadne_landing_rejected",
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    reason=reason,
                    policy=str(landing_safety.get("policy", "unknown")),
                )
                self._journal_event(
                    "quantum_output_rejected",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    pointdir=seed_dir.name,
                    reason=reason,
                )
                continue

            if int(validated["return_code"]) != 0:
                self._journal_event(
                    "ariadne_task_salvaged_from_nonzero_exit",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    result_json=str(result_path.resolve()),
                    return_code=int(validated["return_code"]),
                    reason=str(usability.get("reason", "")),
                    policy=str(landing_safety.get("policy", "unknown")),
                )

            geometry_quality = _ariadne_geometry_quality(
                result_dict,
                validated,
                getattr(self.config, "quality_gates", None),
            )
            if not bool(geometry_quality.get("accepted")):
                reason = ";".join(str(r) for r in geometry_quality.get("reasons", []))
                audit_record["handoff_accepted"] = False
                audit_record["handoff_rejection_reason"] = reason
                rejected.append({
                    "seed_index": seed_index,
                    "seed_dir": str(seed_dir.resolve()),
                    "result_json": str(result_path.resolve()),
                    "reason": reason,
                    "geometry_quality": dict(geometry_quality.get("metrics") or {}),
                })
                audit_record["geometry_quality"] = dict(
                    geometry_quality.get("metrics") or {}
                )
                audit_record["geometry_rejection_reason"] = reason
                self._journal_event(
                    "quantum_output_rejected",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    pointdir=seed_dir.name,
                    reason=reason,
                )
                continue

            try:
                prov_path, provenance_reconstructed = self._ensure_ariadne_seed_provenance(
                    state,
                    picked,
                    seed_record,
                )
            except BackendSubmissionError as exc:
                reason = "ariadne_provenance_missing: " + str(exc)
                audit_record["handoff_accepted"] = False
                audit_record["handoff_rejection_reason"] = reason
                rejected.append({
                    "seed_index": seed_index,
                    "seed_dir": str(seed_dir.resolve()),
                    "result_json": str(result_path.resolve()),
                    "provenance_json": str((seed_dir / PROVENANCE_FILENAME).resolve()),
                    "reason": reason,
                })
                self._journal_event(
                    "quantum_output_rejected",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    pointdir=seed_dir.name,
                    reason=reason,
                )
                continue
            audit_record["provenance_json"] = str(prov_path.resolve())
            audit_record["provenance_reconstructed"] = bool(provenance_reconstructed)
            if provenance_reconstructed:
                self._journal_event(
                    "ariadne_provenance_reconstructed",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    provenance_json=str(prov_path.resolve()),
                )

            enrich_with_ariadne(
                seed_dir,
                alpha_initial=float(validated["alpha_initial"]),
                alpha_final=float(validated["alpha_final"]),
                n_evaluations=int(validated["n_evaluations"]),
                fell_back_to_ds=bool(validated["fell_back_to_ds"]),
                wall_seconds=float(validated["wall_seconds"]),
                return_code=int(validated["return_code"]),
            )
            selection_diagnostics = result_dict.get("selection_diagnostics")
            if isinstance(selection_diagnostics, dict):
                diag_payload = dict(selection_diagnostics)
                diag_payload["model_version"] = int(getattr(state, "models_version", -1))
                diag_payload["seed_index"] = int(seed_index)
                diag_payload["seed_frame_id"] = seed_record.get("frame_id")
                diag_payload["result_json"] = str(result_path.resolve())
                diag_payload["landing_policy"] = str(
                    landing_safety.get(
                        "policy",
                        diag_payload.get("landing_policy", "unknown"),
                    )
                )
                diag_payload["safety_metrics"] = dict(
                    landing_safety.get(
                        "metrics",
                        diag_payload.get("safety_metrics", {}),
                    )
                    or {}
                )
                audit_record["selection_diagnostics"] = dict(diag_payload)
                enrich_with_error_calibration_input(seed_dir, diag_payload)

            class _ResultShim:
                def __init__(self, ai, af, alpha_trajectory):
                    self.alpha_initial = ai
                    self.alpha_final = af
                    self.alpha_trajectory = alpha_trajectory

            shim = _ResultShim(
                float(validated["alpha_initial"]),
                float(validated["alpha_final"]),
                list(validated.get("alpha_trajectory") or []),
            )
            d_w_is_synthetic = False
            if validated.get("whitened_distance_final") is not None:
                d_w = float(validated["whitened_distance_final"])
            else:
                d_w = self._synthetic_whitened_distance(shim)
                d_w_is_synthetic = True
            flag = None
            if d_w is not None:
                min_d, max_d = anti_overlap_whitened_distance_bounds(self.config)
                if d_w < min_d:
                    flag = "moved_too_little"
                elif d_w > max_d:
                    flag = "moved_too_far"
            enrich_with_anti_overlap(
                seed_dir,
                min_whitened_distance_to_training=d_w,
                passed=(flag is None),
                flag=flag,
            )

            rejected_by_anti_overlap = False
            if flag is not None:
                flagged_count += 1
                self._journal_event(
                    "anti_overlap_flagged",
                    iteration=int(state.iteration),
                    seed_dir=seed_dir.name,
                    whitened_distance=float(d_w if d_w is not None else 0.0),
                    flag=str(flag),
                    synthetic_distance=bool(d_w_is_synthetic),
                )
                # only ENFORCE a drop on a REAL whitened distance. the synthetic |delta-alpha| proxy
                # is a different physical quantity (a hartree-scale alpha magnitude, not a feature-
                # space std), so discarding real work because it trips the whitened thresholds is
                # meaningless and would spuriously starve the batch (A45). a missing real distance
                # therefore keeps the seed -- it stays a diagnostic flag, never a silent filter.
                # (enforcement is off by default now anyway, see A43.)
                if (
                    not d_w_is_synthetic
                    and getattr(self.config.anti_overlap, "enforce_post_ariadne", False)
                ):
                    rejected_by_anti_overlap = True

            if rejected_by_anti_overlap:
                audit_record["handoff_accepted"] = False
                audit_record["handoff_rejection_reason"] = str(flag)
                rejected.append({
                    "seed_index": seed_index,
                    "seed_dir": str(seed_dir.resolve()),
                    "result_json": str(result_path.resolve()),
                    "provenance_json": str(prov_path.resolve()),
                    "reason": str(flag),
                })
                self._journal_event(
                    "quantum_output_rejected",
                    phase=phase_name,
                    iteration=int(state.iteration),
                    pointdir=seed_dir.name,
                    reason=str(flag),
                )
                continue

            audit_record["handoff_accepted"] = True
            kept_alphas.append(float(validated["alpha_final"]))
            accepted.append({
                "seed_index": seed_index,
                "seed_dir": str(seed_dir.resolve()),
                "result_json": str(result_path.resolve()),
                "provenance_json": str(prov_path.resolve()),
                "seed_frame_id": seed_record.get("frame_id"),
                "selection_index": int(seed_record.get("selection_index", seed_index)),
                "selection_origin": str(seed_record.get("selection_origin", "unknown")),
                "variance_at_selection": seed_record.get("variance_at_selection"),
                "alpha_initial": float(validated["alpha_initial"]),
                "alpha_final": float(validated["alpha_final"]),
                "whitened_distance_final": d_w,
                "geometry_quality": dict(geometry_quality.get("metrics") or {}),
                "landing_safety": dict(landing_safety),
                "landing_policy": str(landing_safety.get("policy", "unknown")),
                "selection_diagnostics": (
                    dict(selection_diagnostics)
                    if isinstance(selection_diagnostics, dict)
                    else None
                ),
                "return_code": int(validated["return_code"]),
                "task_success": bool(usability.get("usable", False)),
                "task_success_reason": str(usability.get("reason", "")),
            })

        n_kept = len(accepted)
        n_rejected = len(rejected)
        audit_summary = _ariadne_landing_audit_summary(landing_audit_records)
        audit_path = write_ariadne_landing_audit(iter_dir, {
            "iteration": int(state.iteration),
            "summary": audit_summary,
            "seeds": landing_audit_records,
        })
        self.artefact_log.append(str(audit_path))
        maturity_path = write_acquisition_maturity_audit(
            iter_dir,
            acquisition_maturity_audit_payload(
                iteration=int(state.iteration),
                seed_records=landing_audit_records,
            ),
        )
        self.artefact_log.append(str(maturity_path))
        manifest_path = write_ariadne_results_manifest(iter_dir, {
            "schema_version": ARIADNE_RESULTS_SCHEMA_VERSION,
            "iteration": int(state.iteration),
            "trajectory_sha256": str(picked.get("trajectory_sha256", "")),
            "expected_n": int(expected_n),
            "n_accepted": int(n_kept),
            "n_rejected": int(n_rejected),
            "accepted": accepted,
            "rejected": rejected,
        })
        self.artefact_log.append(str(manifest_path))
        self._journal_event(
            "ariadne_landing_summary",
            iteration=int(state.iteration),
            accepted=int(audit_summary.get("accepted", 0)),
            salvaged=int(audit_summary.get("salvaged", 0)),
            backtracked=int(audit_summary.get("backtracked", 0)),
            rejected=int(audit_summary.get("rejected", 0)),
        )

        if n_kept == 0:
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "ariadne_no_seed_results_parsed: " + str(n_rejected)
                ),
            )

        # too many seeds lost (real failures + absentees) against the TRUE submitted count -> fail
        # rather than quietly commit a short batch as if the array had finished. only gated when we
        # actually know the submitted count (seeds_picked.json present); mirrors the quantum phases.
        if expected_n and (n_rejected / float(expected_n)) > float(self.config.failure_threshold_fraction):
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "too_many_seeds_failed: " + str(n_rejected) + "/" + str(expected_n)
                ),
            )

        # per-iteration convergence scalar: a high quantile (p90) over the kept seeds, not the max.
        # tracks the worst-but-one rather than letting one intrinsically-hard seed pin it high
        # forever (A17), and being an order statistic it never reports a value no seed had (A44).
        # n_kept > 0 is guaranteed here, so the list is non-empty.
        last_alpha = _high_quantile(kept_alphas)
        self._journal_event(
            "phase_succeeded_live",
            phase=phase_name,
            iteration=int(state.iteration),
            n_kept=int(n_kept),
            n_rejected=int(n_rejected),
            last_acquisition_alpha0=float(last_alpha),
        )
        return PhaseResult(
            is_complete=True,
            state_updates={
                "last_acquisition_alpha0": float(last_alpha),
                "last_n_anti_overlap_flagged": int(flagged_count),
            },
        )


    #-- POLUS parser body --------------------------------------

    def _parse_polus_postprocess(self, state, phase, observations):
        """Parse the POLUS Phase-A or Phase-B sample output.

        Phase A reads 3_DIVERSITY_SAMPLING/initial/PHASE_A_SAMPLE.json.
        Phase B reads 7_ACTIVE_LEARNING/iteration-N/phase_b_SAMPLE.xyz and
        additionally enriches every seed_*/.provenance.json with the
        phase_b block (selected_after_fps + diversity_rank).

        Validation is presence + parseability via _count_xyz_frames. If the
        sample xyz is empty or unreadable, return a failure reason and
        let _handle_failure decide the next action.
        """
        from pathlib import Path as _Path
        from .phase_executor import PhaseResult

        phase_name = phase.value if hasattr(phase, "value") else str(phase)

        if phase_name == "PHASE_A_POLUS":
            outdir = (
                _Path(self.campaign_dir) / self.diversity_dir_name / "initial"
            )
            if not outdir.is_dir():
                return PhaseResult(
                    is_complete=True,
                    failure_reason="phase_a_outdir_missing: " + str(outdir),
                )
            from ..handoff_manifests import read_phase_a_sample_manifest
            try:
                phase_a_manifest = read_phase_a_sample_manifest(outdir)
            except Exception as exc:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "phase_a_sample_manifest_invalid: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)
                    ),
                )
            sample = _Path(str(phase_a_manifest["sample_xyz"]))
        else:
            iter_dir = self._iter_dir(state.iteration)
            from ..handoff_manifests import read_phase_b_selection_manifest
            try:
                phase_b_manifest = read_phase_b_selection_manifest(
                    iter_dir,
                    expected_iteration=int(state.iteration),
                )
            except Exception as exc:
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "phase_b_selection_manifest_invalid: "
                        + type(exc).__name__
                        + ": "
                        + str(exc)
                    ),
                )
            sample = iter_dir / "phase_b_SAMPLE.xyz"
            if not sample.is_file():
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "phase_b_sample_missing: "
                        + str(iter_dir / "phase_b_SAMPLE.xyz")
                    ),
                )

        n_frames = self._count_xyz_frames(sample)
        if n_frames is None or n_frames <= 0:
            return PhaseResult(
                is_complete=True,
                failure_reason=(
                    "polus_sample_unreadable_or_empty: " + str(sample)
                ),
            )

        if phase_name == "PHASE_A_POLUS":
            expected = int(phase_a_manifest.get("n_select", 0))
            declared = phase_a_manifest.get("n_frames")
            if int(n_frames) != expected or (
                declared is not None and int(declared) != int(n_frames)
            ):
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "phase_a_sample_count_mismatch: sample_frames="
                        + str(int(n_frames))
                        + " manifest_n_select="
                        + str(expected)
                    ),
                )

        if phase_name == "PHASE_B_POLUS":
            final_records = list(phase_b_manifest.get("final", []))
            if int(n_frames) != len(final_records):
                return PhaseResult(
                    is_complete=True,
                    failure_reason=(
                        "phase_b_selection_count_mismatch: sample_frames="
                        + str(int(n_frames))
                        + " manifest_final="
                        + str(len(final_records))
                    ),
                )
            coordinate_error = self._phase_b_sample_coordinate_mismatch(
                sample,
                final_records,
            )
            if coordinate_error is not None:
                return PhaseResult(
                    is_complete=True,
                    failure_reason="phase_b_selection_content_mismatch: " + coordinate_error,
                )
            for rec in final_records:
                seed_dir = _Path(str(rec["seed_dir"]))
                enrich_with_phase_b(
                    seed_dir,
                    selected_after_fps=True,
                    diversity_rank=int(rec["final_index"]),
                    descriptor_used=str(phase_b_manifest.get("descriptor", self.config.phase_b.descriptor)),
                )

        # if the Phase-B dedup ran, surface its counts in the journal.
        dedup_payload = {}
        if phase_name == "PHASE_B_POLUS":
            d = phase_b_manifest.get("dedup", {}) if isinstance(phase_b_manifest, dict) else {}
            if isinstance(d, dict):
                dedup_payload = {
                    "n_kept": int(d.get("n_kept", 0)),
                    "n_dropped": int(d.get("n_dropped", 0)),
                    "min_separation": float(d.get("min_separation", 0.0)),
                }
        self._journal_event(
            "phase_succeeded_live",
            phase=phase_name,
            iteration=int(state.iteration),
            sample_path=str(sample),
            n_frames=int(n_frames),
            **dedup_payload,
        )
        return PhaseResult(is_complete=True, state_updates={})

    def _count_xyz_frames(self, sample_path):
        """Count frames in an xyz file by reading natom  blocks.

        Returns None if the file is unreadable or malformed; an integer
        otherwise.  On CSF4 the real POLUS
        output passes Trajectory.read trivially. We dont require ASE-
        valid xyz here because POLUS may emit a minimal subset.
        """
        from pathlib import Path as _Path
        try:
            text = _Path(sample_path).read_text(encoding="utf-8")
        except OSError:
            return None
        lines = text.splitlines()
        i = 0
        count = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue
            try:
                natoms = int(line)
            except ValueError:
                i += 1
                continue
            if natoms <= 0:
                return None
            i += 2 + natoms
            if i > len(lines):
                return None
            count += 1
        return count

    def _read_xyz_records(self, sample_path):
        from pathlib import Path as _Path

        text = _Path(sample_path).read_text(encoding="utf-8")
        lines = text.splitlines()
        frames = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue
            try:
                natoms = int(line)
            except ValueError:
                raise ValueError("XYZ frame atom count is not an integer")
            if natoms <= 0:
                raise ValueError("XYZ frame atom count must be positive")
            if i + 2 + natoms > len(lines):
                raise ValueError("XYZ frame is truncated")
            atoms = []
            coords = []
            for raw in lines[i + 2 : i + 2 + natoms]:
                parts = raw.split()
                if len(parts) < 4:
                    raise ValueError("XYZ atom line has fewer than four columns")
                atoms.append(str(parts[0]))
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            frames.append({"atom_types": atoms, "coordinates": coords})
            i += 2 + natoms
        return frames

    def _phase_b_sample_coordinate_mismatch(self, sample_path, final_records):
        import json as _json
        import math as _math

        try:
            frames = self._read_xyz_records(sample_path)
        except Exception as exc:
            return "sample_xyz_unreadable: " + type(exc).__name__ + ": " + str(exc)
        if len(frames) != len(final_records):
            return "sample frame count differs from final records"
        tolerance = 5.0e-6
        for index, (frame, rec) in enumerate(zip(frames, final_records)):
            result_path = Path(str(rec.get("result_json", "")))
            try:
                result = _json.loads(result_path.read_text(encoding="utf-8"))
            except Exception as exc:
                return (
                    "result_json_unreadable for final_index "
                    + str(index)
                    + ": "
                    + type(exc).__name__
                )
            expected_atoms = [str(x) for x in (result.get("atom_types") or [])]
            expected_coords = result.get("final_coordinates") or []
            if frame["atom_types"] != expected_atoms:
                return "atom order mismatch at final_index " + str(index)
            if len(frame["coordinates"]) != len(expected_coords):
                return "coordinate row count mismatch at final_index " + str(index)
            for atom_i, (actual, expected) in enumerate(zip(frame["coordinates"], expected_coords)):
                if not isinstance(expected, list) or len(expected) != 3:
                    return "result coordinate shape mismatch at final_index " + str(index)
                for axis, (a, e) in enumerate(zip(actual, expected)):
                    if not _math.isclose(float(a), float(e), rel_tol=0.0, abs_tol=tolerance):
                        return (
                            "coordinate mismatch at final_index "
                            + str(index)
                            + " atom "
                            + str(atom_i)
                            + " axis "
                            + str(axis)
                        )
        return None



# ---module level: sbatch script builder -----------------


def make_live_job_finder(sacct_runner=None, squeue_runner=None):
    """the job_finder the daemon uses in live mode: given (state, phase) return the JobID of an
    already-running job for that exact phase+iteration, or None. lets the daemon adopt a job a crash
    orphaned rather than double-submit (A24/A25)."""
    from ..submit.sacct_poll import JobNameLookup, find_running_job_by_name_detailed

    def _finder(state, phase):
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        uid = getattr(state, "campaign_uid", None)
        iteration = getattr(state, "iteration", 0)
        names = [
            live_job_name(uid, phase_name, iteration),
        ]
        if uid:
            legacy = str(uid)[:8] + "-" + str(phase_name) + "-" + str(int(iteration))
            if legacy not in names:
                names.append(legacy)
        inconclusive: Optional[JobNameLookup] = None
        last_lookup = JobNameLookup(None, inconclusive=False)
        use_squeue_fallback = squeue_runner is not None or sacct_runner is None
        for name in names:
            found = find_running_job_by_name_detailed(
                name,
                sacct_runner=sacct_runner,
                squeue_runner=squeue_runner,
                use_squeue_fallback=use_squeue_fallback,
            )
            last_lookup = found
            if found.job_id:
                return found
            if found.inconclusive and inconclusive is None:
                inconclusive = found
        return inconclusive if inconclusive is not None else last_lookup

    return _finder


def make_live_job_accounting_finder(sacct_runner=None, squeue_runner=None):
    """Return a live-mode expected-job-name accounting lookup."""
    from ..submit.sacct_poll import find_accounted_job_by_name_detailed

    def _finder(state, phase, active_intent):
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        uid = getattr(state, "campaign_uid", None)
        iteration = getattr(state, "iteration", 0)
        expected_tasks = active_intent.get("expected_tasks")
        names = [str(active_intent.get("expected_job_name") or "")]
        live_name = live_job_name(uid, phase_name, iteration)
        if live_name not in names:
            names.append(live_name)
        if uid:
            legacy = str(uid)[:8] + "-" + str(phase_name) + "-" + str(int(iteration))
            if legacy not in names:
                names.append(legacy)
        names = [name for name in names if name]
        inconclusive = None
        last_lookup = None
        use_squeue_fallback = squeue_runner is not None or sacct_runner is None
        for name in names:
            found = find_accounted_job_by_name_detailed(
                name,
                expected_task_count=(
                    None if expected_tasks is None else int(expected_tasks)
                ),
                sacct_runner=sacct_runner,
                squeue_runner=squeue_runner,
                use_squeue_fallback=use_squeue_fallback,
            )
            last_lookup = found
            if found.job_id:
                return found
            if found.inconclusive and inconclusive is None:
                inconclusive = found
        return inconclusive if inconclusive is not None else last_lookup

    return _finder


def make_live_job_liveness_checker(squeue_runner=None):
    """Return a live-mode checker for whether an existing Slurm JobID is active."""
    from ..submit.sacct_poll import find_active_job_by_id_detailed

    def _checker(job_id):
        return find_active_job_by_id_detailed(job_id, squeue_runner=squeue_runner)

    return _checker


def _reject_shell_control_chars(label: str, value: str) -> None:
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise BackendSubmissionError(label + " contains a control character")


def _shell_quote(value: Any) -> str:
    text = str(value)
    _reject_shell_control_chars("shell value", text)
    return shlex.quote(text)


def _shell_executable(value: Any) -> str:
    text = str(value)
    _reject_shell_control_chars("shell executable", text)
    if "$" in text and _SHELL_PATH_FRAGMENT_RE.fullmatch(text):
        return text
    return shlex.quote(text)


def _python_executable_for_script() -> str:
    python_path = profile_value(
        "software", "python", "python_path", default=None
    )
    return _shell_executable(python_path or sys.executable)


def _normalise_module_list(raw: Any, *, label: str) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [raw]
    else:
        try:
            values = list(raw)
        except TypeError as exc:
            raise BackendSubmissionError(
                "configured " + label + " modules must be a string or list"
            ) from exc
    modules: List[str] = []
    for value in values:
        module = str(value).strip()
        if not module:
            continue
        _reject_shell_control_chars("configured " + label + " module", module)
        if not _MODULE_TOKEN_RE.fullmatch(module):
            raise BackendSubmissionError(
                "configured "
                + label
                + " module contains unsafe characters: "
                + repr(module)
            )
        modules.append(module)
    return modules


def _configured_jobscript_shebang() -> str:
    raw = profile_value("hpc", "jobscript_shebang", default=None)
    if raw is None:
        return "#!/bin/bash --login" if active_machine() else "#!/bin/bash"
    value = str(raw).strip()
    _reject_shell_control_chars("configured hpc.jobscript_shebang", value)
    if not _SHEBANG_RE.fullmatch(value):
        raise BackendSubmissionError(
            "configured hpc.jobscript_shebang is unsafe: " + repr(value)
        )
    return value


def _configured_max_array_task_id() -> Optional[int]:
    raw = profile_value("hpc", "max_array_task_id", default=None)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise BackendSubmissionError(
            "configured hpc.max_array_task_id must be an integer"
        ) from exc
    if value < 0:
        raise BackendSubmissionError(
            "configured hpc.max_array_task_id must be >= 0"
        )
    return value


def _configured_scheduler() -> str:
    if active_machine() == "_default":
        raise BackendSubmissionError(
            "_default is a fallback configuration, not a live active-learning "
            "profile; set ICHOR_MACHINE to a real Slurm profile such as csf3 "
            "or csf4"
        )
    raw = profile_value("hpc", "scheduler", default=None)
    if raw is None:
        return "slurm"
    value = str(raw).strip().lower()
    _reject_shell_control_chars("configured hpc.scheduler", value)
    if value != "slurm":
        raise BackendSubmissionError(
            "active-learning live mode supports hpc.scheduler='slurm'; got "
            + repr(value)
        )
    return value


def _configured_ferebus_platform() -> str:
    raw = profile_value(
        "software", "ferebus", "pyferebus_platform", default=None
    )
    if raw:
        value = str(raw).strip()
    else:
        machine = active_machine()
        if machine:
            raise BackendSubmissionError(
                "live FEREBUS requires software.ferebus.pyferebus_platform "
                "in ichor_config.yaml"
            )
        value = "CSF4"
    _reject_shell_control_chars("configured ferebus pyferebus_platform", value)
    if not re.fullmatch(r"^[A-Za-z0-9_.-]+$", value):
        raise BackendSubmissionError(
            "configured ferebus pyferebus_platform is unsafe: " + repr(value)
        )
    return value


def _configured_daemon_runtime_modules() -> List[str]:
    """Modules loaded by daemon-owned live sbatch scripts.

    Python and ARIADNE/MKL runtime modules are kept in ichor_config.yaml so a
    cluster-module update does not require a code edit. Missing config falls
    back to the current CSF4 stack for off-cluster tests and legacy configs.
    FEREBUS is not included: pyferebus writes and submits its own script for
    FEREBUS phases.
    """
    try:
        python_modules = profile_value(
            "software", "python", "modules", default=None
        )
        ariadne_modules = profile_value(
            "software", "ariadne_runtime", "modules", default=None
        )
    except Exception:
        return list(DEFAULT_DAEMON_RUNTIME_MODULES)

    machine = (active_machine() or "").lower()
    legacy_defaults = (not machine) or machine == "csf4"
    default_python_modules = (
        list(DEFAULT_DAEMON_PYTHON_MODULES) if legacy_defaults else []
    )
    default_ariadne_modules = (
        list(DEFAULT_DAEMON_ARIADNE_RUNTIME_MODULES) if legacy_defaults else []
    )
    modules = (
        (_normalise_module_list(python_modules, label="python")
         if python_modules is not None else default_python_modules)
        + (_normalise_module_list(ariadne_modules, label="ariadne_runtime")
           if ariadne_modules is not None else default_ariadne_modules)
    )
    return modules


def build_sbatch_script(
    *,
    phase_name: str,
    iteration: int,
    campaign_dir: Path,
    config: CampaignConfig,
    array_size: Optional[int] = None,
    walltime_hours: Optional[int] = None,
    partition: Optional[str] = None,
    campaign_uid: Optional[str] = None,
    resolved_resources: Optional[ResolvedPhaseResources] = None,
) -> str:
    """Return the body of an sbatch script for the given phase.

    Resources come from config.resources (partition / walltime / mem /
    cpus-per-task / ntasks); walltime_hours and partition may still be passed
    to override them. array_size, when given, turns the job into a
    0..array_size-1 SLURM array -- one task per staged point or seed.

    Paths are absolute (resolved campaign dir) so the script does not depend
    on sbatch being launched from any particular directory.
    """
    _configured_scheduler()
    res = config.resources
    part = str(partition if partition is not None else res.partition)
    resolved = resolved_resources or resolve_phase_resources(
        phase_name=phase_name,
        config=config,
        partition=part,
        campaign_dir=campaign_dir,
        iteration=int(iteration),
        array_size=array_size,
    )
    wall = walltime_hours if walltime_hours is not None else res.walltime_for(phase_name)
    cpus = int(resolved.cpus_per_task)
    ntasks = int(resolved.ntasks)
    mem_per_cpu = str(resolved.mem_per_cpu)
    if phase_name in ("INITIAL_GAUSSIAN", "GAUSSIAN"):
        validate_gaussian_link0_memory(config, resolved)
    camp = str(Path(campaign_dir).resolve())
    try:
        job_name = live_job_name(campaign_uid, phase_name, iteration)
    except ValueError as exc:
        raise BackendSubmissionError(str(exc)) from exc
    _reject_shell_control_chars("Slurm job name", job_name)
    logs = camp + "/.DATA/SCRIPTS"
    is_array = array_size is not None and int(array_size) > 0
    if is_array:
        max_array_task_id = _configured_max_array_task_id()
        highest_task_id = int(array_size) - 1
        if max_array_task_id is not None and highest_task_id > max_array_task_id:
            raise BackendSubmissionError(
                "array task id "
                + str(highest_task_id)
                + " exceeds configured hpc.max_array_task_id "
                + str(max_array_task_id)
            )
    tag = ".%A_%a" if is_array else ".%j"
    lines: List[str] = [
        _configured_jobscript_shebang(),
        "#SBATCH --job-name=" + job_name,
        "#SBATCH --partition=" + str(resolved.partition),
        "#SBATCH --time=" + str(int(wall)) + ":00:00",
        "#SBATCH --mem-per-cpu=" + str(mem_per_cpu),
        "#SBATCH --cpus-per-task=" + str(int(cpus)),
        "#SBATCH --ntasks=" + str(int(ntasks)),
    ]
    if is_array:
        throttle = getattr(res, "array_concurrency_limit", None)
        array_spec = "0-" + str(int(array_size) - 1)
        if throttle is not None:
            try:
                throttle_i = int(throttle)
            except (TypeError, ValueError) as exc:
                raise BackendSubmissionError(
                    "resources.array_concurrency_limit must be a positive integer"
                ) from exc
            if throttle_i <= 0:
                raise BackendSubmissionError(
                    "resources.array_concurrency_limit must be > 0"
                )
            throttle_i = min(throttle_i, int(array_size))
            array_spec += "%" + str(throttle_i)
        lines.append("#SBATCH --array=" + array_spec)
    lines += [
        "#SBATCH --output=" + logs + "/OUTPUTS/" + job_name + tag + ".o",
        "#SBATCH --error="  + logs + "/ERRORS/"  + job_name + tag + ".e",
        "",
        "# Resolved ICHOR resources: backend="
        + str(resolved.backend)
        + " cpu_reason="
        + str(resolved.cpu_reason)
        + " memory_reason="
        + str(resolved.memory_reason),
        "set -euo pipefail",
        "export LC_ALL=C",
        "export LC_NUMERIC=C",
        "",
        *["module load " + m for m in _configured_daemon_runtime_modules()],
        "",
    ]
    bucket = "initial" if phase_name.startswith("INITIAL_") else ("iter_" + str(iteration))
    points_file = camp + "/.DATA/STAGING/" + bucket + "/POINTS.txt"

    if phase_name in ("INITIAL_GAUSSIAN", "GAUSSIAN"):
        lines += _gaussian_invocation_block(
            phase_name,
            iteration,
            camp,
            config,
            points_file,
            resolved_resources=resolved,
        )
    elif phase_name in ("INITIAL_AIMALL", "AIMALL"):
        lines += _aimall_invocation_block(iteration, camp, config, points_file)
    elif phase_name in ("INITIAL_FEREBUS", "FEREBUS"):
        lines += _ferebus_invocation_block(iteration, camp, config)
    elif phase_name == "ARIADNE_ARRAY":
        lines += _ariadne_invocation_block(iteration, camp, config)
    elif phase_name in ("PHASE_A_POLUS", "PHASE_B_POLUS"):
        lines += _polus_invocation_block(phase_name, iteration, camp, config)
    else:
        lines.append("# No invocation block registered for phase " + phase_name)
        lines.append("exit 1")

    lines.append("")
    return "\n".join(lines)


def _gaussian_invocation_block(
    phase_name,
    iteration,
    camp,
    config,
    points_file,
    *,
    resolved_resources: ResolvedPhaseResources,
) -> List[str]:
    gaussian_modules = _configured_backend_modules(
        "gaussian",
        ["gaussian/g16c01_em64t_detectcpu"],
    )
    gaussian_exe = _configured_backend_shell_executable("gaussian", "g16")
    mdef_gb = gaussian_mdef_gb(config, resolved_resources)
    gaussian_memory_mode = str(config.resources.gaussian_memory_mode).strip().lower()
    memory_lines: List[str]
    if gaussian_memory_mode == "slurm_env":
        memory_lines = [
            'export GAUSS_PDEF="${SLURM_CPUS_PER_TASK:-1}"',
            "export GAUSS_MDEF=" + str(int(mdef_gb)) + "GB",
        ]
    else:
        memory_lines = [
            "# Gaussian Link0 memory/core directives are written in input.gjf.",
        ]
    points_file_q = _shell_quote(points_file)
    camp_q = _shell_quote(camp)
    phase_q = _shell_quote(str(phase_name))
    return [
        *["module load " + m for m in gaussian_modules],
        "",
        "# per-point gaussian array: task N runs the Nth staged pointdir.",
        "export ICHOR_CAMPAIGN_DIR=" + camp_q,
        "export ICHOR_GAUSSIAN_PHASE=" + phase_q,
        "export ICHOR_ITERATION=" + str(int(iteration)),
        'export GAUSS_SCRDIR="${ICHOR_CAMPAIGN_DIR}/.DATA/SCRATCH/GAUSSIAN/${ICHOR_GAUSSIAN_PHASE}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"',
        *memory_lines,
        'mkdir -p "$GAUSS_SCRDIR"',
        'echo "GAUSS_SCRDIR=$GAUSS_SCRDIR"',
        "cleanup_gaussian_scratch_success() {",
        '  case "$GAUSS_SCRDIR" in',
        '    "$ICHOR_CAMPAIGN_DIR"/.DATA/SCRATCH/GAUSSIAN/*/"$SLURM_JOB_ID"_*) rm -rf -- "$GAUSS_SCRDIR" ;;',
        '    *) echo "Refusing to remove unexpected Gaussian scratch path: $GAUSS_SCRDIR" >&2 ;;',
        "  esac",
        "}",
        # check the file FIRST -- under set -e a failing sed (missing POINTS.txt) aborts the
        # assignment before the friendly -z guard below ever runs, leaving just a bare sed error.
        "if [ ! -f " + points_file_q + " ]; then echo "
        + _shell_quote("POINTS.txt missing: " + points_file)
        + " >&2; exit 1; fi",
        'POINT_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ' + points_file_q + ")",
        'if [ -z "$POINT_DIR" ]; then echo "no pointdir for index $SLURM_ARRAY_TASK_ID" >&2; exit 1; fi',
        'case "$POINT_DIR" in "$ICHOR_CAMPAIGN_DIR"/.DATA/STAGING/initial/POINT_*.pointdir|"$ICHOR_CAMPAIGN_DIR"/.DATA/STAGING/iter_"$ICHOR_ITERATION"/POINT_*.pointdir) ;; *) echo "POINT_DIR escapes campaign staging: $POINT_DIR" >&2; exit 1 ;; esac',
        'if [ -L "$POINT_DIR" ]; then echo "POINT_DIR is a symlink: $POINT_DIR" >&2; exit 1; fi',
        'if [ ! -f "$POINT_DIR/input.gjf" ]; then echo "input.gjf missing in $POINT_DIR" >&2; exit 1; fi',
        'cd "$POINT_DIR"',
        "GAUSSIAN_EXIT=0",
        gaussian_exe + " < input.gjf > input.gau || GAUSSIAN_EXIT=$?",
        'if [ "$GAUSSIAN_EXIT" -eq 0 ]; then',
        "  cleanup_gaussian_scratch_success",
        "else",
        '  echo "Gaussian failed; keeping scratch at $GAUSS_SCRDIR" >&2',
        '  exit "$GAUSSIAN_EXIT"',
        "fi",
    ]


def _configured_backend_path(backend_name: str, fallback: str) -> str:
    """Look up the executable_path for a software backend from the operator's
    ~/ichor_config.yaml. Falls back to a sensible default (usually the bare
    command) when no active profile/backend path is declared, so unit tests
    and non-cluster environments keep working.
    """
    try:
        raw = expanded_profile_value(
            "software", backend_name, "executable_path", default=None
        )
    except Exception:
        return fallback
    if not raw:
        return fallback
    value = os.path.expanduser(os.path.expandvars(str(raw)))
    _reject_shell_control_chars(
        "configured " + backend_name + " executable_path",
        value,
    )
    return value


def _configured_backend_shell_executable(backend_name: str, fallback: str) -> str:
    try:
        raw = profile_value(
            "software", backend_name, "executable_path", default=None
        )
    except Exception:
        raw = None
    value = os.path.expanduser(str(raw or fallback))
    _reject_shell_control_chars(
        "configured " + backend_name + " executable_path",
        value,
    )
    return _shell_executable(value)


def _configured_backend_modules(backend_name: str, fallback: List[str]) -> List[str]:
    try:
        raw = profile_value("software", backend_name, "modules", default=None)
    except Exception:
        return list(fallback)
    if raw is None:
        return list(fallback)
    return _normalise_module_list(raw, label=backend_name)


def _aimall_invocation_block(iteration, camp, config, points_file) -> List[str]:
    aimall_path = _configured_backend_path("aimall", "~/AIMAll/aimqb.ish")
    aimall_cfg = getattr(config, "aimall", None)
    args: List[str] = []
    if bool(getattr(aimall_cfg, "nogui", True)):
        args.append("-nogui")
    args.append('-nproc="${SLURM_CPUS_PER_TASK:-1}"')
    args.append('-naat="$AIMALL_NAAT"')
    encomp = int(getattr(aimall_cfg, "encomp", 3))
    args.append("-encomp=" + str(encomp))
    boaq = str(getattr(aimall_cfg, "boaq", "auto")).strip().lower()
    if boaq not in VALID_AIMALL_BOAQ_VALUES:
        raise BackendSubmissionError("aimall.boaq is invalid: " + repr(boaq))
    args.append("-boaq=" + boaq)
    iasmesh = str(getattr(aimall_cfg, "iasmesh", "fine")).strip().lower()
    if iasmesh not in VALID_AIMALL_IASMESH_VALUES:
        raise BackendSubmissionError("aimall.iasmesh is invalid: " + repr(iasmesh))
    args.append("-iasmesh=" + iasmesh)
    points_file_q = _shell_quote(points_file)
    camp_q = _shell_quote(camp)
    python = _python_executable_for_script()
    return [
        "# per-point AIMAll array over the .wfn files gaussian produced.",
        "export ICHOR_CAMPAIGN_DIR=" + camp_q,
        "export ICHOR_ITERATION=" + str(int(iteration)),
        # check the file FIRST -- under set -e a failing sed (missing POINTS.txt) aborts the
        # assignment before the friendly -z guard below ever runs, leaving just a bare sed error.
        "if [ ! -f " + points_file_q + " ]; then echo "
        + _shell_quote("POINTS.txt missing: " + points_file)
        + " >&2; exit 1; fi",
        'POINT_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ' + points_file_q + ")",
        'if [ -z "$POINT_DIR" ]; then echo "no pointdir for index $SLURM_ARRAY_TASK_ID" >&2; exit 1; fi',
        'case "$POINT_DIR" in "$ICHOR_CAMPAIGN_DIR"/.DATA/STAGING/initial/POINT_*.pointdir|"$ICHOR_CAMPAIGN_DIR"/.DATA/STAGING/iter_"$ICHOR_ITERATION"/POINT_*.pointdir) ;; *) echo "POINT_DIR escapes campaign staging: $POINT_DIR" >&2; exit 1 ;; esac',
        'if [ -L "$POINT_DIR" ]; then echo "POINT_DIR is a symlink: $POINT_DIR" >&2; exit 1; fi',
        'if [ ! -f "$POINT_DIR/input.wfn" ]; then echo "input.wfn missing in $POINT_DIR" >&2; exit 1; fi',
        'cd "$POINT_DIR"',
        'if [ ! -f AIMALL_TASK.json ]; then echo "AIMALL_TASK.json missing in $POINT_DIR" >&2; exit 1; fi',
        "AIMALL_NAAT=$("
        + python
        + " -c "
        + _shell_quote(
            "import json; print(int(json.load(open('AIMALL_TASK.json', encoding='utf-8'))['naat']))"
        )
        + ")",
        'if [ -z "$AIMALL_NAAT" ]; then echo "AIMALL_NAAT is empty in $POINT_DIR" >&2; exit 1; fi',
        " ".join([_shell_quote(aimall_path)] + args + ["input.wfn"]),
    ]


def _ferebus_invocation_block(iteration, camp, config) -> List[str]:
    return [
        "# FEREBUS live phases are submitted through pyferebus_wrap.submit_ferebus().",
        "# This generic daemon sbatch renderer is intentionally not used for FEREBUS.",
        'echo "FEREBUS must be submitted via pyferebus wrapper, not build_sbatch_script" >&2',
        "exit 2",
    ]


def _ariadne_invocation_block(iteration, camp, config) -> List[str]:
    # single-threaded BLAS so the acquisition-gradient process-pool owns the
    # cores SLURM gave this task.
    python = _python_executable_for_script()
    camp_q = _shell_quote(camp)
    return [
        "# ARIADNE per-seed adversarial attack array.",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        "cd " + camp_q,
        python + " -m ichor.hpc.active_learning.acquisition.ariadne_runner \\",
        "    --seed-index $SLURM_ARRAY_TASK_ID \\",
        "    --iteration " + str(iteration) + " \\",
        "    --campaign-dir " + camp_q,
    ]


def _polus_invocation_block(phase_name, iteration, camp, config) -> List[str]:
    descriptor = (
        "rmsd_massweight"
        if phase_name == "PHASE_A_POLUS"
        else config.phase_b.descriptor
    )
    wrapper_iteration = -1 if phase_name == "PHASE_A_POLUS" else int(iteration)
    python = _python_executable_for_script()
    camp_q = _shell_quote(camp)
    return [
        "# POLUS diversity sub-sample (" + phase_name + ").",
        "cd " + camp_q,
        python + " -m ichor.hpc.active_learning.sampling.polus_wrapper \\",
        "    --descriptor " + _shell_quote(descriptor) + " \\",
        "    --iteration " + str(wrapper_iteration) + " \\",
        "    --campaign-dir " + camp_q,
    ]
