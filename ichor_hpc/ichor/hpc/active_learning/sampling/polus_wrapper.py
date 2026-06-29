"""POLUS wrapper providing a pluggable distance matrix.

Two layers:

* fps_select(distance_matrix, n_select, seed_index=None):
    Standalone greedy furthes-point sampling over an arbitrary distance
    matrix. Used in unit tests and as a small-data path that does not need
    the full POLUS install. Returns FPSResult (indices + diversities).


* make_pluggable_polus_sampler(precomputed_distance_matrix, **divsampler_kw):
    Lazily imports polus.trajectories.diversity.DIVSampler and returns a
    subclass that overrides SetRMSDMatrix to inject a caller-supplied
    distance matrix. Used in production for large trajectories where
    POLUS handles file I/O and the FPS step. The caller computes the
    distance matrix with whichever Descriptor they want.

Both layers produce the same selection sequence given the same input
(modulo any seed-geometry choice POLUS makes from ComputeCentroid).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ichor.core.atoms import Atoms

from .descriptors import Descriptor, MassWeightedRMSDDescriptor


__all__ = [
    "fps_select",
    "FPSResult",
    "make_pluggable_polus_sampler",
    "DEFAULT_DESCRIPTORS",
]


DEFAULT_DESCRIPTORS = {
    "rmsd_massweight": MassWeightedRMSDDescriptor,
}
FPS_TIE_QUANTISATION = 1.0e-12






@dataclass(frozen=True)
class FPSResult:
    indices: List[int]
    diversities: List[float]
    distance_matrix_shape: Tuple[int, int]
    descriptor_name: str

    @property
    def n(self) -> int:
        return len(self.indices)


def fps_select(
    distance_matrix: np.ndarray,
    n_select: int,
    seed_index: Optional[int] = None,
    descriptor_name: str = "unknown",
) -> FPSResult:
    D = np.asarray(distance_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"distance_matrix must be square 2D; got shape {D.shape}")
    if not np.allclose(D, D.T, atol=1.0e-9):
        raise ValueError("distance_matrix must be symmetric")
    n = D.shape[0]
    if n_select <= 0:
        return FPSResult(
            indices=[],
            diversities=[],
            distance_matrix_shape=(n, n),
            descriptor_name=descriptor_name,
        )
    if n_select > n:
        raise ValueError(f"n_select {n_select} > n {n}")

    if seed_index is None:
        row_sums = D.sum(axis=1)
        ranked = np.round(row_sums / FPS_TIE_QUANTISATION) * FPS_TIE_QUANTISATION
        seed = int(np.lexsort((np.arange(n, dtype=int), ranked))[0])
    else:
        if not 0 <= int(seed_index) < n:
            raise ValueError(f"seed_index {seed_index} out of range")
        seed = int(seed_index)

    selected = [seed]
    min_dists = D[seed].copy()
    min_dists[seed] = 0.0
    diversities: List[float] = [0.0]

    for _ in range(1, n_select):
        candidates = np.where(np.isin(np.arange(n), selected, invert=True))[0]
        if candidates.size == 0:
            break
        scores = min_dists[candidates]
        order = np.lexsort((candidates, -scores))
        best_local = int(order[0])
        nxt = int(candidates[best_local])
        diversities.append(float(scores[best_local]))
        selected.append(nxt)
        new_dists = D[nxt]
        min_dists = np.minimum(min_dists, new_dists)
        min_dists[nxt] = 0.0

    return FPSResult(
        indices=selected,
        diversities=diversities,
        distance_matrix_shape=(n, n),
        descriptor_name=descriptor_name,
    )





def make_pluggable_polus_sampler(precomputed_distance_matrix, **divsampler_kwargs):
    from polus.trajectories.diversity import DIVSampler

    D = np.asarray(precomputed_distance_matrix, dtype=np.float32)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("distance matrix must be square 2D")

    class _PluggableMetricDIVSampler(DIVSampler):
        _PRECOMPUTED_MATRIX = D

        def SetRMSDMatrix(self):
            self.RotateTrajectory(self.refGeomFilename, self.rotateTraj, self.rotMethod)
            if not isinstance(self.rotTraj, list):
                raise RuntimeError("POLUS rotation produced no trajectory")
            self.samplePool = list(range(len(self.rotTraj)))
            if self.seedFilename is None:
                self.ComputeCentroid()
            else:
                self.GetSeedGeometry(self.seedFilename)
            expected = (self.ngeoms, self.ngeoms)
            if self._PRECOMPUTED_MATRIX.shape != expected:
                raise ValueError(
                    f"precomputed matrix shape {self._PRECOMPUTED_MATRIX.shape} "
                    f"!= POLUS ngeoms {expected}"
                )
            self.matrRMSD = self._PRECOMPUTED_MATRIX

    return _PluggableMetricDIVSampler(**divsampler_kwargs)

# --- helpers used by main(argv) below -------------------------------


def _write_xyz_file(frames, path):
    """Write a list of ICHOR Atoms objects to a plain xyz file.

    Format: natoms line, comment line, then one atom per line. POLUS
    parses this same shape so the same writer covers both Phase A
    (consumed by the daemon parser) and Phase B (consumed by APPEND).
    """
    out_lines = []
    for k, frame in enumerate(frames):
        out_lines.append(str(len(frame)))
        out_lines.append("frame " + str(k))
        for atom in frame:
            out_lines.append(
                "{symbol} {x:.6f} {y:.6f} {z:.6f}".format(
                    symbol=atom.type, x=float(atom.x),
                    y=float(atom.y), z=float(atom.z),
                )
            )
    Path(path).write_text(chr(10).join(out_lines) + chr(10), encoding="utf-8")


def _write_index_file(indices, path):
    """Write the per-line list of selected frame ids POLUS produces."""
    Path(path).write_text(
        chr(10).join(str(int(i)) for i in indices) + chr(10),
        encoding="utf-8",
    )


def _phase_b_target_size(config, n_candidates, iteration=0):
    """how many candidates to keep after Phase-B FPS, from the batch_sizing block.

    the batch GROWS with iteration so the loop adds more points as the model matures, bounded by
    batch_sizing.cap and never below floor (and never more than the pool actually has):
      * fixed  -- always floor (iteration ignored).
      * linear -- floor + iteration.
      * sqrt   -- floor scaled by sqrt(1 + iteration).
    the old code did min(floor, n, cap) which, since cap >= floor, collapsed to min(floor, n) -- so
    the batch sat at floor forever and cap/policy did nothing at all (A40 + A41). iteration is
    0-based (and < 0 for Phase A, where this is never called), so clamp it to >= 0; at iteration 0
    linear gives exactly floor, which keeps the old behaviour for the first round.
    """
    import math as _math
    b = config.batch_sizing
    floor = max(1, int(b.floor))
    cap = max(floor, int(b.cap))
    it = max(0, int(iteration))
    policy = str(getattr(b, "policy", "linear"))
    if policy == "fixed":
        grown = floor
    elif policy == "sqrt":
        grown = int(round(floor * _math.sqrt(1 + it)))
    else:  # "linear" (default) and anything unrecognised -> linear growth
        grown = floor + it
    # never below floor, never above the cap, never more than the candidate pool holds.
    target = min(max(floor, grown), cap, n_candidates)
    return max(1, int(target))


def _load_committed_training_set(campaign_dir):
    """Read the currently-committed training set as a list of ICHOR
    Atoms objects, one per committed pointdir.

    Returns an empty list when nothing has been committed yet, so the
    anti-overlap filter is a clean no-op against a fresh campaign.
    """
    from ..versioning.training_set import TrainingSetVersioning
    from ichor.core.files import PointDirectory
    v = TrainingSetVersioning(Path(campaign_dir) / "5_TRAINING")
    cur = v.current_version()
    if cur is None:
        return []
    iter_dir = v.iteration_path(int(cur))
    if not iter_dir.is_dir():
        return []
    v.verify_committed_training_inputs(int(cur))
    atoms_list = []
    for child in sorted(iter_dir.iterdir()):
        if not (child.is_dir() and PointDirectory.check_path(child)):
            continue
        try:
            pd = PointDirectory(child)
            atoms_list.append(pd.atoms)
        except Exception:
            # corrupt or partially-written pointdir; skip rather than
            # block the whole phase.
            continue
    return atoms_list


def _build_phase_b_posterior(campaign, config):
    """Load the current committed FEREBUS models for acquisition_weighted."""
    from ..daemon.state import DEFAULT_STATE_FILENAME, read_state
    from ichor.core.adversarial.posterior import TotalEnergyPosterior
    from ichor.core.models import Models

    state_path = (
        Path(campaign)
        / ".DATA"
        / "ACTIVE_LEARNING"
        / DEFAULT_STATE_FILENAME
    )
    state = read_state(state_path)
    models_version = int(getattr(state, "models_version", -1))
    if models_version < 0:
        raise FileNotFoundError("state has no committed models_version")
    models_dir = (
        Path(campaign)
        / "6_TRAINED_MODELS"
        / ("iteration-" + str(models_version).zfill(4))
    )
    if not models_dir.is_dir():
        raise FileNotFoundError("models directory not found: " + str(models_dir))
    property_name = str(config.acquisition.property_name)
    from ..daemon.artifact_contracts import verify_committed_model_version
    verify_committed_model_version(campaign, models_version)
    return TotalEnergyPosterior(
        Models(models_dir),
        property_name=property_name,
        scaled=bool(config.acquisition.use_scaled_posterior_covariance),
    )


def _phase_b_landing_safety_filter(candidate_frames, candidate_records):
    missing = [
        i for i, rec in enumerate(candidate_records)
        if not isinstance(rec.get("landing_safety"), dict)
    ]
    if len(missing) == len(candidate_records):
        return (
            list(candidate_frames),
            list(candidate_records),
            {
                "enabled": True,
                "legacy_missing_safety": True,
                "n_input": int(len(candidate_records)),
                "n_kept": int(len(candidate_records)),
                "n_dropped": 0,
                "dropped": [],
            },
        )
    if missing:
        raise ValueError(
            "partial missing landing_safety metadata for candidate indices "
            + repr([int(i) for i in missing])
        )

    kept_frames = []
    kept_records = []
    dropped = []
    for i, (frame, rec) in enumerate(zip(candidate_frames, candidate_records)):
        safety = rec.get("landing_safety")
        if bool(safety.get("accepted", False)):
            kept_frames.append(frame)
            kept_records.append(rec)
        else:
            dropped.append({
                "candidate_index": int(i),
                "seed_index": int(rec.get("seed_index", i)),
                "policy": str(safety.get("policy", "unknown")),
                "reasons": [str(r) for r in safety.get("reasons", [])],
            })
    return (
        kept_frames,
        kept_records,
        {
            "enabled": True,
            "legacy_missing_safety": False,
            "n_input": int(len(candidate_records)),
            "n_kept": int(len(kept_records)),
            "n_dropped": int(len(dropped)),
            "dropped": dropped,
        },
    )


def _run_phase_a(campaign, config):
    """POLUS Phase-A: pick a diverse subsample from the imported
    trajectory pool to seed the campaign with initial training points.

    Phase A always uses mass-weighted RMSD as the distance metric --
    no posterior exists yet, so any descriptor that needs one (e.g.
    acquisition-weighted) does not apply. The two outputs match what
    POLUS itself would write so the daemon postprocess parser does not
    care which path produced them.
    """
    from ..acquisition.trajectory_pool import TrajectoryPool
    from ..acquisition.trajectory_pool import POOL_MANIFEST_FILENAME, POOL_SUBDIR
    from ..handoff_manifests import write_phase_a_sample_manifest
    from .descriptors import MassWeightedRMSDDescriptor
    import sys as _sys

    try:
        pool = TrajectoryPool.load(campaign)
    except (FileNotFoundError, ValueError) as exc:
        print(
            "trajectory pool not loadable: " + str(exc),
            file=_sys.stderr,
        )
        return 3

    frames = pool.to_atoms_list()
    if not frames:
        print("trajectory pool is empty", file=_sys.stderr)
        return 3

    n_target = int(config.initial_train_size) + int(config.initial_val_size)
    n_select = min(max(1, n_target), len(frames))

    descriptor = MassWeightedRMSDDescriptor()
    matrix = descriptor.pairwise_distance_matrix(frames)
    sel = fps_select(matrix, n_select, descriptor_name=descriptor.name)

    outdir = campaign / "3_DIVERSITY_SAMPLING" / "initial"
    outdir.mkdir(parents=True, exist_ok=True)
    sample_path = outdir / ("initial-SAMPLE-" + str(n_select) + ".xyz")
    index_path = outdir / ("initial-INDEX-" + str(n_select) + ".dat")
    _write_xyz_file([frames[i] for i in sel.indices], sample_path)
    _write_index_file(sel.indices, index_path)
    write_phase_a_sample_manifest(outdir, {
        "phase": "PHASE_A_POLUS",
        "iteration": -1,
        "sample_xyz": str(sample_path.resolve()),
        "index_path": str(index_path.resolve()),
        "n_select": int(n_select),
        "n_frames": int(n_select),
        "selected_indices": [int(i) for i in sel.indices],
        "descriptor": str(descriptor.name),
        "n_pool_frames": int(len(frames)),
        "trajectory_sha256": str(pool.sha256),
        "source_pool_manifest": str((campaign / POOL_SUBDIR / POOL_MANIFEST_FILENAME).resolve()),
        "initial_train_size": int(config.initial_train_size),
        "initial_val_size": int(config.initial_val_size),
    })

    print(
        "Phase A: wrote " + str(n_select) + " frames to "
        + str(sample_path),
    )
    return 0


def _run_phase_b(args, campaign, config):
    """POLUS Phase-B: pick a diverse subsample from the adversarial
    pool ARIADNE just produced, then run the optional anti-overlap
    pass against the committed training set.

    Two outputs always written:
      phase_b_SAMPLE_raw.xyz  -- the raw FPS selection (whatever POLUS picked).
      phase_b_SAMPLE.xyz      -- the dedup-filtered final selection.
                                  identical to RAW when min_separation is 0.
    Plus phase_b_dedup.json   -- the DedupReport for journal / reconcile.
    """
    from .descriptors import build_descriptor_from_config
    from .anti_overlap import filter_candidates_against_training
    from ..handoff_manifests import (
        PHASE_B_SELECTION_SCHEMA_VERSION,
        ariadne_candidate_frames,
        ariadne_results_path,
        write_phase_b_selection_manifest,
    )
    import sys as _sys
    import json as _json

    iter_dir = (
        campaign / "7_ACTIVE_LEARNING"
        / ("iteration-" + str(int(args.iteration)).zfill(4))
    )
    manifest_path = ariadne_results_path(iter_dir)
    if not manifest_path.is_file():
        print(
            "ARIADNE results manifest not found: " + str(manifest_path),
            file=_sys.stderr,
        )
        return 3

    try:
        ariadne_manifest, candidate_frames, candidate_records = ariadne_candidate_frames(
            iter_dir,
            expected_iteration=int(args.iteration),
        )
    except Exception as exc:
        print(
            "ARIADNE results manifest invalid: "
            + type(exc).__name__
            + ": "
            + str(exc),
            file=_sys.stderr,
        )
        return 3

    if not candidate_frames:
        print("ARIADNE results manifest has no accepted candidates", file=_sys.stderr)
        return 3

    safety_filter = {
        "enabled": False,
        "n_input": int(len(candidate_records)),
        "n_kept": int(len(candidate_records)),
        "n_dropped": 0,
        "dropped": [],
    }
    if bool(getattr(config.adversarial_safety, "phase_b_filter_enabled", True)):
        try:
            candidate_frames, candidate_records, safety_filter = (
                _phase_b_landing_safety_filter(candidate_frames, candidate_records)
            )
        except Exception as exc:
            print(
                "Phase B landing safety filter failed: "
                + type(exc).__name__
                + ": "
                + str(exc),
                file=_sys.stderr,
            )
            return 3
        if not candidate_frames:
            print(
                "Phase B landing safety filter removed every candidate",
                file=_sys.stderr,
            )
            return 3

    posterior = None
    if config.phase_b.descriptor == "acquisition_weighted":
        try:
            posterior = _build_phase_b_posterior(campaign, config)
        except Exception as exc:
            print(
                "acquisition_weighted descriptor could not load posterior for property "
                + repr(config.acquisition.property_name)
                + ": "
                + type(exc).__name__
                + ": "
                + str(exc),
                file=_sys.stderr,
            )
            return 3
    descriptor = build_descriptor_from_config(config, posterior=posterior)
    try:
        matrix = descriptor.pairwise_distance_matrix(candidate_frames)
    except Exception as exc:
        print(
            "Phase B descriptor failed: "
            + type(exc).__name__
            + ": "
            + str(exc),
            file=_sys.stderr,
        )
        return 3
    n_select = _phase_b_target_size(config, len(candidate_frames), int(args.iteration))
    sel = fps_select(matrix, n_select, descriptor_name=descriptor.name)
    selected_frames = [candidate_frames[i] for i in sel.indices]
    selected_records = [candidate_records[i] for i in sel.indices]

    iter_dir.mkdir(parents=True, exist_ok=True)
    raw_path = iter_dir / "phase_b_SAMPLE_raw.xyz"
    _write_xyz_file(selected_frames, raw_path)

    # anti-overlap case (d): drop any selected candidate that lands too
    # close to an existing training point. off by default (min_sep=0).
    min_sep = float(config.phase_b.min_separation)
    final_path = iter_dir / "phase_b_SAMPLE.xyz"
    dedup_path = iter_dir / "phase_b_dedup.json"
    try:
        training = _load_committed_training_set(campaign) if min_sep > 0.0 else []
    except Exception as exc:
        print(
            "committed training set invalid for Phase B anti-overlap: "
            + type(exc).__name__
            + ": "
            + str(exc),
            file=_sys.stderr,
        )
        return 3
    report = filter_candidates_against_training(
        selected_frames, training, min_separation=min_sep,
    )
    dedup_payload = {
        "kept_indices": list(report.kept_indices),
        "dropped_indices": list(report.dropped_indices),
        "distances_to_nearest": list(report.distances_to_nearest),
        "min_separation": report.min_separation,
        "n_kept": report.n_kept,
        "n_dropped": report.n_dropped,
        "n_candidates": len(selected_frames),
        "descriptor_used": descriptor.name,
    }
    dedup_path.write_text(_json.dumps(dedup_payload, indent=2), encoding="utf-8")
    if int(report.n_kept) <= 0:
        print(
            "phase_b_anti_overlap_removed_every_candidate: "
            + "kept 0/"
            + str(len(selected_frames))
            + " candidates after anti-overlap; diagnostics written to "
            + str(dedup_path),
            file=_sys.stderr,
        )
        return 3

    kept_frames = [selected_frames[i] for i in report.kept_indices]
    _write_xyz_file(kept_frames, final_path)

    kept_lookup = {int(raw_index): final_index for final_index, raw_index in enumerate(report.kept_indices)}
    raw_records = []
    final_records = []
    for raw_index, rec in enumerate(selected_records):
        out_rec = dict(rec)
        out_rec["candidate_index"] = int(sel.indices[raw_index])
        out_rec["raw_index"] = int(raw_index)
        out_rec["kept_after_dedup"] = int(raw_index) in kept_lookup
        out_rec["final_index"] = (
            int(kept_lookup[int(raw_index)])
            if int(raw_index) in kept_lookup else None
        )
        out_rec["drop_reason"] = None if out_rec["kept_after_dedup"] else "min_separation"
        raw_records.append(dict(out_rec))
        if out_rec["kept_after_dedup"]:
            final_records.append(dict(out_rec))
    manifest_path = write_phase_b_selection_manifest(iter_dir, {
        "schema_version": PHASE_B_SELECTION_SCHEMA_VERSION,
        "iteration": int(args.iteration),
        "descriptor": str(descriptor.name),
        "source_ariadne_manifest": str((iter_dir / "ARIADNE_RESULTS.json").resolve()),
        "n_candidates": int(len(candidate_frames)),
        "n_selected_raw": int(len(raw_records)),
        "n_kept": int(len(final_records)),
        "raw": raw_records,
        "final": final_records,
        "dedup": dedup_payload,
        "safety_filter": safety_filter,
        "source_expected_n": int(ariadne_manifest.get("expected_n", len(candidate_frames))),
    })

    print(
        "Phase B: kept " + str(report.n_kept) + "/"
        + str(len(selected_frames)) + " candidates after dedup; "
        + "wrote " + str(final_path)
        + " and "
        + str(manifest_path),
    )
    return 0


def main(argv=None) -> int:
    """Command-line entrypoint for POLUS diversity sub-sampling.

    The daemon calls this from inside a sbatch script for both Phase A
    (initial pool sub-sample, run once at the start of a campaign) and
    Phase B (per-iteration sub-sample over the adversarial pool).

    --iteration < 0 means Phase A on the trajectory pool.
    --iteration >= 0 means Phase B for that iteration.

    Exit codes:
      0 -- success, sample xyz + index files written.
      2 -- bad command line (missing files, malformed args).
      3 -- per-iteration state was not in a runnable shape.
    """
    import argparse
    import sys as _sys
    from pathlib import Path as _Path

    parser = argparse.ArgumentParser(
        prog="python -m ichor.hpc.active_learning.sampling.polus_wrapper",
        description=(
            "Run POLUS diversity sub-sampling for either the initial "
            "pool (Phase A) or the per-iteration adversarial pool "
            "(Phase B). Negative --iteration means Phase A."
        ),
    )
    parser.add_argument(
        "--descriptor", type=str, required=True,
        choices=["rmsd_massweight", "hybrid_alf_rmsd", "acquisition_weighted"],
        help="Distance metric used to build the pairwise matrix POLUS chews on.",
    )
    parser.add_argument(
        "--iteration", type=int, required=True,
        help="Campaign iteration. Negative value means Phase A on the initial pool.",
    )
    parser.add_argument(
        "--campaign-dir", type=str, required=True,
        help="Path to the campaign root (where campaign.yaml lives).",
    )
    args = parser.parse_args(argv)

    campaign = _Path(args.campaign_dir).resolve()
    if not campaign.is_dir():
        print(
            "campaign-dir does not exist or is not a directory: "
            + str(campaign), file=_sys.stderr,
        )
        return 2
    cfg_path = campaign / "campaign.yaml"
    if not cfg_path.is_file():
        print(
            "campaign.yaml not found at " + str(cfg_path),
            file=_sys.stderr,
        )
        return 2

    from ..config import CampaignConfig
    config = CampaignConfig.from_yaml(cfg_path)

    # the descriptor actually used comes from campaign.yaml; --descriptor is
    # only the launching script stating its intent. surface a mismatch loudly
    # but let the config win rather than failing the whole job over it.
    if args.descriptor and args.descriptor != config.phase_b.descriptor:
        print(
            "note: --descriptor " + repr(args.descriptor) + " is overridden by "
            + "campaign.yaml phase_b.descriptor "
            + repr(config.phase_b.descriptor),
            file=_sys.stderr,
        )

    if int(args.iteration) < 0:
        return _run_phase_a(campaign, config)
    return _run_phase_b(args, campaign, config)


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
