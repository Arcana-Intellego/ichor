"""Per-point input staging for the live sbatch phases.

Lays out the POINT_<k>.pointdir tree (and the POINTS.txt the array jobs index
into) on the login node before submit, reusing the core file writers. Nothing
here talks to SLURM -- it only writes files.

Layout, matching what the postprocess parsers expect:

    .DATA/STAGING/<bucket>/
        POINT_0000.pointdir/input.gjf      # gaussian input (writes input.wfn)
        POINT_0001.pointdir/input.gjf
        ...
        POINTS.txt                         # one absolute pointdir path per line

<bucket> is "initial" for the INITIAL_* phases and "iter_<N>" otherwise.
"""
from __future__ import annotations

import json
import os  # stage_ferebus_inputs cd's into the staging dir to export csvs; this was missing and only bit on a live run
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from ichor.core.atoms import Atoms
from ichor.core.files import PointDirectory
from ichor.core.files.xyz import Trajectory

from .state import atomic_write_json


QUANTUM_ACCEPTANCE_MANIFEST = "accepted_pointdirs.json"
QUANTUM_ACCEPTANCE_SCHEMA_VERSION = 1
AIMALL_TASK_METADATA = "AIMALL_TASK.json"
AIMALL_TASK_METADATA_SCHEMA_VERSION = 1
FEREBUS_TASK_MANIFEST = "FEREBUS_TASKS.json"
FEREBUS_TASK_SCHEMA_VERSION = 1
FEREBUS_JOB_DETAILS = "job-details"
SAFE_PATH_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def bucket_dir(campaign_dir, phase_name: str, iteration: int) -> Path:
    bucket = "initial" if phase_name.startswith("INITIAL_") else ("iter_" + str(iteration))
    return Path(campaign_dir) / ".DATA" / "STAGING" / bucket


def _load_frames(sample_xyz) -> List[Atoms]:
    traj = Trajectory(Path(sample_xyz))
    traj.read()
    return [atoms.copy() for atoms in traj]


def write_points_file(staging_dir: Path, pointdirs: Sequence[Path]) -> Path:
    """One absolute pointdir path per line; the array sed-lookup reads this."""
    points_file = Path(staging_dir) / "POINTS.txt"
    body = "\n".join(str(Path(p).resolve()) for p in pointdirs)
    # force LF. the array sbatch sed-reads this on a linux node, and if we ever stage from
    # windows the default crlf leaves a trailing \r so cd "$POINT_DIR" quietly breaks.
    points_file.write_text(body + ("\n" if body else ""), encoding="utf-8", newline="\n")
    return points_file


def _pointdir_name(pointdir: Any) -> str:
    path = getattr(pointdir, "path", pointdir)
    return Path(path).name


def validate_safe_path_token(label: str, value: str) -> None:
    text = str(value)
    if not SAFE_PATH_TOKEN_RE.fullmatch(text):
        raise ValueError(
            label
            + " must be a safe path token matching ^[A-Za-z0-9][A-Za-z0-9_.-]*$: "
            + repr(text)
        )


def _checked_rmtree(path: Path) -> None:
    target = Path(path)
    if not target.exists():
        return
    shutil.rmtree(str(target), ignore_errors=False)
    if target.exists():
        raise OSError("failed to remove stale staging directory: " + str(target))


def _reject_symlink_tree(root: Path) -> None:
    root = Path(root)
    if root.is_symlink():
        raise ValueError("refusing to copy symlinked path: " + str(root))
    for child in root.rglob("*"):
        if child.is_symlink():
            raise ValueError("refusing to copy tree containing symlink: " + str(child))


def _copytree_no_symlinks(src: Path, dest: Path) -> None:
    _reject_symlink_tree(Path(src))
    shutil.copytree(str(src), str(dest), symlinks=False)


def quantum_acceptance_manifest_path(staging_dir: Path) -> Path:
    return Path(staging_dir) / QUANTUM_ACCEPTANCE_MANIFEST


def write_quantum_acceptance_manifest(
    staging_dir: Path,
    *,
    phase_name: str,
    iteration: int,
    accepted: Sequence[Any],
    rejected: Sequence[Tuple[str, str]],
) -> Path:
    """Persist the exact validated quantum handoff for downstream live stages."""
    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    accepted_names = [_pointdir_name(p) for p in accepted]
    rejected_payload = [
        {"pointdir": str(name), "reason": str(reason)}
        for name, reason in rejected
    ]
    payload: Dict[str, Any] = {
        "schema_version": QUANTUM_ACCEPTANCE_SCHEMA_VERSION,
        "phase": str(phase_name),
        "iteration": int(iteration),
        "accepted_pointdirs": accepted_names,
        "rejected": rejected_payload,
        "n_total": int(len(accepted_names) + len(rejected_payload)),
    }
    path = quantum_acceptance_manifest_path(staging)
    atomic_write_json(path, payload)
    return path


def read_quantum_acceptance_manifest(
    staging_dir: Path,
    *,
    expected_phase: str,
    expected_iteration: int,
    require_nonempty: bool = True,
) -> Tuple[List[Path], Dict[str, Any]]:
    """Read and validate the live quantum acceptance manifest.

    Downstream consumers use this instead of globbing POINT_*.pointdir so
    rejected quantum outputs cannot be silently committed or reprocessed.
    """
    staging = Path(staging_dir)
    path = quantum_acceptance_manifest_path(staging)
    if not path.is_file():
        raise FileNotFoundError("quantum acceptance manifest missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("quantum acceptance manifest unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise ValueError("quantum acceptance manifest must be a JSON object: " + str(path))
    if int(data.get("schema_version", -1)) != QUANTUM_ACCEPTANCE_SCHEMA_VERSION:
        raise ValueError("unsupported quantum acceptance manifest schema: " + str(path))
    if data.get("phase") != expected_phase:
        raise ValueError(
            "quantum acceptance manifest phase mismatch: expected "
            + expected_phase + " got " + str(data.get("phase"))
        )
    try:
        iteration = int(data.get("iteration"))
    except (TypeError, ValueError) as exc:
        raise ValueError("quantum acceptance manifest iteration is not an integer") from exc
    if iteration != int(expected_iteration):
        raise ValueError(
            "quantum acceptance manifest iteration mismatch: expected "
            + str(int(expected_iteration)) + " got " + str(iteration)
        )
    accepted = data.get("accepted_pointdirs")
    if not isinstance(accepted, list):
        raise ValueError("quantum acceptance manifest accepted_pointdirs must be a list")
    rejected = data.get("rejected", [])
    if not isinstance(rejected, list):
        raise ValueError("quantum acceptance manifest rejected must be a list")
    n_total = data.get("n_total")
    if n_total is not None:
        try:
            parsed_total = int(n_total)
        except (TypeError, ValueError) as exc:
            raise ValueError("quantum acceptance manifest n_total is not an integer") from exc
        if parsed_total != len(accepted) + len(rejected):
            raise ValueError("quantum acceptance manifest n_total does not match payload lengths")

    seen = set()
    resolved: List[Path] = []
    for raw_name in accepted:
        if not isinstance(raw_name, str):
            raise ValueError("accepted pointdir name is not a string")
        name = raw_name.strip()
        if Path(name).name != name or not name.endswith(".pointdir"):
            raise ValueError("unsafe accepted pointdir name in manifest: " + repr(raw_name))
        if name in seen:
            raise ValueError("duplicate accepted pointdir in manifest: " + name)
        seen.add(name)
        pointdir = staging / name
        if not pointdir.is_dir():
            raise FileNotFoundError(
                "accepted pointdir listed in manifest is missing: " + str(pointdir)
            )
        resolved.append(pointdir)
    if require_nonempty and not resolved:
        raise ValueError("quantum acceptance manifest accepted_pointdirs is empty: " + str(path))
    return resolved, data


def ferebus_manifest_path(staging_dir: Path) -> Path:
    return Path(staging_dir) / FEREBUS_TASK_MANIFEST


def read_ferebus_manifest(staging_dir: Path) -> Dict[str, Any]:
    path = ferebus_manifest_path(Path(staging_dir))
    if not path.is_file():
        raise FileNotFoundError("FEREBUS task manifest missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("FEREBUS task manifest unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise ValueError("FEREBUS task manifest must be a JSON object: " + str(path))
    if int(data.get("schema_version", -1)) != FEREBUS_TASK_SCHEMA_VERSION:
        raise ValueError("unsupported FEREBUS task manifest schema: " + str(path))
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("FEREBUS task manifest has no tasks: " + str(path))
    return data


def _write_ferebus_manifest(staging_dir: Path, payload: Dict[str, Any]) -> Path:
    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    path = ferebus_manifest_path(staging)
    atomic_write_json(path, payload)
    return path


def _alf_to_ferebus(alf: Any, atom: str) -> List[int]:
    try:
        raw = [int(x) for x in alf]
    except Exception as exc:
        raise ValueError("ALF for atom " + atom + " is not iterable/int-like") from exc
    if len(raw) != 3:
        raise ValueError("ALF for atom " + atom + " must contain exactly three indexes")
    if any(x < 0 for x in raw):
        raise ValueError("ALF for atom " + atom + " contains a negative zero-indexed entry")
    return [x + 1 for x in raw]


def _write_pyferebus_job_details(
    path: Path,
    *,
    system: str,
    atoms: Sequence[str],
    properties: Sequence[str],
    alf_by_atom: Dict[str, Sequence[int]],
    stats_by_prop_atom: Dict[Tuple[str, str], Dict[str, float]],
) -> Path:
    lines = [
        "system_name " + str(system),
        "natoms " + str(len(atoms)),
        "atoms " + " ".join(str(a) for a in atoms),
        "props " + " ".join(str(p) for p in properties),
    ]
    for atom in atoms:
        lines.append(str(atom) + " " + " ".join(str(int(x)) for x in alf_by_atom[atom]))
    stat_keys = ("min", "max", "range", "mean", "median", "std", "cv")
    for prop in properties:
        for atom in atoms:
            stats = stats_by_prop_atom.get((str(prop), str(atom)), {})
            missing = [k for k in stat_keys if k not in stats]
            if missing:
                raise ValueError(
                    "missing target-property stats for "
                    + str(prop)
                    + "-"
                    + str(atom)
                    + ": "
                    + repr(missing)
                )
            lines.append(
                str(prop)
                + "-"
                + str(atom)
                + " "
                + " ".join(str(float(stats[k])) for k in stat_keys)
            )
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def stage_gaussian_inputs(campaign_dir, config, phase_name, iteration, sample_xyz) -> Tuple[Path, int]:
    """Write one POINT_<k>.pointdir/input.gjf per frame in sample_xyz, plus
    POINTS.txt. Returns (staging_dir, n_points)."""
    from ichor.core.files.gaussian.gjf import GJF

    frames = _load_frames(sample_xyz)
    phase_b_records: List[Dict[str, Any]] = []
    if str(phase_name) == "GAUSSIAN":
        from ..handoff_manifests import read_phase_b_selection_manifest

        phase_b_manifest = read_phase_b_selection_manifest(
            Path(sample_xyz).parent,
            expected_iteration=int(iteration),
        )
        phase_b_records = list(phase_b_manifest.get("final", []))
        if len(phase_b_records) != len(frames):
            raise ValueError(
                "Phase B selection manifest final count "
                + str(len(phase_b_records))
                + " != sample frame count "
                + str(len(frames))
            )
    g = config.gaussian
    staging = bucket_dir(campaign_dir, phase_name, iteration)
    # clear the bucket first so a same-iteration crash-retry does not leave stale POINT_*.pointdir
    # from a partial earlier attempt mixed in with the fresh staging. gaussian is the bucket's FIRST
    # writer so clearing here is safe -- AIMAll staging must NOT clear because it consumes the
    # Gaussian acceptance manifest and the accepted pointdirs in this same bucket.
    if staging.exists():
        _checked_rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    keywords = ["nosymm", "output=wfn", "force", "geom=notest"]
    if str(g.extra_keywords).strip():
        keywords += str(g.extra_keywords).split()

    pointdirs: List[Path] = []
    for k, atoms in enumerate(frames):
        pd = staging / ("POINT_" + str(k).zfill(4) + ".pointdir")
        pd.mkdir(parents=True, exist_ok=True)
        gjf = GJF(
            pd / "input.gjf",
            method=str(g.method),
            basis_set=str(g.basis_set),
            keywords=list(keywords),
            charge=int(g.charge),
            spin_multiplicity=int(g.spin_multiplicity),
            atoms=atoms,
        )
        if str(getattr(g, "memory_mode", "slurm_env")) == "link0":
            gjf.set_nproc(int(g.nproc))
            gjf.set_mem(str(g.mem))
        gjf.write()
        if phase_b_records:
            from ..versioning.provenance import PROVENANCE_FILENAME

            src_prov = Path(str(phase_b_records[k].get("provenance_json", "")))
            if not src_prov.is_file():
                raise FileNotFoundError(
                    "Phase B provenance missing for staged Gaussian point "
                    + str(k)
                    + ": "
                    + str(src_prov)
                )
            shutil.copy2(str(src_prov), str(pd / PROVENANCE_FILENAME))
        pointdirs.append(pd)

    write_points_file(staging, pointdirs)
    return staging, len(pointdirs)


def stage_aimall_inputs(campaign_dir, config, phase_name, iteration) -> Tuple[Path, int]:
    """AIMAll runs on the .wfn files Gaussian produced in the same bucket. The
    pointdirs already exist; rewrite POINTS.txt over the Gaussian-accepted
    pointdirs so the array only indexes ready points. Returns (dir, n_points)."""
    staging = bucket_dir(campaign_dir, phase_name, iteration)
    expected_phase = "INITIAL_GAUSSIAN" if phase_name.startswith("INITIAL_") else "GAUSSIAN"
    pointdirs, _manifest = read_quantum_acceptance_manifest(
        staging,
        expected_phase=expected_phase,
        expected_iteration=int(iteration),
    )
    aimall_cpus = int(config.resources.aimall_cpus_per_task)
    raw_naat = getattr(config.aimall, "naat", "auto")
    for pointdir in pointdirs:
        if not (pointdir / "input.wfn").is_file():
            raise FileNotFoundError(
                "Gaussian-accepted pointdir is missing input.wfn: " + str(pointdir)
            )
        try:
            atom_count = len(PointDirectory(pointdir).atoms)
        except Exception as exc:
            raise ValueError(
                "failed to count atoms for AIMAll pointdir: " + str(pointdir)
            ) from exc
        if atom_count <= 0:
            raise ValueError("AIMAll pointdir has no atoms: " + str(pointdir))
        if isinstance(raw_naat, str) and raw_naat.strip().lower() == "auto":
            resolved_naat = min(int(atom_count), int(aimall_cpus))
        else:
            resolved_naat = int(raw_naat)
            if resolved_naat < 1 or resolved_naat > int(aimall_cpus):
                raise ValueError(
                    "aimall.naat must be in [1, resources.aimall_cpus_per_task]"
                )
        atomic_write_json(
            pointdir / AIMALL_TASK_METADATA,
            {
                "schema_version": AIMALL_TASK_METADATA_SCHEMA_VERSION,
                "atom_count": int(atom_count),
                "nproc": int(aimall_cpus),
                "naat": int(resolved_naat),
            },
        )
    write_points_file(staging, pointdirs)
    return staging, len(pointdirs)


def commit_initial_training_set(campaign_dir) -> bool:
    """build + commit 5_TRAINING/iteration-0 from the initial quantum staging bucket
    (.DATA/STAGING/initial/POINT_*). returns True if it actually committed, False if
    iteration-0 was already there.

    idempotent on purpose -- it gets called at INITIAL_FEREBUS staging (so the feature export
    has a training set to read) and again at INITIAL_FEREBUS postprocess, and has to be a no-op
    the second time + safe on a crash-retry. the per-iteration loop never needs this because its
    inline APPEND commits 5_TRAINING before FEREBUS stages; only the initial bootstrap is missing
    that step, which is the whole reason this helper exists.
    """
    from ..versioning.training_set import TrainingSetVersioning

    campaign = Path(campaign_dir)
    v_train = TrainingSetVersioning(campaign / "5_TRAINING")
    if 0 in v_train.list_committed_versions():
        v_train.ensure_current(0)
        return False
    initial_staging = campaign / ".DATA" / "STAGING" / "initial"
    accepted_pointdirs, _manifest = read_quantum_acceptance_manifest(
        initial_staging,
        expected_phase="INITIAL_AIMALL",
        expected_iteration=0,
    )
    # bin any half-built staging left by a dead attempt, then stage an empty iter-0 and copy
    # the validated initial pointdirs into it.
    v_train.recover_dangling_staging()
    train_staging = v_train.stage(source_version=None, target_version=0)
    for pdir in accepted_pointdirs:
        dest = train_staging / pdir.name
        if dest.exists():
            _checked_rmtree(dest)
        _copytree_no_symlinks(pdir, dest)
    v_train.commit(0)
    v_train.update_current(0)
    return True


def stage_ferebus_inputs(campaign_dir, config, training_version, is_initial=False) -> Tuple[Path, int]:
    """Stage pyferebus/FEREBUS inputs and return (staging_dir, n_tasks).

    pyferebus owns the FEREBUS folder/config/command/slurm contract. The daemon stages the flat
    property directories that pyferebus expects, plus the pyferebus job-details file and a daemon
    manifest used for strict postprocess validation.
    """
    from ichor.core.files import PointsDirectory
    from ichor.core.calculators import calculate_alf_atom_sequence

    campaign = Path(campaign_dir)
    # the initial bootstrap has no APPEND phase ahead of it, so 5_TRAINING/iteration-0 may not
    # exist yet when INITIAL_FEREBUS stages. build it from the initial quantum staging so the
    # export has something to read. idempotent; and we only create the dir here -- the
    # training_set_version bump stays in postprocess so a restart mid-flight reconciles cleanly.
    if is_initial:
        training_version = 0
        commit_initial_training_set(campaign)
    training_dir = campaign / "5_TRAINING" / ("iteration-" + str(int(training_version)).zfill(4))
    if not training_dir.is_dir():
        raise FileNotFoundError(
            "no committed training set to build a FEREBUS dataset from: "
            + str(training_dir)
        )
    from ..versioning.training_set import TrainingSetVersioning
    TrainingSetVersioning(campaign / "5_TRAINING").verify_committed_training_inputs(
        int(training_version)
    )

    staging = campaign / "6_TRAINED_MODELS" / "iteration-staging"
    # iteration-staging is ONE shared scratch dir reused every iteration, so wipe it first.
    # otherwise last iteration's *_train.csv / *.model lying around get globbed back in and we
    # either split stale data or re-commit an old model as a fresh version (both silent + nasty).
    if staging.exists():
        _checked_rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    pd = PointsDirectory(training_dir)
    pointdir_names = [Path(getattr(p, "path", p)).name for p in pd]
    if not pointdir_names:
        raise ValueError("committed training set contains no pointdirs: " + str(training_dir))
    # system ALF defines the per-atom local frame the features are built in.
    system_alf = pd.alf_dict(calculate_alf_atom_sequence)
    f = config.ferebus
    properties = [str(p) for p in getattr(f, "properties", ["iqa"])]
    if not properties:
        raise ValueError("ferebus.properties must contain at least one property")

    # write one <atom>_train.csv per atom containing every configured target property.
    cwd = os.getcwd()
    try:
        os.chdir(staging)
        pd.features_with_properties_to_csv(
            system_alf,
            str_to_append_to_fname="_train.csv",
            property_types=properties,
        )
    finally:
        os.chdir(cwd)
    feature_csvs = sorted(staging.glob("*_train.csv"))
    if not feature_csvs:
        raise ValueError("PointsDirectory export produced no *_train.csv files for FEREBUS")

    system = str(getattr(config, "system_name", "SYSTEM"))
    fractions = (
        float(f.train_fraction),
        float(f.int_val_fraction),
        float(f.ext_val_fraction),
    )
    from . import ferebus_dataset as _fds
    from .ferebus_split_ledger import ensure_split_assignments

    split_ledger = ensure_split_assignments(
        campaign,
        pointdir_names,
        training_version=int(training_version),
        fractions=fractions,
    )
    ledger_row_ids = dict(split_ledger["row_ids"])
    atom_labels = []
    alf_by_atom: Dict[str, List[int]] = {}
    split_counts: Dict[str, Dict[str, int]] = {}
    row_ids_by_atom: Dict[str, Dict[str, List[int]]] = {}
    stats_by_prop_atom: Dict[Tuple[str, str], Dict[str, float]] = {}
    prop_dirs = {prop: staging / prop for prop in properties}
    for prop_dir in prop_dirs.values():
        prop_dir.mkdir(parents=True, exist_ok=True)
    degenerate_property_stats: List[Dict[str, Any]] = []
    for atom_csv in feature_csvs:
        # filename is "<atom>_train.csv"; recover the atom label.
        atom = atom_csv.name[:-len("_train.csv")]
        validate_safe_path_token("FEREBUS atom label", atom)
        if atom not in system_alf:
            raise ValueError("system ALF is missing atom " + atom)
        row_count = _fds._row_count(atom_csv)
        if int(row_count) != len(pointdir_names):
            raise ValueError(
                "FEREBUS CSV row count for "
                + str(atom_csv)
                + " is "
                + str(row_count)
                + " but training pointdir count is "
                + str(len(pointdir_names))
            )
        alf_by_atom[atom] = _alf_to_ferebus(system_alf[atom], atom)
        split = _fds.split_atom_csv_to_property_dirs(
            atom_csv,
            prop_dirs,
            system,
            atom,
            properties,
            fractions,
            row_ids=ledger_row_ids,
        )
        counts = dict(split["counts"])
        atom_labels.append(atom)
        split_counts[atom] = counts
        row_ids_by_atom[atom] = dict(split["row_ids"])
        for prop in properties:
            train_csv = prop_dirs[prop] / (system + "_" + atom + "_TRAINING_SET.csv")
            stats = _fds.prop_stats(train_csv, prop)
            if not stats:
                raise ValueError(
                    "could not compute target-property stats for "
                    + prop
                    + "-"
                    + atom
                    + " from "
                    + str(train_csv)
                )
            stats_by_prop_atom[(prop, atom)] = stats
            if bool(stats.get("degenerate_property_stats", False)):
                degenerate_property_stats.append({
                    "property": str(prop),
                    "atom": str(atom),
                    "std": float(stats["std"]),
                    "range": float(stats["range"]),
                })
    n_atoms = len(atom_labels)
    atoms_file = staging / "ATOMS.txt"
    atoms_file.write_text(
        chr(10).join(atom_labels) + (chr(10) if atom_labels else ""),
        encoding="utf-8",
        newline="\n",
    )
    props_file = staging / "PROPERTIES.txt"
    props_file.write_text(
        chr(10).join(properties) + chr(10),
        encoding="utf-8",
        newline="\n",
    )

    job_details = _write_pyferebus_job_details(
        staging / FEREBUS_JOB_DETAILS,
        system=system,
        atoms=atom_labels,
        properties=properties,
        alf_by_atom=alf_by_atom,
        stats_by_prop_atom=stats_by_prop_atom,
    )

    tasks: List[Dict[str, Any]] = []
    task_index = 1
    for prop in properties:
        for atom in atom_labels:
            output_dir = staging / prop / atom
            input_dir = output_dir / "datasets"
            config_path = output_dir / "ferebus.config"
            alf_cli = "_".join(str(x) for x in alf_by_atom[atom])
            training_csv = input_dir / (system + "_" + atom + "_TRAINING_SET.csv")
            int_csv = input_dir / (system + "_" + atom + "_INT_VALIDATION_SET.csv")
            ext_csv = input_dir / (system + "_" + atom + "_EXT_VALIDATION_SET.csv")
            model_path = output_dir / (system + "_" + prop + "_" + atom + ".model")
            command = (
                "-c "
                + str(config_path)
                + "  -I "
                + str(input_dir)
                + " -O "
                + str(output_dir)
                + " -P "
                + prop
                + " -A "
                + atom
                + " -ALF "
                + alf_cli
            )
            tasks.append(
                {
                    "task_index": int(task_index),
                    "property": prop,
                    "atom": atom,
                    "alf_1_indexed": [int(x) for x in alf_by_atom[atom]],
                    "alf_cli": alf_cli,
                    "property_dir": str((staging / prop).resolve()),
                    "output_dir": str(output_dir.resolve()),
                    "input_dir": str(input_dir.resolve()),
                    "config_path": str(config_path.resolve()),
                    "training_csv": str(training_csv.resolve()),
                    "int_validation_csv": str(int_csv.resolve()),
                    "ext_validation_csv": str(ext_csv.resolve()),
                    "expected_model_path": str(model_path.resolve()),
                    "command": command,
                    "row_counts": dict(split_counts[atom]),
                    "row_ids": dict(row_ids_by_atom[atom]),
                    "degenerate_property_stats": bool(
                        stats_by_prop_atom.get((prop, atom), {}).get(
                            "degenerate_property_stats", False
                        )
                    ),
                }
            )
            task_index += 1

    _write_ferebus_manifest(
        staging,
        {
            "schema_version": FEREBUS_TASK_SCHEMA_VERSION,
            "system": system,
            "training_version": int(training_version),
            "pointdir_row_order": list(pointdir_names),
            "properties": properties,
            "atoms": atom_labels,
            "n_atoms": int(n_atoms),
            "n_tasks": int(len(tasks)),
            "degenerate_property_stats": list(degenerate_property_stats),
            "job_details": str(job_details.resolve()),
            "split_ledger": {
                "path": str(split_ledger["path"]),
                "counts": dict(split_ledger["counts"]),
            },
            "tasks": tasks,
        },
    )
    return staging, len(tasks)
