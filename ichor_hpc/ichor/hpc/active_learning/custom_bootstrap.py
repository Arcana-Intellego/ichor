"""Discovery and immutable admission of operator bootstrap inputs.

Bootstrap CSV files are geometry containers: only their leading ALF feature
columns are used.  Gaussian/AIMAll remain authoritative for scientific labels.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np

from ichor.core.atoms import ALF, Atom, Atoms
from ichor.core.calculators import (
    alf_features_to_coordinates,
    calculate_alf_features,
    default_connectivity_calculator,
)
from ichor.core.common.units import AtomicDistance
from ichor.core.files.xyz import Trajectory

from .daemon.state import atomic_write_json, atomic_write_text
from .sampling.descriptors import mass_weighted_rmsd
from .versioning.manifest import sha256_file


BOOTSTRAP_DIRECTORY = "bootstrap"
CUSTOM_BOOTSTRAP_MANIFEST = "CUSTOM_BOOTSTRAP.json"
MODEL_BOOTSTRAP_MANIFEST = "MODEL_BOOTSTRAP.json"
CUSTOM_BOOTSTRAP_SCHEMA_VERSION = 1
MODEL_BOOTSTRAP_SCHEMA_VERSION = 1
BOOTSTRAP_DUPLICATE_RMSD_ANGSTROM = 1.0e-6
SPLITS = ("train", "int_val", "ext_val")
SPLIT_STEMS = {
    "train": "training_set_bootstrap",
    "int_val": "internal_validation_set_bootstrap",
    "ext_val": "external_validation_set_bootstrap",
}
ALF_SIDECAR_FILENAME = "alf.yaml"
MODEL_DIRECTORY_NAME = "model_krig"


class BootstrapInputError(ValueError):
    """Raised when bootstrap discovery cannot produce one safe plan."""


@dataclass
class BootstrapSource:
    split: str
    kind: str
    path: Path
    sha256: str
    frames: List[Atoms]
    alf_zero_indexed: Optional[Tuple[int, ...]] = None
    extra_columns: Tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return len(self.frames)


@dataclass
class ModelBootstrap:
    source_dir: Path
    files: List[Dict[str, Any]]
    training_count: int
    frames: List[Atoms]
    properties: Tuple[str, ...]
    atoms: Tuple[str, ...]
    system: str


@dataclass
class BootstrapPlan:
    campaign_dir: Path
    custom_enabled: bool
    pool_sha256: str
    pool_frames: List[Atoms]
    configured_targets: Dict[str, int]
    sources: Dict[str, Optional[BootstrapSource]]
    polus_deficits: Dict[str, int]
    excluded_pool_frame_ids: Tuple[int, ...]
    model: Optional[ModelBootstrap] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def effective_qm_targets(self) -> Dict[str, int]:
        targets = dict(self.configured_targets)
        if self.model is not None:
            targets["train"] = 0
        targets["total"] = sum(targets[split] for split in SPLITS)
        return targets

    @property
    def supplied_counts(self) -> Dict[str, int]:
        return {
            split: 0 if self.sources.get(split) is None else self.sources[split].count
            for split in SPLITS
        }

    @property
    def effective_training_count(self) -> int:
        if self.model is not None:
            return int(self.model.training_count)
        return int(self.configured_targets["train"])


def bootstrap_dir(campaign_dir: str | Path) -> Path:
    return Path(campaign_dir) / BOOTSTRAP_DIRECTORY


def custom_bootstrap_manifest_path(campaign_dir: str | Path) -> Path:
    return (
        Path(campaign_dir)
        / ".DATA"
        / "ACTIVE_LEARNING"
        / CUSTOM_BOOTSTRAP_MANIFEST
    )


def bootstrap_inputs_dir(campaign_dir: str | Path) -> Path:
    return Path(campaign_dir) / ".DATA" / "ACTIVE_LEARNING" / "bootstrap_inputs"


def _sha256_directory_records(records: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(list(records), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_xyz(path: Path, expected_types: Sequence[str]) -> List[Atoms]:
    try:
        trajectory = Trajectory(path)
        trajectory.read()
        frames = [frame.copy() for frame in trajectory]
    except Exception as exc:
        raise BootstrapInputError(
            "failed to read bootstrap XYZ " + str(path) + ": " + str(exc)
        ) from exc
    if not frames:
        raise BootstrapInputError("bootstrap XYZ contains no geometries: " + str(path))
    for index, frame in enumerate(frames):
        _validate_frame(
            frame,
            expected_types,
            label=path.name + " frame " + str(index),
        )
    return frames


def _validate_frame(frame: Atoms, expected_types: Sequence[str], *, label: str) -> None:
    observed = tuple(str(value) for value in frame.types_extended)
    expected = tuple(str(value) for value in expected_types)
    if observed != expected:
        raise BootstrapInputError(
            label + " atom order/type mismatch: expected " + repr(expected)
            + ", found " + repr(observed)
        )
    coordinates = np.asarray(frame.coordinates, dtype=float)
    if coordinates.shape != (len(expected), 3) or not np.all(np.isfinite(coordinates)):
        raise BootstrapInputError(label + " has invalid or non-finite coordinates")


def _expected_feature_count(natoms: int) -> int:
    if natoms == 2:
        return 1
    if natoms > 2:
        return 3 * natoms - 6
    raise BootstrapInputError(
        "CSV ALF bootstrap requires at least two atoms; use XYZ for a monoatomic system"
    )


def _normalise_alf(
    values: Sequence[Any],
    *,
    natoms: int,
    label: str,
) -> Tuple[int, ...]:
    required = 2 if natoms == 2 else 3
    if len(values) != required:
        raise BootstrapInputError(
            label + " ALF must contain exactly " + str(required) + " atom numbers"
        )
    parsed: List[int] = []
    for value in values:
        if isinstance(value, bool):
            raise BootstrapInputError(label + " ALF entries must be integers")
        try:
            item = int(value)
        except (TypeError, ValueError) as exc:
            raise BootstrapInputError(label + " ALF entries must be integers") from exc
        if str(value).strip() not in {str(item), "+" + str(item)}:
            raise BootstrapInputError(label + " ALF entries must be integers")
        if not 1 <= item <= natoms:
            raise BootstrapInputError(
                label + " ALF atom number is outside 1.." + str(natoms)
            )
        parsed.append(item - 1)
    if len(set(parsed)) != len(parsed):
        raise BootstrapInputError(label + " ALF atom numbers must be distinct")
    return tuple(parsed)


def _alf_object(values: Sequence[int], natoms: int) -> ALF:
    if natoms == 2:
        return ALF(int(values[0]), int(values[1]), None)
    return ALF(int(values[0]), int(values[1]), int(values[2]))


def _frame_from_feature_row(
    features: np.ndarray,
    *,
    pool_frame: Atoms,
    alf_values: Sequence[int],
) -> Atoms:
    natoms = len(pool_frame)
    alf = _alf_object(alf_values, natoms)
    local = np.asarray(alf_features_to_coordinates(features), dtype=float)[0]
    ordering = [int(alf.origin_idx), int(alf.x_axis_idx)]
    if alf.xy_plane_idx is not None:
        ordering.append(int(alf.xy_plane_idx))
    ordering.extend(index for index in range(natoms) if index not in ordering)
    if local.shape != (natoms, 3) or len(ordering) != natoms:
        raise BootstrapInputError("CSV ALF reconstruction returned an invalid shape")
    restored = np.empty_like(local)
    for local_index, original_index in enumerate(ordering):
        restored[original_index] = local[local_index]
    frame = Atoms(
        [
            Atom(
                str(atom_type),
                float(coords[0]),
                float(coords[1]),
                float(coords[2]),
                units=AtomicDistance.Bohr,
            )
            for atom_type, coords in zip(pool_frame.types_extended, restored)
        ]
    ).to_angstroms()
    return frame


def _read_csv_geometry(
    path: Path,
    *,
    pool_frame: Atoms,
    alf_1_indexed: Sequence[Any],
) -> Tuple[List[Atoms], Tuple[int, ...], Tuple[str, ...]]:
    natoms = len(pool_frame)
    expected_features = _expected_feature_count(natoms)
    alf_values = _normalise_alf(
        alf_1_indexed,
        natoms=natoms,
        label=path.name,
    )
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except OSError as exc:
        raise BootstrapInputError("bootstrap CSV is unreadable: " + str(path)) from exc
    if len(rows) < 2:
        raise BootstrapInputError("bootstrap CSV contains no geometries: " + str(path))
    header = [str(value).strip() for value in rows[0]]
    if not header or any(not value for value in header) or len(set(header)) != len(header):
        raise BootstrapInputError("bootstrap CSV headers must be non-empty and unique")
    expected_header = ["f" + str(index) for index in range(1, expected_features + 1)]
    if header[:expected_features] != expected_header:
        raise BootstrapInputError(
            "bootstrap CSV must begin with consecutive feature columns "
            + expected_header[0] + ".." + expected_header[-1] + ": " + str(path)
        )
    for extra in header[expected_features:]:
        if extra.startswith("f") and extra[1:].isdigit():
            raise BootstrapInputError(
                "bootstrap CSV contains a feature-like column after the feature prefix: "
                + extra
            )
    frames: List[Atoms] = []
    alf = _alf_object(alf_values, natoms)
    for row_number, raw_row in enumerate(rows[1:], start=2):
        if not raw_row or not any(str(value).strip() for value in raw_row):
            raise BootstrapInputError(
                "bootstrap CSV contains a blank row at line " + str(row_number)
            )
        if len(raw_row) != len(header):
            raise BootstrapInputError(
                "bootstrap CSV row " + str(row_number) + " has "
                + str(len(raw_row)) + " columns; expected " + str(len(header))
            )
        numbers: List[float] = []
        for column, raw in zip(header, raw_row):
            text = str(raw).strip()
            if not text:
                raise BootstrapInputError(
                    "bootstrap CSV has a blank value at line " + str(row_number)
                    + ", column " + column
                )
            try:
                value = float(text)
            except ValueError as exc:
                raise BootstrapInputError(
                    "bootstrap CSV has a non-numeric value at line "
                    + str(row_number) + ", column " + column
                ) from exc
            if not np.isfinite(value):
                raise BootstrapInputError(
                    "bootstrap CSV has a non-finite value at line "
                    + str(row_number) + ", column " + column
                )
            numbers.append(value)
        features = np.asarray(numbers[:expected_features], dtype=float)
        frame = _frame_from_feature_row(
            features,
            pool_frame=pool_frame,
            alf_values=alf_values,
        )
        _validate_frame(
            frame,
            pool_frame.types_extended,
            label=path.name + " row " + str(row_number),
        )
        try:
            round_trip = calculate_alf_features(frame[int(alf.origin_idx)], alf)
        except Exception as exc:
            raise BootstrapInputError(
                "bootstrap CSV ALF round trip failed at line " + str(row_number)
                + ": " + str(exc)
            ) from exc
        if not np.allclose(round_trip, features, rtol=1.0e-8, atol=1.0e-8):
            raise BootstrapInputError(
                "bootstrap CSV ALF round trip does not reproduce line "
                + str(row_number)
            )
        frames.append(frame)
    return frames, alf_values, tuple(header[expected_features:])


def _connectivity_signature(frame: Atoms) -> Tuple[Tuple[int, ...], ...]:
    matrix = np.asarray(frame.connectivity(default_connectivity_calculator), dtype=int)
    return tuple(tuple(int(value) for value in row) for row in matrix)


def _validate_molecular_identity(frames: Sequence[Atoms], pool_frame: Atoms, label: str) -> None:
    expected_connectivity = _connectivity_signature(pool_frame)
    for index, frame in enumerate(frames):
        _validate_frame(
            frame,
            pool_frame.types_extended,
            label=label + " geometry " + str(index),
        )
        if _connectivity_signature(frame) != expected_connectivity:
            raise BootstrapInputError(
                label + " geometry " + str(index)
                + " does not encode the same molecular connectivity as pool.xyz"
            )


def _duplicate_pairs(frames: Sequence[Atoms]) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for left in range(len(frames)):
        for right in range(left + 1, len(frames)):
            if float(mass_weighted_rmsd(frames[left], frames[right])) <= BOOTSTRAP_DUPLICATE_RMSD_ANGSTROM:
                pairs.append((left, right))
    return pairs


def _pool_matches(frames: Sequence[Atoms], pool_frames: Sequence[Atoms]) -> Tuple[int, ...]:
    matched = set()
    for custom in frames:
        for frame_id, pool in enumerate(pool_frames):
            if float(mass_weighted_rmsd(custom, pool)) <= BOOTSTRAP_DUPLICATE_RMSD_ANGSTROM:
                matched.add(int(frame_id))
    return tuple(sorted(matched))


def _load_alf_sidecar(path: Path) -> Dict[str, Sequence[Any]]:
    if not path.is_file():
        return {}
    if path.is_symlink():
        raise BootstrapInputError("bootstrap ALF sidecar refuses a symlink: " + str(path))
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise BootstrapInputError("bootstrap ALF sidecar is unreadable: " + str(path)) from exc
    if not isinstance(payload, Mapping):
        raise BootstrapInputError("bootstrap ALF sidecar must be a mapping")
    aliases = {
        "training": "train",
        "train": "train",
        "internal_validation": "int_val",
        "int_val": "int_val",
        "external_validation": "ext_val",
        "ext_val": "ext_val",
    }
    out: Dict[str, Sequence[Any]] = {}
    for key, value in payload.items():
        canonical = aliases.get(str(key))
        if canonical is None:
            raise BootstrapInputError("unknown bootstrap ALF sidecar key: " + str(key))
        if not isinstance(value, (list, tuple)):
            raise BootstrapInputError("bootstrap ALF sidecar entries must be lists")
        out[canonical] = list(value)
    return out


def _model_metadata(
    model_dir: Path,
    *,
    pool_frame: Atoms,
    config: Any,
) -> ModelBootstrap:
    try:
        from ichor.core.models import Model
    except Exception as exc:
        raise BootstrapInputError("FEREBUS model loader is unavailable: " + str(exc)) from exc
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise BootstrapInputError("bootstrap/model_krig must be a regular directory")
    for entry in model_dir.rglob("*"):
        if entry.is_symlink():
            raise BootstrapInputError(
                "model bootstrap refuses a symlink: " + str(entry)
            )
        if entry.is_file() and entry.suffix != ".model":
            raise BootstrapInputError(
                "bootstrap/model_krig contains a non-model file: " + str(entry)
            )
    model_paths = sorted(model_dir.rglob("*.model"))
    if not model_paths:
        raise BootstrapInputError("bootstrap/model_krig contains no .model files")
    expected_atoms = tuple(str(value) for value in pool_frame.atom_names)
    expected_properties = tuple(
        sorted({"iqa"} | {str(value) for value in config.ferebus.properties})
    )
    by_key: Dict[Tuple[str, str], Tuple[Any, Path]] = {}
    system_name: Optional[str] = None
    ntrain: Optional[int] = None
    records: List[Dict[str, Any]] = []
    for path in model_paths:
        if path.is_symlink():
            raise BootstrapInputError("model bootstrap refuses a symlink: " + str(path))
        try:
            model = Model(path)
            atom = str(model.atom)
            prop = str(model.type)
            system = str(model.system_name)
            task_ntrain = int(model.ntrain)
            nfeats = int(model.nfeats)
            x = np.asarray(model.x, dtype=float)
            y = np.asarray(model.y, dtype=float).reshape(-1)
            weights = np.asarray(model.weights, dtype=float).reshape(-1)
            ialf = tuple(int(value) for value in np.asarray(model.ialf, dtype=int).reshape(-1))
        except Exception as exc:
            raise BootstrapInputError(
                "failed to parse imported model " + str(path) + ": " + str(exc)
            ) from exc
        key = (prop, atom)
        if key in by_key:
            raise BootstrapInputError("duplicate imported model task: " + repr(key))
        if atom not in expected_atoms:
            raise BootstrapInputError("unexpected imported model atom: " + atom)
        if prop not in expected_properties:
            raise BootstrapInputError(
                "unexpected imported model property not listed in ferebus.properties: " + prop
            )
        if system_name is None:
            system_name = system
        elif system != system_name:
            raise BootstrapInputError("imported models disagree on system name")
        if system != str(config.campaign.system_name):
            raise BootstrapInputError(
                "imported model system " + repr(system)
                + " does not match campaign.system_name "
                + repr(config.campaign.system_name)
            )
        if ntrain is None:
            ntrain = task_ntrain
        elif task_ntrain != ntrain:
            raise BootstrapInputError("imported models disagree on training-row count")
        if nfeats != _expected_feature_count(len(pool_frame)):
            raise BootstrapInputError("imported model feature count does not match pool.xyz")
        if task_ntrain <= 0:
            raise BootstrapInputError("imported model has no training rows: " + str(path))
        if x.shape != (task_ntrain, nfeats) or y.size != task_ntrain or weights.size != task_ntrain:
            raise BootstrapInputError("imported model training array shape is invalid: " + str(path))
        if not all(np.all(np.isfinite(value)) for value in (x, y, weights)):
            raise BootstrapInputError("imported model contains non-finite training data: " + str(path))
        _normalise_alf([value + 1 for value in ialf if value is not None], natoms=len(pool_frame), label=path.name)
        by_key[key] = (model, path)
        records.append({
            "property": prop,
            "atom": atom,
            "source_path": str(path.resolve()),
            "source_relative_path": path.relative_to(model_dir).as_posix(),
            "sha256": sha256_file(path),
            "ntrain": task_ntrain,
            "nfeatures": nfeats,
            "alf_zero_indexed": list(ialf),
        })
    expected_keys = {
        (prop, atom) for prop in expected_properties for atom in expected_atoms
    }
    if set(by_key) != expected_keys:
        missing = sorted(expected_keys - set(by_key))
        extra = sorted(set(by_key) - expected_keys)
        raise BootstrapInputError(
            "imported model atom/property coverage mismatch: missing="
            + repr(missing) + " extra=" + repr(extra)
        )
    for (prop, atom), (model, path) in sorted(by_key.items()):
        try:
            from .daemon.model_contract import validate_imported_model_file

            validate_imported_model_file(
                path,
                system=str(model.system_name),
                property_name=prop,
                atom=atom,
                alf_zero_indexed=np.asarray(model.ialf, dtype=int).reshape(-1),
                train_rows=int(model.ntrain),
            )
        except Exception as exc:
            raise BootstrapInputError(
                "imported model fails the full runtime contract: "
                + str(path)
                + ": "
                + str(exc)
            ) from exc
    canonical_key = ("iqa", expected_atoms[0])
    canonical_model = by_key[canonical_key][0]
    canonical_alf = tuple(
        int(value) for value in np.asarray(canonical_model.ialf, dtype=int).reshape(-1)
    )
    frames = [
        _frame_from_feature_row(
            row,
            pool_frame=pool_frame,
            alf_values=canonical_alf,
        )
        for row in np.asarray(canonical_model.x, dtype=float)
    ]
    _validate_molecular_identity(frames, pool_frame, "imported model training")
    for (prop, atom), (model, path) in sorted(by_key.items()):
        atom_index = expected_atoms.index(atom)
        alf_values = tuple(
            int(value) for value in np.asarray(model.ialf, dtype=int).reshape(-1)
        )
        alf = _alf_object(alf_values, len(pool_frame))
        observed = np.asarray(
            [calculate_alf_features(frame[atom_index], alf) for frame in frames],
            dtype=float,
        )
        expected = np.asarray(model.x, dtype=float)
        if not np.allclose(observed, expected, rtol=1.0e-7, atol=1.0e-7):
            raise BootstrapInputError(
                "imported model training rows are inconsistent for "
                + prop + "/" + atom + ": " + str(path)
            )
    records.sort(key=lambda value: (str(value["property"]), str(value["atom"])))
    return ModelBootstrap(
        source_dir=model_dir,
        files=records,
        training_count=int(ntrain or 0),
        frames=frames,
        properties=expected_properties,
        atoms=expected_atoms,
        system=str(system_name or ""),
    )


def inspect_bootstrap_inputs(
    campaign_dir: str | Path,
    config: Any,
    pool_frames: Sequence[Atoms],
    *,
    pool_sha256: str,
    alf_values: Optional[Mapping[str, Sequence[Any]]] = None,
    alf_prompt: Optional[Callable[[str, Path, Sequence[str]], Sequence[Any]]] = None,
) -> BootstrapPlan:
    campaign = Path(campaign_dir).resolve()
    frames = [frame.copy() for frame in pool_frames]
    if not frames:
        raise BootstrapInputError("pool.xyz contains no geometries")
    pool_frame = frames[0]
    _validate_molecular_identity(frames, pool_frame, "pool")
    directory = bootstrap_dir(campaign)
    if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
        raise BootstrapInputError("bootstrap must be a regular directory: " + str(directory))
    sidecar = _load_alf_sidecar(directory / ALF_SIDECAR_FILENAME) if directory.is_dir() else {}
    supplied_alfs = dict(sidecar)
    supplied_alfs.update({str(key): value for key, value in (alf_values or {}).items()})
    recognised: Dict[str, Dict[str, Path]] = {split: {} for split in SPLITS}
    model_path = directory / MODEL_DIRECTORY_NAME
    if directory.is_dir():
        allowed_names = {MODEL_DIRECTORY_NAME, ALF_SIDECAR_FILENAME}
        for split, stem in SPLIT_STEMS.items():
            for suffix in ("xyz", "csv"):
                path = directory / (stem + "." + suffix)
                allowed_names.add(path.name)
                if path.exists():
                    if path.is_symlink() or not path.is_file():
                        raise BootstrapInputError("bootstrap input must be a regular file: " + str(path))
                    recognised[split][suffix] = path
        unknown = sorted(
            child.name for child in directory.iterdir()
            if not child.name.startswith(".") and child.name not in allowed_names
        )
        if unknown and bool(config.campaign.custom_bootstrap):
            raise BootstrapInputError(
                "unrecognised bootstrap entries: " + repr(unknown)
            )
    model_present = model_path.exists()
    recognised_count = sum(len(value) for value in recognised.values()) + int(model_present)
    if not bool(config.campaign.custom_bootstrap):
        if recognised_count:
            raise BootstrapInputError(
                "bootstrap inputs are present but campaign.custom_bootstrap is false"
            )
    elif recognised_count == 0:
        raise BootstrapInputError(
            "campaign.custom_bootstrap is true but bootstrap/ contains no recognised inputs"
        )
    for split, formats in recognised.items():
        if len(formats) > 1:
            raise BootstrapInputError(
                SPLIT_STEMS[split] + " is supplied as both XYZ and CSV"
            )
    if model_present and recognised["train"]:
        raise BootstrapInputError(
            "bootstrap/model_krig and a training-set bootstrap file both supply training data"
        )
    sources: Dict[str, Optional[BootstrapSource]] = {split: None for split in SPLITS}
    for split in SPLITS:
        formats = recognised[split]
        if not formats:
            continue
        kind, path = next(iter(formats.items()))
        if kind == "xyz":
            source_frames = _load_xyz(path, pool_frame.types_extended)
            alf = None
            extras: Tuple[str, ...] = ()
        else:
            raw_alf = supplied_alfs.get(split)
            if raw_alf is None:
                if alf_prompt is None:
                    raise BootstrapInputError(
                        "bootstrap CSV requires an ALF for " + split
                    )
                raw_alf = alf_prompt(split, path, pool_frame.atom_names)
            source_frames, alf, extras = _read_csv_geometry(
                path,
                pool_frame=pool_frame,
                alf_1_indexed=raw_alf,
            )
        _validate_molecular_identity(source_frames, pool_frame, path.name)
        duplicates = _duplicate_pairs(source_frames)
        if duplicates:
            raise BootstrapInputError(
                path.name + " contains duplicate geometries: " + repr(duplicates[:10])
            )
        sources[split] = BootstrapSource(
            split=split,
            kind=kind,
            path=path,
            sha256=sha256_file(path),
            frames=source_frames,
            alf_zero_indexed=alf,
            extra_columns=extras,
        )
    model = _model_metadata(
        model_path,
        pool_frame=pool_frame,
        config=config,
    ) if model_present else None
    configured = {
        "train": int(config.point_allocation.bootstrap_training_size),
        "int_val": int(config.point_allocation.bootstrap_internal_validation_size),
        "ext_val": int(config.point_allocation.bootstrap_external_validation_size),
    }
    for split, source in sources.items():
        if source is not None and source.count > configured[split]:
            raise BootstrapInputError(
                source.path.name + " contains " + str(source.count)
                + " geometries but point_allocation permits at most "
                + str(configured[split]) + " for " + split
            )
    all_named_frames: List[Tuple[str, int, Atoms]] = []
    if model is not None:
        all_named_frames.extend(("model_train", index, frame) for index, frame in enumerate(model.frames))
    for split, source in sources.items():
        if source is not None:
            all_named_frames.extend((split, index, frame) for index, frame in enumerate(source.frames))
    for left in range(len(all_named_frames)):
        for right in range(left + 1, len(all_named_frames)):
            left_split, left_index, left_frame = all_named_frames[left]
            right_split, right_index, right_frame = all_named_frames[right]
            if left_split == right_split:
                continue
            if float(mass_weighted_rmsd(left_frame, right_frame)) <= BOOTSTRAP_DUPLICATE_RMSD_ANGSTROM:
                raise BootstrapInputError(
                    "bootstrap split leakage: " + left_split + "[" + str(left_index)
                    + "] duplicates " + right_split + "[" + str(right_index) + "]"
                )
    custom_frames = [frame for _split, _index, frame in all_named_frames]
    excluded = _pool_matches(custom_frames, frames)
    deficits = {
        split: (
            0 if split == "train" and model is not None
            else configured[split] - (0 if sources[split] is None else sources[split].count)
        )
        for split in SPLITS
    }
    if sum(deficits.values()) > len(frames) - len(excluded):
        raise BootstrapInputError(
            "pool.xyz cannot fill bootstrap deficits after excluding custom/model geometries: "
            + "need " + str(sum(deficits.values())) + ", available "
            + str(len(frames) - len(excluded))
        )
    return BootstrapPlan(
        campaign_dir=campaign,
        custom_enabled=bool(config.campaign.custom_bootstrap),
        pool_sha256=str(pool_sha256),
        pool_frames=frames,
        configured_targets=configured,
        sources=sources,
        polus_deficits=deficits,
        excluded_pool_frame_ids=excluded,
        model=model,
    )


def _xyz_text(frames: Sequence[Atoms]) -> str:
    lines: List[str] = []
    for index, frame in enumerate(frames):
        lines.extend((str(len(frame)), "bootstrap frame " + str(index)))
        for atom in frame:
            lines.append(
                str(atom.type) + " " + format(float(atom.x), ".16g") + " "
                + format(float(atom.y), ".16g") + " " + format(float(atom.z), ".16g")
            )
    return "\n".join(lines) + ("\n" if lines else "")


def _copy_atomic_checked(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> None:
    """Copy the inspected regular-file inode and verify its confirmed bytes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        "." + destination.name + "." + uuid4().hex[:8] + ".tmp"
    )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: Optional[int] = None
    try:
        if source.is_symlink() or not source.is_file():
            raise BootstrapInputError(
                "bootstrap source changed and is no longer a regular file: " + str(source)
            )
        descriptor = os.open(source, flags)
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise BootstrapInputError("bootstrap source is not a regular file: " + str(source))
        with os.fdopen(descriptor, "rb", closefd=True) as input_file:
            descriptor = None
            with open(temporary, "xb") as output_file:
                shutil.copyfileobj(input_file, output_file)
                output_file.flush()
                os.fsync(output_file.fileno())
        copied_sha = sha256_file(temporary)
        if copied_sha != str(expected_sha256):
            raise BootstrapInputError(
                "bootstrap source changed after inspection: " + str(source)
            )
        os.replace(temporary, destination)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _validate_copied_source(source: BootstrapSource, copied: Path, pool_frame: Atoms) -> None:
    if source.kind == "xyz":
        frames = _load_xyz(copied, pool_frame.types_extended)
    elif source.kind == "csv":
        if source.alf_zero_indexed is None:
            raise BootstrapInputError("copied CSV bootstrap has no confirmed ALF")
        frames, copied_alf, _extra = _read_csv_geometry(
            copied,
            pool_frame=pool_frame,
            alf_1_indexed=[int(value) + 1 for value in source.alf_zero_indexed],
        )
        if tuple(copied_alf) != tuple(source.alf_zero_indexed):
            raise BootstrapInputError("copied CSV bootstrap ALF changed")
    else:
        raise BootstrapInputError("unknown bootstrap source kind: " + str(source.kind))
    if len(frames) != len(source.frames):
        raise BootstrapInputError("copied bootstrap frame count changed")
    for index, (observed, expected) in enumerate(zip(frames, source.frames)):
        if float(mass_weighted_rmsd(observed, expected)) > BOOTSTRAP_DUPLICATE_RMSD_ANGSTROM:
            raise BootstrapInputError(
                "copied bootstrap geometry changed at frame " + str(index)
            )


def commit_bootstrap_plan(plan: BootstrapPlan) -> Dict[str, Any]:
    """Snapshot one confirmed plan into immutable daemon-owned storage."""
    manifest_path = custom_bootstrap_manifest_path(plan.campaign_dir)
    destination = bootstrap_inputs_dir(plan.campaign_dir)
    if manifest_path.is_file():
        existing = read_custom_bootstrap_manifest(plan.campaign_dir)
        candidate = _plan_identity(plan)
        if existing.get("plan_identity_sha256") != candidate:
            raise BootstrapInputError(
                "accepted bootstrap inputs already exist with a different identity"
            )
        return existing
    if destination.exists():
        embedded = destination / CUSTOM_BOOTSTRAP_MANIFEST
        try:
            recovered = json.loads(embedded.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BootstrapInputError(
                "uncommitted bootstrap input staging already exists: "
                + str(destination)
            ) from exc
        if not isinstance(recovered, dict) or recovered.get(
            "plan_identity_sha256"
        ) != _plan_identity(plan):
            raise BootstrapInputError(
                "uncommitted bootstrap input staging has a different identity: "
                + str(destination)
            )
        pointer = dict(recovered)
        pointer["bootstrap_inputs_root"] = destination.relative_to(
            plan.campaign_dir
        ).as_posix()
        atomic_write_json(manifest_path, pointer)
        return read_custom_bootstrap_manifest(plan.campaign_dir)
    staging = destination.with_name(".bootstrap-" + uuid4().hex[:8])
    staging.mkdir(parents=True, exist_ok=False)
    source_records: Dict[str, Optional[Dict[str, Any]]] = {}
    try:
        for split in SPLITS:
            source = plan.sources.get(split)
            if source is None:
                source_records[split] = None
                continue
            raw_destination = staging / "sources" / source.path.name
            _copy_atomic_checked(
                source.path,
                raw_destination,
                expected_sha256=source.sha256,
            )
            _validate_copied_source(source, raw_destination, plan.pool_frames[0])
            canonical_xyz = staging / "canonical" / (SPLIT_STEMS[split] + ".xyz")
            canonical_xyz.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(canonical_xyz, _xyz_text(source.frames))
            source_records[split] = {
                "kind": source.kind,
                "operator_path": str(source.path.resolve()),
                "source_path": raw_destination.relative_to(staging).as_posix(),
                "source_sha256": source.sha256,
                "canonical_xyz": canonical_xyz.relative_to(staging).as_posix(),
                "canonical_xyz_sha256": sha256_file(canonical_xyz),
                "count": int(source.count),
                "alf_zero_indexed": (
                    None if source.alf_zero_indexed is None
                    else [int(value) for value in source.alf_zero_indexed]
                ),
                "alf_one_indexed": (
                    None if source.alf_zero_indexed is None
                    else [int(value) + 1 for value in source.alf_zero_indexed]
                ),
                "extra_columns": list(source.extra_columns),
            }
        model_payload = None
        if plan.model is not None:
            copied_records: List[Dict[str, Any]] = []
            for record in plan.model.files:
                source = Path(str(record["source_path"]))
                target = staging / MODEL_DIRECTORY_NAME / str(record["source_relative_path"])
                expected_sha = str(record["sha256"])
                _copy_atomic_checked(
                    source,
                    target,
                    expected_sha256=expected_sha,
                )
                copied = dict(record)
                copied["path"] = target.relative_to(staging).as_posix()
                copied["sha256"] = expected_sha
                try:
                    from .daemon.model_contract import validate_imported_model_file

                    validate_imported_model_file(
                        target,
                        system=plan.model.system,
                        property_name=str(record["property"]),
                        atom=str(record["atom"]),
                        alf_zero_indexed=record["alf_zero_indexed"],
                        train_rows=int(record["ntrain"]),
                    )
                except Exception as exc:
                    raise BootstrapInputError(
                        "copied imported model fails the runtime contract: "
                        + str(target)
                        + ": "
                        + str(exc)
                    ) from exc
                copied_records.append(copied)
            model_xyz = staging / "canonical" / "model_training_reconstructed.xyz"
            model_xyz.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(model_xyz, _xyz_text(plan.model.frames))
            model_payload = {
                "schema_version": MODEL_BOOTSTRAP_SCHEMA_VERSION,
                "system": plan.model.system,
                "properties": list(plan.model.properties),
                "atoms": list(plan.model.atoms),
                "training_count": int(plan.model.training_count),
                "files": copied_records,
                "model_set_sha256": _sha256_directory_records(copied_records),
                "reconstructed_training_xyz": model_xyz.relative_to(staging).as_posix(),
                "reconstructed_training_xyz_sha256": sha256_file(model_xyz),
            }
            atomic_write_json(staging / MODEL_BOOTSTRAP_MANIFEST, model_payload)
        payload = {
            "schema_version": CUSTOM_BOOTSTRAP_SCHEMA_VERSION,
            "campaign_schema_version": 11,
            "custom_bootstrap": bool(plan.custom_enabled),
            "pool_sha256": str(plan.pool_sha256),
            "configured_targets": dict(plan.configured_targets),
            "effective_qm_targets": plan.effective_qm_targets,
            "supplied_counts": plan.supplied_counts,
            "polus_deficits": dict(plan.polus_deficits),
            "effective_training_count": int(plan.effective_training_count),
            "excluded_pool_frame_ids": [int(value) for value in plan.excluded_pool_frame_ids],
            "sources": source_records,
            "model": model_payload,
            "plan_identity_sha256": _plan_identity(plan),
            "confirmed": True,
        }
        atomic_write_json(staging / CUSTOM_BOOTSTRAP_MANIFEST, payload)
        os.replace(staging, destination)
        pointer = dict(payload)
        pointer["bootstrap_inputs_root"] = destination.relative_to(plan.campaign_dir).as_posix()
        atomic_write_json(manifest_path, pointer)
        return pointer
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _plan_identity(plan: BootstrapPlan) -> str:
    identity = {
        "custom_bootstrap": bool(plan.custom_enabled),
        "pool_sha256": str(plan.pool_sha256),
        "targets": dict(plan.configured_targets),
        "sources": {
            split: (
                None if plan.sources.get(split) is None else {
                    "kind": plan.sources[split].kind,
                    "sha256": plan.sources[split].sha256,
                    "count": plan.sources[split].count,
                    "alf": plan.sources[split].alf_zero_indexed,
                }
            ) for split in SPLITS
        },
        "model": (
            None if plan.model is None else {
                "training_count": plan.model.training_count,
                "files": [
                    (record["property"], record["atom"], record["sha256"])
                    for record in plan.model.files
                ],
            }
        ),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_custom_bootstrap_manifest(campaign_dir: str | Path) -> Dict[str, Any]:
    path = custom_bootstrap_manifest_path(campaign_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BootstrapInputError("custom bootstrap manifest is unreadable: " + str(path)) from exc
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != CUSTOM_BOOTSTRAP_SCHEMA_VERSION:
        raise BootstrapInputError("custom bootstrap manifest schema is invalid")
    root = Path(campaign_dir) / str(payload.get("bootstrap_inputs_root") or "")
    if root.resolve(strict=False) != bootstrap_inputs_dir(campaign_dir).resolve(strict=False):
        raise BootstrapInputError("custom bootstrap manifest root is invalid")
    if not root.is_dir() or root.is_symlink():
        raise BootstrapInputError("immutable bootstrap input root is missing")
    embedded = root / CUSTOM_BOOTSTRAP_MANIFEST
    if not embedded.is_file() or embedded.is_symlink():
        raise BootstrapInputError("immutable bootstrap manifest is missing")
    try:
        embedded_payload = json.loads(embedded.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BootstrapInputError("immutable bootstrap manifest is unreadable") from exc
    expected_pointer = dict(embedded_payload) if isinstance(embedded_payload, dict) else {}
    expected_pointer["bootstrap_inputs_root"] = root.resolve().relative_to(
        Path(campaign_dir).resolve()
    ).as_posix()
    if payload != expected_pointer:
        raise BootstrapInputError(
            "custom bootstrap pointer disagrees with its immutable manifest"
        )
    if payload.get("confirmed") is not True:
        raise BootstrapInputError("custom bootstrap manifest is not confirmed")
    raw_excluded = payload.get("excluded_pool_frame_ids")
    if not isinstance(raw_excluded, list):
        raise BootstrapInputError(
            "custom bootstrap excluded pool-frame IDs must be a list"
        )
    excluded: List[int] = []
    for value in raw_excluded:
        if isinstance(value, bool):
            raise BootstrapInputError(
                "custom bootstrap excluded pool-frame IDs must be integers"
            )
        try:
            frame_id = int(value)
        except (TypeError, ValueError) as exc:
            raise BootstrapInputError(
                "custom bootstrap excluded pool-frame IDs must be integers"
            ) from exc
        if frame_id < 0 or str(value).strip() != str(frame_id):
            raise BootstrapInputError(
                "custom bootstrap excluded pool-frame IDs are invalid"
            )
        excluded.append(frame_id)
    if len(set(excluded)) != len(excluded):
        raise BootstrapInputError(
            "custom bootstrap excluded pool-frame IDs contain duplicates"
        )

    def committed_file(raw_path: Any, label: str) -> Path:
        text = str(raw_path or "")
        if not text or "\\" in text:
            raise BootstrapInputError(label + " path is invalid")
        relative = Path(*text.split("/"))
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise BootstrapInputError(label + " path is invalid")
        candidate = root / relative
        try:
            candidate.resolve(strict=False).relative_to(root.resolve())
        except ValueError as exc:
            raise BootstrapInputError(label + " path escapes immutable inputs") from exc
        if candidate.is_symlink() or not candidate.is_file():
            raise BootstrapInputError(label + " is missing")
        return candidate

    for split, record in dict(payload.get("sources") or {}).items():
        if split not in SPLITS:
            raise BootstrapInputError("custom bootstrap manifest has an invalid split")
        if record is None:
            continue
        if not isinstance(record, dict):
            raise BootstrapInputError("custom bootstrap source record is invalid")
        source = committed_file(record.get("source_path"), split + " source")
        canonical = committed_file(record.get("canonical_xyz"), split + " canonical XYZ")
        if sha256_file(source) != str(record.get("source_sha256") or ""):
            raise BootstrapInputError(split + " bootstrap source SHA mismatch")
        if sha256_file(canonical) != str(record.get("canonical_xyz_sha256") or ""):
            raise BootstrapInputError(split + " canonical bootstrap SHA mismatch")

    model = payload.get("model")
    if model is not None:
        if not isinstance(model, dict):
            raise BootstrapInputError("model-bootstrap record is invalid")
        model_manifest = committed_file(MODEL_BOOTSTRAP_MANIFEST, "model-bootstrap manifest")
        try:
            model_payload = json.loads(model_manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BootstrapInputError("model-bootstrap manifest is unreadable") from exc
        if model_payload != model:
            raise BootstrapInputError(
                "custom bootstrap and model-bootstrap manifests disagree"
            )
        records = model.get("files")
        if not isinstance(records, list) or not records:
            raise BootstrapInputError("model-bootstrap file inventory is empty")
        for record in records:
            if not isinstance(record, dict):
                raise BootstrapInputError("model-bootstrap file record is invalid")
            path = committed_file(record.get("path"), "model-bootstrap model")
            if sha256_file(path) != str(record.get("sha256") or ""):
                raise BootstrapInputError("model-bootstrap model SHA mismatch: " + str(path))
        if _sha256_directory_records(records) != str(model.get("model_set_sha256") or ""):
            raise BootstrapInputError("model-bootstrap set SHA mismatch")
        reconstructed = committed_file(
            model.get("reconstructed_training_xyz"),
            "model-bootstrap reconstructed training XYZ",
        )
        if sha256_file(reconstructed) != str(
            model.get("reconstructed_training_xyz_sha256") or ""
        ):
            raise BootstrapInputError(
                "model-bootstrap reconstructed training XYZ SHA mismatch"
            )
    return payload


def load_committed_bootstrap_frames(
    campaign_dir: str | Path,
) -> Tuple[Dict[str, List[Atoms]], Dict[str, Any]]:
    payload = read_custom_bootstrap_manifest(campaign_dir)
    root = bootstrap_inputs_dir(campaign_dir)
    frames: Dict[str, List[Atoms]] = {split: [] for split in SPLITS}
    for split, record in dict(payload.get("sources") or {}).items():
        if split not in frames or record is None:
            continue
        path = root / str(record.get("canonical_xyz") or "")
        if not path.is_file() or path.is_symlink():
            raise BootstrapInputError("canonical bootstrap XYZ is missing: " + str(path))
        if sha256_file(path) != str(record.get("canonical_xyz_sha256") or ""):
            raise BootstrapInputError("canonical bootstrap XYZ hash mismatch: " + str(path))
        trajectory = Trajectory(path)
        trajectory.read()
        frames[split] = [frame.copy() for frame in trajectory]
        if len(frames[split]) != int(record.get("count", -1)):
            raise BootstrapInputError("canonical bootstrap XYZ count mismatch")
    return frames, payload


__all__ = [
    "BootstrapInputError",
    "BootstrapPlan",
    "BootstrapSource",
    "CUSTOM_BOOTSTRAP_MANIFEST",
    "MODEL_BOOTSTRAP_MANIFEST",
    "SPLITS",
    "SPLIT_STEMS",
    "bootstrap_dir",
    "bootstrap_inputs_dir",
    "custom_bootstrap_manifest_path",
    "inspect_bootstrap_inputs",
    "commit_bootstrap_plan",
    "read_custom_bootstrap_manifest",
    "load_committed_bootstrap_frames",
]
