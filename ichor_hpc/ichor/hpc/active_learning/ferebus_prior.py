"""Physical prior-mean contract shared by active-learning FEREBUS stages.

FEREBUS mean type 21 returns an isolated-atom IQA energy in Hartree for IQA
models and zero for auxiliary properties.  The values below intentionally
mirror ``FEREBUS_CPU/src/utils/atomic_energies.f90``.  Keeping the table and a
version identifier here lets the daemon validate generated configurations and
trained models before they become campaign state.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


PRIOR_MEAN_TYPE = 21
PRIOR_MEAN_UNITS = "ha"
PRIOR_CONTRACT_SCHEMA_VERSION = 1
ATOMIC_ENERGY_REGISTRY_VERSION = "ferebus_cpu_atomic_energies_v1"


class FerebusPriorError(ValueError):
    """Raised when a physical FEREBUS prior contract is invalid."""


_ATOMIC_ENERGIES_HA: Dict[str, Dict[str, float]] = {
    "b3lyp/aug-cc-pvtz": {
        "C": -37.8590608392,
        "H": -0.502259675743,
        "O": -75.0941778191,
        "N": -54.6028906778,
        "S": -398.139308517,
    },
    "b3lyp/6-31+g(d,p)": {
        "C": -37.851334,
        "H": -0.500273,
        "O": -75.067605,
        "N": -54.587774,
        "S": -398.106712319,
    },
    "b3lyp/6-311+g(d,p)": {
        "C": -35.0,
        "H": -0.5,
        "O": -75.0,
        "N": -54.0,
        "S": -398.0,
    },
    "ccsd/6-31+g(d,p)": {
        "C": -37.755631,
        "H": -0.4982329,
        "O": -74.9015088,
        "N": -54.475602,
        "S": -398.0,
    },
    "gold": {
        "C": -37.779653,
        "H": -0.4998212,
        "O": -74.9755223,
        "N": -54.5143678,
        "S": -397.6511594,
    },
    "b3lyp/def2-tzvp": {
        "C": -37.8594781,
        "H": -0.5021542,
        "O": -75.0962754,
        "N": -54.6039774,
        "S": -398.1323742,
    },
}

SUPPORTED_LEVELS = frozenset(_ATOMIC_ENERGIES_HA)
SUPPORTED_ELEMENTS = frozenset({"H", "C", "N", "O", "S"})
_ATOM_LABEL_RE = re.compile(r"^([A-Z][a-z]?)(?:[0-9].*)?$")


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonicalise_level_of_theory(value: Any) -> str:
    """Return the exact FEREBUS token for one supported level of theory."""
    if not isinstance(value, str) or not value.strip():
        raise FerebusPriorError(
            "ferebus.prior_mean_level_of_theory must be 'auto' or a non-empty string"
        )
    compact = re.sub(r"\s+", "", value).lower()
    if compact == "auto":
        return "auto"
    if compact not in SUPPORTED_LEVELS:
        raise FerebusPriorError(
            "unsupported physical-prior level of theory "
            + repr(value)
            + "; expected one of "
            + repr(sorted(SUPPORTED_LEVELS))
            + " or 'auto'"
        )
    return compact


def gaussian_level_of_theory(method: Any, basis_set: Any) -> str:
    """Resolve a Gaussian method/basis pair to a FEREBUS atomic-energy key."""
    if not isinstance(method, str) or not method.strip():
        raise FerebusPriorError("gaussian.method must be a non-empty string")
    if not isinstance(basis_set, str) or not basis_set.strip():
        raise FerebusPriorError("gaussian.basis_set must be a non-empty string")
    return canonicalise_level_of_theory(method.strip() + "/" + basis_set.strip())


def element_from_atom_label(atom: Any) -> str:
    """Extract and validate the chemical element represented by an atom label."""
    label = str(atom or "")
    match = _ATOM_LABEL_RE.fullmatch(label)
    if match is None:
        raise FerebusPriorError(
            "FEREBUS atom label does not expose an unambiguous element: " + repr(label)
        )
    element = match.group(1)
    if element not in SUPPORTED_ELEMENTS:
        raise FerebusPriorError(
            "FEREBUS physical prior has no isolated-atom IQA energy for element "
            + repr(element)
            + " from atom "
            + repr(label)
            + "; supported elements are "
            + repr(sorted(SUPPORTED_ELEMENTS))
        )
    return element


@dataclass(frozen=True)
class FerebusPriorContract:
    mean_type: int
    level_of_theory: str
    iqa_deviation_factor: float
    feature_scaling: bool
    property_scaling: bool

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": PRIOR_CONTRACT_SCHEMA_VERSION,
            "mean_type": int(self.mean_type),
            "level_of_theory": self.level_of_theory,
            "iqa_deviation_factor": float(self.iqa_deviation_factor),
            "units": PRIOR_MEAN_UNITS,
            "feature_scaling": bool(self.feature_scaling),
            "property_scaling": bool(self.property_scaling),
            "atomic_energy_registry_version": ATOMIC_ENERGY_REGISTRY_VERSION,
        }

    @property
    def contract_sha256(self) -> str:
        return _canonical_json_sha256(self.identity_payload())

    def to_dict(self) -> Dict[str, Any]:
        payload = self.identity_payload()
        payload["contract_sha256"] = self.contract_sha256
        return payload

    def expected_mean_ha(self, property_name: Any, atom: Any) -> float:
        if str(property_name) != "iqa":
            return 0.0
        element = element_from_atom_label(atom)
        return float(
            self.iqa_deviation_factor
            * _ATOMIC_ENERGIES_HA[self.level_of_theory][element]
        )

    def task_payload(self, property_name: Any, atom: Any) -> Dict[str, Any]:
        return {
            "contract_sha256": self.contract_sha256,
            "mean_type": int(self.mean_type),
            "level_of_theory": self.level_of_theory,
            "iqa_deviation_factor": float(self.iqa_deviation_factor),
            "units": PRIOR_MEAN_UNITS,
            "expected_mean_ha": self.expected_mean_ha(property_name, atom),
            "feature_scaling": bool(self.feature_scaling),
            "property_scaling": bool(self.property_scaling),
        }


def resolve_ferebus_prior_contract(
    config: Any,
    *,
    atom_labels: Optional[Sequence[Any]] = None,
) -> FerebusPriorContract:
    """Resolve and validate the campaign's physical prior-mean contract."""
    ferebus = getattr(config, "ferebus", config)
    gaussian = getattr(config, "gaussian", None)
    mean_type = getattr(ferebus, "prior_mean_type", PRIOR_MEAN_TYPE)
    try:
        parsed_mean_type = int(mean_type)
    except (TypeError, ValueError) as exc:
        raise FerebusPriorError(
            "ferebus.prior_mean_type must be 21 for active-learning campaigns"
        ) from exc
    if isinstance(mean_type, bool) or parsed_mean_type != PRIOR_MEAN_TYPE:
        raise FerebusPriorError(
            "ferebus.prior_mean_type must be 21 for active-learning campaigns"
        )
    raw_level = canonicalise_level_of_theory(
        getattr(ferebus, "prior_mean_level_of_theory", "auto")
    )
    if raw_level == "auto":
        if gaussian is None:
            raise FerebusPriorError(
                "ferebus.prior_mean_level_of_theory='auto' requires Gaussian settings"
            )
        level = gaussian_level_of_theory(
            getattr(gaussian, "method", None),
            getattr(gaussian, "basis_set", None),
        )
    else:
        level = raw_level
        if gaussian is not None:
            gaussian_level = gaussian_level_of_theory(
                getattr(gaussian, "method", None),
                getattr(gaussian, "basis_set", None),
            )
            if level != gaussian_level:
                raise FerebusPriorError(
                    "ferebus.prior_mean_level_of_theory "
                    + repr(level)
                    + " does not match Gaussian training-data level "
                    + repr(gaussian_level)
                )
    factor = getattr(ferebus, "prior_mean_iqa_deviation_factor", 1.0)
    if isinstance(factor, bool):
        raise FerebusPriorError(
            "ferebus.prior_mean_iqa_deviation_factor must be a finite positive number"
        )
    try:
        factor_float = float(factor)
    except (TypeError, ValueError) as exc:
        raise FerebusPriorError(
            "ferebus.prior_mean_iqa_deviation_factor must be a finite positive number"
        ) from exc
    if not math.isfinite(factor_float) or factor_float <= 0.0:
        raise FerebusPriorError(
            "ferebus.prior_mean_iqa_deviation_factor must be a finite positive number"
        )
    feature_scaling = getattr(ferebus, "feature_scaling", True)
    property_scaling = getattr(ferebus, "property_scaling", False)
    if not isinstance(feature_scaling, bool):
        raise FerebusPriorError("ferebus.feature_scaling must be a boolean")
    if not isinstance(property_scaling, bool):
        raise FerebusPriorError("ferebus.property_scaling must be a boolean")
    if property_scaling:
        raise FerebusPriorError(
            "ferebus.property_scaling must be false with prior_mean_type 21 because "
            "the physical prior and IQA targets must both remain in Hartree"
        )
    contract = FerebusPriorContract(
        mean_type=PRIOR_MEAN_TYPE,
        level_of_theory=level,
        iqa_deviation_factor=factor_float,
        feature_scaling=feature_scaling,
        property_scaling=False,
    )
    for atom in atom_labels or ():
        element_from_atom_label(atom)
    return contract


def contract_from_payload(payload: Any) -> FerebusPriorContract:
    """Parse and authenticate a serialised physical-prior contract."""
    if not isinstance(payload, Mapping):
        raise FerebusPriorError("FEREBUS prior contract must be an object")
    if int(payload.get("schema_version", -1)) != PRIOR_CONTRACT_SCHEMA_VERSION:
        raise FerebusPriorError("unsupported FEREBUS prior contract schema")
    if str(payload.get("units") or "") != PRIOR_MEAN_UNITS:
        raise FerebusPriorError("FEREBUS prior contract units must be 'ha'")
    if str(payload.get("atomic_energy_registry_version") or "") != (
        ATOMIC_ENERGY_REGISTRY_VERSION
    ):
        raise FerebusPriorError("unsupported FEREBUS atomic-energy registry")
    contract = FerebusPriorContract(
        mean_type=int(payload.get("mean_type", -1)),
        level_of_theory=canonicalise_level_of_theory(
            payload.get("level_of_theory")
        ),
        iqa_deviation_factor=float(payload.get("iqa_deviation_factor")),
        feature_scaling=payload.get("feature_scaling"),
        property_scaling=payload.get("property_scaling"),
    )
    if contract.mean_type != PRIOR_MEAN_TYPE:
        raise FerebusPriorError("FEREBUS prior contract mean_type must be 21")
    if not isinstance(contract.feature_scaling, bool):
        raise FerebusPriorError("FEREBUS prior contract feature_scaling is invalid")
    if not isinstance(contract.property_scaling, bool) or contract.property_scaling:
        raise FerebusPriorError("FEREBUS prior contract property_scaling must be false")
    if (
        not math.isfinite(contract.iqa_deviation_factor)
        or contract.iqa_deviation_factor <= 0.0
    ):
        raise FerebusPriorError("FEREBUS prior contract factor is invalid")
    if str(payload.get("contract_sha256") or "") != contract.contract_sha256:
        raise FerebusPriorError("FEREBUS prior contract hash mismatch")
    return contract


def model_constant_mean_ha(model: Any) -> float:
    """Read one parsed model's scalar constant mean in model-output units."""
    try:
        value = model.mean.value(np.zeros((1, int(model.nfeats)), dtype=float))
        mean = float(value[0])
    except Exception as exc:
        raise FerebusPriorError(
            "FEREBUS model does not expose a scalar constant prior mean"
        ) from exc
    if not math.isfinite(mean):
        raise FerebusPriorError("FEREBUS model prior mean is non-finite")
    return mean


def validate_model_prior_mean(
    model: Any,
    *,
    contract: FerebusPriorContract,
    property_name: Any,
    atom: Any,
    relative_tolerance: float = 1.0e-10,
    absolute_tolerance: float = 1.0e-10,
) -> Dict[str, Any]:
    """Validate and report the physical prior stored in a trained model."""
    observed = model_constant_mean_ha(model)
    expected = contract.expected_mean_ha(property_name, atom)
    if not math.isclose(
        observed,
        expected,
        rel_tol=float(relative_tolerance),
        abs_tol=float(absolute_tolerance),
    ):
        raise FerebusPriorError(
            "FEREBUS model prior mean mismatch for "
            + str(property_name)
            + "/"
            + str(atom)
            + ": expected "
            + repr(expected)
            + " Ha, observed "
            + repr(observed)
            + " Ha"
        )
    return {
        "contract_sha256": contract.contract_sha256,
        "expected_mean_ha": float(expected),
        "observed_mean_ha": float(observed),
        "units": PRIOR_MEAN_UNITS,
    }


def parse_ferebus_config_contract(path: Path) -> Dict[str, Any]:
    """Parse the six configuration fields that define the prior/scaling contract."""
    required = {
        "mean_type": "mean_type",
        "level_of_theory": "level_of_theory",
        "iqadeviationfactor": "iqa_deviation_factor",
        "scaling": "scaling",
        "scale_feats": "scale_feats",
        "scale_prop": "scale_prop",
    }
    found: Dict[str, str] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FerebusPriorError("FEREBUS config is unreadable: " + str(path)) from exc
    for raw in lines:
        match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", raw)
        if match is None:
            continue
        folded = match.group(1).casefold()
        if folded not in required:
            continue
        key = required[folded]
        if key in found:
            raise FerebusPriorError(
                "FEREBUS config contains duplicate prior field " + repr(match.group(1))
            )
        found[key] = match.group(2)
    missing = sorted(set(required.values()) - set(found))
    if missing:
        raise FerebusPriorError(
            "FEREBUS config is missing prior/scaling fields: " + repr(missing)
        )
    try:
        level = found["level_of_theory"].strip().strip('"').strip("'")
        return {
            "mean_type": int(found["mean_type"]),
            "level_of_theory": canonicalise_level_of_theory(level),
            "iqa_deviation_factor": float(found["iqa_deviation_factor"]),
            "scaling": bool(int(found["scaling"])),
            "scale_feats": bool(int(found["scale_feats"])),
            "scale_prop": bool(int(found["scale_prop"])),
        }
    except (TypeError, ValueError) as exc:
        raise FerebusPriorError(
            "FEREBUS config prior/scaling values are invalid: " + str(path)
        ) from exc


def validate_ferebus_config_contract(
    path: Path,
    contract: FerebusPriorContract,
) -> Dict[str, Any]:
    """Prove that a generated FEREBUS config matches the campaign contract."""
    parsed = parse_ferebus_config_contract(path)
    expected = {
        "mean_type": PRIOR_MEAN_TYPE,
        "level_of_theory": contract.level_of_theory,
        "iqa_deviation_factor": float(contract.iqa_deviation_factor),
        "scaling": bool(contract.feature_scaling or contract.property_scaling),
        "scale_feats": bool(contract.feature_scaling),
        "scale_prop": bool(contract.property_scaling),
    }
    for key, value in expected.items():
        observed = parsed[key]
        if isinstance(value, float):
            equal = math.isclose(float(observed), value, rel_tol=1.0e-12, abs_tol=1.0e-15)
        else:
            equal = observed == value
        if not equal:
            raise FerebusPriorError(
                "FEREBUS config "
                + key
                + " mismatch: expected "
                + repr(value)
                + ", observed "
                + repr(observed)
            )
    return parsed


__all__ = [
    "ATOMIC_ENERGY_REGISTRY_VERSION",
    "FerebusPriorContract",
    "FerebusPriorError",
    "PRIOR_CONTRACT_SCHEMA_VERSION",
    "PRIOR_MEAN_TYPE",
    "PRIOR_MEAN_UNITS",
    "SUPPORTED_ELEMENTS",
    "SUPPORTED_LEVELS",
    "canonicalise_level_of_theory",
    "contract_from_payload",
    "element_from_atom_label",
    "gaussian_level_of_theory",
    "model_constant_mean_ha",
    "parse_ferebus_config_contract",
    "resolve_ferebus_prior_contract",
    "validate_ferebus_config_contract",
    "validate_model_prior_mean",
]
