"""Semantic prior-mean contract shared by active-learning FEREBUS stages."""
from __future__ import annotations

import hashlib
from .strict_json import strict_json as json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


PRIOR_MEAN_TYPES = {
    "zero": 0,
    "training_mean": 1,
    "training_median": 2,
    "physical_atomic_iqa": 21,
}
FEREBUS_KERNEL_TOKENS = {
    "periodic_rbf": "rbfc_per",
    "rbf": "rbf",
}
PRIOR_MEAN_TYPE = PRIOR_MEAN_TYPES["physical_atomic_iqa"]
PRIOR_MEAN_UNITS = "ha"
PRIOR_CONTRACT_SCHEMA_VERSION = 2
ATOMIC_ENERGY_REGISTRY_VERSION = "ichor_verified_ferebus_atomic_energies_v2"


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
    "ccsd/6-31+g(d,p)": {
        "C": -37.755631,
        "H": -0.4982329,
        "O": -74.9015088,
        "N": -54.475602,
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


def backend_kernel_token(value: Any) -> str:
    """Translate one public kernel family to its exact native token."""
    if not isinstance(value, str) or value not in FEREBUS_KERNEL_TOKENS:
        raise FerebusPriorError(
            "unsupported FEREBUS kernel; expected one of "
            + repr(sorted(FEREBUS_KERNEL_TOKENS))
        )
    return FEREBUS_KERNEL_TOKENS[value]


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
    strategy: str
    mean_type: int
    level_of_theory: Optional[str]
    physical_prior_scale: float

    @property
    def iqa_deviation_factor(self) -> float:
        """Return the native FEREBUS name for the physical scale."""
        return self.physical_prior_scale

    @property
    def feature_scaling(self) -> bool:
        return True

    @property
    def property_scaling(self) -> bool:
        return False

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": PRIOR_CONTRACT_SCHEMA_VERSION,
            "strategy": self.strategy,
            "mean_type": int(self.mean_type),
            "level_of_theory": self.level_of_theory,
            "physical_prior_scale": float(self.physical_prior_scale),
            "units": PRIOR_MEAN_UNITS,
            "feature_scaling": True,
            "property_scaling": False,
            "atomic_energy_registry_version": (
                ATOMIC_ENERGY_REGISTRY_VERSION
                if self.strategy == "physical_atomic_iqa"
                else None
            ),
        }

    @property
    def contract_sha256(self) -> str:
        return _canonical_json_sha256(self.identity_payload())

    def to_dict(self) -> Dict[str, Any]:
        payload = self.identity_payload()
        payload["contract_sha256"] = self.contract_sha256
        return payload

    def expected_mean_ha(
        self,
        property_name: Any,
        atom: Any,
        *,
        training_values: Optional[Sequence[float]] = None,
    ) -> float:
        if self.strategy == "zero":
            return 0.0
        if self.strategy == "physical_atomic_iqa":
            if str(property_name) != "iqa":
                return 0.0
            element = element_from_atom_label(atom)
            level_values = _ATOMIC_ENERGIES_HA.get(str(self.level_of_theory), {})
            if element not in level_values:
                raise FerebusPriorError(
                    "FEREBUS physical prior has no verified value for "
                    + element
                    + " at "
                    + str(self.level_of_theory)
                )
            return float(self.physical_prior_scale * level_values[element])
        if training_values is None:
            raise FerebusPriorError(
                self.strategy + " requires the exact task training values"
            )
        values = np.asarray(training_values, dtype=float).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise FerebusPriorError(
                self.strategy + " requires non-empty finite training values"
            )
        if self.strategy == "training_mean":
            return float(np.mean(values))
        if self.strategy == "training_median":
            return float(np.median(values))
        raise FerebusPriorError("unsupported FEREBUS prior strategy " + repr(self.strategy))

    def task_payload(
        self,
        property_name: Any,
        atom: Any,
        *,
        training_values: Optional[Sequence[float]] = None,
        training_dataset_sha256: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "contract_sha256": self.contract_sha256,
            "strategy": self.strategy,
            "mean_type": int(self.mean_type),
            "level_of_theory": self.level_of_theory,
            "physical_prior_scale": float(self.physical_prior_scale),
            "units": PRIOR_MEAN_UNITS,
            "expected_mean_ha": self.expected_mean_ha(
                property_name,
                atom,
                training_values=training_values,
            ),
            "training_dataset_sha256": training_dataset_sha256,
            "feature_scaling": True,
            "property_scaling": False,
        }


def resolve_ferebus_prior_contract(
    config: Any,
    *,
    atom_labels: Optional[Sequence[Any]] = None,
) -> FerebusPriorContract:
    """Resolve and validate the campaign's semantic prior-mean contract."""
    ferebus = getattr(config, "ferebus", config)
    gaussian = getattr(config, "gaussian", None)
    strategy = getattr(ferebus, "prior_mean_strategy", "physical_atomic_iqa")
    if not isinstance(strategy, str) or strategy not in PRIOR_MEAN_TYPES:
        raise FerebusPriorError(
            "ferebus.prior_mean_strategy must be one of "
            + repr(sorted(PRIOR_MEAN_TYPES))
        )
    level: Optional[str] = None
    if strategy == "physical_atomic_iqa":
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
    factor = getattr(ferebus, "physical_prior_scale", 1.0)
    if isinstance(factor, bool):
        raise FerebusPriorError(
            "ferebus.physical_prior_scale must be a finite positive number"
        )
    try:
        factor_float = float(factor)
    except (TypeError, ValueError) as exc:
        raise FerebusPriorError(
            "ferebus.physical_prior_scale must be a finite positive number"
        ) from exc
    if not math.isfinite(factor_float) or factor_float <= 0.0:
        raise FerebusPriorError(
            "ferebus.physical_prior_scale must be a finite positive number"
        )
    contract = FerebusPriorContract(
        strategy=strategy,
        mean_type=PRIOR_MEAN_TYPES[strategy],
        level_of_theory=level,
        physical_prior_scale=factor_float,
    )
    if strategy == "physical_atomic_iqa":
        for atom in atom_labels or ():
            contract.expected_mean_ha("iqa", atom)
    return contract


def contract_from_payload(payload: Any) -> FerebusPriorContract:
    """Parse and authenticate a serialised physical-prior contract."""
    if not isinstance(payload, Mapping):
        raise FerebusPriorError("FEREBUS prior contract must be an object")
    schema = payload.get("schema_version")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != PRIOR_CONTRACT_SCHEMA_VERSION:
        raise FerebusPriorError("unsupported FEREBUS prior contract schema")
    if str(payload.get("units") or "") != PRIOR_MEAN_UNITS:
        raise FerebusPriorError("FEREBUS prior contract units must be 'ha'")
    strategy = payload.get("strategy")
    if not isinstance(strategy, str) or strategy not in PRIOR_MEAN_TYPES:
        raise FerebusPriorError("FEREBUS prior contract strategy is invalid")
    level_value = payload.get("level_of_theory")
    level = None
    if strategy == "physical_atomic_iqa":
        if payload.get("atomic_energy_registry_version") != ATOMIC_ENERGY_REGISTRY_VERSION:
            raise FerebusPriorError("unsupported FEREBUS atomic-energy registry")
        level = canonicalise_level_of_theory(level_value)
    elif level_value is not None or payload.get("atomic_energy_registry_version") is not None:
        raise FerebusPriorError("non-physical FEREBUS prior must not bind an atomic registry")
    mean_type = payload.get("mean_type")
    if isinstance(mean_type, bool) or not isinstance(mean_type, int):
        raise FerebusPriorError("FEREBUS prior contract mean_type must be an integer")
    factor = payload.get("physical_prior_scale")
    if isinstance(factor, bool):
        raise FerebusPriorError("FEREBUS prior contract factor is invalid")
    try:
        factor_float = float(factor)
    except (TypeError, ValueError) as exc:
        raise FerebusPriorError("FEREBUS prior contract factor is invalid") from exc
    contract = FerebusPriorContract(
        strategy=strategy,
        mean_type=mean_type,
        level_of_theory=level,
        physical_prior_scale=factor_float,
    )
    if contract.mean_type != PRIOR_MEAN_TYPES[strategy]:
        raise FerebusPriorError("FEREBUS prior contract mean_type/strategy mismatch")
    if payload.get("feature_scaling") is not True or payload.get("property_scaling") is not False:
        raise FerebusPriorError("FEREBUS prior contract scaling is invalid")
    if (
        not math.isfinite(contract.physical_prior_scale)
        or contract.physical_prior_scale <= 0.0
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
    training_values: Optional[Sequence[float]] = None,
    expected_mean_ha: Optional[float] = None,
    relative_tolerance: float = 1.0e-10,
    absolute_tolerance: float = 1.0e-10,
) -> Dict[str, Any]:
    """Validate and report the semantic prior stored in a trained model."""
    observed = model_constant_mean_ha(model)
    if expected_mean_ha is None:
        expected = contract.expected_mean_ha(
            property_name,
            atom,
            training_values=training_values,
        )
    else:
        if isinstance(expected_mean_ha, bool):
            raise FerebusPriorError("FEREBUS expected prior mean is invalid")
        expected = float(expected_mean_ha)
        if not math.isfinite(expected):
            raise FerebusPriorError("FEREBUS expected prior mean is non-finite")
        if contract.strategy in {"zero", "physical_atomic_iqa"}:
            derived = contract.expected_mean_ha(property_name, atom)
            if not math.isclose(
                expected,
                derived,
                rel_tol=float(relative_tolerance),
                abs_tol=float(absolute_tolerance),
            ):
                raise FerebusPriorError(
                    "FEREBUS recorded prior mean disagrees with its semantic contract"
                )
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
        if not level:
            raise ValueError("empty level_of_theory")
        return {
            "mean_type": int(found["mean_type"]),
            "level_of_theory": level.lower(),
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
        "mean_type": int(contract.mean_type),
        "level_of_theory": (
            contract.level_of_theory
            if contract.level_of_theory is not None
            else "not_applicable"
        ),
        "iqa_deviation_factor": float(contract.physical_prior_scale),
        "scaling": True,
        "scale_feats": True,
        "scale_prop": False,
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
    "PRIOR_MEAN_TYPES",
    "FEREBUS_KERNEL_TOKENS",
    "PRIOR_MEAN_TYPE",
    "PRIOR_MEAN_UNITS",
    "SUPPORTED_ELEMENTS",
    "SUPPORTED_LEVELS",
    "canonicalise_level_of_theory",
    "backend_kernel_token",
    "contract_from_payload",
    "element_from_atom_label",
    "gaussian_level_of_theory",
    "model_constant_mean_ha",
    "parse_ferebus_config_contract",
    "resolve_ferebus_prior_contract",
    "validate_ferebus_config_contract",
    "validate_model_prior_mean",
]
