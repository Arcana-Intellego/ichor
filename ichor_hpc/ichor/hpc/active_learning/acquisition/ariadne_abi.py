"""Strict ICHOR-side contract probe for the ARIADNE Python wrapper."""
from __future__ import annotations

import inspect
import math
from typing import Any, Dict, Iterable


ARIADNE_ABI_CONTRACT_VERSION = 1
TRQN_STATUS_LENGTH = 89
DS_STATUS_LENGTH = 42
_REQUIRED_METHODS = (
    "init",
    "step_py",
    "get_state_flat_py",
    "get_status_py",
    "set_invalid_trial_reason_py",
)


class AriadneABIError(RuntimeError):
    """Raised when the imported wrapper cannot satisfy ICHOR's ABI contract."""


def _normalise_descriptor(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AriadneABIError("ARIADNE ABI descriptor contains a non-finite value")
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AriadneABIError("ARIADNE ABI descriptor contains non-UTF-8 bytes") from exc
    if isinstance(value, (list, tuple)):
        return [_normalise_descriptor(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise AriadneABIError("ARIADNE ABI descriptor keys must be strings")
        return {key: _normalise_descriptor(item) for key, item in value.items()}
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return _normalise_descriptor(to_list())
    raise AriadneABIError(
        "ARIADNE ABI descriptor contains unsupported value " + type(value).__name__
    )


def _require_callable(owner: Any, name: str, *, label: str) -> Any:
    value = getattr(owner, name, None)
    if not callable(value):
        raise AriadneABIError(label + " is missing callable " + name)
    return value


def _check_signature(
    value: Any,
    *,
    required_parameters: Iterable[str],
    label: str,
) -> Dict[str, Any]:
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return {"available": False, "parameters": []}
    names = list(signature.parameters)
    missing = [name for name in required_parameters if name not in names]
    if missing:
        raise AriadneABIError(
            label + " signature is missing parameters " + repr(missing)
        )
    return {"available": True, "parameters": names}


def _probe_optimiser(
    namespace: Any,
    constructor_name: str,
    *,
    label: str,
) -> Dict[str, Any]:
    constructor = _require_callable(namespace, constructor_name, label=label)
    try:
        optimiser = constructor()
    except Exception as exc:
        raise AriadneABIError(label + " constructor failed: " + str(exc)) from exc
    for method in _REQUIRED_METHODS:
        _require_callable(optimiser, method, label=label)
    return {
        "constructor": constructor_name,
        "required_methods": list(_REQUIRED_METHODS),
        "step_signature": _check_signature(
            optimiser.step_py,
            required_parameters=("stage", "f_old", "f_new", "g_xyz_new"),
            label=label + ".step_py",
        ),
    }


def probe_ariadne_module(module: Any) -> Dict[str, Any]:
    """Validate wrapper symbols and return a serialisable ABI receipt.

    ARIADNE does not yet publish ``get_abi_info_py``. Its absence is recorded
    explicitly while ICHOR enforces the concrete constructor, method and
    status-layout contract it consumes. Runtime status tuples are checked
    against the exact lengths recorded here after optimiser initialisation.
    """

    trqn_namespace = getattr(module, "Geometric_Trqn", None)
    ds_namespace = getattr(module, "Ds_Optimiser", None)
    if trqn_namespace is None or ds_namespace is None:
        raise AriadneABIError(
            "ARIADNE wrapper must expose Geometric_Trqn and Ds_Optimiser"
        )
    trqn = _probe_optimiser(
        trqn_namespace,
        "trust_region_qn",
        label="ARIADNE TRQN",
    )
    ds = _probe_optimiser(
        ds_namespace,
        "dissipative_symplectic",
        label="ARIADNE DS",
    )
    descriptor = getattr(module, "get_abi_info_py", None)
    descriptor_payload = None
    if descriptor is not None:
        if not callable(descriptor):
            raise AriadneABIError("ARIADNE get_abi_info_py is not callable")
        try:
            descriptor_payload = _normalise_descriptor(descriptor())
        except Exception as exc:
            raise AriadneABIError("ARIADNE ABI descriptor failed: " + str(exc)) from exc
    return {
        "contract_version": ARIADNE_ABI_CONTRACT_VERSION,
        "trqn": {**trqn, "status_length": TRQN_STATUS_LENGTH},
        "ds": {**ds, "status_length": DS_STATUS_LENGTH},
        "backend_descriptor_symbol": "get_abi_info_py",
        "backend_descriptor_available": bool(callable(descriptor)),
        "backend_descriptor": descriptor_payload,
    }


__all__ = [
    "ARIADNE_ABI_CONTRACT_VERSION",
    "AriadneABIError",
    "DS_STATUS_LENGTH",
    "TRQN_STATUS_LENGTH",
    "probe_ariadne_module",
]
