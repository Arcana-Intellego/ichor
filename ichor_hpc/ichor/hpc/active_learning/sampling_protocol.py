"""Versioned resolution of the public sampling-aggressiveness control."""
from __future__ import annotations

import copy
import hashlib
import math
from .strict_json import strict_json as json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import (
    AntiOverlapConfigBlock,
    CampaignConfig,
    ErrorCalibrationConfigBlock,
    GeometryNoveltyConfigBlock,
    PhaseBConfigBlock,
)
from .daemon.state import atomic_write_json
from .versioning.manifest import sha256_file
from .geometry_protocol import (
    MOVEMENT_BAND_HARD_MAX_FRACTION,
    MOVEMENT_BAND_HARD_MIN_FRACTION,
    MOVEMENT_BAND_TARGET_HIGH_FRACTION,
    MOVEMENT_BAND_TARGET_LOW_FRACTION,
    MOVEMENT_BAND_TARGET_PEAK_FRACTION,
    PHASE_B_MIN_SEPARATION_SCALE,
)


SAMPLING_PROTOCOL_SCHEMA_VERSION = 2
SAMPLING_PROTOCOL_AUDIT_SCHEMA_VERSION = 2
SAMPLING_AGGRESSIVENESS_POLICY_VERSION = 1
SAMPLING_PROTOCOL_RESOLVED_FILENAME = "SAMPLING_PROTOCOL_RESOLVED.json"
SAMPLING_PROTOCOL_AUDIT_FILENAME = "SAMPLING_PROTOCOL_AUDIT.json"
SAMPLING_PROTOCOL_CALIBRATION_FILENAME = "ERROR_CALIBRATION_MODEL.json"


@dataclass(frozen=True)
class SamplingAggressivenessPolicy:
    fallback_scale_angstrom: float
    phase_b_min_separation_scale: float
    movement_trust_multiplier: float
    trust_max_to_initial_ratio: float
    lambda_distance: float
    lambda_residual: float
    lambda_rmsd: float
    max_whitened_distance: float
    max_atom_displacement_ang: float
    min_pair_distance_ang: float
    max_scaled_atom_move: float
    max_scaled_rmsd: float
    max_scaled_fullspace_residual: float
    normalised_chemistry_penalty_cap: float
    bond_ratio_lower: float = 0.70
    bond_ratio_upper: float = 1.35
    angle_ratio_lower: float = 0.65
    angle_ratio_upper: float = 1.45


@dataclass
class ResolvedAdversarialSafety:
    """Mandatory safety gates plus the public recovery controls."""

    salvage_safe_iterate: bool
    backtrack_to_safe_landing: bool
    backtrack_points: int
    under_move_retry: bool
    under_move_retry_max: int
    min_whitened_distance: float = 0.0
    max_whitened_distance: float = 10.0
    enforce_min_whitened_distance: bool = False
    max_predicted_energy_delta_ha: Optional[float] = None
    max_energy_variance: Optional[float] = None
    max_chemistry_penalty: Optional[float] = None
    enforce_movement_band: bool = True
    reject_over_moved: bool = True
    reject_unsafe_landings: bool = True
    allow_seed_fallback: bool = False
    phase_b_filter_enabled: bool = True


_POLICIES: Dict[int, SamplingAggressivenessPolicy] = {
    1: SamplingAggressivenessPolicy(0.020, 0.70, 0.65, 5.000, 2.00, 1.00, 0.50, 4.0, 0.70, 0.65, 20.0, 2.5, 4.0, 10.0),
    2: SamplingAggressivenessPolicy(0.025, 0.65, 0.75, 5.000, 1.70, 0.85, 0.42, 5.0, 0.80, 0.65, 24.0, 3.0, 4.8, 12.0),
    3: SamplingAggressivenessPolicy(0.030, 0.60, 0.85, 5.000, 1.45, 0.70, 0.35, 6.0, 0.90, 0.65, 28.0, 3.4, 5.5, 14.0),
    4: SamplingAggressivenessPolicy(0.040, 0.55, 0.93, 4.375, 1.20, 0.60, 0.30, 8.0, 1.05, 0.62, 32.0, 3.8, 6.2, 17.0),
    5: SamplingAggressivenessPolicy(0.050, 0.50, 1.00, 4.000, 1.00, 0.50, 0.25, 10.0, 1.25, 0.60, 36.0, 4.2, 7.0, 20.0),
    6: SamplingAggressivenessPolicy(0.060, 0.48, 1.08, 3.750, 0.90, 0.45, 0.22, 11.0, 1.35, 0.60, 40.0, 4.8, 7.8, 22.0),
    7: SamplingAggressivenessPolicy(0.075, 0.46, 1.15, 3.571, 0.80, 0.40, 0.20, 12.0, 1.50, 0.60, 44.0, 5.4, 8.6, 24.0),
    8: SamplingAggressivenessPolicy(0.090, 0.44, 1.23, 3.438, 0.70, 0.35, 0.18, 13.0, 1.60, 0.60, 48.0, 6.0, 9.4, 26.0),
    9: SamplingAggressivenessPolicy(0.110, 0.42, 1.32, 3.333, 0.60, 0.30, 0.16, 14.0, 1.70, 0.60, 52.0, 6.6, 10.2, 28.0),
    10: SamplingAggressivenessPolicy(0.130, 0.40, 1.40, 3.250, 0.50, 0.25, 0.14, 15.0, 1.80, 0.60, 56.0, 7.2, 11.0, 30.0),
}


_HIDDEN_TOP_LEVEL_BLOCKS = (
    "anti_overlap",
    "geometry_novelty",
)


@dataclass(frozen=True)
class ResolvedSamplingProtocol:
    schema_version: int
    iteration: int
    sampling_aggressiveness: int
    policy: SamplingAggressivenessPolicy
    effective_config: CampaignConfig
    geometry_scale_payload: Dict[str, Any]
    resolved_geometry_scale_angstrom: Optional[float]
    acquisition_config: Any
    ariadne_run_config: Any
    adversarial_safety: Any
    quality_gates: Any
    phase_b: Dict[str, Any]
    anti_overlap: Dict[str, Any]
    error_calibration_snapshot: Dict[str, Any] = field(default_factory=dict)
    scale_model_payload: Dict[str, Any] = field(default_factory=dict)
    sources: Dict[str, Any] = field(default_factory=dict)
    hidden_overrides_detected: List[Dict[str, Any]] = field(default_factory=list)
    manifest_path: Optional[Path] = None
    scale_model_path: Optional[Path] = None
    audit_manifest_path: Optional[Path] = None


def sampling_protocol_resolved_path(iter_dir: Union[str, Path]) -> Path:
    from .layout import active_protocol_dir

    return active_protocol_dir(iter_dir) / SAMPLING_PROTOCOL_RESOLVED_FILENAME


def sampling_protocol_audit_path(iter_dir: Union[str, Path]) -> Path:
    from .layout import active_protocol_dir

    return active_protocol_dir(iter_dir) / SAMPLING_PROTOCOL_AUDIT_FILENAME


def _iteration_dir(campaign_dir: Union[str, Path], iteration: int) -> Path:
    from .layout import active_iteration_dir

    return active_iteration_dir(campaign_dir, int(iteration))


def _policy_for(config: CampaignConfig) -> SamplingAggressivenessPolicy:
    level = int(config.campaign.sampling_aggressiveness)
    try:
        policy = _POLICIES[level]
    except KeyError as exc:
        raise ValueError("campaign.sampling_aggressiveness must be in [1, 10]") from exc
    return policy


def dimensionless_policy_payload(policy: SamplingAggressivenessPolicy) -> Dict[str, Any]:
    """Return the safety caps resolved by one immutable policy entry."""
    return {
        "max_scaled_atom_move": float(policy.max_scaled_atom_move),
        "max_scaled_rmsd": float(policy.max_scaled_rmsd),
        "max_scaled_fullspace_residual": float(policy.max_scaled_fullspace_residual),
        "max_scaled_whitened_distance": float(policy.max_whitened_distance),
        "bond_ratio_lower": float(policy.bond_ratio_lower),
        "bond_ratio_upper": float(policy.bond_ratio_upper),
        "angle_ratio_lower": float(policy.angle_ratio_lower),
        "angle_ratio_upper": float(policy.angle_ratio_upper),
        "normalised_chemistry_penalty_cap": float(
            policy.normalised_chemistry_penalty_cap
        ),
    }


def _flatten(payload: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(payload, dict):
        out: Dict[str, Any] = {}
        for key, value in payload.items():
            path = str(key) if not prefix else prefix + "." + str(key)
            out.update(_flatten(value, path))
        return out
    return {prefix: payload}


def hidden_sampling_overrides(config: CampaignConfig) -> List[Dict[str, Any]]:
    current = _flatten(config.to_dict())
    defaults = _flatten(CampaignConfig().to_dict())
    hidden_paths = []
    for block in _HIDDEN_TOP_LEVEL_BLOCKS:
        hidden_paths.extend(path for path in current if path == block or path.startswith(block + "."))
    hidden_paths.extend(
        [
            "ariadne.delta0",
            "ariadne.delta_max",
            "quality_gates.ariadne_max_displacement_angstrom",
            "quality_gates.ariadne_min_pair_distance_angstrom",
        ]
    )
    out: List[Dict[str, Any]] = []
    for path in sorted(set(hidden_paths)):
        if current.get(path) != defaults.get(path):
            out.append(
                {
                    "path": path,
                    "configured_value": current.get(path),
                    "default_value": defaults.get(path),
                    "runtime_policy": "sampling_protocol_resolver_controls_effective_value",
                }
            )
    return out


def _effective_campaign_config(
    config: CampaignConfig,
    policy: SamplingAggressivenessPolicy,
) -> CampaignConfig:
    effective = copy.deepcopy(config)
    effective.anti_overlap = AntiOverlapConfigBlock(
        max_post_ariadne_whitened_distance=float(policy.max_whitened_distance),
    )
    effective.phase_b = copy.deepcopy(config.phase_b)
    effective.geometry_novelty = GeometryNoveltyConfigBlock(
        enabled=True,
        scale_source="local_motion",
        statistic="median",
        scale_floor_angstrom=1.0e-3,
        history_window_iterations=5,
        fallback_scale_angstrom=float(policy.fallback_scale_angstrom),
    )
    effective.acquisition = copy.deepcopy(config.acquisition)
    effective.ariadne = copy.deepcopy(config.ariadne)
    effective.adversarial_safety = ResolvedAdversarialSafety(
        salvage_safe_iterate=bool(config.adversarial_safety.salvage_safe_iterate),
        backtrack_to_safe_landing=bool(
            config.adversarial_safety.backtrack_to_safe_landing
        ),
        backtrack_points=int(config.adversarial_safety.backtrack_points),
        under_move_retry=bool(config.adversarial_safety.under_move_retry),
        under_move_retry_max=int(config.adversarial_safety.under_move_retry_max),
        min_whitened_distance=0.0,
        max_whitened_distance=float(policy.max_whitened_distance),
        enforce_min_whitened_distance=False,
        enforce_movement_band=True,
        reject_over_moved=True,
    )

    effective.quality_gates = copy.deepcopy(config.quality_gates)
    effective.quality_gates.ariadne_max_displacement_angstrom = float(
        policy.max_atom_displacement_ang
    )
    effective.quality_gates.ariadne_min_pair_distance_angstrom = float(
        policy.min_pair_distance_ang
    )
    return effective


def _apply_policy_to_acquisition_config(
    acquisition_config: Any,
    policy: SamplingAggressivenessPolicy,
) -> Any:
    movement = acquisition_config.movement_band
    scaled_movement = replace(
        movement,
        hard_min_fraction=float(MOVEMENT_BAND_HARD_MIN_FRACTION) * float(policy.movement_trust_multiplier),
        target_low_fraction=float(MOVEMENT_BAND_TARGET_LOW_FRACTION) * float(policy.movement_trust_multiplier),
        target_peak_fraction=float(MOVEMENT_BAND_TARGET_PEAK_FRACTION) * float(policy.movement_trust_multiplier),
        target_high_fraction=float(MOVEMENT_BAND_TARGET_HIGH_FRACTION) * float(policy.movement_trust_multiplier),
        hard_max_fraction=float(MOVEMENT_BAND_HARD_MAX_FRACTION) * float(policy.movement_trust_multiplier),
    )
    return replace(
        acquisition_config,
        movement_band=scaled_movement,
        weights=replace(
            acquisition_config.weights,
            lambda_distance=float(policy.lambda_distance),
        ),
        fullspace_confinement=replace(
            acquisition_config.fullspace_confinement,
            lambda_residual=float(policy.lambda_residual),
            lambda_rmsd=float(policy.lambda_rmsd),
        ),
    )


def _positive_model_value(
    scale_model_payload: Dict[str, Any],
    *path: str,
) -> Optional[float]:
    cur: Any = scale_model_payload
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    try:
        value = float(cur)
    except (TypeError, ValueError):
        return None
    if value > 0.0:
        return value
    return None


def _apply_scale_model_to_acquisition_config(
    acquisition_config: Any,
    scale_model_payload: Dict[str, Any],
) -> Any:
    geom = _positive_model_value(
        scale_model_payload, "geometry_motion_scale", "value_angstrom"
    )
    aligned = _positive_model_value(
        scale_model_payload, "aligned_rmsd_scale", "value_angstrom"
    )
    residual = _positive_model_value(
        scale_model_payload, "residual_fullspace_scale", "value_angstrom"
    )
    movement = acquisition_config.movement_band
    if geom is not None:
        movement = replace(movement, geometry_novelty_scale_angstrom=float(geom))
    fullspace = acquisition_config.fullspace_confinement
    if aligned is not None:
        fullspace = replace(fullspace, rmsd_scale_ang=float(aligned))
    if residual is not None:
        fullspace = replace(
            fullspace,
            residual_scale="fixed",
            fixed_residual_scale_ang=float(residual),
        )
    return replace(
        acquisition_config,
        movement_band=movement,
        fullspace_confinement=fullspace,
    )


def _attach_trust_radius_policy(
    scale_model_payload: Dict[str, Any],
    policy: SamplingAggressivenessPolicy,
) -> Dict[str, Any]:
    payload = dict(scale_model_payload)
    payload["trust_radius_policy"] = {
        "enabled": True,
        "normalisation": "weighted_mobility_sqrt_effective_atoms",
        "aggressiveness_multiplier": float(policy.movement_trust_multiplier),
        "max_to_initial_ratio": float(policy.trust_max_to_initial_ratio),
        "under_move_feedback_min_factor": 1.0,
        "under_move_feedback_max_factor": 2.0,
        "formula": (
            "trust0 = weighted_per_atom_mobility_angstrom * "
            "sqrt(n_effective_movement_atoms) * aggressiveness_multiplier"
        ),
    }
    return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _canonical_sha(payload: Dict[str, Any]) -> str:
    text = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sampling_policy_table_payload() -> Dict[str, Any]:
    return {
        "policy_version": SAMPLING_AGGRESSIVENESS_POLICY_VERSION,
        "levels": {
            str(level): asdict(policy)
            for level, policy in sorted(_POLICIES.items())
        },
    }


def sampling_policy_table_sha256() -> str:
    return _canonical_sha(sampling_policy_table_payload())


def _validate_policy_table() -> None:
    if set(_POLICIES) != set(range(1, 11)):
        raise RuntimeError("sampling-aggressiveness policy must define levels 1 to 10")
    previous: Optional[SamplingAggressivenessPolicy] = None
    for level, policy in sorted(_POLICIES.items()):
        values = asdict(policy)
        for name, raw in values.items():
            value = float(raw)
            if not math.isfinite(value) or value <= 0.0:
                raise RuntimeError(
                    "sampling policy level " + str(level) + " has invalid " + name
                )
        if policy.min_pair_distance_ang < 0.60:
            raise RuntimeError("sampling policy pair-distance floor is below 0.60 Angstrom")
        if not (
            policy.bond_ratio_lower < policy.bond_ratio_upper
            and policy.angle_ratio_lower < policy.angle_ratio_upper
        ):
            raise RuntimeError("sampling policy chemistry ratios are unordered")
        if previous is not None:
            if policy.movement_trust_multiplier < previous.movement_trust_multiplier:
                raise RuntimeError("sampling policy movement multiplier decreases")
            if policy.max_atom_displacement_ang < previous.max_atom_displacement_ang:
                raise RuntimeError("sampling policy displacement cap decreases")
            if policy.phase_b_min_separation_scale > previous.phase_b_min_separation_scale:
                raise RuntimeError("sampling policy Phase B separation increases")
            for name in ("lambda_distance", "lambda_residual", "lambda_rmsd"):
                if getattr(policy, name) > getattr(previous, name):
                    raise RuntimeError("sampling policy confinement penalty increases")
        previous = policy


_validate_policy_table()


def _resolved_manifest_payload(resolved: ResolvedSamplingProtocol) -> Dict[str, Any]:
    policy = asdict(resolved.policy)
    geometry_payload = dict(resolved.geometry_scale_payload or {})
    scale_model = dict(resolved.scale_model_payload or {})
    scale = resolved.resolved_geometry_scale_angstrom
    phase_b = dict(resolved.phase_b)
    movement_band = {}
    try:
        mb = resolved.acquisition_config.movement_band
        if scale is not None:
            movement_band = {
                "metric": str(mb.metric),
                "scale_source": "geometry_novelty",
                "scale_angstrom": float(scale),
                "min_angstrom": float(mb.hard_min_fraction) * float(scale),
                "low_angstrom": float(mb.target_low_fraction) * float(scale),
                "peak_angstrom": float(mb.target_peak_fraction) * float(scale),
                "high_angstrom": float(mb.target_high_fraction) * float(scale),
                "max_angstrom": float(mb.hard_max_fraction) * float(scale),
                "fractions": {
                    "hard_min": float(mb.hard_min_fraction),
                    "target_low": float(mb.target_low_fraction),
                    "target_peak": float(mb.target_peak_fraction),
                    "target_high": float(mb.target_high_fraction),
                    "hard_max": float(mb.hard_max_fraction),
                },
            }
    except Exception:
        movement_band = {"error": "movement_band_unavailable"}
    payload = {
        "schema_version": SAMPLING_PROTOCOL_SCHEMA_VERSION,
        "iteration": int(resolved.iteration),
        "sampling_aggressiveness": int(resolved.sampling_aggressiveness),
        "generated_at_iso": datetime.now(timezone.utc).isoformat(),
        "sources": dict(resolved.sources),
        "scale_sources": {
            "geometry": "sampling_scale_model.geometry_motion_scale",
            "per_atom_mobility": "sampling_scale_model.per_atom_mobility_scales",
            "pair_reference": "sampling_scale_model.pair_distance_reference",
            "bond_angle_reference": "sampling_scale_model.bond_angle_reference",
            "component_scale": "existing_reference_scales_only",
        },
        "sampling_scale_model": scale_model,
        "sampling_scale_model_manifest": (
            None if resolved.scale_model_path is None
            else str(resolved.scale_model_path)
        ),
        "sampling_protocol_audit_manifest": (
            None if resolved.audit_manifest_path is None
            else str(resolved.audit_manifest_path)
        ),
        "sampling_policy_version": SAMPLING_AGGRESSIVENESS_POLICY_VERSION,
        "sampling_policy_table_sha256": sampling_policy_table_sha256(),
        "sampling_policy": policy,
        "dimensionless_policy": dimensionless_policy_payload(resolved.policy),
        "geometry_novelty_scale": geometry_payload,
        "resolved_geometry_scale_angstrom": scale,
        "resolved_phase_b": phase_b,
        "resolved_anti_overlap": dict(resolved.anti_overlap),
        "resolved_error_calibration": dict(resolved.error_calibration_snapshot),
        "resolved_movement_band": movement_band,
        "resolved_adversarial_safety": {
            "reject_unsafe_landings": bool(resolved.adversarial_safety.reject_unsafe_landings),
            "salvage_safe_iterate": bool(resolved.adversarial_safety.salvage_safe_iterate),
            "backtrack_to_safe_landing": bool(resolved.adversarial_safety.backtrack_to_safe_landing),
            "backtrack_points": int(resolved.adversarial_safety.backtrack_points),
            "allow_seed_fallback": bool(resolved.adversarial_safety.allow_seed_fallback),
            "max_whitened_distance": float(resolved.adversarial_safety.max_whitened_distance),
            "phase_b_filter_enabled": bool(resolved.adversarial_safety.phase_b_filter_enabled),
            "enforce_movement_band": bool(resolved.adversarial_safety.enforce_movement_band),
            "under_move_retry": bool(resolved.adversarial_safety.under_move_retry),
            "reject_over_moved": bool(resolved.adversarial_safety.reject_over_moved),
        },
        "resolved_quality_gates": {
            "ariadne_max_displacement_ang": (
                resolved.quality_gates.ariadne_max_displacement_angstrom
            ),
            "ariadne_min_pair_distance_ang": (
                resolved.quality_gates.ariadne_min_pair_distance_angstrom
            ),
            "min_pair_distance_policy": {
                "mode": "scale_model_minimum_safe_pair_ratio",
                "hard_floor_angstrom": (
                    resolved.quality_gates.ariadne_min_pair_distance_angstrom
                ),
                "reference_min_pair_distance_angstrom": (
                    scale_model.get("pair_distance_reference", {})
                    .get("reference_min_pair_distance_angstrom")
                ),
                "ratio_floor": (
                    scale_model.get("pair_distance_reference", {})
                    .get("ratio_floor")
                ),
            },
            "max_displacement_policy": {
                "mode": "scale_model_scaled_atom_move_with_absolute_cap",
                "cap_angstrom": (
                    resolved.quality_gates.ariadne_max_displacement_angstrom
                ),
                "scale_angstrom": (
                    scale_model.get("geometry_motion_scale", {})
                    .get("value_angstrom")
                ),
            },
        },
        "resolved_acquisition_weights": {
            "lambda_distance": float(resolved.acquisition_config.weights.lambda_distance),
        },
        "resolved_fullspace_confinement": {
            "enabled": True,
            "lambda_residual": float(resolved.acquisition_config.fullspace_confinement.lambda_residual),
            "lambda_rmsd": float(resolved.acquisition_config.fullspace_confinement.lambda_rmsd),
            "residual_scale": str(resolved.acquisition_config.fullspace_confinement.residual_scale),
            "fixed_residual_scale_ang": resolved.acquisition_config.fullspace_confinement.fixed_residual_scale_ang,
            "rmsd_scale_ang": float(resolved.acquisition_config.fullspace_confinement.rmsd_scale_ang),
        },
        "resolved_ariadne": {
            "optimiser": str(resolved.ariadne_run_config.optimiser),
            "max_iter": int(resolved.ariadne_run_config.max_iter),
            "delta0": float(resolved.ariadne_run_config.delta0),
            "delta_max": float(resolved.ariadne_run_config.delta_max),
            "trqn_target_initial_grad_rms": float(resolved.ariadne_run_config.trqn_target_initial_grad_rms),
            "trqn_retry_target_initial_grad_rms": float(resolved.ariadne_run_config.trqn_retry_target_initial_grad_rms),
            "under_move_retry_target_initial_grad_rms": 6.0e-4,
        },
        "hard_safety_rails": {
            "reject_unsafe_landings": True,
            "salvage_safe_iterate": True,
            "backtrack_to_safe_landing": True,
            "allow_seed_fallback": False,
            "phase_b_filter_enabled": True,
            "connectivity_barrier": True,
            "dimensionless_scale_gates": True,
        },
        "hidden_overrides_detected": list(resolved.hidden_overrides_detected),
        "source_scale_hashes": {
            "geometry_scale_sha256": _canonical_sha(geometry_payload),
            "sampling_scale_model_sha256": _canonical_sha(scale_model),
        },
    }
    payload["input_fingerprint"] = {
        "sha256": _canonical_sha(
            {
                "sampling_aggressiveness": resolved.sampling_aggressiveness,
                "sampling_policy_version": SAMPLING_AGGRESSIVENESS_POLICY_VERSION,
                "sampling_policy_table_sha256": sampling_policy_table_sha256(),
                "sampling_policy": policy,
                "geometry_input_fingerprint": geometry_payload.get("input_fingerprint"),
                "hidden_overrides": resolved.hidden_overrides_detected,
                "error_calibration": resolved.error_calibration_snapshot,
            }
        )
    }
    return _json_ready(payload)


def _resolve_error_calibration_snapshot(
    campaign_dir: Union[str, Path],
    config: CampaignConfig,
    iteration: int,
    *,
    write_manifest: bool,
) -> Dict[str, Any]:
    from .daemon.error_calibration import load_calibration_model_for_acquisition
    from .daemon.error_calibration_contract import (
        estimator_settings,
        estimator_settings_sha256,
        validate_model,
    )
    from .layout import active_protocol_dir

    settings = asdict(config.error_calibration)
    model, reason = load_calibration_model_for_acquisition(
        campaign_dir,
        config,
        current_iteration=int(iteration),
    )
    if model is not None:
        model = validate_model(model)
    snapshot: Dict[str, Any] = {
        "settings": settings,
        "estimator_settings": estimator_settings(config),
        "estimator_settings_sha256": estimator_settings_sha256(config),
        "model_reason": str(reason),
        "model_path": None,
        "model_sha256": None,
        "model": None if model is None else dict(model),
    }
    if model is not None:
        snapshot.update(
            {
                "calibration_context_sha256": model[
                    "calibration_context_sha256"
                ],
                "environment_generation_digest_sha256": model[
                    "environment_generation_digest_sha256"
                ],
                "source_records_sha256": model["source_records_sha256"],
            }
        )
    if write_manifest and model is not None:
        path = active_protocol_dir(_iteration_dir(campaign_dir, int(iteration))) / (
            SAMPLING_PROTOCOL_CALIBRATION_FILENAME
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, dict(model))
        snapshot["model_path"] = path.name
        snapshot["model_sha256"] = sha256_file(path)
    return snapshot


def _load_error_calibration_snapshot(
    iter_dir: Path,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    snapshot = payload.get("resolved_error_calibration")
    if not isinstance(snapshot, dict):
        raise ValueError("resolved calibration snapshot is missing")
    out = dict(snapshot)
    model_path = out.get("model_path")
    if model_path is None:
        out["model"] = None
        return out
    if Path(str(model_path)).name != str(model_path):
        raise ValueError("resolved calibration snapshot path is invalid")
    path = sampling_protocol_resolved_path(iter_dir).parent / str(model_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("resolved calibration model snapshot is missing")
    if sha256_file(path) != str(out.get("model_sha256") or ""):
        raise ValueError("resolved calibration model snapshot SHA mismatch")
    model = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(model, dict):
        raise ValueError("resolved calibration model snapshot must be an object")
    from .daemon.error_calibration_contract import validate_model

    try:
        validated = validate_model(model)
    except Exception as exc:
        raise ValueError("resolved calibration model snapshot is invalid") from exc
    if validated.get("calibration_context_sha256") != out.get(
        "calibration_context_sha256"
    ):
        raise ValueError("resolved calibration context binding mismatch")
    if validated.get("source_records_sha256") != out.get(
        "source_records_sha256"
    ):
        raise ValueError("resolved calibration source-record binding mismatch")
    out["model"] = validated
    return out


def write_sampling_protocol_resolved(iter_dir: Union[str, Path], resolved: ResolvedSamplingProtocol) -> Path:
    path = sampling_protocol_resolved_path(iter_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, _resolved_manifest_payload(resolved))
    return path


def read_sampling_protocol_resolved(
    iter_dir: Union[str, Path],
    *,
    expected_iteration: Optional[int] = None,
) -> Dict[str, Any]:
    path = sampling_protocol_resolved_path(iter_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SAMPLING_PROTOCOL_RESOLVED.json must contain an object")
    if int(payload.get("schema_version", -1)) != SAMPLING_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported sampling protocol resolved schema")
    if expected_iteration is not None and int(payload.get("iteration", -1)) != int(expected_iteration):
        raise ValueError("sampling protocol resolved iteration mismatch")
    return payload


def _sampling_protocol_audit_payload(resolved: ResolvedSamplingProtocol) -> Dict[str, Any]:
    scale_model = dict(resolved.scale_model_payload or {})
    dimensionless = dimensionless_policy_payload(resolved.policy)
    diagnostics = dict(scale_model.get("diagnostics") or {})
    history = dict(scale_model.get("history") or {})
    pair_reference = dict(scale_model.get("pair_distance_reference") or {})
    bond_angle = dict(scale_model.get("bond_angle_reference") or {})
    return _json_ready(
        {
            "schema_version": SAMPLING_PROTOCOL_AUDIT_SCHEMA_VERSION,
            "iteration": int(resolved.iteration),
            "generated_at_iso": datetime.now(timezone.utc).isoformat(),
            "sampling_aggressiveness": int(resolved.sampling_aggressiveness),
            "public_user_surface": {
                "editable_campaign_field": "campaign.sampling_aggressiveness",
                "hidden_low_level_blocks": list(_HIDDEN_TOP_LEVEL_BLOCKS),
            },
            "sampling_policy_version": SAMPLING_AGGRESSIVENESS_POLICY_VERSION,
            "sampling_policy_table_sha256": sampling_policy_table_sha256(),
            "sampling_policy": asdict(resolved.policy),
            "dimensionless_policy": dimensionless,
            "enforced_landing_gates": {
                "scaled_whitened_distance_max": float(
                    dimensionless["max_scaled_whitened_distance"]
                ),
                "scaled_max_atom_move_max": float(
                    dimensionless["max_scaled_atom_move"]
                ),
                "scaled_aligned_rmsd_max": float(dimensionless["max_scaled_rmsd"]),
                "scaled_fullspace_residual_max": float(
                    dimensionless["max_scaled_fullspace_residual"]
                ),
                "absolute_min_pair_distance_angstrom": float(
                    resolved.policy.min_pair_distance_ang
                ),
                "normalised_chemistry_penalty_cap": float(
                    dimensionless["normalised_chemistry_penalty_cap"]
                ),
            },
            "record_only_reference_ranges": {
                "bond_ratio_lower": float(dimensionless["bond_ratio_lower"]),
                "bond_ratio_upper": float(dimensionless["bond_ratio_upper"]),
                "angle_ratio_lower": float(dimensionless["angle_ratio_lower"]),
                "angle_ratio_upper": float(dimensionless["angle_ratio_upper"]),
            },
            "scale_model_summary": {
                "schema_version": scale_model.get("schema_version"),
                "model_version": scale_model.get("model_version"),
                "geometry_motion_scale": scale_model.get("geometry_motion_scale"),
                "aligned_rmsd_scale": scale_model.get("aligned_rmsd_scale"),
                "residual_fullspace_scale": scale_model.get("residual_fullspace_scale"),
                "per_atom_mobility": scale_model.get("per_atom_mobility_scales"),
                "pair_distance_reference": pair_reference,
                "bond_angle_reference": bond_angle,
                "history": {
                    "window_iterations": history.get("window_iterations"),
                    "n_records": history.get("n_records"),
                    "n_result_json": history.get("n_result_json"),
                    "filter": history.get("filter"),
                },
            },
            "fallback_warnings": list(diagnostics.get("fallback_warnings") or []),
            "hidden_overrides_detected": list(resolved.hidden_overrides_detected),
            "manifest_paths": {
                "resolved": None if resolved.manifest_path is None else str(resolved.manifest_path),
                "scale_model": None if resolved.scale_model_path is None else str(resolved.scale_model_path),
            },
            "scheduler_impact": {
                "new_scheduler_jobs": 0,
                "uses_only_existing_campaign_data": True,
            },
        }
    )


def write_sampling_protocol_audit(
    iter_dir: Union[str, Path],
    resolved: ResolvedSamplingProtocol,
) -> Path:
    path = sampling_protocol_audit_path(iter_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, _sampling_protocol_audit_payload(resolved))
    return path


def read_sampling_protocol_audit(
    iter_dir: Union[str, Path],
    *,
    expected_iteration: Optional[int] = None,
) -> Dict[str, Any]:
    path = sampling_protocol_audit_path(iter_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SAMPLING_PROTOCOL_AUDIT.json must contain an object")
    if int(payload.get("schema_version", -1)) != SAMPLING_PROTOCOL_AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported sampling protocol audit schema")
    if expected_iteration is not None and int(payload.get("iteration", -1)) != int(expected_iteration):
        raise ValueError("sampling protocol audit iteration mismatch")
    return payload


def resolve_sampling_protocol(
    campaign_dir: Union[str, Path],
    config: CampaignConfig,
    iteration: int,
    *,
    write_manifest: bool = True,
) -> ResolvedSamplingProtocol:
    from .geometry_novelty import (
        apply_geometry_novelty_to_acquisition_config,
        ensure_geometry_novelty_scale,
        resolve_geometry_novelty_consumers,
    )
    from .sampling_scale_model import (
        build_sampling_scale_model,
        sampling_scale_model_path,
    )

    level = int(config.campaign.sampling_aggressiveness)
    policy = _policy_for(config)
    effective = _effective_campaign_config(config, policy)
    calibration_snapshot = _resolve_error_calibration_snapshot(
        campaign_dir,
        effective,
        int(iteration),
        write_manifest=write_manifest,
    )
    geometry_payload = ensure_geometry_novelty_scale(
        campaign_dir,
        effective,
        iteration=int(iteration),
    )
    scale_model_payload = build_sampling_scale_model(
        campaign_dir,
        effective,
        int(iteration),
        geometry_scale_payload=dict(geometry_payload),
        write_manifest=write_manifest,
    )
    scale_model_payload = dict(scale_model_payload)
    scale_model_payload = _attach_trust_radius_policy(
        scale_model_payload,
        policy,
    )
    scale_model_payload["dimensionless_policy"] = dimensionless_policy_payload(policy)
    scale_model_payload["sampling_policy"] = {
        "version": SAMPLING_AGGRESSIVENESS_POLICY_VERSION,
        "table_sha256": sampling_policy_table_sha256(),
        "level": level,
        "entry": asdict(policy),
    }
    scale_model_payload["bond_angle_reference"] = {
        "source": "sampling_policy_fixed_chemistry_invariants",
        "fallback_used": False,
        "bond_ratio_lower": float(policy.bond_ratio_lower),
        "bond_ratio_upper": float(policy.bond_ratio_upper),
        "angle_ratio_lower": float(policy.angle_ratio_lower),
        "angle_ratio_upper": float(policy.angle_ratio_upper),
    }
    diagnostics = dict(scale_model_payload.get("diagnostics") or {})
    diagnostics["size_independence_wave"] = 3
    scale_model_payload["diagnostics"] = diagnostics
    acquisition_config = apply_geometry_novelty_to_acquisition_config(
        effective.to_acquisition_config(),
        effective,
        geometry_payload,
    )
    acquisition_config = _apply_policy_to_acquisition_config(acquisition_config, policy)
    acquisition_config = _apply_scale_model_to_acquisition_config(
        acquisition_config,
        scale_model_payload,
    )
    ariadne_run_config = effective.to_ariadne_run_config()
    resolved_consumers = resolve_geometry_novelty_consumers(effective, geometry_payload)
    phase_b = dict(resolved_consumers.get("phase_b") or {})
    scale = geometry_payload.get("scale_angstrom") if isinstance(geometry_payload, dict) else None
    try:
        scale_value = float(scale)
    except (TypeError, ValueError):
        scale_value = None
    if scale_value is not None and scale_value > 0.0:
        phase_b_scale = scale_value
        try:
            phase_b_scale = float(
                scale_model_payload.get("aligned_rmsd_scale", {}).get("value_angstrom")
            )
        except (TypeError, ValueError):
            phase_b_scale = scale_value
        if not (phase_b_scale > 0.0):
            phase_b_scale = scale_value
        phase_b["scaled_threshold"] = float(policy.phase_b_min_separation_scale)
        phase_b["min_separation_scaled"] = float(policy.phase_b_min_separation_scale)
        phase_b["effective_min_separation_angstrom"] = (
            float(policy.phase_b_min_separation_scale) * float(phase_b_scale)
        )
        phase_b["scale_model_source"] = "aligned_rmsd_scale"
        phase_b["scale_model_value_angstrom"] = float(phase_b_scale)
    phase_b["descriptor"] = str(effective.phase_b.descriptor)
    phase_b["beta"] = float(effective.phase_b.beta)
    scale_path = (
        sampling_scale_model_path(_iteration_dir(campaign_dir, int(iteration)))
        if write_manifest else None
    )
    if write_manifest and scale_path is not None:
        scale_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(scale_path, _json_ready(scale_model_payload))

    resolved = ResolvedSamplingProtocol(
        schema_version=SAMPLING_PROTOCOL_SCHEMA_VERSION,
        iteration=int(iteration),
        sampling_aggressiveness=level,
        policy=policy,
        effective_config=effective,
        geometry_scale_payload=dict(geometry_payload),
        resolved_geometry_scale_angstrom=scale_value,
        acquisition_config=acquisition_config,
        ariadne_run_config=ariadne_run_config,
        adversarial_safety=effective.adversarial_safety,
        quality_gates=effective.quality_gates,
        phase_b=phase_b,
        anti_overlap=asdict(effective.anti_overlap),
        error_calibration_snapshot=calibration_snapshot,
        scale_model_payload=dict(scale_model_payload),
        sources={
            "level_5_policy": "matches_current_defaults",
            "geometry_scale": "sampling_scale_model",
            "pair_distance": "sampling_scale_model_minimum_safe_pair_ratio",
        },
        hidden_overrides_detected=hidden_sampling_overrides(config),
        scale_model_path=scale_path,
    )
    if write_manifest:
        iter_dir = _iteration_dir(campaign_dir, int(iteration))
        path = write_sampling_protocol_resolved(iter_dir, resolved)
        resolved = replace(resolved, manifest_path=path)
        audit_path = write_sampling_protocol_audit(iter_dir, resolved)
        resolved = replace(resolved, audit_manifest_path=audit_path)
        write_sampling_protocol_resolved(iter_dir, resolved)
    return resolved


def load_sampling_protocol(
    campaign_dir: Union[str, Path],
    config: CampaignConfig,
    iteration: int,
) -> ResolvedSamplingProtocol:
    """Load one immutable protocol snapshot without rewriting its sidecars."""
    from .geometry_novelty import (
        apply_geometry_novelty_to_acquisition_config,
        resolve_geometry_novelty_consumers,
    )
    from .sampling_scale_model import (
        read_sampling_scale_model,
        sampling_scale_model_path,
    )

    campaign = Path(campaign_dir)
    iter_dir = _iteration_dir(campaign, int(iteration))
    protocol_path = sampling_protocol_resolved_path(iter_dir)
    audit_path = sampling_protocol_audit_path(iter_dir)
    scale_path = sampling_scale_model_path(iter_dir)
    for label, path in (
        ("resolved sampling protocol", protocol_path),
        ("sampling protocol audit", audit_path),
        ("sampling scale model", scale_path),
    ):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(label + " is not a regular file: " + str(path))

    protocol_payload = read_sampling_protocol_resolved(
        iter_dir,
        expected_iteration=int(iteration),
    )
    calibration_snapshot = _load_error_calibration_snapshot(iter_dir, protocol_payload)
    read_sampling_protocol_audit(
        iter_dir,
        expected_iteration=int(iteration),
    )
    scale_model_payload = read_sampling_scale_model(
        iter_dir,
        expected_iteration=int(iteration),
    )
    if protocol_payload.get("sampling_scale_model") != scale_model_payload:
        raise ValueError(
            "resolved sampling protocol does not match its scale-model sidecar"
        )
    geometry_payload = protocol_payload.get("geometry_novelty_scale")
    if not isinstance(geometry_payload, dict):
        raise ValueError(
            "resolved sampling protocol geometry novelty scale is missing"
        )

    level = int(config.campaign.sampling_aggressiveness)
    if int(protocol_payload.get("sampling_aggressiveness", -1)) != level:
        raise ValueError(
            "resolved sampling protocol aggressiveness does not match campaign config"
        )
    policy = _policy_for(config)
    effective = _effective_campaign_config(config, policy)
    frozen_settings = calibration_snapshot.get("settings")
    if isinstance(frozen_settings, dict) and frozen_settings:
        effective.error_calibration = ErrorCalibrationConfigBlock(
            **dict(frozen_settings)
        )
    acquisition_config = apply_geometry_novelty_to_acquisition_config(
        effective.to_acquisition_config(),
        effective,
        geometry_payload,
    )
    acquisition_config = _apply_policy_to_acquisition_config(
        acquisition_config,
        policy,
    )
    acquisition_config = _apply_scale_model_to_acquisition_config(
        acquisition_config,
        scale_model_payload,
    )
    ariadne_run_config = effective.to_ariadne_run_config()
    resolved_consumers = resolve_geometry_novelty_consumers(
        effective,
        geometry_payload,
    )
    phase_b = dict(resolved_consumers.get("phase_b") or {})
    scale_raw = geometry_payload.get("scale_angstrom")
    try:
        scale_value = float(scale_raw)
    except (TypeError, ValueError):
        scale_value = None
    if scale_value is not None and scale_value > 0.0:
        phase_b_scale = scale_value
        try:
            phase_b_scale = float(
                scale_model_payload.get("aligned_rmsd_scale", {}).get(
                    "value_angstrom"
                )
            )
        except (TypeError, ValueError):
            phase_b_scale = scale_value
        if not (phase_b_scale > 0.0):
            phase_b_scale = scale_value
        phase_b["scaled_threshold"] = float(
            policy.phase_b_min_separation_scale
        )
        phase_b["min_separation_scaled"] = float(
            policy.phase_b_min_separation_scale
        )
        phase_b["effective_min_separation_angstrom"] = float(
            policy.phase_b_min_separation_scale
        ) * float(phase_b_scale)
        phase_b["scale_model_source"] = "aligned_rmsd_scale"
        phase_b["scale_model_value_angstrom"] = float(phase_b_scale)
    phase_b["descriptor"] = str(effective.phase_b.descriptor)
    phase_b["beta"] = float(effective.phase_b.beta)

    resolved = ResolvedSamplingProtocol(
        schema_version=SAMPLING_PROTOCOL_SCHEMA_VERSION,
        iteration=int(iteration),
        sampling_aggressiveness=level,
        policy=policy,
        effective_config=effective,
        geometry_scale_payload=dict(geometry_payload),
        resolved_geometry_scale_angstrom=scale_value,
        acquisition_config=acquisition_config,
        ariadne_run_config=ariadne_run_config,
        adversarial_safety=effective.adversarial_safety,
        quality_gates=effective.quality_gates,
        phase_b=phase_b,
        anti_overlap=asdict(effective.anti_overlap),
        error_calibration_snapshot=calibration_snapshot,
        scale_model_payload=dict(scale_model_payload),
        sources={
            "level_5_policy": "matches_current_defaults",
            "geometry_scale": "sampling_scale_model",
            "pair_distance": "sampling_scale_model_minimum_safe_pair_ratio",
        },
        hidden_overrides_detected=hidden_sampling_overrides(config),
        manifest_path=protocol_path,
        scale_model_path=scale_path,
        audit_manifest_path=audit_path,
    )
    expected_payload = _resolved_manifest_payload(resolved)
    invariant_fields = (
        "sampling_aggressiveness",
        "sampling_policy_version",
        "sampling_policy_table_sha256",
        "sampling_policy",
        "dimensionless_policy",
        "source_scale_hashes",
        "geometry_novelty_scale",
        "sampling_scale_model",
        "resolved_acquisition_weights",
        "resolved_fullspace_confinement",
        "resolved_movement_band",
        "resolved_adversarial_safety",
        "resolved_quality_gates",
        "resolved_phase_b",
        "resolved_anti_overlap",
        "resolved_error_calibration",
        "resolved_ariadne",
        "hard_safety_rails",
        "hidden_overrides_detected",
    )
    for field_name in invariant_fields:
        if protocol_payload.get(field_name) != expected_payload.get(field_name):
            raise ValueError(
                "resolved sampling protocol/config mismatch: " + field_name
            )
    stored_fingerprint = dict(protocol_payload.get("input_fingerprint") or {}).get(
        "sha256"
    )
    expected_fingerprint = dict(expected_payload.get("input_fingerprint") or {}).get(
        "sha256"
    )
    if stored_fingerprint != expected_fingerprint:
        raise ValueError("resolved sampling protocol input fingerprint mismatch")
    return resolved


def resolve_or_load_sampling_protocol(
    campaign_dir: Union[str, Path],
    config: CampaignConfig,
    iteration: int,
) -> ResolvedSamplingProtocol:
    """Resolve a protocol once, then reuse the immutable snapshot."""
    from .sampling_scale_model import sampling_scale_model_path

    iter_dir = _iteration_dir(campaign_dir, int(iteration))
    paths = (
        sampling_protocol_resolved_path(iter_dir),
        sampling_protocol_audit_path(iter_dir),
        sampling_scale_model_path(iter_dir),
    )
    present = [path.is_file() or path.is_symlink() for path in paths]
    if any(present):
        if not all(present):
            raise ValueError(
                "sampling protocol snapshot is incomplete; refusing to regenerate it"
            )
        return load_sampling_protocol(campaign_dir, config, int(iteration))
    return resolve_sampling_protocol(campaign_dir, config, int(iteration))


def preview_sampling_protocol(
    config: CampaignConfig,
    *,
    campaign_dir: Union[str, Path, None] = None,
    iteration: int = 1,
    geometry_scale_payload: Optional[Dict[str, Any]] = None,
) -> ResolvedSamplingProtocol:
    """Resolve the sampling protocol without writing sidecars.

    CLI summaries use this path so merely viewing the menu does not mutate a
    campaign directory.  When no sidecar payload is supplied, the preview uses
    the policy's geometry fallback scale and labels it as a preview fallback.
    """
    from .geometry_novelty import (
        apply_geometry_novelty_to_acquisition_config,
        resolve_geometry_novelty_consumers,
    )
    from .sampling_scale_model import build_sampling_scale_model

    level = int(config.campaign.sampling_aggressiveness)
    policy = _policy_for(config)
    effective = _effective_campaign_config(config, policy)
    if geometry_scale_payload is None:
        geometry_scale_payload = {
            "schema_version": 1,
            "iteration": int(iteration),
            "scale_angstrom": float(policy.fallback_scale_angstrom),
            "fallback_used": True,
            "scale_resolution_mode": "preview_policy_fallback",
            "n_values": 0,
        }
    preview_campaign_dir = Path(campaign_dir) if campaign_dir is not None else Path(".")
    scale_model_payload = build_sampling_scale_model(
        preview_campaign_dir,
        effective,
        int(iteration),
        geometry_scale_payload=dict(geometry_scale_payload),
        write_manifest=False,
    )
    scale_model_payload = dict(scale_model_payload)
    scale_model_payload = _attach_trust_radius_policy(
        scale_model_payload,
        policy,
    )
    scale_model_payload["dimensionless_policy"] = dimensionless_policy_payload(policy)
    scale_model_payload["sampling_policy"] = {
        "version": SAMPLING_AGGRESSIVENESS_POLICY_VERSION,
        "table_sha256": sampling_policy_table_sha256(),
        "level": level,
        "entry": asdict(policy),
    }
    scale_model_payload["bond_angle_reference"] = {
        "source": "sampling_policy_fixed_chemistry_invariants",
        "fallback_used": False,
        "bond_ratio_lower": float(policy.bond_ratio_lower),
        "bond_ratio_upper": float(policy.bond_ratio_upper),
        "angle_ratio_lower": float(policy.angle_ratio_lower),
        "angle_ratio_upper": float(policy.angle_ratio_upper),
    }
    diagnostics = dict(scale_model_payload.get("diagnostics") or {})
    diagnostics["size_independence_wave"] = 3
    scale_model_payload["diagnostics"] = diagnostics
    acquisition_config = apply_geometry_novelty_to_acquisition_config(
        effective.to_acquisition_config(),
        effective,
        geometry_scale_payload,
    )
    acquisition_config = _apply_policy_to_acquisition_config(acquisition_config, policy)
    acquisition_config = _apply_scale_model_to_acquisition_config(
        acquisition_config,
        scale_model_payload,
    )
    ariadne_run_config = effective.to_ariadne_run_config()
    resolved_consumers = resolve_geometry_novelty_consumers(effective, geometry_scale_payload)
    phase_b = dict(resolved_consumers.get("phase_b") or {})
    scale = geometry_scale_payload.get("scale_angstrom")
    try:
        scale_value = float(scale)
    except (TypeError, ValueError):
        scale_value = None
    if scale_value is not None and scale_value > 0.0:
        phase_b_scale = scale_value
        try:
            phase_b_scale = float(
                scale_model_payload.get("aligned_rmsd_scale", {}).get("value_angstrom")
            )
        except (TypeError, ValueError):
            phase_b_scale = scale_value
        if not (phase_b_scale > 0.0):
            phase_b_scale = scale_value
        phase_b["scaled_threshold"] = float(policy.phase_b_min_separation_scale)
        phase_b["min_separation_scaled"] = float(policy.phase_b_min_separation_scale)
        phase_b["effective_min_separation_angstrom"] = (
            float(policy.phase_b_min_separation_scale) * float(phase_b_scale)
        )
        phase_b["scale_model_source"] = "aligned_rmsd_scale"
        phase_b["scale_model_value_angstrom"] = float(phase_b_scale)
    phase_b["descriptor"] = str(effective.phase_b.descriptor)
    phase_b["beta"] = float(effective.phase_b.beta)
    return ResolvedSamplingProtocol(
        schema_version=SAMPLING_PROTOCOL_SCHEMA_VERSION,
        iteration=int(iteration),
        sampling_aggressiveness=level,
        policy=policy,
        effective_config=effective,
        geometry_scale_payload=dict(geometry_scale_payload),
        resolved_geometry_scale_angstrom=scale_value,
        acquisition_config=acquisition_config,
        ariadne_run_config=ariadne_run_config,
        adversarial_safety=effective.adversarial_safety,
        quality_gates=effective.quality_gates,
        phase_b=phase_b,
        anti_overlap=asdict(effective.anti_overlap),
        scale_model_payload=dict(scale_model_payload),
        sources={
            "level_5_policy": "matches_current_defaults",
            "geometry_scale": "preview_sampling_scale_model",
            "pair_distance": "sampling_scale_model_minimum_safe_pair_ratio",
        },
        hidden_overrides_detected=hidden_sampling_overrides(config),
    )


def phase_b_min_separation_from_resolved(
    resolved: ResolvedSamplingProtocol,
) -> tuple[float, str]:
    phase_b = dict(resolved.phase_b or {})
    return (
        float(phase_b.get("effective_min_separation_angstrom", 0.0)),
        str(phase_b.get("threshold_mode", "absolute")),
    )


__all__ = [
    "SAMPLING_PROTOCOL_AUDIT_FILENAME",
    "SAMPLING_PROTOCOL_AUDIT_SCHEMA_VERSION",
    "SAMPLING_PROTOCOL_RESOLVED_FILENAME",
    "SAMPLING_PROTOCOL_SCHEMA_VERSION",
    "ResolvedSamplingProtocol",
    "SAMPLING_AGGRESSIVENESS_POLICY_VERSION",
    "SamplingAggressivenessPolicy",
    "dimensionless_policy_payload",
    "sampling_policy_table_payload",
    "sampling_policy_table_sha256",
    "hidden_sampling_overrides",
    "load_sampling_protocol",
    "phase_b_min_separation_from_resolved",
    "preview_sampling_protocol",
    "read_sampling_protocol_audit",
    "read_sampling_protocol_resolved",
    "resolve_or_load_sampling_protocol",
    "resolve_sampling_protocol",
    "sampling_protocol_audit_path",
    "sampling_protocol_resolved_path",
    "write_sampling_protocol_audit",
    "write_sampling_protocol_resolved",
]
