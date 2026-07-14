"""Benchmark acquisition-gradient modes for an existing campaign seed."""
from __future__ import annotations

import argparse
from ..strict_json import strict_json as json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from ichor.core.adversarial.acquisition import SeedLocalAdversarialAcquisition

from ..acquisition.ase_calculator import AdversarialASECalculator
from ..acquisition.gradient_diagnostics import (
    calculator_gradient_diagnostics,
    static_gradient_diagnostics,
)
from ..acquisition.trajectory_pool import TrajectoryPool
from ..config import CampaignConfig, VALID_GRADIENT_MODES
from ..daemon.state import DEFAULT_STATE_FILENAME, read_state
from ..handoff_manifests import load_seeds_picked


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    return out if np.isfinite(out) else str(out)


def _parse_gradient_modes(text: str) -> List[str]:
    modes = [part.strip() for part in str(text).split(",") if part.strip()]
    if not modes:
        raise ValueError("at least one gradient mode is required")
    invalid = [mode for mode in modes if mode not in VALID_GRADIENT_MODES]
    if invalid:
        raise ValueError(
            "gradient mode(s) must be one of "
            + repr(sorted(VALID_GRADIENT_MODES))
            + "; got "
            + repr(invalid)
        )
    return modes


def _load_reference_scales(iter_dir: Path) -> Dict[str, float] | None:
    from ..layout import active_protocol_dir

    path = active_protocol_dir(iter_dir) / "reference_scales.json"
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("reference_scales.json must contain an object")
    required = ("energy", "force", "omega", "anh", "anh_std")
    out: Dict[str, float] = {}
    for key, value in payload.items():
        try:
            fvalue = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("reference scale " + str(key) + " is not numeric") from exc
        if not np.isfinite(fvalue) or fvalue <= 0.0:
            raise ValueError("reference scale " + str(key) + " must be finite and positive")
        out[str(key)] = fvalue
    missing = [key for key in required if key not in out]
    if missing:
        raise ValueError("reference_scales.json missing " + ", ".join(missing))
    return out


def _load_context(campaign_dir: Path, iteration: int, seed_id: int) -> Dict[str, Any]:
    campaign = campaign_dir.resolve()
    config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
    pool = TrajectoryPool.load(campaign)
    state = read_state(
        campaign / ".DATA" / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME
    )
    if int(state.models_version) < 0:
        raise ValueError("state.models_version is negative; no trained model version is committed")
    from ..versioning.trained_models import load_trained_models

    _, models = load_trained_models(
        campaign,
        int(state.models_version),
        verification="deep",
    )
    from ..layout import active_iteration_dir

    iter_dir = active_iteration_dir(campaign, int(iteration))
    picked = load_seeds_picked(iter_dir, expected_iteration=int(iteration))
    seed_records = list(picked.get("seed_records") or [])
    if not 1 <= int(seed_id) <= len(seed_records):
        raise IndexError(
            "seed-id "
            + str(seed_id)
            + " out of range of "
            + str(len(seed_records))
            + " picked seeds"
        )
    record = seed_records[int(seed_id) - 1]
    if int(record["seed_id"]) != int(seed_id):
        raise ValueError("seed selection is not ordered by seed_id")
    frame_id = int(record["frame_id"])
    return {
        "campaign": campaign,
        "config": config,
        "pool": pool,
        "state": state,
        "models": models,
        "iter_dir": iter_dir,
        "seed_frame_id": frame_id,
        "seed_atoms": pool.frame(frame_id),
        "trajectory": pool.to_atoms_list(),
        "reference_scales": _load_reference_scales(iter_dir),
    }


def _reset_posterior_diagnostics(acquisition: SeedLocalAdversarialAcquisition) -> None:
    posterior = getattr(acquisition, "posterior", None)
    diagnostics = getattr(posterior, "diagnostics", None)
    if isinstance(diagnostics, dict):
        for key in list(diagnostics):
            diagnostics[key] = 0


def _posterior_diagnostics(acquisition: SeedLocalAdversarialAcquisition) -> Dict[str, int]:
    posterior = getattr(acquisition, "posterior", None)
    diagnostics = getattr(posterior, "diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        return {}
    out: Dict[str, int] = {}
    for key, value in diagnostics.items():
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def _build_acquisition(
    context: Mapping[str, Any],
    mode: str,
    *,
    objective: str = "full",
    driver_gradient_backend: str | None = None,
) -> SeedLocalAdversarialAcquisition:
    config = context["config"].to_acquisition_config()
    config = replace(config, gradient=replace(config.gradient, mode=str(mode)))
    if driver_gradient_backend:
        config = replace(
            config,
            driver=replace(
                config.driver,
                enabled=(str(objective) == "cheap_driver" or bool(config.driver.enabled)),
                gradient_backend=str(driver_gradient_backend),
            ),
        )
    return SeedLocalAdversarialAcquisition(
        models=context["models"],
        seed=context["seed_atoms"],
        trajectory=context["trajectory"],
        config=config,
        seed_frame_id=int(context["seed_frame_id"]),
        external_reference_scales=context["reference_scales"],
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    avec = np.asarray(a, dtype=float).reshape(-1)
    bvec = np.asarray(b, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(avec) * np.linalg.norm(bvec))
    if not np.isfinite(denom) or denom <= 0.0:
        return None
    return float(np.dot(avec, bvec) / denom)


def run_benchmark(
    *,
    campaign_dir: Path,
    iteration: int,
    seed_id: int,
    gradient_modes: Sequence[str],
    repeat: int,
    gradient_backend: str = "direct",
    objective: str = "full",
    driver_gradient_backend: str | None = None,
    workers: int | None = None,
) -> Dict[str, Any]:
    context = _load_context(campaign_dir, int(iteration), int(seed_id))
    runs: List[Dict[str, Any]] = []
    gradients_by_mode: Dict[str, np.ndarray] = {}
    median_by_mode: Dict[str, float] = {}

    for mode in gradient_modes:
        acquisition = _build_acquisition(
            context,
            str(mode),
            objective=str(objective),
            driver_gradient_backend=driver_gradient_backend,
        )
        # Warm the reference/acquisition path once outside timing.
        acquisition.components(context["seed_atoms"], objective=str(objective))
        mode_walls: List[float] = []
        for repeat_index in range(int(repeat)):
            _reset_posterior_diagnostics(acquisition)
            static = static_gradient_diagnostics(
                acquisition,
                context["seed_atoms"],
                gradient_mode=str(mode),
                gradient_backend=(
                    "serial" if str(gradient_backend) == "direct" else str(gradient_backend)
                ),
            )
            t0 = time.perf_counter()
            if str(gradient_backend) == "direct":
                grad = np.asarray(
                    acquisition.gradient(
                        context["seed_atoms"],
                        mode=str(mode),
                        objective=str(objective),
                    ),
                    dtype=float,
                )
                calc_diag = {}
            else:
                previous_workers = os.environ.get("ICHOR_GRADIENT_WORKERS")
                if workers is not None:
                    os.environ["ICHOR_GRADIENT_WORKERS"] = str(int(workers))
                try:
                    calculator = AdversarialASECalculator(
                        acquisition,
                        gradient_mode=str(mode),
                        gradient_backend=str(gradient_backend),
                        objective=str(objective),
                        project_rigid=False,
                        max_acquisition_grad_per_ang=0.0,
                    )
                    _energy, forces = calculator.evaluate(context["seed_atoms"])
                    hartree = calculator._hartree()
                    grad = np.asarray(forces, dtype=float) / max(float(hartree), 1.0e-300)
                    calc_diag = calculator_gradient_diagnostics(calculator)
                finally:
                    if workers is not None:
                        if previous_workers is None:
                            os.environ.pop("ICHOR_GRADIENT_WORKERS", None)
                        else:
                            os.environ["ICHOR_GRADIENT_WORKERS"] = previous_workers
            wall = float(time.perf_counter() - t0)
            mode_walls.append(wall)
            gradients_by_mode[str(mode)] = grad
            posterior_diag = _posterior_diagnostics(acquisition)
            runs.append({
                "gradient_mode": str(mode),
                "repeat": int(repeat_index),
                "wall_seconds": wall,
                "grad_norm": float(np.linalg.norm(grad)),
                "grad_shape": list(grad.shape),
                "n_estimated_acquisition_value_calls": (
                    static.get("n_estimated_acquisition_value_calls_per_gradient")
                ),
                "subspace_dim": static.get("subspace_dim"),
                "n_cartesian_dof": static.get("n_cartesian_dof"),
                "n_live_cartesian_dof": static.get("n_live_cartesian_dof"),
                "gradient_backend": str(gradient_backend),
                "objective": str(objective),
                "driver_gradient_backend": driver_gradient_backend,
                "workers": workers,
                "posterior_diagnostics": posterior_diag,
                "calculator_diagnostics": calc_diag,
            })
        median_by_mode[str(mode)] = float(np.median(mode_walls)) if mode_walls else 0.0

    comparisons: Dict[str, Any] = {}
    if "active_fd" in gradients_by_mode and "cartesian_fd" in gradients_by_mode:
        active = gradients_by_mode["active_fd"]
        cart = gradients_by_mode["cartesian_fd"]
        active_median = median_by_mode.get("active_fd", 0.0)
        cart_median = median_by_mode.get("cartesian_fd", 0.0)
        comparisons["active_fd_vs_cartesian_fd"] = {
            "cosine": _cosine(active, cart),
            "relative_norm": (
                None
                if float(np.linalg.norm(cart)) <= 0.0
                else float(np.linalg.norm(active) / np.linalg.norm(cart))
            ),
            "cartesian_wall_seconds_median": cart_median,
            "active_wall_seconds_median": active_median,
            "speedup": (
                None
                if active_median <= 0.0
                else float(cart_median / active_median)
            ),
        }

    seed_atoms = context["seed_atoms"]
    return {
        "schema_version": 1,
        "campaign_dir": str(context["campaign"]),
        "iteration": int(iteration),
        "seed_id": int(seed_id),
        "seed_frame_id": int(context["seed_frame_id"]),
        "models_version": int(context["state"].models_version),
        "natoms": int(len(seed_atoms)),
        "gradient_modes": list(gradient_modes),
        "gradient_backend": str(gradient_backend),
        "objective": str(objective),
        "driver_gradient_backend": driver_gradient_backend,
        "workers": workers,
        "repeat": int(repeat),
        "runs": runs,
        "comparisons": comparisons,
    }


def _print_table(payload: Mapping[str, Any]) -> None:
    print("mode        repeat  wall_seconds  grad_norm    value_calls backend objective")
    for row in payload.get("runs", []):
        if not isinstance(row, Mapping):
            continue
        print(
            f"{str(row.get('gradient_mode', '')):<11} "
            f"{int(row.get('repeat', 0)):>6} "
            f"{float(row.get('wall_seconds', 0.0)):>13.6f} "
            f"{float(row.get('grad_norm', 0.0)):>10.4e} "
            f"{str(row.get('n_estimated_acquisition_value_calls')):>11} "
            f"{str(row.get('gradient_backend', '')):<7} "
            f"{str(row.get('objective', '')):<12}"
        )
    comparisons = payload.get("comparisons")
    if isinstance(comparisons, Mapping) and comparisons:
        print("comparisons:")
        for name, data in comparisons.items():
            print("  " + str(name) + ": " + json.dumps(_json_safe(data), sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ichor-al-benchmark-acquisition-gradient",
        description="Benchmark ICHOR active-learning acquisition gradient modes.",
    )
    parser.add_argument("--campaign-dir", required=True, help="Campaign root directory.")
    parser.add_argument("--iteration", required=True, type=int, help="Campaign iteration.")
    parser.add_argument("--seed-id", required=True, type=int, help="One-based seed ID in seed_selection/SELECTION.json.")
    parser.add_argument(
        "--gradient-mode",
        default="cartesian_fd,active_fd",
        help="Comma-separated gradient modes to benchmark.",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Repeats per mode.")
    parser.add_argument(
        "--gradient-backend",
        default="direct",
        choices=("direct", "serial", "process", "thread"),
        help="direct calls acquisition.gradient(); others route through the live calculator backend.",
    )
    parser.add_argument(
        "--objective",
        default="full",
        choices=("full", "cheap_driver"),
        help="Acquisition objective to benchmark.",
    )
    parser.add_argument(
        "--driver-gradient-backend",
        default="",
        choices=("", "fd", "hybrid_geometry"),
        help="Override acquisition.driver.gradient_backend for cheap-driver benchmarks.",
    )
    parser.add_argument("--workers", type=int, default=0, help="Worker override for process/thread backends.")
    parser.add_argument("--json", dest="json_path", default="", help="Optional JSON output path.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        modes = _parse_gradient_modes(args.gradient_mode)
        if int(args.repeat) <= 0:
            raise ValueError("--repeat must be > 0")
        payload = run_benchmark(
            campaign_dir=Path(args.campaign_dir),
            iteration=int(args.iteration),
            seed_id=int(args.seed_id),
            gradient_modes=modes,
            repeat=int(args.repeat),
            gradient_backend=str(args.gradient_backend),
            objective=str(args.objective),
            driver_gradient_backend=(args.driver_gradient_backend or None),
            workers=(None if int(args.workers) <= 0 else int(args.workers)),
        )
        _print_table(payload)
        if args.json_path:
            out = Path(args.json_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
                handle.write("\n")
            print("wrote " + str(out))
        return 0
    except Exception as exc:
        print(
            "acquisition gradient benchmark failed: "
            + type(exc).__name__
            + ": "
            + str(exc),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
