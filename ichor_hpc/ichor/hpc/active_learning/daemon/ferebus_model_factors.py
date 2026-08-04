"""Non-authoritative task-side Cholesky evidence for FEREBUS models."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import platform
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from ..strict_json import strict_json as json
from .filesystem import campaign_owned_path
from .state import atomic_write_json


FEREBUS_MODEL_FACTOR_CACHE_SCHEMA_VERSION = 1
FEREBUS_MODEL_FACTOR_ALGORITHM = "numpy.linalg.cholesky:model.R:v1"


class FerebusModelFactorError(ValueError):
    """Raised when optional task-factor evidence is invalid."""


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _campaign_root(staging_dir: Path, campaign_dir: Optional[Path]) -> Path:
    staging = Path(staging_dir).absolute()
    valid_model_root = (
        staging.parent.name == "TRAINED_MODELS"
        and (
            staging.name == "iteration-staging"
            or re.fullmatch(r"iteration-[0-9]{6}", staging.name) is not None
        )
    )
    if campaign_dir is None:
        if not valid_model_root:
            raise FerebusModelFactorError(
                "FEREBUS model root is not canonical beneath TRAINED_MODELS"
            )
        campaign = staging.parent.parent
    else:
        campaign = Path(campaign_dir).absolute()
        if staging.parent != campaign / "TRAINED_MODELS" or not valid_model_root:
            raise FerebusModelFactorError("FEREBUS factor staging path mismatch")
    if campaign.is_symlink() or not campaign.is_dir():
        raise FerebusModelFactorError("FEREBUS campaign root is unsafe")
    return campaign


def _task_context(
    staging_dir: Path,
    logical_task_id: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    from . import input_staging as staging_api
    from .ferebus_task_runner import (
        FEREBUS_TASK_MAP_FILENAME,
        _read_task_map,
        validate_task_receipt,
    )

    staging = Path(staging_dir).resolve()
    task_map = _read_task_map(staging / FEREBUS_TASK_MAP_FILENAME)
    task_manifest = staging_api.read_ferebus_manifest(
        staging,
        verify_dataset_files=False,
    )
    task_id = int(logical_task_id)
    map_tasks = task_map.get("tasks")
    manifest_tasks = task_manifest.get("tasks")
    if (
        task_id < 0
        or not isinstance(map_tasks, list)
        or not isinstance(manifest_tasks, list)
        or task_id >= len(map_tasks)
        or task_id >= len(manifest_tasks)
    ):
        raise FerebusModelFactorError("FEREBUS factor task is out of range")
    map_task = map_tasks[task_id]
    manifest_task = manifest_tasks[task_id]
    if (
        not isinstance(map_task, dict)
        or not isinstance(manifest_task, dict)
        or map_task.get("task_index") != task_id + 1
        or manifest_task.get("task_index") != task_id + 1
        or map_task.get("property") != manifest_task.get("property")
        or map_task.get("atom") != manifest_task.get("atom")
    ):
        raise FerebusModelFactorError("FEREBUS factor task identity mismatch")
    receipt = validate_task_receipt(staging, task_id)
    return task_map, task_manifest, map_task, receipt


def _factor_identity(
    task_map: Mapping[str, Any],
    task_manifest: Mapping[str, Any],
    map_task: Mapping[str, Any],
    receipt: Mapping[str, Any],
    model: Any,
) -> Dict[str, Any]:
    import numpy as np
    import scipy
    from .seed_selection_runtime import _numpy_build_sha256

    task_index = int(map_task["task_index"])
    manifest_task = task_manifest["tasks"][task_index - 1]
    return {
        "schema_version": FEREBUS_MODEL_FACTOR_CACHE_SCHEMA_VERSION,
        "campaign_uid": str(task_manifest.get("campaign_uid") or ""),
        "reference_data_version": int(
            task_manifest.get("reference_data_version", -1)
        ),
        "task_map_sha256": str(task_map["task_map_sha256"]),
        "task_manifest_sha256": str(task_map["task_manifest_sha256"]),
        "task_index": task_index,
        "property": str(map_task.get("property") or ""),
        "atom": str(map_task.get("atom") or ""),
        "model_sha256": str(receipt.get("model_sha256") or ""),
        "datasets": dict(manifest_task.get("datasets") or {}),
        "numeric_model_identity": str(model.numeric_identity),
        "ntrain": int(model.ntrain),
        "nfeats": int(model.nfeats),
        "jitter": float(model.jitter),
        "factor_algorithm": FEREBUS_MODEL_FACTOR_ALGORITHM,
        "dtype": np.dtype(np.float64).str,
        "numpy_version": str(np.__version__),
        "scipy_version": str(scipy.__version__),
        "numpy_build_sha256": _numpy_build_sha256(),
        "machine": str(platform.machine()),
        "byteorder": sys.byteorder,
        "numerical_threads": 1,
    }


def _paths(
    campaign: Path,
    task_map_sha256: str,
    task_index: int,
) -> Tuple[Path, Path, Path]:
    if len(task_map_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in task_map_sha256
    ):
        raise FerebusModelFactorError("FEREBUS task-map digest is invalid")
    root = campaign_owned_path(
        campaign,
        Path(".DATA")
        / "CACHE"
        / "FEREBUS_MODEL_FACTORS"
        / task_map_sha256
        / ("task-" + str(int(task_index)).zfill(6)),
    )
    return root / "factor.npy", root / "MANIFEST.json", root / ".lock"


def _validate_factor(model: Any, factor: Any) -> Any:
    import numpy as np
    from .seed_selection_runtime import SeedSelectionRuntimeCache

    values = np.asarray(factor)
    expected_shape = (int(model.ntrain), int(model.ntrain))
    if values.dtype != np.dtype(np.float64) or values.shape != expected_shape:
        raise FerebusModelFactorError("FEREBUS factor shape or dtype mismatch")
    if not np.all(np.isfinite(values)) or np.any(np.diag(values) <= 0.0):
        raise FerebusModelFactorError("FEREBUS factor values are invalid")
    scale = max(1.0, float(np.max(np.abs(values))))
    tolerance = (
        np.finfo(np.float64).eps
        * max(1, int(model.ntrain))
        * scale
        * 16.0
    )
    if np.any(np.abs(np.triu(values, k=1)) > tolerance):
        raise FerebusModelFactorError("FEREBUS factor is not lower triangular")
    if not SeedSelectionRuntimeCache._factor_residual_is_valid(model, values):
        raise FerebusModelFactorError("FEREBUS factor residual is invalid")
    return values


def _read_task_factor_unlocked(
    staging_dir: Path,
    logical_task_id: int,
    *,
    model: Any,
    campaign_dir: Optional[Path] = None,
) -> Any:
    """Read one factor while the caller owns its task-cache lock."""
    import numpy as np
    from ..versioning.manifest import sha256_file

    staging = Path(staging_dir).resolve()
    campaign = _campaign_root(staging, campaign_dir)
    task_map, task_manifest, map_task, receipt = _task_context(
        staging, int(logical_task_id)
    )
    identity = _factor_identity(
        task_map, task_manifest, map_task, receipt, model
    )
    data_path, manifest_path, unused_lock = _paths(
        campaign, str(task_map["task_map_sha256"]), int(map_task["task_index"])
    )
    del unused_lock
    if (
        data_path.is_symlink()
        or manifest_path.is_symlink()
        or not data_path.is_file()
        or not manifest_path.is_file()
    ):
        raise FerebusModelFactorError("FEREBUS task factor is absent or unsafe")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FerebusModelFactorError("FEREBUS factor manifest is unreadable") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if (
        payload.get("schema_version")
        != FEREBUS_MODEL_FACTOR_CACHE_SCHEMA_VERSION
        or payload.get("identity") != identity
        or payload.get("identity_sha256") != _canonical_sha256(identity)
        or not isinstance(data, dict)
        or data.get("path") != "factor.npy"
        or int(data.get("size", -1)) != int(data_path.stat().st_size)
        or str(data.get("sha256") or "") != sha256_file(data_path)
        or data.get("shape") != [int(model.ntrain), int(model.ntrain)]
        or data.get("dtype") != np.dtype(np.float64).str
    ):
        raise FerebusModelFactorError("FEREBUS task factor identity mismatch")
    factor = np.load(data_path, mmap_mode="r", allow_pickle=False)
    return _validate_factor(model, factor)


@contextmanager
def _locked_task_factor(
    staging_dir: Path,
    logical_task_id: int,
    *,
    model: Any,
    campaign_dir: Optional[Path] = None,
) -> Iterator[Any]:
    import portalocker

    staging = Path(staging_dir).resolve()
    campaign = _campaign_root(staging, campaign_dir)
    task_map, unused_manifest, map_task, unused_receipt = _task_context(
        staging, int(logical_task_id)
    )
    del unused_manifest, unused_receipt
    unused_data, unused_manifest_path, lock_path = _paths(
        campaign,
        str(task_map["task_map_sha256"]),
        int(map_task["task_index"]),
    )
    del unused_data, unused_manifest_path
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.parent.is_symlink() or lock_path.is_symlink():
        raise FerebusModelFactorError("FEREBUS task-factor cache is symlinked")
    with portalocker.Lock(str(lock_path), mode="a", timeout=600):
        yield _read_task_factor_unlocked(
            staging,
            int(logical_task_id),
            model=model,
            campaign_dir=campaign,
        )


def read_task_factor(
    staging_dir: Path,
    logical_task_id: int,
    *,
    model: Any,
    campaign_dir: Optional[Path] = None,
) -> Any:
    """Read and authenticate one optional task-produced factor."""
    import numpy as np

    with _locked_task_factor(
        staging_dir,
        logical_task_id,
        model=model,
        campaign_dir=campaign_dir,
    ) as factor:
        # A detached value cannot change after the producer-cache lock is
        # released. Adoption uses the locked context directly to copy bytes.
        return np.array(factor, dtype=np.float64, copy=True, order="C")


def publish_task_factor(
    staging_dir: Path,
    logical_task_id: int,
    *,
    model: Optional[Any] = None,
    campaign_dir: Optional[Path] = None,
) -> Path:
    """Atomically publish one optional factor without touching task receipts."""
    import numpy as np
    import portalocker
    from ichor.core.models import Model
    from ..versioning.manifest import sha256_file
    from . import input_staging as staging_api
    from .seed_selection_runtime import _write_npy_atomic

    staging = Path(staging_dir).resolve()
    campaign = _campaign_root(staging, campaign_dir)
    task_map, task_manifest, map_task, receipt = _task_context(
        staging, int(logical_task_id)
    )
    if model is None:
        model_path = staging_api.resolve_ferebus_task_path(
            staging, receipt["model_path"], "factor model"
        )
        model = Model(model_path)
    identity = _factor_identity(
        task_map, task_manifest, map_task, receipt, model
    )
    data_path, manifest_path, lock_path = _paths(
        campaign, str(task_map["task_map_sha256"]), int(map_task["task_index"])
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if data_path.parent.is_symlink() or lock_path.is_symlink():
        raise FerebusModelFactorError("FEREBUS task-factor cache is symlinked")
    with portalocker.Lock(str(lock_path), mode="a", timeout=600):
        try:
            _read_task_factor_unlocked(
                staging,
                int(logical_task_id),
                model=model,
                campaign_dir=campaign,
            )
            return manifest_path
        except Exception:
            data_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
        factor = _validate_factor(
            model, np.asarray(model.lower_cholesky, dtype=np.float64)
        )
        _write_npy_atomic(data_path, factor)
        payload = {
            "schema_version": FEREBUS_MODEL_FACTOR_CACHE_SCHEMA_VERSION,
            "identity": identity,
            "identity_sha256": _canonical_sha256(identity),
            "data": {
                "path": "factor.npy",
                "size": int(data_path.stat().st_size),
                "sha256": sha256_file(data_path),
                "shape": [int(model.ntrain), int(model.ntrain)],
                "dtype": np.dtype(np.float64).str,
            },
        }
        atomic_write_json(manifest_path, payload)
        _read_task_factor_unlocked(
            staging,
            int(logical_task_id),
            model=model,
            campaign_dir=campaign,
        )
        return manifest_path


def _run_factor_subprocess(
    staging_dir: Path,
    logical_task_id: int,
    python_executable: str,
) -> None:
    import subprocess

    environment = dict(os.environ)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[variable] = "1"
    completed = subprocess.run(
        [
            str(python_executable),
            "-m",
            "ichor.hpc.active_learning.daemon.ferebus_model_factors",
            "--publish",
            "--staging-dir",
            str(Path(staging_dir).resolve()),
            "--logical-task-id",
            str(int(logical_task_id)),
        ],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout or "").strip()
        raise FerebusModelFactorError(
            "isolated FEREBUS factor calculation failed: " + diagnostic[-1000:]
        )


def adopt_ferebus_model_factors(
    staging_dir: Path,
    *,
    committed_model_set: Any,
    cache: Any,
    python_executable: str,
    progress: Optional[Any] = None,
) -> Dict[str, str]:
    """Adopt task evidence and calculate only genuinely missing factors."""
    staging = Path(staging_dir).resolve()
    tasks = [
        task
        for task in committed_model_set.tasks
        if str(task.property) == "iqa"
    ]
    tasks.sort(key=lambda task: int(task.task_index))
    models = cache.posterior._property_models
    statuses: Dict[str, str] = {}
    missing = []
    total = len(tasks)

    def report(stage: str, completed: int) -> None:
        if progress is not None:
            progress(
                stage,
                completed=int(completed),
                total=int(total),
                unit="models",
            )

    report("model_factor_validation", 0)
    for position, task in enumerate(tasks, start=1):
        atom = str(task.atom)
        model = models.get(atom)
        if model is None:
            statuses[atom] = "failed_optional"
            continue
        try:
            cache.restore_model_factor(atom)
            statuses[atom] = "existing_hit"
        except Exception:
            try:
                with _locked_task_factor(
                    staging,
                    int(task.task_index) - 1,
                    model=model,
                    campaign_dir=Path(cache.campaign_dir),
                ) as factor:
                    statuses[atom] = cache.adopt_model_factor(atom, factor)
            except Exception:
                missing.append((task, model))
        report("model_factor_validation", position)

    if missing:
        report("model_factor_fallback", 0)
        failures = set()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, min(6, len(missing)))
        ) as executor:
            futures = {
                executor.submit(
                    _run_factor_subprocess,
                    staging,
                    int(task.task_index) - 1,
                    str(python_executable),
                ): str(task.atom)
                for task, unused_model in missing
            }
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures),
                start=1,
            ):
                atom = futures[future]
                try:
                    future.result()
                except Exception:
                    failures.add(atom)
                report("model_factor_fallback", completed)

        still_missing = []
        report("model_factor_adoption", 0)
        for position, (task, model) in enumerate(missing, start=1):
            atom = str(task.atom)
            if atom in failures:
                still_missing.append(atom)
            else:
                try:
                    with _locked_task_factor(
                        staging,
                        int(task.task_index) - 1,
                        model=model,
                        campaign_dir=Path(cache.campaign_dir),
                    ) as factor:
                        cache.adopt_model_factor(atom, factor)
                    statuses[atom] = "local_fallback"
                except Exception:
                    still_missing.append(atom)
            report("model_factor_adoption", position)

        if still_missing:
            cache.ensure_model_factors()
            for atom in still_missing:
                statuses[atom] = "local_fallback"
    return statuses


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--staging-dir")
    parser.add_argument("--logical-task-id", type=int)
    args = parser.parse_args(argv)
    if not args.publish:
        parser.error("an internal factor action is required")
    if args.staging_dir is None or args.logical_task_id is None:
        parser.error("--staging-dir and --logical-task-id are required")
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = "1"
    publish_task_factor(Path(args.staging_dir), int(args.logical_task_id))
    return 0


__all__ = [
    "FEREBUS_MODEL_FACTOR_ALGORITHM",
    "FEREBUS_MODEL_FACTOR_CACHE_SCHEMA_VERSION",
    "FerebusModelFactorError",
    "adopt_ferebus_model_factors",
    "publish_task_factor",
    "read_task_factor",
]


if __name__ == "__main__":
    raise SystemExit(main())
