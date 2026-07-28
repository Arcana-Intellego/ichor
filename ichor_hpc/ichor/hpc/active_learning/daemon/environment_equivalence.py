"""Proofs for reusing scheduler output across environment generations.

These records are operational evidence.  They do not make output
authoritative; phase-specific validators still decide whether each completed
task is reusable.
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple, Union

from ..strict_json import strict_json as json
from ..versioning.manifest import sha256_file
from .filesystem import campaign_owned_path
from .resource_records import read_resolution, verify_scientific_evidence
from .state import CampaignPhase, atomic_write_json


ENVIRONMENT_EQUIVALENCE_SCHEMA_VERSION = 1
ENVIRONMENT_EQUIVALENCE_DIRNAME = "environment_equivalences"

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ACTIVE_LEARNING_PREFIX = "ichor.hpc.active_learning"
_ACTIVE_LEARNING_REPO_ROOT = Path(
    "ichor_hpc/ichor/hpc/active_learning"
)

_PHASE_BACKEND = {
    "PHASE_A_DIVERSITY": "diversity",
    "PHASE_B_DIVERSITY": "diversity",
    "INITIAL_GAUSSIAN": "gaussian",
    "GAUSSIAN": "gaussian",
    "INITIAL_REPLACEMENT_GAUSSIAN": "gaussian",
    "REPLACEMENT_GAUSSIAN": "gaussian",
    "INITIAL_AIMALL": "aimall",
    "AIMALL": "aimall",
    "INITIAL_REPLACEMENT_AIMALL": "aimall",
    "REPLACEMENT_AIMALL": "aimall",
    "ARIADNE_ARRAY": "ariadne",
    "INITIAL_FEREBUS": "ferebus",
    "FEREBUS": "ferebus",
}

_PRODUCER_ROOTS: Dict[str, Tuple[Tuple[str, Tuple[str, ...]], ...]] = {
    "gaussian": (
        (
            "ichor_hpc/ichor/hpc/active_learning/daemon/live_executor.py",
            ("_gaussian_invocation_block",),
        ),
        (
            "ichor_hpc/ichor/hpc/active_learning/daemon/input_staging.py",
            ("stage_gaussian_inputs",),
        ),
    ),
    "aimall": (
        (
            "ichor_hpc/ichor/hpc/active_learning/daemon/live_executor.py",
            ("_aimall_invocation_block",),
        ),
        (
            "ichor_hpc/ichor/hpc/active_learning/daemon/input_staging.py",
            ("stage_aimall_inputs", "rewrite_wfn_for_aimall"),
        ),
    ),
    "ariadne": (
        (
            "ichor_hpc/ichor/hpc/active_learning/acquisition/ariadne_runner.py",
            (
                "main",
                "optimise_seed",
                "ariadne_result_usability_payload",
            ),
        ),
        (
            "ichor_hpc/ichor/hpc/active_learning/acquisition/ariadne_local_runner.py",
            (
                "run_optimisation_against_calculator",
                "probe_ariadne_runtime",
            ),
        ),
    ),
    "ferebus": (
        (
            "ichor_hpc/ichor/hpc/active_learning/daemon/ferebus_task_runner.py",
            ("execute_task",),
        ),
    ),
    "diversity": (
        (
            "ichor_hpc/ichor/hpc/active_learning/sampling/diversity.py",
            ("_run_phase_a", "_run_phase_b", "fps_select"),
        ),
        (
            "ichor_hpc/ichor/hpc/active_learning/sampling/descriptors.py",
            ("build_descriptor_from_config", "build_condensed_distance_store"),
        ),
    ),
}

_RESOURCE_EVIDENCE_ROOTS: Dict[
    str, Tuple[Tuple[str, Tuple[str, ...]], ...]
] = {
    "ariadne": (
        (
            "ichor_core/ichor/core/adversarial/geometry.py",
            (
                "select_local_neighbours",
                "aligned_mass_weighted_distance",
            ),
        ),
        (
            "ichor_core/ichor/core/adversarial/subspace.py",
            ("build_local_subspace",),
        ),
        (
            "ichor_hpc/ichor/hpc/active_learning/acquisition/trajectory_pool.py",
            ("TrajectoryPool",),
        ),
    ),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _phase_name(value: Any) -> str:
    text = value.value if isinstance(value, CampaignPhase) else str(value or "")
    if text not in _PHASE_BACKEND:
        raise ValueError(
            "phase does not support scheduler environment equivalence: " + text
        )
    return text


def _exact_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(field + " must be a non-negative integer")
    return int(value)


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(field + " must be a non-empty string")
    return value


def _safe_identity(value: Any, field: str) -> str:
    text = _required_text(value, field)
    if not _SAFE_ID_RE.fullmatch(text):
        raise ValueError(field + " contains unsafe characters")
    return text


def _repository_root() -> Path:
    try:
        output = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(Path(__file__).resolve().parent),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("ICHOR Git repository cannot be located") from exc
    if not output:
        raise ValueError("ICHOR Git repository cannot be located")
    return Path(output).resolve()


def _git_text(repo: Path, commit: str, relative_path: str) -> str:
    try:
        raw = subprocess.run(
            ["git", "show", commit + ":" + relative_path],
            cwd=str(repo),
            check=True,
            capture_output=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(
            "required producer source is unavailable at Git commit "
            + commit
            + ": "
            + relative_path
        ) from exc
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "producer source is not UTF-8: " + relative_path
        ) from exc


def _require_clean_available_commit(
    repo: Path,
    generation: Mapping[str, Any],
    *,
    label: str,
    require_worktree_clean: bool,
) -> str:
    git_identity = generation.get("ichor_git")
    if not isinstance(git_identity, Mapping):
        raise ValueError(label + " environment has no Git identity")
    commit = _required_text(git_identity.get("commit"), label + " Git commit")
    if git_identity.get("tracked_tree_clean") is not True:
        raise ValueError(label + " ICHOR source tree was not clean")
    try:
        subprocess.run(
            ["git", "cat-file", "-e", commit + "^{commit}"],
            cwd=str(repo),
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(label + " Git commit is not locally available") from exc
    if require_worktree_clean:
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=str(repo),
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("current ICHOR worktree cleanliness is unknown") from exc
        if status.strip():
            raise ValueError("current ICHOR worktree contains tracked changes")
    return commit


def _top_level_symbols(tree: ast.Module) -> Dict[str, ast.AST]:
    symbols: Dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
            )
            for target in targets:
                if isinstance(target, ast.Name):
                    symbols[target.id] = node
    return symbols


def _without_docstrings(node: ast.AST) -> ast.AST:
    node = ast.fix_missing_locations(ast.parse(ast.unparse(node)).body[0])
    for candidate in ast.walk(node):
        body = getattr(candidate, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]
    return node


def _node_digest(node: ast.AST) -> str:
    normalised = _without_docstrings(node)
    return hashlib.sha256(
        ast.dump(normalised, annotate_fields=True, include_attributes=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _local_module_path(current_path: str, node: ast.ImportFrom) -> Optional[str]:
    if node.level:
        base = Path(current_path).parent
        for _ in range(max(0, int(node.level) - 1)):
            base = base.parent
        target = base / Path(*(str(node.module or "").split(".")))
    elif str(node.module or "") == _ACTIVE_LEARNING_PREFIX:
        target = _ACTIVE_LEARNING_REPO_ROOT
    elif str(node.module or "").startswith(_ACTIVE_LEARNING_PREFIX + "."):
        suffix = str(node.module)[len(_ACTIVE_LEARNING_PREFIX) + 1 :]
        target = _ACTIVE_LEARNING_REPO_ROOT / Path(*suffix.split("."))
    else:
        return None
    return target.as_posix() + ".py"


def _local_imports(
    current_path: str,
    tree: ast.Module,
) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, str]]:
    direct: Dict[str, Tuple[str, str]] = {}
    modules: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            target_path = _local_module_path(current_path, node)
            if target_path is None:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                direct[str(alias.asname or alias.name)] = (
                    target_path,
                    str(alias.name),
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = str(alias.name)
                if name == _ACTIVE_LEARNING_PREFIX:
                    target = _ACTIVE_LEARNING_REPO_ROOT
                elif name.startswith(_ACTIVE_LEARNING_PREFIX + "."):
                    suffix = name[len(_ACTIVE_LEARNING_PREFIX) + 1 :]
                    target = _ACTIVE_LEARNING_REPO_ROOT / Path(*suffix.split("."))
                else:
                    continue
                modules[str(alias.asname or name.split(".")[0])] = (
                    target.as_posix() + ".py"
                )
    return direct, modules


def _referenced_names(node: ast.AST) -> Set[str]:
    return {
        str(candidate.id)
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Name) and isinstance(candidate.ctx, ast.Load)
    }


def _referenced_attributes(node: ast.AST) -> Set[Tuple[str, str]]:
    values = set()
    for candidate in ast.walk(node):
        if (
            isinstance(candidate, ast.Attribute)
            and isinstance(candidate.value, ast.Name)
        ):
            values.add((str(candidate.value.id), str(candidate.attr)))
    return values


def _fingerprint_roots(
    repo: Path,
    commit: str,
    backend: str,
    roots: Sequence[Tuple[str, Tuple[str, ...]]],
) -> Dict[str, Any]:
    queue = [
        (path, symbol)
        for path, symbols in roots
        for symbol in symbols
    ]
    visited: Set[Tuple[str, str]] = set()
    source_cache: Dict[str, Tuple[ast.Module, Dict[str, ast.AST]]] = {}
    records = []
    while queue:
        path, symbol = queue.pop(0)
        identity = (path, symbol)
        if identity in visited:
            continue
        visited.add(identity)
        if len(visited) > 512:
            raise ValueError("producer symbol closure exceeds the safety limit")
        if path not in source_cache:
            source = _git_text(repo, commit, path)
            try:
                tree = ast.parse(source, filename=path)
            except SyntaxError as exc:
                raise ValueError(
                    "producer source cannot be parsed at " + commit + ": " + path
                ) from exc
            source_cache[path] = (tree, _top_level_symbols(tree))
        tree, symbols = source_cache[path]
        node = symbols.get(symbol)
        if node is None:
            direct_imports, _module_imports = _local_imports(path, tree)
            if symbol in direct_imports:
                queue.append(direct_imports[symbol])
                continue
            raise ValueError(
                "producer symbol is unavailable at "
                + commit
                + ": "
                + path
                + ":"
                + symbol
            )
        records.append(
            {
                "path": path,
                "symbol": symbol,
                "ast_sha256": _node_digest(node),
            }
        )
        direct_imports, module_imports = _local_imports(path, tree)
        for name in sorted(_referenced_names(node)):
            if name in symbols:
                queue.append((path, name))
            elif name in direct_imports:
                queue.append(direct_imports[name])
        for module_name, attribute in sorted(_referenced_attributes(node)):
            target_path = module_imports.get(module_name)
            if target_path is not None:
                queue.append((target_path, attribute))
    records.sort(key=lambda item: (item["path"], item["symbol"]))
    return {
        "backend": backend,
        "symbols": records,
        "fingerprint_sha256": _sha256_json(records),
    }


def _producer_fingerprint(
    repo: Path,
    commit: str,
    backend: str,
) -> Dict[str, Any]:
    return _fingerprint_roots(
        repo,
        commit,
        backend,
        _PRODUCER_ROOTS[backend],
    )


def assess_resource_evidence_code_equivalence(
    producer_generation: Mapping[str, Any],
    current_generation: Mapping[str, Any],
    *,
    backend: str,
) -> Dict[str, Any]:
    """Compare only the code that constructs reusable resource evidence."""
    backend_name = str(backend).strip().casefold()
    roots = _RESOURCE_EVIDENCE_ROOTS.get(backend_name)
    if roots is None:
        raise ValueError(
            "backend does not support resource-evidence equivalence: "
            + backend_name
        )
    repo = _repository_root()
    producer_commit = _require_clean_available_commit(
        repo,
        producer_generation,
        label="producer",
        require_worktree_clean=False,
    )
    current_commit = _require_clean_available_commit(
        repo,
        current_generation,
        label="current",
        require_worktree_clean=True,
    )
    producer = _fingerprint_roots(
        repo,
        producer_commit,
        backend_name,
        roots,
    )
    current = _fingerprint_roots(
        repo,
        current_commit,
        backend_name,
        roots,
    )
    return {
        "equivalent": (
            producer["fingerprint_sha256"]
            == current["fingerprint_sha256"]
        ),
        "producer_fingerprint": producer,
        "current_fingerprint": current,
    }


def _resolution_for_intent(
    campaign: Path,
    intent: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_path = _required_text(
        intent.get("resource_resolution_path"),
        "submission intent resource resolution path",
    )
    expected_sha = _required_text(
        intent.get("resource_resolution_sha256"),
        "submission intent resource resolution SHA-256",
    )
    if not _SHA256_RE.fullmatch(expected_sha):
        raise ValueError("submission intent resource resolution SHA-256 is invalid")
    path = campaign_owned_path(campaign, Path(raw_path))
    if path.is_symlink() or not path.is_file():
        raise ValueError("submission resource resolution is missing")
    if sha256_file(path) != expected_sha:
        raise ValueError("submission resource resolution digest mismatch")
    resolution = read_resolution(path)
    expected_identity = (
        str(intent.get("campaign_uid") or ""),
        str(intent.get("phase") or ""),
        int(intent.get("iteration", -1)),
        str(intent.get("attempt_id") or ""),
        str(intent.get("submission_identity") or ""),
    )
    observed_identity = (
        str(resolution.get("campaign_uid") or ""),
        str(resolution.get("phase") or ""),
        int(resolution.get("iteration", -1)),
        str(resolution.get("attempt_id") or ""),
        str(resolution.get("submission_identity") or ""),
    )
    if observed_identity != expected_identity:
        raise ValueError("submission resource resolution identity mismatch")
    verify_scientific_evidence(campaign, resolution["evidence"])
    return resolution


def _native_identity_reasons(
    backend: str,
    producer: Mapping[str, Any],
    current: Mapping[str, Any],
    resolution: Mapping[str, Any],
) -> Sequence[str]:
    reasons = []
    if backend == "ariadne" and producer.get("ariadne") != current.get("ariadne"):
        reasons.append("ariadne_native_identity_changed")
    if backend == "ferebus":
        if producer.get("pyferebus") != current.get("pyferebus"):
            reasons.append("pyferebus_identity_changed")
        if producer.get("ferebus_executable") != current.get("ferebus_executable"):
            reasons.append("ferebus_executable_changed")
    if backend in {"gaussian", "aimall"}:
        implementation = resolution.get("implementation_identity")
        executable = (
            implementation.get("backend_executable")
            if isinstance(implementation, Mapping)
            else None
        )
        resolved_file = (
            executable.get("resolved_file")
            if isinstance(executable, Mapping)
            else None
        )
        if isinstance(resolved_file, Mapping):
            path = Path(str(resolved_file.get("path") or ""))
            digest = str(resolved_file.get("sha256") or "")
            size = resolved_file.get("size")
            if (
                not path.is_absolute()
                or path.is_symlink()
                or not path.is_file()
                or isinstance(size, bool)
                or not isinstance(size, int)
                or int(path.stat().st_size) != int(size)
                or sha256_file(path) != digest
            ):
                reasons.append(backend + "_executable_changed")
    return reasons


def _profile_module_sequence(
    generation: Mapping[str, Any],
) -> Any:
    profile = generation.get("machine_profile")
    if not isinstance(profile, Mapping):
        return None
    sequence = profile.get("module_sequence")
    return dict(sequence) if isinstance(sequence, Mapping) else None


def environment_equivalence_path(
    campaign_dir: Union[str, Path],
    *,
    phase: Any,
    iteration: int,
    replacement_round: int,
    submission_identity: str,
    current_generation: int,
) -> Path:
    phase_name = _phase_name(phase)
    identity = _safe_identity(submission_identity, "submission_identity")
    filename = (
        phase_name
        + "-"
        + f"{_exact_nonnegative_int(iteration, 'iteration'):06d}"
        + "-r"
        + f"{_exact_nonnegative_int(replacement_round, 'replacement_round'):04d}"
        + "-"
        + identity
        + "-to-g"
        + f"{_exact_nonnegative_int(current_generation, 'current_generation'):06d}"
        + ".json"
    )
    return (
        Path(campaign_dir).expanduser().resolve()
        / ".DATA"
        / "ACTIVE_LEARNING"
        / ENVIRONMENT_EQUIVALENCE_DIRNAME
        / filename
    )


def _validate_proof(payload: Mapping[str, Any]) -> Dict[str, Any]:
    data = dict(payload)
    if data.get("schema_version") != ENVIRONMENT_EQUIVALENCE_SCHEMA_VERSION:
        raise ValueError("environment equivalence schema version is unsupported")
    _required_text(data.get("campaign_uid"), "campaign_uid")
    _phase_name(data.get("phase"))
    _exact_nonnegative_int(data.get("iteration"), "iteration")
    _exact_nonnegative_int(data.get("replacement_round"), "replacement_round")
    _safe_identity(data.get("submission_identity"), "submission_identity")
    _required_text(data.get("job_id"), "job_id")
    _exact_nonnegative_int(data.get("producer_generation"), "producer_generation")
    _exact_nonnegative_int(data.get("current_generation"), "current_generation")
    for field in (
        "producer_generation_digest_sha256",
        "current_generation_digest_sha256",
        "proof_sha256",
    ):
        if not _SHA256_RE.fullmatch(str(data.get(field) or "")):
            raise ValueError("environment equivalence " + field + " is invalid")
    if not isinstance(data.get("equivalent"), bool):
        raise ValueError("environment equivalence verdict is invalid")
    if not isinstance(data.get("reasons"), list) or any(
        not isinstance(value, str) or not value for value in data["reasons"]
    ):
        raise ValueError("environment equivalence reasons are invalid")
    unsigned = dict(data)
    recorded = unsigned.pop("proof_sha256", None)
    if _sha256_json(unsigned) != recorded:
        raise ValueError("environment equivalence proof digest mismatch")
    return data


def read_environment_equivalence(path: Union[str, Path]) -> Dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("environment equivalence proof is missing")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("environment equivalence proof is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("environment equivalence proof must be an object")
    return _validate_proof(payload)


def assess_recovery_environment(
    campaign_dir: Union[str, Path],
    *,
    intent: Mapping[str, Any],
    current_scheduler_kind: str,
) -> Dict[str, Any]:
    """Prove whether one producer attempt may contribute reusable output."""
    from ..execution_identity import (
        read_active_environment_generation,
        read_environment_generation,
    )

    campaign = Path(campaign_dir).expanduser().resolve()
    phase = _phase_name(intent.get("phase"))
    backend = _PHASE_BACKEND[phase]
    campaign_uid = _required_text(intent.get("campaign_uid"), "campaign_uid")
    scheduler_kind = _required_text(
        intent.get("scheduler_identity_kind") or "slurm",
        "scheduler_identity_kind",
    ).lower()
    producer_generation_number = _exact_nonnegative_int(
        intent.get("environment_generation"),
        "producer environment generation",
    )
    producer_digest = _required_text(
        intent.get("environment_generation_digest_sha256"),
        "producer environment generation digest",
    )
    producer_generation = read_environment_generation(
        campaign,
        generation=producer_generation_number,
        expected_campaign_uid=campaign_uid,
    )
    if str(producer_generation.get("digest_sha256")) != producer_digest:
        raise ValueError("producer environment generation digest mismatch")
    active = read_active_environment_generation(
        campaign,
        expected_campaign_uid=campaign_uid,
    )
    current_generation = active["generation"]
    current_generation_number = int(current_generation["generation"])
    resolution = _resolution_for_intent(campaign, intent)

    reasons = []
    checks: Dict[str, Any] = {
        "scheduler_kind": scheduler_kind == str(current_scheduler_kind).lower(),
        "python_executable": (
            producer_generation.get("python_executable")
            == current_generation.get("python_executable")
        ),
        "python_version": (
            producer_generation.get("python_version")
            == current_generation.get("python_version")
        ),
        "dependencies": (
            producer_generation.get("dependencies")
            == current_generation.get("dependencies")
        ),
        "loaded_modules": (
            producer_generation.get("loaded_modules")
            == current_generation.get("loaded_modules")
        ),
        "native_library_paths": (
            producer_generation.get("native_library_paths")
            == current_generation.get("native_library_paths")
        ),
        "profile_module_sequence": (
            _profile_module_sequence(producer_generation)
            == _profile_module_sequence(current_generation)
        ),
        "scientific_inputs": True,
    }
    if not checks["scheduler_kind"]:
        reasons.append("scheduler_kind_changed")
    if not checks["python_executable"]:
        reasons.append("python_executable_changed")
    if not checks["python_version"]:
        reasons.append("python_version_changed")
    if not checks["dependencies"]:
        reasons.append("dependency_environment_changed")
    if not checks["loaded_modules"]:
        reasons.append("loaded_modules_changed")
    if not checks["native_library_paths"]:
        reasons.append("native_library_paths_changed")
    if not checks["profile_module_sequence"]:
        reasons.append("profile_module_sequence_changed")
    reasons.extend(
        _native_identity_reasons(
            backend,
            producer_generation,
            current_generation,
            resolution,
        )
    )

    producer_fingerprint: Optional[Dict[str, Any]] = None
    current_fingerprint: Optional[Dict[str, Any]] = None
    if producer_digest != str(current_generation["digest_sha256"]):
        try:
            repo = _repository_root()
            producer_commit = _require_clean_available_commit(
                repo,
                producer_generation,
                label="producer",
                require_worktree_clean=False,
            )
            current_commit = _require_clean_available_commit(
                repo,
                current_generation,
                label="current",
                require_worktree_clean=True,
            )
            producer_fingerprint = _producer_fingerprint(
                repo,
                producer_commit,
                backend,
            )
            current_fingerprint = _producer_fingerprint(
                repo,
                current_commit,
                backend,
            )
            checks["producer_code"] = (
                producer_fingerprint["fingerprint_sha256"]
                == current_fingerprint["fingerprint_sha256"]
            )
            if not checks["producer_code"]:
                reasons.append("scientific_producer_code_changed")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            checks["producer_code"] = False
            reasons.append(
                "producer_code_equivalence_unproven:"
                + type(exc).__name__
                + ":"
                + str(exc)[:180]
            )
    else:
        checks["producer_code"] = True

    equivalent = not reasons
    payload: Dict[str, Any] = {
        "schema_version": ENVIRONMENT_EQUIVALENCE_SCHEMA_VERSION,
        "campaign_uid": campaign_uid,
        "phase": phase,
        "iteration": _exact_nonnegative_int(intent.get("iteration"), "iteration"),
        "replacement_round": _exact_nonnegative_int(
            intent.get("replacement_round", 0),
            "replacement_round",
        ),
        "submission_identity": _safe_identity(
            intent.get("submission_identity"),
            "submission_identity",
        ),
        "attempt_id": _safe_identity(intent.get("attempt_id"), "attempt_id"),
        "job_id": _required_text(intent.get("job_id"), "job_id"),
        "scheduler_identity_kind": scheduler_kind,
        "backend": backend,
        "producer_generation": producer_generation_number,
        "producer_generation_digest_sha256": producer_digest,
        "current_generation": current_generation_number,
        "current_generation_digest_sha256": str(
            current_generation["digest_sha256"]
        ),
        "resource_resolution_path": str(
            intent.get("resource_resolution_path")
        ),
        "resource_resolution_sha256": str(
            intent.get("resource_resolution_sha256")
        ),
        "checks": checks,
        "producer_fingerprint": producer_fingerprint,
        "current_fingerprint": current_fingerprint,
        "equivalent": bool(equivalent),
        "reasons": list(dict.fromkeys(reasons)),
        "recorded_at_iso": _now_iso(),
    }
    payload["proof_sha256"] = _sha256_json(payload)
    validated = _validate_proof(payload)
    path = environment_equivalence_path(
        campaign,
        phase=phase,
        iteration=int(validated["iteration"]),
        replacement_round=int(validated["replacement_round"]),
        submission_identity=str(validated["submission_identity"]),
        current_generation=current_generation_number,
    )
    if path.exists() or path.is_symlink():
        existing = read_environment_equivalence(path)
        comparable_existing = dict(existing)
        comparable_new = dict(validated)
        for item in (comparable_existing, comparable_new):
            item.pop("recorded_at_iso", None)
            item.pop("proof_sha256", None)
        if comparable_existing != comparable_new:
            raise ValueError(
                "environment equivalence proof already exists with different evidence"
            )
        validated = existing
    else:
        if path.parent.is_symlink():
            raise ValueError("environment equivalence directory is a symlink")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, validated)
    return {
        "equivalent": bool(validated["equivalent"]),
        "reasons": list(validated["reasons"]),
        "proof": validated,
        "path": str(path),
        "sha256": str(validated["proof_sha256"]),
    }


__all__ = [
    "ENVIRONMENT_EQUIVALENCE_DIRNAME",
    "ENVIRONMENT_EQUIVALENCE_SCHEMA_VERSION",
    "assess_recovery_environment",
    "assess_resource_evidence_code_equivalence",
    "environment_equivalence_path",
    "read_environment_equivalence",
]
