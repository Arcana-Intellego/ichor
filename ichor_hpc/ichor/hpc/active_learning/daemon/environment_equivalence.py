"""Proofs for reusing scheduler output across environment generations.

These records are operational evidence.  They do not make output
authoritative; phase-specific validators still decide whether each completed
task is reusable.
"""
from __future__ import annotations

import ast
import hashlib
import io
import os
import re
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple, Union

from ..strict_json import strict_json as json
from ..versioning.manifest import sha256_file
from .filesystem import campaign_owned_path
from .resource_records import read_resolution, verify_scientific_evidence
from .state import CampaignPhase, atomic_write_json


ENVIRONMENT_EQUIVALENCE_SCHEMA_VERSION = 2
LEGACY_ENVIRONMENT_EQUIVALENCE_SCHEMA_VERSION = 1
ENVIRONMENT_EQUIVALENCE_DIRNAME = "environment_equivalences"
SCIENTIFIC_FINGERPRINT_ALGORITHM = "repository_module_closure_v2"

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_REPOSITORY_NAMESPACE_ROOTS = {
    "ichor.core": Path("ichor_core/ichor/core"),
    "ichor.hpc": Path("ichor_hpc/ichor/hpc"),
    "ichor.cli": Path("ichor_cli/ichor/cli"),
}
_NON_RUNTIME_PATH_PREFIXES = (
    ".github/",
    "docs/",
    "ichor_cli/tests/",
    "ichor_core/tests/",
    "ichor_hpc/tests/",
)
_NON_RUNTIME_BASENAMES = {
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
}
_GIT_SOURCE_TREE_CACHE: Dict[Tuple[str, str], Dict[str, str]] = {}

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


def _git_python_sources(repo: Path, commit: str) -> Dict[str, str]:
    identity = (str(repo.resolve()), str(commit))
    cached = _GIT_SOURCE_TREE_CACHE.get(identity)
    if cached is not None:
        return cached
    try:
        raw = subprocess.run(
            [
                "git",
                "archive",
                "--format=tar",
                str(commit),
                "ichor_core/ichor/core",
                "ichor_hpc/ichor/hpc",
                "ichor_cli/ichor/cli",
            ],
            cwd=str(repo),
            check=True,
            capture_output=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(
            "ICHOR Python source tree is unavailable at Git commit " + commit
        ) from exc
    sources: Dict[str, str] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            for member in archive.getmembers():
                name = str(member.name).replace("\\", "/")
                if not member.isfile() or not name.endswith(".py"):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError("Git archive member is unreadable: " + name)
                sources[name] = handle.read().decode("utf-8")
    except (tarfile.TarError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(
            "ICHOR Python source tree cannot be decoded at Git commit " + commit
        ) from exc
    if len(_GIT_SOURCE_TREE_CACHE) >= 4:
        _GIT_SOURCE_TREE_CACHE.pop(next(iter(_GIT_SOURCE_TREE_CACHE)))
    _GIT_SOURCE_TREE_CACHE[identity] = sources
    return sources


def _git_text(repo: Path, commit: str, relative_path: str) -> str:
    sources = _git_python_sources(repo, commit)
    try:
        return sources[str(relative_path).replace("\\", "/")]
    except KeyError as exc:
        raise ValueError(
            "required producer source is unavailable at Git commit "
            + commit
            + ": "
            + relative_path
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
    if not _GIT_COMMIT_RE.fullmatch(commit):
        raise ValueError(label + " environment has an invalid Git commit")
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


def _module_identity_for_path(relative_path: str) -> Tuple[str, bool]:
    path = Path(relative_path)
    for namespace, root in sorted(
        _REPOSITORY_NAMESPACE_ROOTS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        try:
            suffix = path.relative_to(root)
        except ValueError:
            continue
        if suffix.suffix != ".py":
            raise ValueError("repository-local Python path is invalid: " + relative_path)
        parts = list(suffix.with_suffix("").parts)
        is_package = bool(parts and parts[-1] == "__init__")
        if is_package:
            parts.pop()
        module = namespace + ("." + ".".join(parts) if parts else "")
        return module, is_package
    raise ValueError("producer path is outside the ICHOR package roots: " + relative_path)


def _repository_namespace(module: str) -> Optional[Tuple[str, Path]]:
    for namespace, root in sorted(
        _REPOSITORY_NAMESPACE_ROOTS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if module == namespace or module.startswith(namespace + "."):
            return namespace, root
    return None


def _git_path_exists(
    repo: Path,
    commit: str,
    relative_path: str,
    cache: Dict[Tuple[str, str], bool],
) -> bool:
    identity = (commit, relative_path)
    if identity not in cache:
        cache[identity] = (
            str(relative_path).replace("\\", "/")
            in _git_python_sources(repo, commit)
        )
    return bool(cache[identity])


def _resolve_repository_module(
    repo: Path,
    commit: str,
    module: str,
    existence_cache: Dict[Tuple[str, str], bool],
    *,
    required: bool = False,
) -> Optional[str]:
    root_identity = _repository_namespace(str(module))
    if root_identity is None:
        return None
    namespace, root = root_identity
    suffix = str(module)[len(namespace) :].lstrip(".")
    base = root / Path(*suffix.split(".")) if suffix else root
    candidates = (
        base.with_suffix(".py").as_posix(),
        (base / "__init__.py").as_posix(),
    )
    for candidate in candidates:
        if _git_path_exists(repo, commit, candidate, existence_cache):
            return candidate
    if required:
        raise ValueError(
            "repository-local import is unavailable at "
            + commit
            + ": "
            + str(module)
        )
    return None


def _absolute_import_module(current_path: str, node: ast.ImportFrom) -> str:
    if not node.level:
        return str(node.module or "")
    current_module, is_package = _module_identity_for_path(current_path)
    package = current_module if is_package else current_module.rpartition(".")[0]
    parts = package.split(".") if package else []
    remove = int(node.level) - 1
    if remove > len(parts):
        raise ValueError("relative repository import escapes its package")
    base = parts[: len(parts) - remove] if remove else parts
    if node.module:
        base.extend(str(node.module).split("."))
    return ".".join(base)


def _module_digest(tree: ast.Module) -> str:
    normalised = ast.parse(ast.unparse(tree))
    for candidate in ast.walk(normalised):
        body = getattr(candidate, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]
    return hashlib.sha256(
        ast.dump(
            normalised,
            annotate_fields=True,
            include_attributes=False,
        ).encode("utf-8")
    ).hexdigest()


def _repository_import_targets(
    repo: Path,
    commit: str,
    current_path: str,
    tree: ast.Module,
    existence_cache: Dict[Tuple[str, str], bool],
) -> Tuple[Set[str], Tuple[str, ...]]:
    targets: Set[str] = set()
    unresolved_dynamic = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = _resolve_repository_module(
                    repo,
                    commit,
                    str(alias.name),
                    existence_cache,
                    required=_repository_namespace(str(alias.name)) is not None,
                )
                if target is not None:
                    targets.add(target)
        elif isinstance(node, ast.ImportFrom):
            module = _absolute_import_module(current_path, node)
            target = _resolve_repository_module(
                repo,
                commit,
                module,
                existence_cache,
                required=_repository_namespace(module) is not None,
            )
            if target is not None:
                targets.add(target)
            for alias in node.names:
                if alias.name == "*":
                    continue
                child_module = (
                    module + "." + str(alias.name)
                    if module
                    else str(alias.name)
                )
                child = _resolve_repository_module(
                    repo,
                    commit,
                    child_module,
                    existence_cache,
                    required=False,
                )
                if child is not None:
                    targets.add(child)
        elif isinstance(node, ast.Call):
            function = node.func
            dynamic_import = (
                isinstance(function, ast.Name) and function.id == "__import__"
            ) or (
                isinstance(function, ast.Attribute)
                and function.attr == "import_module"
            )
            if not dynamic_import:
                continue
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                module = str(node.args[0].value)
                target = _resolve_repository_module(
                    repo,
                    commit,
                    module,
                    existence_cache,
                    required=_repository_namespace(module) is not None,
                )
                if target is not None:
                    targets.add(target)
            else:
                unresolved_dynamic.append(current_path)
    return targets, tuple(sorted(set(unresolved_dynamic)))


def _changed_paths(repo: Path, producer_commit: str, current_commit: str) -> Set[str]:
    if producer_commit == current_commit:
        return set()
    try:
        output = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                "--diff-filter=ACDMRTUXB",
                producer_commit,
                current_commit,
                "--",
            ],
            cwd=str(repo),
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("changed repository paths cannot be established") from exc
    return {line.strip().replace("\\", "/") for line in output.splitlines() if line.strip()}


def _is_non_runtime_path(path: str) -> bool:
    normalised = str(path).replace("\\", "/")
    return (
        normalised in _NON_RUNTIME_BASENAMES
        or normalised.endswith(".md")
        or any(normalised.startswith(prefix) for prefix in _NON_RUNTIME_PATH_PREFIXES)
        or "/tests/" in normalised
    )


def _fingerprint_roots(
    repo: Path,
    commit: str,
    backend: str,
    roots: Sequence[Tuple[str, Tuple[str, ...]]],
) -> Dict[str, Any]:
    root_symbols = {
        str(path): tuple(str(symbol) for symbol in symbols)
        for path, symbols in roots
    }
    queue = list(root_symbols)
    visited: Set[str] = set()
    existence_cache: Dict[Tuple[str, str], bool] = {}
    records = []
    unresolved_dynamic: Set[str] = set()
    while queue:
        path = str(queue.pop(0)).replace("\\", "/")
        if path in visited:
            continue
        visited.add(path)
        if len(visited) > 2048:
            raise ValueError("producer module closure exceeds the safety limit")
        source = _git_text(repo, commit, path)
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError as exc:
            raise ValueError(
                "producer source cannot be parsed at " + commit + ": " + path
            ) from exc
        symbols = _top_level_symbols(tree)
        for symbol in root_symbols.get(path, ()):
            if symbol not in symbols:
                raise ValueError(
                    "producer symbol is unavailable at "
                    + commit
                    + ": "
                    + path
                    + ":"
                    + symbol
                )
        targets, unresolved = _repository_import_targets(
            repo,
            commit,
            path,
            tree,
            existence_cache,
        )
        unresolved_dynamic.update(unresolved)
        queue.extend(sorted(targets.difference(visited)))
        module_name, _is_package = _module_identity_for_path(path)
        records.append(
            {
                "path": path,
                "module": module_name,
                "ast_sha256": _module_digest(tree),
            }
        )
    records.sort(key=lambda item: item["path"])
    fingerprint_payload = {
        "algorithm": SCIENTIFIC_FINGERPRINT_ALGORITHM,
        "backend": backend,
        "roots": [
            {"path": path, "symbols": list(root_symbols[path])}
            for path in sorted(root_symbols)
        ],
        "modules": records,
        "unresolved_dynamic_import_modules": sorted(unresolved_dynamic),
    }
    return {
        **fingerprint_payload,
        "covered_paths": sorted(visited),
        "fingerprint_sha256": _sha256_json(fingerprint_payload),
    }


def _uncovered_runtime_changes(
    repo: Path,
    producer_commit: str,
    current_commit: str,
    *fingerprints: Mapping[str, Any],
) -> Tuple[str, ...]:
    covered = {
        str(path)
        for fingerprint in fingerprints
        for path in fingerprint.get("covered_paths", [])
    }
    return tuple(
        sorted(
            path
            for path in _changed_paths(repo, producer_commit, current_commit)
            if path not in covered and not _is_non_runtime_path(path)
        )
    )


def _unresolved_dynamic_import_modules(
    *fingerprints: Mapping[str, Any],
) -> Tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(module)
                for fingerprint in fingerprints
                for module in fingerprint.get(
                    "unresolved_dynamic_import_modules",
                    [],
                )
                if str(module)
            }
        )
    )


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
    uncovered = _uncovered_runtime_changes(
        repo,
        producer_commit,
        current_commit,
        producer,
        current,
    )
    unresolved_dynamic = _unresolved_dynamic_import_modules(
        producer,
        current,
    )
    return {
        "equivalent": (
            producer["fingerprint_sha256"]
            == current["fingerprint_sha256"]
            and not uncovered
            and not unresolved_dynamic
        ),
        "fingerprint_algorithm": SCIENTIFIC_FINGERPRINT_ALGORITHM,
        "producer_fingerprint": producer,
        "current_fingerprint": current,
        "uncovered_runtime_changes": list(uncovered),
        "unresolved_dynamic_import_modules": list(unresolved_dynamic),
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
    schema_version: int = ENVIRONMENT_EQUIVALENCE_SCHEMA_VERSION,
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
        + (
            ""
            if int(schema_version) == LEGACY_ENVIRONMENT_EQUIVALENCE_SCHEMA_VERSION
            else "-v" + str(int(schema_version))
        )
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
    schema_version = data.get("schema_version")
    if schema_version not in {
        LEGACY_ENVIRONMENT_EQUIVALENCE_SCHEMA_VERSION,
        ENVIRONMENT_EQUIVALENCE_SCHEMA_VERSION,
    }:
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
    if schema_version == ENVIRONMENT_EQUIVALENCE_SCHEMA_VERSION:
        if data.get("fingerprint_algorithm") != SCIENTIFIC_FINGERPRINT_ALGORITHM:
            raise ValueError("environment equivalence fingerprint algorithm is invalid")
        for field in ("producer_fingerprint", "current_fingerprint"):
            value = data.get(field)
            if value is not None and (
                not isinstance(value, Mapping)
                or value.get("algorithm") != SCIENTIFIC_FINGERPRINT_ALGORITHM
            ):
                raise ValueError(
                    "environment equivalence " + field + " is invalid"
                )
        uncovered = data.get("uncovered_runtime_changes")
        if not isinstance(uncovered, list) or any(
            not isinstance(value, str) or not value for value in uncovered
        ):
            raise ValueError(
                "environment equivalence uncovered runtime changes are invalid"
            )
        if bool(data.get("equivalent")) and uncovered:
            raise ValueError(
                "equivalent environment proof has uncovered runtime changes"
            )
        unresolved_dynamic = data.get("unresolved_dynamic_import_modules")
        if not isinstance(unresolved_dynamic, list) or any(
            not isinstance(value, str) or not value
            for value in unresolved_dynamic
        ):
            raise ValueError(
                "environment equivalence unresolved dynamic imports are invalid"
            )
        if bool(data.get("equivalent")) and unresolved_dynamic:
            raise ValueError(
                "equivalent environment proof has unresolved dynamic imports"
            )
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
    uncovered_runtime_changes: Tuple[str, ...] = ()
    unresolved_dynamic_import_modules: Tuple[str, ...] = ()
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
            uncovered_runtime_changes = _uncovered_runtime_changes(
                repo,
                producer_commit,
                current_commit,
                producer_fingerprint,
                current_fingerprint,
            )
            unresolved_dynamic_import_modules = (
                _unresolved_dynamic_import_modules(
                    producer_fingerprint,
                    current_fingerprint,
                )
            )
            checks["producer_code"] = (
                producer_fingerprint["fingerprint_sha256"]
                == current_fingerprint["fingerprint_sha256"]
                and not uncovered_runtime_changes
                and not unresolved_dynamic_import_modules
            )
            if not checks["producer_code"]:
                if unresolved_dynamic_import_modules:
                    reasons.append("dynamic_repository_import_unresolved")
                elif uncovered_runtime_changes:
                    reasons.append("uncovered_runtime_code_changed")
                else:
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
        "fingerprint_algorithm": SCIENTIFIC_FINGERPRINT_ALGORITHM,
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
        "uncovered_runtime_changes": list(uncovered_runtime_changes),
        "unresolved_dynamic_import_modules": list(
            unresolved_dynamic_import_modules
        ),
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
    "LEGACY_ENVIRONMENT_EQUIVALENCE_SCHEMA_VERSION",
    "SCIENTIFIC_FINGERPRINT_ALGORITHM",
    "assess_recovery_environment",
    "assess_resource_evidence_code_equivalence",
    "environment_equivalence_path",
    "read_environment_equivalence",
]
