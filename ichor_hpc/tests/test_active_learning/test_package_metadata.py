import ast
import configparser
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIRS = {
    "ichor_core": REPO_ROOT / "ichor_core",
    "ichor_hpc": REPO_ROOT / "ichor_hpc",
    "ichor_cli": REPO_ROOT / "ichor_cli",
}
IMPORT_TO_PACKAGE = {
    "consolemenu": "console-menu",
    "concurrent_log_handler": "concurrent-log-handler",
    "ruamel": "ruamel.yaml",
    "sqlalchemy": "SQLAlchemy",
    "yaml": "pyyaml",
}
EXTERNAL_BACKENDS = {"ariadne", "pyferebus"}


def _normalise_package_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _declared_runtime_dependencies(package_dir: Path) -> set[str]:
    config = configparser.ConfigParser()
    config.read(package_dir / "setup.cfg")
    deps = set()
    for line in config.get("options", "install_requires").splitlines():
        line = line.strip()
        if not line:
            continue
        package_name = re.split(r"[<>=;\[]", line, maxsplit=1)[0].strip()
        deps.add(_normalise_package_name(package_name))
    return deps


def _console_scripts(package_dir: Path) -> dict[str, str]:
    config = configparser.ConfigParser()
    config.read(package_dir / "setup.cfg")
    scripts: dict[str, str] = {}
    if not config.has_section("options.entry_points"):
        return scripts
    for line in config.get("options.entry_points", "console_scripts").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        name, target = line.split("=", 1)
        scripts[name.strip()] = target.strip()
    return scripts


def _stdlib_modules() -> set[str]:
    if not hasattr(sys, "stdlib_module_names"):
        raise RuntimeError("ICHOR requires Python 3.11 or newer")
    modules = set(sys.stdlib_module_names)
    modules.update(sys.builtin_module_names)
    return modules


def _production_imports(source_root: Path) -> dict[str, set[str]]:
    imports: dict[str, set[str]] = {}
    for py_file in source_root.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        rel = str(py_file.relative_to(REPO_ROOT))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            elif _literal_dynamic_import(node):
                names = [node.args[0].value.split(".")[0]]
            for name in names:
                imports.setdefault(name, set()).add(f"{rel}:{node.lineno}")
    return imports


def _literal_dynamic_import(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or not node.args:
        return False
    if not isinstance(node.args[0], ast.Constant) or not isinstance(
        node.args[0].value, str
    ):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in {"__import__", "import_module"}
    if isinstance(func, ast.Attribute):
        return func.attr == "import_module"
    return False


def _third_party_imports(package_name: str) -> dict[str, set[str]]:
    stdlib = _stdlib_modules()
    imports = _production_imports(PACKAGE_DIRS[package_name] / "ichor")
    return {
        name: locations
        for name, locations in imports.items()
        if name != "ichor" and name not in stdlib and not name.startswith("_")
    }


def test_runtime_dependency_metadata_covers_production_imports():
    missing = []
    for package_name, package_dir in PACKAGE_DIRS.items():
        declared = _declared_runtime_dependencies(package_dir)
        for import_name, locations in _third_party_imports(package_name).items():
            if import_name in EXTERNAL_BACKENDS:
                continue
            package = IMPORT_TO_PACKAGE.get(import_name, import_name)
            if _normalise_package_name(package) not in declared:
                missing.append(
                    f"{package_name}: import {import_name!r} should declare "
                    f"{package!r}; seen at {sorted(locations)[:3]}"
                )
    assert missing == []


def test_csf4_critical_runtime_dependencies_are_declared():
    core_deps = _declared_runtime_dependencies(PACKAGE_DIRS["ichor_core"])
    hpc_deps = _declared_runtime_dependencies(PACKAGE_DIRS["ichor_hpc"])
    cli_deps = _declared_runtime_dependencies(PACKAGE_DIRS["ichor_cli"])

    assert {"ase", "xtb", "plumed", "rdkit", "tqdm"} <= core_deps
    assert {
        "numpy",
        "pandas",
        "ase",
        "xtb",
        "plumed",
        "portalocker",
        "threadpoolctl",
        "tqdm",
        "ruamel-yaml",
    } <= hpc_deps
    assert {"console-menu", "termcolor", "ase"} <= cli_deps


def test_active_learning_console_scripts_are_declared():
    scripts = _console_scripts(PACKAGE_DIRS["ichor_cli"])

    assert scripts["ichor-al-daemon"] == "ichor.hpc.active_learning.cli:main"
    assert (
        scripts["ichor-al-benchmark-acquisition-gradient"]
        == "ichor.hpc.active_learning.benchmark.acquisition_gradient:main"
    )


def test_all_ichor_packages_require_python_311():
    for package_dir in PACKAGE_DIRS.values():
        config = configparser.ConfigParser()
        config.read(package_dir / "setup.cfg")
        assert config.get("options", "python_requires") == ">=3.11"
        assert config.get("bdist_wheel", "python-tag") == "py3"
