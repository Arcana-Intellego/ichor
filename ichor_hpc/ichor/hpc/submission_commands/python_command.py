import os
import shlex
from pathlib import Path
from typing import List, Optional

import ichor.hpc.global_variables

from ichor.core.common.functools import classproperty
from ichor.hpc.global_variables import get_param_from_config
from ichor.hpc.submission_command import SubmissionCommand


class PythonEnvironmentNotFound(Exception):
    pass


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _expand_path(value: str) -> str:
    return str(Path(os.path.expanduser(os.path.expandvars(str(value)))))


def _config_value(*keys, default=None):
    return get_param_from_config(
        ichor.hpc.global_variables.ICHOR_CONFIG or {},
        ichor.hpc.global_variables.MACHINE,
        *keys,
        default=default,
    )


def _configured_modules(section: str) -> list:
    return _as_list(_config_value("software", section, "modules", default=[]))


def _configured_plumed_exports() -> List[str]:
    exports = []
    kernel_path = _config_value("software", "plumed", "kernel_path")
    library_path = _config_value("software", "plumed", "library_path")
    loader_paths = [
        _expand_path(value)
        for value in _as_list(
            _config_value("software", "python", "library_path", default=[])
        )
        if str(value).strip()
    ]

    if kernel_path:
        expanded_kernel = _expand_path(kernel_path)
        exports.append(f"export PLUMED_KERNEL={shlex.quote(expanded_kernel)}")
        if not library_path:
            library_path = str(Path(expanded_kernel).parent)

    if library_path:
        expanded_library = _expand_path(library_path)
        if expanded_library not in loader_paths:
            loader_paths.append(expanded_library)
    if loader_paths:
        exports.append(
            "export LD_LIBRARY_PATH="
            f"{shlex.quote(':'.join(loader_paths))}"
            "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        )

    return exports


class PythonCommand(SubmissionCommand):
    """A class which is used for any jobs that are going to run Python code

    :param python_script: A path object to the python script that is being ran
    :param args: A list of arguments (strings) which need to be passed to the python script via the command line
    """

    def __init__(self, python_script: Path, args: Optional[List[str]] = None):
        self.script = Path(python_script)
        self.args = args if args is not None else []

    @classproperty
    def modules(self) -> list:
        """Returns the python executable that the current ichor program is running from."""
        return _configured_modules("python") + _configured_modules("plumed")

    @property
    def data(self) -> None:
        pass

    @classproperty
    def command(self) -> str:
        """For a Python command, this loads in the virtual environment. The same python environment is going to be used
        as the one that is used for ichor."""
        # load in environment
        python_env = ichor.hpc.global_variables.CURRENT_PYTHON_ENVIRONMENT_PATH
        if python_env.uses_venv:
            env_path = python_env.venv_path.absolute()
            activate_script = env_path / "bin" / "activate"
            return f"source {shlex.quote(str(activate_script))}"
        elif python_env.uses_conda:
            env_path = python_env.conda_path.absolute()
            return f"source activate {shlex.quote(str(env_path))}"

        raise PythonEnvironmentNotFound(
            "Python environment was not found. Cannot submit Python command."
        )

    def repr(self, variables: Optional[List[str]] = None) -> str:
        """Returns a string which is then written into the submission script in order to run a python job."""
        env_lines = [PythonCommand.command]
        env_lines.extend(_configured_plumed_exports())
        args = " ".join(shlex.quote(str(arg)) for arg in self.args)
        python_script_to_run = f"python3 {shlex.quote(str(self.script))}"
        if args:
            python_script_to_run = f"{python_script_to_run} {args}"
        env_lines.append(python_script_to_run)
        return "\n".join(env_lines)
