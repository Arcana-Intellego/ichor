"""Import helpers for daemon probes that should stay quiet."""
from __future__ import annotations

import contextlib
import importlib
import io
from types import ModuleType


def quiet_import_module(module_name: str) -> ModuleType:
    """Import a module while swallowing import-time banner output.

    Some scientific helper packages print a banner at import time. That is fine
    in batch job logs, but daemon preflight/status output should stay machine
    readable and operator-focused.
    """
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        return importlib.import_module(module_name)
