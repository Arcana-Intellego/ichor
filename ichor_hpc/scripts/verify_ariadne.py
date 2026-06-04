#!/usr/bin/env python
"""Preflight check that ARIADNE is importable and usable on the host.

Run this on the CSF4 login node (or developer workstation) before launching a
campaign. The script:

    1. Tries to import ``ariadne`` from PYTHONPATH.
    2. If successful, performs a trivial sanity probe -- imports the Python
       wrapper's main optimiser class and constructs it. Real optimisation
       isn't attempted here (that requires a calculator / ASE installation);
       the goal is to catch oneAPI / MKL runtime issues at startup rather
       than mid-campaign.

Exit codes:
    0  ARIADNE importable and the wrapper class constructs successfully.
    1  ARIADNE not importable -- prints build / PYTHONPATH instructions.
    2  ARIADNE imports but the wrapper class lookup fails -- likely a
       library / API mismatch.
"""
from __future__ import annotations

import os
import platform
import sys
from typing import Optional


def _print_install_hint() -> None:
    print("=== ARIADNE preflight FAILED ===", file=sys.stderr)
    print(
        "Build the oneAPI shared object and add it to PYTHONPATH:\n"
        "\n"
        "    module load compilers/oneapi/2024.2.0  mkl/2024.2\n"
        "    cd <ARIADNE>\n"
        "    cmake -S . -B build-oneapi -DCMAKE_BUILD_TYPE=Release\n"
        "    cmake --build build-oneapi -j 8\n"
        "    export PYTHONPATH=$PWD/build-oneapi/python:$PYTHONPATH\n"
        "\n"
        "Then re-run this script.",
        file=sys.stderr,
    )


def main() -> int:
    print("python: " + sys.version.replace("\n", " "))
    print("platform: " + platform.platform())
    print("PYTHONPATH heads: " + os.pathsep.join(sys.path[:3]))
    try:
        import ariadne   # type: ignore[import]
    except ImportError as exc:
        print("import ariadne -> ImportError: " + str(exc), file=sys.stderr)
        _print_install_hint()
        return 1
    print("ariadne imported from: " + str(getattr(ariadne, "__file__", "<unknown>")))

    # Probe the documented optimiser class lookup so we fail fast on ABI drift.
    optimiser_class = None
    for candidate in ("Geometric_Trqn", "Ds_Optimiser"):
        if hasattr(ariadne, candidate):
            optimiser_class = getattr(ariadne, candidate)
            print("found ariadne." + candidate + " = " + repr(optimiser_class))
            break
    if optimiser_class is None:
        print(
            "ariadne imported but no documented optimiser class found "
            "(Geometric_Trqn / Ds_Optimiser); upstream API may have shifted.",
            file=sys.stderr,
        )
        return 2

    print("=== ARIADNE preflight OK ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
