"""pytest configuration for the ichor_hpc test suite.

Registers the ``live`` marker used by the M8.10 backend-smoke harness.
Run live tests on a host where all backends are present:

    pytest -m live tests/test_active_learning/test_live_backends_smoke.py

Run everything except live tests (the default off-cluster):

    pytest -m "not live"
"""
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: requires live cluster backends "
        "(sbatch, sacct, Gaussian, AIMAll, FEREBUS, ariadne). "
        "Skip cleanly when the binaries are absent.",
    )
