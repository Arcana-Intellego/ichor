# M16 live-parser fixture pack

Each subdirectory mirrors a canonical staging layout the live
executor parsers consume.

- `initial_quantum/` and `iter_quantum/` -- pointdirs as the
  Gaussian/AIMAll cluster jobs leave them on disk. Real .wfn + .int
  files copied from `example_files/example_points_directory/`.
- `iter_quantum_scf_failure/` -- one pointdir whose .gaussianoutput
  has had its Normal-termination line replaced with a Convergence
  failure marker, for testing the SCF-failure rejection path.
- `ferebus_staging/` -- a placeholder FEREBUS .model + config.toml.
  The .model bytes are synthetic; off-cluster tests only check
  file existence + non-empty size. CSF4 manual smoke tests the
  real pyferebus loader.
- `ferebus_staging_empty/` -- missing .model failure path.
- `ariadne_pool/seed_*/result.json` -- source-only per-seed ARIADNE results.
  Tests translate these historical fixture names into canonical
  `ariadne/seeds/seed-NNNNNN/` task outputs before invoking daemon readers.
- `polus_phase_a/` and `polus_phase_b/` -- source-only POLUS samples. Tests
  publish them under the canonical `.DATA/BOOTSTRAP/selection/` and `phase_b/`
  contracts before postprocessing.
