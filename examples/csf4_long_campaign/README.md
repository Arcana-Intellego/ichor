# CSF4 Long-Campaign Example

Before launch, inspect the complete phase resource model with
`ichor-al-daemon resource-plan --campaign-dir . --all`. Submitted scripts and
per-array-member logs are retained under `.DATA/SCRIPTS/JOBS`; temporary job
data is kept under campaign-owned `.DATA/SCRATCH`, and failed leaves are
removed only through guarded `reconcile --clean-scratch --apply`. CSF4 scratch
is unbacked, so copy important committed results to backed-up storage
periodically.

This is a production-style starting point, not a first-live smoke. Run a
single-iteration campaign first, inspect the ARIADNE landing audit, queue
timings, AIMAll quality gates, and FEREBUS quality, then copy this template
into a real campaign directory.

The template keeps calibration in `record_only` mode. Schema 13 requires
finite, explicitly safety-labelled ARIADNE landings; there is no legacy bypass.
