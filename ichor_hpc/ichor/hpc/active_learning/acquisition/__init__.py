"""Adversarial acquisition driver components.

Holds the ARIADNE-facing wrappers around 'ichor.core.adversarial':

- :mod: '.rigid_projection' -- mass-weighted rigid-body projector used at the
  ASE-calculator boundary to remove unphysical translation / rotation
  components from FD gradients.
- :mod: '.ase_calculator' -- :class:'AdversarialASECalculator' (ASE Calculator
  subclass) wrapping :class:'SeedLocalAdversarialAcquisition' for ARIADNE.
- :mod:'.ariadne_runner' -- single-seed driver ('optimise_seed') returning a
  structured :class: 'AriadneRunResult'. Live-ARIADNE driver landing provides full mock-mode coverage.
- :mod: '.seed_selection' -- explore/exploit seed picker
  (random + variance-weighted) for the per-iteration adversarial batch.
"""
