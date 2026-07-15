"""Shared pseudo-energy units for the ASE-to-ARIADNE boundary."""

from __future__ import annotations


def hartree_ev() -> float:
    """Return the exact Hartree-to-eV factor supplied by installed ASE."""
    from ase.units import Hartree

    return float(Hartree)


__all__ = ["hartree_ev"]
