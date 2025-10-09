"""
Phase space analysis module for THOR.

This module provides tools for analyzing orbital phase space coverage,
quantifying overlaps between test orbit volumes, and identifying gaps
in coverage for strategic orbit placement.
"""

from .coverage import (
    generate_even_coverage_test_orbits, 
    generate_orbit_volumes_for_target_coverage,
    generate_orbits_for_coverage_with_fixed_volumes,
    analyze_orbit_coverage_diagnostics,
    OrbitVolumes,
    PhaseSpaceBounds
)

__all__ = [
    "generate_even_coverage_test_orbits",
    "generate_orbit_volumes_for_target_coverage",
    "generate_orbits_for_coverage_with_fixed_volumes",
    "analyze_orbit_coverage_diagnostics",
    "OrbitVolumes",
    "PhaseSpaceBounds",
]

__version__ = "0.1.0"