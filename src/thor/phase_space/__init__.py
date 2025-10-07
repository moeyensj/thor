"""
Phase space analysis module for THOR.

This module provides tools for analyzing orbital phase space coverage,
quantifying overlaps between test orbit volumes, and identifying gaps
in coverage for strategic orbit placement.
"""

from .coverage import (
    generate_even_coverage_test_orbits, 
    generate_orbits_for_target_coverage,
    generate_orbits_for_coverage_with_fixed_volumes,
    OrbitVolumes
)

__all__ = [
    "generate_even_coverage_test_orbits",
    "generate_orbits_for_target_coverage",
    "generate_orbits_for_coverage_with_fixed_volumes",
    "OrbitVolumes",
]

__version__ = "0.1.0"