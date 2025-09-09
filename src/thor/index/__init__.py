"""
BVH indexing stage for THOR.

This module provides functionality to build BVH shards from TestOrbit objects,
creating spatial indices for efficient geometric queries.
"""

from .build_orbit_bvh import build_from_test_orbits

__all__ = ["build_from_test_orbits"]
