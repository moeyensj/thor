"""
Observation-to-orbit mapping stage for THOR.

This module provides functionality to map observations to test orbits using
BVH spatial indices for efficient geometric queries.
"""

from .map_observations import map_observations_to_test_orbits

__all__ = ["map_observations_to_test_orbits"]
