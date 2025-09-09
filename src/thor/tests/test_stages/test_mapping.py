"""
Tests for observation-to-orbit mapping stage.
"""

import tempfile
from pathlib import Path

import pytest
import numpy as np

from adam_core.coordinates import CartesianCoordinates, KeplerianCoordinates, Origin, OriginCodes, SphericalCoordinates
from adam_core.time import Timestamp
from adam_core.rays import ObservationRays

from thor.orbit import TestOrbits
from thor.index import build_from_test_orbits
from thor.mapping import map_observations_to_test_orbits


@pytest.fixture
def sample_test_orbits():
    """Create sample TestOrbits for testing."""
    times = Timestamp.from_mjd([60000.0, 60000.0], scale="utc")
    origins = Origin.from_kwargs(code=[OriginCodes.SUN.name] * 2)
    
    kep_coords = KeplerianCoordinates.from_kwargs(
        time=times,
        a=[2.5, 3.0],  # AU
        e=[0.1, 0.2],
        i=[5.0, 10.0],  # degrees
        raan=[45.0, 90.0],  # degrees
        ap=[30.0, 60.0],  # degrees
        M=[0.0, 45.0],  # degrees
        origin=origins,
    )
    
    cart_coords = CartesianCoordinates.from_keplerian(kep_coords)
    
    return TestOrbits.from_kwargs(
        orbit_id=["test_orbit_1", "test_orbit_2"],
        object_id=["obj_1", "obj_2"],
        coordinates=cart_coords,
    )


@pytest.fixture
def sample_observation_rays():
    """Create sample ObservationRays for testing."""
    times = Timestamp.from_mjd([60000.1, 60000.2], scale="utc")
    origins = Origin.from_kwargs(code=[OriginCodes.EARTH.name] * 2)
    
    # Create some spherical coordinates (RA, Dec from Earth)
    coords = SphericalCoordinates.from_kwargs(
        time=times,
        lon=[45.0, 90.0],  # RA in degrees
        lat=[10.0, -5.0],  # Dec in degrees
        rho=[1.0, 1.0],  # Distance (not used for rays)
        origin=origins,
        frame="equatorial",
    )
    
    return ObservationRays.from_kwargs(
        det_id=["det_1", "det_2"],
        coordinates=coords,
    )


def test_map_observations_to_test_orbits_smoke(sample_test_orbits, sample_observation_rays):
    """Test basic functionality of map_observations_to_test_orbits."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Build BVH index
        manifest_path = build_from_test_orbits(
            sample_test_orbits,
            out_dir=temp_dir,
            target_shard_bytes=1_000_000,
        )
        
        # Map observations
        hits = map_observations_to_test_orbits(
            sample_observation_rays,
            manifest_path=manifest_path,
            guard_arcmin=60.0,  # Large guard for testing
            use_ray=False,  # Use local for simplicity
        )
        
        # Should return OverlapHits table (may be empty)
        assert hasattr(hits, 'det_id')
        assert hasattr(hits, 'orbit_id')
        assert hasattr(hits, 'seg_id')
        assert hasattr(hits, 'leaf_id')


def test_map_observations_ray_vs_local(sample_test_orbits, sample_observation_rays):
    """Test that Ray and local implementations give consistent results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Build BVH index
        manifest_path = build_from_test_orbits(
            sample_test_orbits,
            out_dir=temp_dir,
            target_shard_bytes=1_000_000,
        )
        
        # Map observations with local
        hits_local = map_observations_to_test_orbits(
            sample_observation_rays,
            manifest_path=manifest_path,
            guard_arcmin=60.0,
            use_ray=False,
        )
        
        # Map observations with Ray
        hits_ray = map_observations_to_test_orbits(
            sample_observation_rays,
            manifest_path=manifest_path,
            guard_arcmin=60.0,
            use_ray=True,
            ray_batch_size=1000,
        )
        
        # Results should be consistent
        assert len(hits_local) == len(hits_ray)
        if len(hits_local) > 0:
            # Sort both by det_id and orbit_id for comparison
            local_sorted = hits_local.sort_by([("det_id", "ascending"), ("orbit_id", "ascending")])
            ray_sorted = hits_ray.sort_by([("det_id", "ascending"), ("orbit_id", "ascending")])
            
            assert local_sorted.det_id.to_pylist() == ray_sorted.det_id.to_pylist()
            assert local_sorted.orbit_id.to_pylist() == ray_sorted.orbit_id.to_pylist()
