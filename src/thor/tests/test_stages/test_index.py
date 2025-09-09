"""
Tests for BVH indexing stage.
"""

import tempfile
from pathlib import Path

import pytest

from adam_core.coordinates import CartesianCoordinates, KeplerianCoordinates, Origin, OriginCodes
from adam_core.time import Timestamp

from thor.orbit import TestOrbits
from thor.index import build_from_test_orbits


@pytest.fixture
def sample_test_orbits():
    """Create sample TestOrbits for testing."""
    # Create some test Keplerian elements
    times = Timestamp.from_mjd([60000.0, 60000.0, 60000.0], scale="utc")
    origins = Origin.from_kwargs(code=[OriginCodes.SUN.name] * 3)
    
    kep_coords = KeplerianCoordinates.from_kwargs(
        time=times,
        a=[2.5, 3.0, 1.8],  # AU
        e=[0.1, 0.2, 0.05],
        i=[5.0, 10.0, 2.0],  # degrees
        raan=[45.0, 90.0, 180.0],  # degrees
        ap=[30.0, 60.0, 120.0],  # degrees
        M=[0.0, 45.0, 90.0],  # degrees
        origin=origins,
    )
    
    cart_coords = CartesianCoordinates.from_keplerian(kep_coords)
    
    return TestOrbits.from_kwargs(
        orbit_id=["test_orbit_1", "test_orbit_2", "test_orbit_3"],
        object_id=["obj_1", "obj_2", "obj_3"],
        coordinates=cart_coords,
    )


def test_build_from_test_orbits_smoke(sample_test_orbits):
    """Test basic functionality of build_from_test_orbits."""
    with tempfile.TemporaryDirectory() as temp_dir:
        manifest_path = build_from_test_orbits(
            sample_test_orbits,
            out_dir=temp_dir,
            max_chord_arcmin=60.0,
            target_shard_bytes=1_000_000,  # Small for testing
        )
        
        # Check that manifest file exists
        assert Path(manifest_path).exists()
        assert manifest_path.endswith("manifest.json")
        
        # Check that shard files exist
        shard_dir = Path(manifest_path).parent
        shard_files = list(shard_dir.glob("shard_*.npz"))
        assert len(shard_files) > 0


def test_build_from_test_orbits_temp_dir(sample_test_orbits):
    """Test build_from_test_orbits with temporary directory."""
    manifest_path = build_from_test_orbits(
        sample_test_orbits,
        out_dir=None,  # Use temp dir
        target_shard_bytes=1_000_000,
    )
    
    # Check that manifest file exists
    assert Path(manifest_path).exists()
    
    # Check that it's in a temp directory
    assert "thor_bvh_" in manifest_path


def test_build_from_test_orbits_empty():
    """Test build_from_test_orbits with empty input."""
    with pytest.raises(ValueError, match="No test orbits provided"):
        build_from_test_orbits([])


def test_build_from_test_orbits_multiple_tables(sample_test_orbits):
    """Test build_from_test_orbits with multiple TestOrbits tables."""
    # Split the sample into two tables
    table1 = sample_test_orbits.take([0, 1])
    table2 = sample_test_orbits.take([2])
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manifest_path = build_from_test_orbits(
            [table1, table2],
            out_dir=temp_dir,
            target_shard_bytes=1_000_000,
        )
        
        assert Path(manifest_path).exists()
