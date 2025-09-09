"""
Tests for Kepler clock gating stage.
"""

import pytest
import numpy as np

from adam_core.coordinates import SphericalCoordinates, Origin, OriginCodes
from adam_core.time import Timestamp
from adam_core.rays import ObservationRays
from adam_core.geometry.overlap import OverlapHits

from thor.clock_gating import apply_clock_gating, Chains, ChainMembers


@pytest.fixture
def sample_overlap_hits():
    """Create sample OverlapHits for testing."""
    return OverlapHits.from_kwargs(
        det_id=["det_1", "det_2", "det_3", "det_4"],
        orbit_id=["orbit_A", "orbit_A", "orbit_B", "orbit_B"],
        seg_id=[0, 1, 0, 1],
        leaf_id=[10, 11, 20, 21],
        distance_au=[0.1, 0.2, 0.15, 0.25],
    )


@pytest.fixture
def sample_observation_rays():
    """Create sample ObservationRays for testing."""
    times = Timestamp.from_mjd([60000.1, 60000.2, 60000.3, 60000.4], scale="utc")
    origins = Origin.from_kwargs(code=[OriginCodes.EARTH.name] * 4)
    
    coords = SphericalCoordinates.from_kwargs(
        time=times,
        lon=[45.0, 90.0, 135.0, 180.0],  # RA in degrees
        lat=[10.0, -5.0, 0.0, 15.0],  # Dec in degrees
        rho=[1.0, 1.0, 1.0, 1.0],
        origin=origins,
        frame="equatorial",
    )
    
    return ObservationRays.from_kwargs(
        det_id=["det_1", "det_2", "det_3", "det_4"],
        coordinates=coords,
    )


def test_apply_clock_gating_stub_outputs_kchains(sample_overlap_hits, sample_observation_rays):
    """Test that the stub implementation outputs K-chains."""
    chains, chain_members = apply_clock_gating(
        sample_overlap_hits,
        sample_observation_rays,
        cfg=None,
    )
    
    # Should return Chains and ChainMembers tables
    assert isinstance(chains, Chains)
    assert isinstance(chain_members, ChainMembers)
    
    # Should have non-empty results
    assert len(chains) > 0
    assert len(chain_members) > 0
    
    # Check that we have one chain per orbit
    unique_orbit_ids = set(sample_overlap_hits.orbit_id.to_pylist())
    chain_orbit_ids = set(chains.orbit_id.to_pylist())
    assert chain_orbit_ids == unique_orbit_ids
    
    # Check that chain_ids are consistent between tables
    chain_ids_in_chains = set(chains.chain_id.to_pylist())
    chain_ids_in_members = set(chain_members.chain_id.to_pylist())
    assert chain_ids_in_chains == chain_ids_in_members


def test_apply_clock_gating_empty_input():
    """Test clock gating with empty input."""
    empty_hits = OverlapHits.empty()
    empty_rays = ObservationRays.empty()
    
    chains, chain_members = apply_clock_gating(empty_hits, empty_rays)
    
    assert len(chains) == 0
    assert len(chain_members) == 0


def test_apply_clock_gating_chain_properties(sample_overlap_hits, sample_observation_rays):
    """Test that chain properties are computed correctly."""
    chains, chain_members = apply_clock_gating(
        sample_overlap_hits,
        sample_observation_rays,
    )
    
    # Check chain properties
    for i in range(len(chains)):
        chain_id = chains.chain_id[i].as_py()
        orbit_id = chains.orbit_id[i].as_py()
        size = chains.size[i].as_py()
        t_min = chains.t_min[i].as_py()
        t_max = chains.t_max[i].as_py()
        
        # Find members for this chain
        members_mask = [
            chain_members.chain_id[j].as_py() == chain_id 
            for j in range(len(chain_members))
        ]
        member_times = [
            chain_members.time_mjd[j].as_py() 
            for j in range(len(chain_members)) 
            if members_mask[j]
        ]
        
        # Check consistency
        assert size == len(member_times)
        assert t_min == min(member_times)
        assert t_max == max(member_times)


def test_apply_clock_gating_unique_detections(sample_overlap_hits, sample_observation_rays):
    """Test that duplicate detections are handled correctly."""
    # Create hits with duplicate det_id for same orbit
    duplicate_hits = OverlapHits.from_kwargs(
        det_id=["det_1", "det_1", "det_2"],  # det_1 appears twice
        orbit_id=["orbit_A", "orbit_A", "orbit_A"],
        seg_id=[0, 1, 2],  # Different segments
        leaf_id=[10, 11, 12],
        distance_au=[0.1, 0.2, 0.15],
    )
    
    chains, chain_members = apply_clock_gating(
        duplicate_hits,
        sample_observation_rays,
    )
    
    # Should have one chain for orbit_A
    assert len(chains) == 1
    assert chains.orbit_id[0].as_py() == "orbit_A"
    
    # Should have only unique detections in members
    member_det_ids = chain_members.det_id.to_pylist()
    assert len(set(member_det_ids)) == len(member_det_ids)  # All unique
    assert set(member_det_ids) == {"det_1", "det_2"}
