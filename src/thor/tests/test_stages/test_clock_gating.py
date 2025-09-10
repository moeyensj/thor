"""
Tests for Kepler clock gating stage.
"""

import pytest
import numpy as np

from adam_core.coordinates import SphericalCoordinates, CartesianCoordinates, Origin, OriginCodes
from adam_core.time import Timestamp
from adam_core.observations.rays import ObservationRays
from adam_core.geometry.overlap import OverlapHits
from adam_core.geometry.anomaly import AnomalyLabels

from thor.clock_gating import apply_clock_gating, Chains, ChainMembers
from thor.clock_gating.apply_clock_gating import accept_time_consistency


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
    import numpy as np

    times = Timestamp.from_mjd([60000.1, 60000.2, 60000.3, 60000.4], scale="utc")

    sun_origin = Origin.from_kwargs(code=[OriginCodes.SUN.name] * 4)
    observer_coords = CartesianCoordinates.from_kwargs(
        x=[0.0, 0.0, 0.0, 0.0], y=[0.0, 0.0, 0.0, 0.0], z=[0.0, 0.0, 0.0, 0.0],
        vx=[0.0, 0.0, 0.0, 0.0], vy=[0.0, 0.0, 0.0, 0.0], vz=[0.0, 0.0, 0.0, 0.0],
        time=times,
        origin=sun_origin,
        frame="ecliptic",
    )

    ra_deg = np.array([45.0, 90.0, 135.0, 180.0])
    dec_deg = np.array([10.0, -5.0, 0.0, 15.0])
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    u_x = np.cos(dec) * np.cos(ra)
    u_y = np.cos(dec) * np.sin(ra)
    u_z = np.sin(dec)

    return ObservationRays.from_kwargs(
        det_id=["det_1", "det_2", "det_3", "det_4"],
        time=times,
        observer_code=["X05", "X05", "X05", "X05"],
        observer=observer_coords,
        u_x=u_x,
        u_y=u_y,
        u_z=u_z,
    )


def test_apply_clock_gating_stub_outputs_kchains(sample_overlap_hits, sample_observation_rays):
    """Test that real implementation outputs K-chains with consistent labels."""
    labels = _make_consistent_anomaly_labels(sample_overlap_hits, sample_observation_rays)
    chains, chain_members = apply_clock_gating(
        sample_overlap_hits,
        sample_observation_rays,
        labels,
        cfg={},
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
    empty_labels = AnomalyLabels.empty()
    
    chains, chain_members = apply_clock_gating(empty_hits, empty_rays, empty_labels)
    
    assert len(chains) == 0
    assert len(chain_members) == 0


def test_apply_clock_gating_chain_properties(sample_overlap_hits, sample_observation_rays):
    """Test that chain properties are computed correctly."""
    labels = _make_consistent_anomaly_labels(sample_overlap_hits, sample_observation_rays)
    chains, chain_members = apply_clock_gating(
        sample_overlap_hits,
        sample_observation_rays,
        labels,
        cfg={"min_chain_size": 1, "min_chain_days": 0.0},
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
    
    labels = _make_consistent_anomaly_labels(duplicate_hits, sample_observation_rays)
    chains, chain_members = apply_clock_gating(
        duplicate_hits,
        sample_observation_rays,
        labels,
        cfg={"min_chain_size": 1, "min_chain_days": 0.0},
    )
    
    # Should have one chain for orbit_A
    assert len(chains) == 1
    assert chains.orbit_id[0].as_py() == "orbit_A"
    
    # Should have only unique detections in members
    member_det_ids = chain_members.det_id.to_pylist()
    assert len(set(member_det_ids)) == len(member_det_ids)  # All unique
    assert set(member_det_ids) == {"det_1", "det_2"}


def _make_consistent_anomaly_labels(hits: OverlapHits, rays: ObservationRays) -> AnomalyLabels:
    """Create anomaly labels where M advances consistently with time for each orbit.

    Sets a fixed n per orbit and computes M = n * (t - t0), ensuring acceptance.
    """
    if len(hits) == 0 or len(rays) == 0:
        return AnomalyLabels.empty()

    det_to_time = dict(zip(rays.det_id.to_pylist(), rays.time.mjd().to_pylist()))
    det_ids = hits.det_id.to_pylist()
    orbit_ids = hits.orbit_id.to_pylist()
    seg_ids = hits.seg_id.to_pylist()

    # Use per-orbit n to avoid mixing rates; choose moderate mean motion
    unique_orbits = sorted(set(orbit_ids))
    orbit_n_deg_per_day = {oid: 20.0 + 3.0 * i for i, oid in enumerate(unique_orbits)}
    t0 = 60000.0

    out = {
        "det_id": [],
        "orbit_id": [],
        "seg_id": [],
        "time_mjd": [],
        "M_deg": [],
        "n_deg_per_day": [],
        "r_au": [],
        "plane_distance": [],
        "snap_error": [],
        "E_rad": [],
        "f_rad": [],
    }

    seen = set()
    for d, o, s in zip(det_ids, orbit_ids, seg_ids):
        key = (o, d)
        if key in seen:
            continue
        seen.add(key)
        if d not in det_to_time:
            continue
        t = det_to_time[d]
        n = orbit_n_deg_per_day[o]
        M = (n * (t - t0)) % 360.0
        out["det_id"].append(d)
        out["orbit_id"].append(o)
        out["seg_id"].append(s)
        out["time_mjd"].append(t)
        out["M_deg"].append(M)
        out["n_deg_per_day"].append(n)
        out["r_au"].append(1.0)
        out["plane_distance"].append(0.0)
        out["snap_error"].append(0.0)
        out["E_rad"].append(0.0)
        out["f_rad"].append(0.0)

    return AnomalyLabels.from_kwargs(**out)


def test_promotion_min_chain_size_filters(sample_overlap_hits, sample_observation_rays):
    """Chains smaller than threshold are removed."""
    labels = _make_consistent_anomaly_labels(sample_overlap_hits, sample_observation_rays)
    chains, members = apply_clock_gating(
        sample_overlap_hits,
        sample_observation_rays,
        labels,
        cfg={"min_chain_size": 3, "min_chain_days": 0.0},
    )
    assert len(chains) == 0
    assert len(members) == 0


def test_promotion_min_chain_days_filters(sample_overlap_hits, sample_observation_rays):
    """Chains shorter than the minimum span are removed."""
    labels = _make_consistent_anomaly_labels(sample_overlap_hits, sample_observation_rays)
    # sample_observation_rays spans ~0.3 days; require >1 day
    chains, members = apply_clock_gating(
        sample_overlap_hits,
        sample_observation_rays,
        labels,
        cfg={"min_chain_size": 1, "min_chain_days": 1.0},
    )
    assert len(chains) == 0
    assert len(members) == 0


def test_promotion_thresholds_accept():
    """A chain meeting size and day thresholds should be kept."""
    # Build one orbit with 6 detections over 4 days
    from adam_core.time import Timestamp
    from adam_core.coordinates import CartesianCoordinates, Origin, OriginCodes
    from adam_core.observations.rays import ObservationRays
    from adam_core.geometry.overlap import OverlapHits

    det_ids = [f"d{i}" for i in range(6)]
    times = Timestamp.from_mjd([60000.0 + i for i in range(6)], scale="utc")
    origin = Origin.from_kwargs(code=[OriginCodes.SUN.name] * 6)
    observer = CartesianCoordinates.from_kwargs(
        x=[0.0] * 6, y=[0.0] * 6, z=[0.0] * 6,
        vx=[0.0] * 6, vy=[0.0] * 6, vz=[0.0] * 6,
        time=times,
        origin=origin,
        frame="ecliptic",
    )
    # Unit vectors arbitrary
    u_x = np.ones(6)
    u_y = np.zeros(6)
    u_z = np.zeros(6)
    rays = ObservationRays.from_kwargs(
        det_id=det_ids,
        time=times,
        observer_code=["500"] * 6,
        observer=observer,
        u_x=u_x, u_y=u_y, u_z=u_z,
    )
    hits = OverlapHits.from_kwargs(
        det_id=det_ids,
        orbit_id=["O1"] * 6,
        seg_id=list(range(6)),
        leaf_id=list(range(6)),
        distance_au=[0.1] * 6,
    )
    labels = _make_consistent_anomaly_labels(hits, rays)
    chains, members = apply_clock_gating(
        hits,
        rays,
        labels,
        cfg={"min_chain_size": 6, "min_chain_days": 3.0},
    )
    assert len(chains) == 1
    assert chains.size[0].as_py() == 6
    assert abs(chains.t_max[0].as_py() - chains.t_min[0].as_py() - 5.0) < 1e-9


def test_clock_gating_benchmark_edges(benchmark):
    """Micro-benchmark for gating throughput on moderate N."""
    # Build one orbit with N detections over ~2 days
    N = 400
    from adam_core.time import Timestamp
    from adam_core.coordinates import CartesianCoordinates, Origin, OriginCodes
    from adam_core.observations.rays import ObservationRays
    from adam_core.geometry.overlap import OverlapHits

    det_ids = [f"d{i}" for i in range(N)]
    times = Timestamp.from_mjd(np.linspace(60000.0, 60002.0, N), scale="utc")
    origin = Origin.from_kwargs(code=[OriginCodes.SUN.name] * N)
    observer = CartesianCoordinates.from_kwargs(
        x=[0.0] * N, y=[0.0] * N, z=[0.0] * N,
        vx=[0.0] * N, vy=[0.0] * N, vz=[0.0] * N,
        time=times,
        origin=origin,
        frame="ecliptic",
    )
    u_x = np.ones(N)
    u_y = np.zeros(N)
    u_z = np.zeros(N)
    rays = ObservationRays.from_kwargs(
        det_id=det_ids,
        time=times,
        observer_code=["500"] * N,
        observer=observer,
        u_x=u_x, u_y=u_y, u_z=u_z,
    )
    hits = OverlapHits.from_kwargs(
        det_id=det_ids,
        orbit_id=["O1"] * N,
        seg_id=list(range(N)),
        leaf_id=list(range(N)),
        distance_au=[0.1] * N,
    )
    labels = _make_consistent_anomaly_labels(hits, rays)

    def run():
        return apply_clock_gating(hits, rays, labels, cfg={"min_chain_size": 1, "min_chain_days": 0.0})

    chains, members = benchmark(run)
    assert len(chains) >= 1


def test_time_consistency_accept_simple_case():
    """Edge should be accepted when anomalies advance exactly with n and k=0."""
    M_i = 0.0
    n = 0.5 * 2 * np.pi / 360.0  # arbitrary small n in rad/day (not used literally)
    # Better: pick simple units: let n = 0.1 rad/day
    n = 0.1
    dt = 5.0  # days
    M_j = M_i + n * dt
    accepted, k, resid = accept_time_consistency(M_i, 60000.0, M_j, 60005.0, n)
    assert accepted is True
    assert k == 0
    assert resid < 1e-6


def test_time_consistency_wraps_angle_correctly():
    """Angle wrapping should allow acceptance across 2π boundary with k chosen properly."""
    n = 0.2  # rad/day
    dt = 20.0
    # Construct M_i near +pi, M_j near -pi (wrap crossing)
    M_i = np.pi - 1e-3
    M_j_true = M_i + n * dt
    # Wrap M_j into (-pi, pi]
    M_j = (M_j_true + np.pi) % (2 * np.pi) - np.pi
    accepted, k, resid = accept_time_consistency(M_i, 60000.0, M_j, 60000.0 + dt, n)
    assert accepted is True
    assert abs(resid) < 1e-6


def test_time_consistency_requires_positive_dt_and_n():
    """Non-positive dt or n should be rejected."""
    ok, k, resid = accept_time_consistency(0.0, 60001.0, 0.1, 60000.0, 0.1)
    assert ok is False
    ok, k, resid = accept_time_consistency(0.0, 60000.0, 0.1, 60001.0, 0.0)
    assert ok is False


def test_time_consistency_tolerance_schedule():
    """Stricter tolerance should reject small residuals beyond tau, looser should accept."""
    n = 0.3
    dt = 3.0
    M_i = 0.0
    # Make M_j slightly off
    eps = 0.0005  # rad error
    M_j = (M_i + n * dt + eps)
    ok_strict, _, _ = accept_time_consistency(M_i, 60000.0, M_j, 60000.0 + dt, n, tau0_minutes=0.0, alpha_minutes_per_day=0.0)
    ok_loose, _, _ = accept_time_consistency(M_i, 60000.0, M_j, 60000.0 + dt, n, tau0_minutes=5.0, alpha_minutes_per_day=0.1)
    assert ok_strict is False
    assert ok_loose is True
