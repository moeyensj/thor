import numpy as np
import pyarrow as pa
from adam_core.coordinates import CartesianCoordinates, Origin
from adam_core.observers import Observers
from adam_core.orbits import Ephemeris
from adam_core.time import Timestamp

from thor.clustering.tracklets import (
    TrackletMembers,
    Tracklets,
    _UnionFind,
    form_tracklets,
)
from thor.orbit import TestOrbitEphemeris
from thor.projections import GnomonicCoordinates
from thor.projections.covariances import ProjectionCovariances
from thor.range_and_transform import TransformedDetections


def _make_transformed_detections(
    obs_ids,
    nights,
    state_ids,
    theta_x,
    theta_y,
    mjds,
    test_orbit_id="test_orbit_001",
):
    """Helper to create TransformedDetections from arrays."""
    n = len(obs_ids)
    times = Timestamp.from_mjd(mjds, scale="utc")
    coords = GnomonicCoordinates.from_kwargs(
        theta_x=theta_x,
        theta_y=theta_y,
        time=times,
        origin=Origin.from_kwargs(code=pa.repeat("SUN", n)),
        frame="testorbit",
    )
    return TransformedDetections.from_kwargs(
        id=obs_ids,
        test_orbit_id=pa.repeat(test_orbit_id, n),
        night=nights,
        coordinates=coords,
        state_id=state_ids,
    )


def _make_test_orbit_ephemeris(
    mjds,
    sigma_vx=0.01,
    sigma_vy=0.01,
    sigma_x=0.001,
    sigma_y=0.001,
):
    """Helper to create a minimal TestOrbitEphemeris with velocity covariances."""
    n = len(mjds)
    times = Timestamp.from_mjd(mjds, scale="utc")

    # Build 4x4 covariance matrices for gnomonic coordinates
    cov_matrices = np.zeros((n, 4, 4))
    cov_matrices[:, 0, 0] = sigma_x**2
    cov_matrices[:, 1, 1] = sigma_y**2
    cov_matrices[:, 2, 2] = sigma_vx**2
    cov_matrices[:, 3, 3] = sigma_vy**2

    gnomonic = GnomonicCoordinates.from_kwargs(
        theta_x=np.zeros(n),
        theta_y=np.zeros(n),
        vtheta_x=np.zeros(n),
        vtheta_y=np.zeros(n),
        time=times,
        covariance=ProjectionCovariances.from_matrix(cov_matrices),
        origin=Origin.from_kwargs(code=pa.repeat("SUN", n)),
        frame="testorbit",
    )

    # Build minimal Ephemeris and Observers required by TestOrbitEphemeris
    cart = CartesianCoordinates.from_kwargs(
        x=np.ones(n),
        y=np.zeros(n),
        z=np.zeros(n),
        vx=np.zeros(n),
        vy=np.full(n, 0.01),
        vz=np.zeros(n),
        time=times,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=pa.repeat("SUN", n)),
    )
    ephemeris = Ephemeris.from_kwargs(
        orbit_id=pa.repeat("test_orbit_001", n),
        object_id=pa.repeat("obj", n),
        coordinates=cart,
        aberrated_coordinates=cart,
    )
    observers = Observers.from_kwargs(
        code=pa.repeat("500", n),
        coordinates=cart,
    )

    # Gnomonic rotation matrix: 6x6 identity flattened to 36 elements
    rot_matrix = np.tile(np.eye(6).flatten(), (n, 1)).tolist()

    return TestOrbitEphemeris.from_kwargs(
        id=[f"state_{i}" for i in range(n)],
        test_orbit_id=pa.repeat("test_orbit_001", n),
        ephemeris=ephemeris,
        observer=observers,
        gnomonic=gnomonic,
        gnomonic_rotation_matrix=rot_matrix,
    )


class TestUnionFind:
    def test_basic(self):
        uf = _UnionFind(5)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(1, 3)
        assert uf.find(0) == uf.find(3)
        assert uf.find(0) != uf.find(4)

    def test_singleton(self):
        uf = _UnionFind(3)
        assert uf.find(0) != uf.find(1)
        assert uf.find(1) != uf.find(2)


class TestFormTracklets:

    def test_empty_detections(self):
        td = TransformedDetections.empty()
        toe = _make_test_orbit_ephemeris([60000.0, 60001.0])
        tracklets, members = form_tracklets(td, toe)
        assert len(tracklets) == 0
        assert len(members) == 0

    def test_single_state_per_night(self):
        """All observations on one night with one exposure → all singletons."""
        td = _make_transformed_detections(
            obs_ids=["a", "b", "c"],
            nights=[1, 1, 1],
            state_ids=["s1", "s1", "s1"],
            theta_x=[0.0, 0.01, 0.02],
            theta_y=[0.0, 0.01, 0.02],
            mjds=[60000.0, 60000.0, 60000.0],
        )
        toe = _make_test_orbit_ephemeris([60000.0])
        tracklets, members = form_tracklets(td, toe)

        # All should be singletons
        assert len(tracklets) == 3
        assert len(members) == 3
        assert all(n == 1 for n in tracklets.num_obs.to_pylist())

    def test_basic_pair_formation(self):
        """Two exposures on the same night, one object with consistent velocity."""
        # Object at (0, 0) at t=60000.0 and (0.001, 0.001) at t=60000.05
        # Velocity = (0.02, 0.02) deg/day — well within sigma_v=0.1
        td = _make_transformed_detections(
            obs_ids=["obj_t1", "noise_t1", "obj_t2", "noise_t2"],
            nights=[1, 1, 1, 1],
            state_ids=["s1", "s1", "s2", "s2"],
            theta_x=[0.0, 0.5, 0.001, -0.5],
            theta_y=[0.0, 0.5, 0.001, -0.5],
            mjds=[60000.0, 60000.0, 60000.05, 60000.05],
        )
        toe = _make_test_orbit_ephemeris([60000.0, 60000.05], sigma_vx=0.1, sigma_vy=0.1)
        tracklets, members = form_tracklets(td, toe, min_obs=2, mahalanobis_distance=3.0)

        # The "obj" pair should form a tracklet; "noise" pair too far away
        tracklet_list = tracklets.num_obs.to_pylist()
        assert 2 in tracklet_list, f"Expected a 2-obs tracklet, got {tracklet_list}"

        # Verify the 2-obs tracklet contains the right observations
        multi_tracklet_ids = [
            tid for tid, n in zip(tracklets.tracklet_id.to_pylist(), tracklet_list) if n == 2
        ]
        for tid in multi_tracklet_ids:
            member_mask = members.select("tracklet_id", tid)
            obs_ids = set(member_mask.obs_id.to_pylist())
            assert obs_ids == {"obj_t1", "obj_t2"}

    def test_velocity_filter_rejects_fast_pairs(self):
        """Pairs whose velocity exceeds the Mahalanobis threshold are rejected."""
        # Large displacement in short time → high velocity
        td = _make_transformed_detections(
            obs_ids=["a1", "a2"],
            nights=[1, 1],
            state_ids=["s1", "s2"],
            theta_x=[0.0, 1.0],  # 1 degree in 0.05 days = 20 deg/day
            theta_y=[0.0, 0.0],
            mjds=[60000.0, 60000.05],
        )
        toe = _make_test_orbit_ephemeris([60000.0, 60000.05], sigma_vx=0.01, sigma_vy=0.01)
        tracklets, members = form_tracklets(td, toe, min_obs=2, mahalanobis_distance=3.0)

        # Both should be singletons because velocity (20 deg/day) >> 3 * 0.01 = 0.03 deg/day
        assert len(tracklets) == 2
        assert all(n == 1 for n in tracklets.num_obs.to_pylist())

    def test_connected_components_three_states(self):
        """Object observed in 3 states on same night → single 3-obs tracklet."""
        # Consistent velocity across all three states
        v = 0.01  # deg/day
        dt1 = 0.04  # 58 min
        dt2 = 0.08  # 115 min
        td = _make_transformed_detections(
            obs_ids=["obj_t1", "obj_t2", "obj_t3"],
            nights=[1, 1, 1],
            state_ids=["s1", "s2", "s3"],
            theta_x=[0.0, v * dt1, v * dt2],
            theta_y=[0.0, v * dt1, v * dt2],
            mjds=[60000.0, 60000.0 + dt1, 60000.0 + dt2],
        )
        toe = _make_test_orbit_ephemeris(
            [60000.0, 60000.0 + dt1, 60000.0 + dt2], sigma_vx=0.1, sigma_vy=0.1
        )
        tracklets, members = form_tracklets(td, toe, min_obs=2, mahalanobis_distance=3.0)

        # Should form one 3-observation tracklet
        assert 3 in tracklets.num_obs.to_pylist()

    def test_multi_night_independence(self):
        """Tracklets form independently per night — no cross-night linking."""
        td = _make_transformed_detections(
            obs_ids=["n1_a", "n1_b", "n2_a", "n2_b"],
            nights=[1, 1, 2, 2],
            state_ids=["s1", "s2", "s3", "s4"],
            theta_x=[0.0, 0.0005, 0.0, 0.0005],
            theta_y=[0.0, 0.0005, 0.0, 0.0005],
            mjds=[60000.0, 60000.05, 60001.0, 60001.05],
        )
        toe = _make_test_orbit_ephemeris(
            [60000.0, 60000.05, 60001.0, 60001.05], sigma_vx=0.1, sigma_vy=0.1
        )
        tracklets, members = form_tracklets(td, toe, min_obs=2, mahalanobis_distance=3.0)

        # Should form two separate 2-obs tracklets (one per night)
        multi = [n for n in tracklets.num_obs.to_pylist() if n == 2]
        assert len(multi) == 2

        # Each tracklet should have observations from only one night
        for tid in tracklets.tracklet_id.to_pylist():
            member_rows = members.select("tracklet_id", tid)
            obs_ids = member_rows.obs_id.to_pylist()
            # Find the night of each member observation
            member_nights = set()
            for oid in obs_ids:
                idx = td.id.to_pylist().index(oid)
                member_nights.add(td.night.to_pylist()[idx])
            assert len(member_nights) == 1, f"Tracklet {tid} spans multiple nights: {member_nights}"

    def test_tracklet_centroid_computation(self):
        """Verify centroid position, time, and velocity are correctly computed."""
        x1, y1, t1 = 0.0, 0.0, 60000.0
        x2, y2, t2 = 0.002, 0.004, 60000.05
        td = _make_transformed_detections(
            obs_ids=["a", "b"],
            nights=[1, 1],
            state_ids=["s1", "s2"],
            theta_x=[x1, x2],
            theta_y=[y1, y2],
            mjds=[t1, t2],
        )
        toe = _make_test_orbit_ephemeris([t1, t2], sigma_vx=1.0, sigma_vy=1.0)
        tracklets, members = form_tracklets(td, toe, min_obs=2, mahalanobis_distance=10.0)

        multi = [i for i, n in enumerate(tracklets.num_obs.to_pylist()) if n == 2]
        assert len(multi) == 1
        idx = multi[0]

        # Check centroid
        cx = tracklets.theta_x.to_pylist()[idx]
        cy = tracklets.theta_y.to_pylist()[idx]
        np.testing.assert_allclose(cx, (x1 + x2) / 2, atol=1e-10)
        np.testing.assert_allclose(cy, (y1 + y2) / 2, atol=1e-10)

        # Check velocity
        expected_vx = (x2 - x1) / (t2 - t1)
        expected_vy = (y2 - y1) / (t2 - t1)
        vx = tracklets.vtheta_x.to_pylist()[idx]
        vy = tracklets.vtheta_y.to_pylist()[idx]
        np.testing.assert_allclose(vx, expected_vx, rtol=1e-6)
        np.testing.assert_allclose(vy, expected_vy, rtol=1e-6)

    def test_singleton_tracklets_have_null_velocity(self):
        """Singleton tracklets should have null velocity."""
        td = _make_transformed_detections(
            obs_ids=["a"],
            nights=[1],
            state_ids=["s1"],
            theta_x=[0.0],
            theta_y=[0.0],
            mjds=[60000.0],
        )
        toe = _make_test_orbit_ephemeris([60000.0])
        tracklets, members = form_tracklets(td, toe)

        assert len(tracklets) == 1
        assert tracklets.vtheta_x.to_pylist()[0] is None
        assert tracklets.vtheta_y.to_pylist()[0] is None

    def test_every_observation_accounted_for(self):
        """Every input observation should appear in exactly one tracklet."""
        np.random.seed(42)
        n_obs = 50
        td = _make_transformed_detections(
            obs_ids=[f"obs_{i}" for i in range(n_obs)],
            nights=[1] * 25 + [2] * 25,
            state_ids=["s1"] * 12 + ["s2"] * 13 + ["s3"] * 12 + ["s4"] * 13,
            theta_x=np.random.uniform(-0.01, 0.01, n_obs),
            theta_y=np.random.uniform(-0.01, 0.01, n_obs),
            mjds=[60000.0] * 12 + [60000.05] * 13 + [60001.0] * 12 + [60001.05] * 13,
        )
        toe = _make_test_orbit_ephemeris(
            [60000.0, 60000.05, 60001.0, 60001.05], sigma_vx=0.5, sigma_vy=0.5
        )
        tracklets, members = form_tracklets(td, toe, min_obs=2, mahalanobis_distance=5.0)

        # Every obs_id should appear exactly once in members
        member_obs_ids = members.obs_id.to_pylist()
        input_obs_ids = td.id.to_pylist()
        assert sorted(member_obs_ids) == sorted(input_obs_ids)

    def test_max_velocity_override(self):
        """max_velocity parameter should override the covariance-derived bound."""
        # Pair with velocity = 0.02 deg/day
        td = _make_transformed_detections(
            obs_ids=["a", "b"],
            nights=[1, 1],
            state_ids=["s1", "s2"],
            theta_x=[0.0, 0.001],
            theta_y=[0.0, 0.0],
            mjds=[60000.0, 60000.05],
        )
        # Very wide covariance would normally allow this pair
        toe = _make_test_orbit_ephemeris([60000.0, 60000.05], sigma_vx=1.0, sigma_vy=1.0)

        # With tight max_velocity, the pair should be rejected
        tracklets, _ = form_tracklets(td, toe, min_obs=2, max_velocity=0.01)
        assert all(n == 1 for n in tracklets.num_obs.to_pylist())

        # Without max_velocity, the pair should form
        tracklets2, _ = form_tracklets(td, toe, min_obs=2, max_velocity=None)
        assert 2 in tracklets2.num_obs.to_pylist()
