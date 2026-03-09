import numpy as np
import pyarrow as pa
import pytest
from adam_core.coordinates import CartesianCoordinates, Origin, SphericalCoordinates
from adam_core.observers import Observers
from adam_core.orbits import Ephemeris
from adam_core.time import Timestamp

from ..clustering.velocity_grid import VelocityGridBase, estimate_window_duration
from ..orbit import TestOrbitEphemeris
from ..projections.covariances import ProjectionCovariances
from ..projections.gnomonic import GnomonicCoordinates
from ..range_and_transform import TransformedDetections


def _make_ephemeris(
    n_times=30,
    mjd0=59000.0,
    accel_x=0.0,
    accel_y=0.0,
    base_sigma=1e-3,
):
    """Build a TestOrbitEphemeris with known quadratic covariance growth.

    The position covariance sigma grows as:
        sigma(t) = base_sigma + 0.5 * accel * t^2

    This models the differential curvature that causes nearby objects
    to spread faster than linearly in the co-rotating gnomonic frame.

    Parameters
    ----------
    accel_x, accel_y : float
        Quadratic growth rate of position sigma (deg/day^2).
        The effective acceleration is sqrt(accel_x^2 + accel_y^2).
    base_sigma : float
        Base position sigma at t=0 (degrees).
    """
    times_mjd = mjd0 + np.arange(n_times, dtype=np.float64)
    t = Timestamp.from_mjd(times_mjd, scale="utc")
    dt = np.arange(n_times, dtype=np.float64)

    # Gnomonic positions are zero (test orbit is at origin in co-rotating frame)
    theta_x = np.zeros(n_times)
    theta_y = np.zeros(n_times)

    # Position covariance grows quadratically to model differential curvature
    sigma_x = base_sigma + 0.5 * abs(accel_x) * dt**2
    sigma_y = base_sigma + 0.5 * abs(accel_y) * dt**2

    cov = np.zeros((n_times, 4, 4), dtype=np.float64)
    cov[:, 0, 0] = sigma_x**2
    cov[:, 1, 1] = sigma_y**2
    cov[:, 2, 2] = 1e-8
    cov[:, 3, 3] = 1e-8
    covariances = ProjectionCovariances.from_matrix(cov)
    origin = Origin.from_kwargs(code=["SUN"] * n_times)

    gnomonic = GnomonicCoordinates.from_kwargs(
        time=t,
        theta_x=theta_x,
        theta_y=theta_y,
        covariance=covariances,
        origin=origin,
        frame="testorbit",
    )

    spherical = SphericalCoordinates.from_kwargs(
        lon=np.zeros(n_times),
        lat=np.zeros(n_times),
        time=t,
        origin=origin,
        frame="equatorial",
    )
    ephemeris = Ephemeris.from_kwargs(
        orbit_id=["test_orbit"] * n_times,
        coordinates=spherical,
    )

    observer_coords = CartesianCoordinates.from_kwargs(
        x=np.ones(n_times),
        y=np.zeros(n_times),
        z=np.zeros(n_times),
        vx=np.zeros(n_times),
        vy=np.zeros(n_times),
        vz=np.zeros(n_times),
        time=t,
        origin=origin,
        frame="ecliptic",
    )
    observers = Observers.from_kwargs(
        code=["500"] * n_times,
        coordinates=observer_coords,
    )

    rotation_matrices = [np.eye(6).flatten().tolist() for _ in range(n_times)]
    return TestOrbitEphemeris.from_kwargs(
        id=[f"state_{i}" for i in range(n_times)],
        test_orbit_id=["test_orbit"] * n_times,
        ephemeris=ephemeris,
        observer=observers,
        gnomonic=gnomonic,
        gnomonic_rotation_matrix=rotation_matrices,
    )


def _make_detections(n_obs, mjd0=59000.0, arc_days=29.0):
    """Build minimal TransformedDetections spanning a given arc."""
    obs_times = np.linspace(mjd0, mjd0 + arc_days, n_obs)
    obs_t = Timestamp.from_mjd(obs_times, scale="utc")

    cov = np.zeros((n_obs, 4, 4), dtype=np.float64)
    cov[:, 0, 0] = 1e-6
    cov[:, 1, 1] = 1e-6
    cov[:, 2, 2] = 1e-8
    cov[:, 3, 3] = 1e-8
    covariances = ProjectionCovariances.from_matrix(cov)
    origin = Origin.from_kwargs(code=["SUN"] * n_obs)

    rng = np.random.default_rng(42)
    coords = GnomonicCoordinates.from_kwargs(
        time=obs_t,
        theta_x=rng.normal(scale=1e-3, size=n_obs),
        theta_y=rng.normal(scale=1e-3, size=n_obs),
        covariance=covariances,
        origin=origin,
        frame="testorbit",
    )

    nights = ((obs_times - mjd0)).astype(np.int64)
    return TransformedDetections.from_kwargs(
        id=[f"obs_{i}" for i in range(n_obs)],
        test_orbit_id=["test_orbit"] * n_obs,
        night=nights,
        coordinates=coords,
        state_id=[f"state_{i % 30}" for i in range(n_obs)],
    )


class TestEstimateWindowDuration:

    def test_zero_curvature_returns_inf(self):
        """Constant covariance (zero acceleration) should return inf."""
        ephem = _make_ephemeris(accel_x=0.0, accel_y=0.0)
        result = estimate_window_duration(ephem, cluster_radius=0.005)
        assert np.isinf(result)

    def test_known_curvature(self):
        """With known quadratic covariance growth, verify window formula.

        The position sigma grows as sigma(t) = base + 0.5 * accel * t^2.
        The quadratic fit to sigma(t) gives coefficient a = 0.5 * accel,
        so the effective acceleration = 2 * a = accel.
        Window = sqrt(8 * threshold / accel) where threshold = safety * radius.
        """
        accel = 1e-4
        ephem = _make_ephemeris(accel_x=accel, accel_y=0.0)
        result = estimate_window_duration(ephem, cluster_radius=0.005, safety_factor=0.5)
        expected = np.sqrt(8 * 0.0025 / accel)
        assert abs(result - expected) < 1.0  # within 1 day

    def test_high_curvature_gives_short_window(self):
        """Higher acceleration should give shorter window."""
        ephem_low = _make_ephemeris(accel_x=1e-4, accel_y=0.0)
        ephem_high = _make_ephemeris(accel_x=1e-3, accel_y=0.0)
        dur_low = estimate_window_duration(ephem_low, cluster_radius=0.005)
        dur_high = estimate_window_duration(ephem_high, cluster_radius=0.005)
        assert dur_high < dur_low

    def test_min_window_enforced(self):
        """Very high curvature should be clamped to min_window."""
        ephem = _make_ephemeris(accel_x=1.0, accel_y=0.0)  # extreme curvature
        result = estimate_window_duration(ephem, cluster_radius=0.005, min_window=3.0)
        assert result == 3.0

    def test_fewer_than_3_points_returns_inf(self):
        """With fewer than 3 ephemeris points, can't fit quadratic."""
        ephem = _make_ephemeris(n_times=2)
        result = estimate_window_duration(ephem, cluster_radius=0.005)
        assert np.isinf(result)

    def test_larger_radius_gives_longer_window(self):
        """Larger cluster radius should allow longer windows."""
        ephem = _make_ephemeris(accel_x=1e-4, accel_y=0.0)
        dur_small = estimate_window_duration(ephem, cluster_radius=0.001)
        dur_large = estimate_window_duration(ephem, cluster_radius=0.01)
        assert dur_large > dur_small

    def test_2d_curvature(self):
        """Larger accel in the dominant axis gives shorter window.

        When accel_y is larger, the max eigenvalue of the position
        covariance grows faster, giving a shorter estimated window.
        """
        ephem_small = _make_ephemeris(accel_x=1e-4, accel_y=0.0)
        ephem_large = _make_ephemeris(accel_x=0.0, accel_y=2e-4)
        dur_small = estimate_window_duration(ephem_small, cluster_radius=0.005)
        dur_large = estimate_window_duration(ephem_large, cluster_radius=0.005)
        assert dur_large < dur_small


class TestComputeTimeWindows:

    def _make_base(self, window_duration=None):
        """Create a minimal VelocityGridBase subclass for testing."""
        from ..clustering.dbscan import VelocityGridDBSCAN

        return VelocityGridDBSCAN(
            radius=0.005,
            min_obs=6,
            min_arc_length=1.0,
            min_nights=3,
            window_duration=window_duration,
        )

    def test_no_windowing_returns_single_window(self):
        """With window_duration=None and no ephemeris, returns full arc."""
        alg = self._make_base(window_duration=None)
        td = _make_detections(100, arc_days=29.0)
        windows = alg._compute_time_windows(td, test_orbit_ephemeris=None)
        assert len(windows) == 1
        assert abs(windows[0][1] - windows[0][0] - 29.0) < 0.01

    def test_explicit_window_larger_than_arc(self):
        """Window duration >= arc returns single window."""
        alg = self._make_base(window_duration=50.0)
        td = _make_detections(100, arc_days=29.0)
        windows = alg._compute_time_windows(td, test_orbit_ephemeris=None)
        assert len(windows) == 1

    def test_explicit_window_splits_arc(self):
        """A 7-day window on a 29-day arc should produce multiple windows."""
        alg = self._make_base(window_duration=7.0)
        td = _make_detections(300, arc_days=29.0)
        windows = alg._compute_time_windows(td, test_orbit_ephemeris=None)

        # With 7-day windows and 3.5-day step:
        # starts at 0, 3.5, 7, 10.5, 14, 17.5, 21, 22 (last to cover end)
        assert len(windows) > 1

        # All windows should be 7 days
        for w_start, w_end in windows:
            assert abs((w_end - w_start) - 7.0) < 0.01

        # First window starts at arc start
        assert abs(windows[0][0] - 59000.0) < 0.01

        # Last window covers the end of the arc
        assert windows[-1][1] >= 59000.0 + 29.0 - 0.01

    def test_50_percent_overlap(self):
        """Windows should overlap by 50%."""
        alg = self._make_base(window_duration=10.0)
        td = _make_detections(300, arc_days=29.0)
        windows = alg._compute_time_windows(td, test_orbit_ephemeris=None)

        # Check step size between consecutive windows is window_duration / 2
        for i in range(len(windows) - 2):  # Exclude last which may be shifted
            step = windows[i + 1][0] - windows[i][0]
            assert abs(step - 5.0) < 0.01

    def test_auto_window_with_curved_ephemeris(self):
        """Auto window should produce multiple windows for curved ephemeris."""
        alg = self._make_base(window_duration=None)
        td = _make_detections(300, arc_days=29.0)
        ephem = _make_ephemeris(n_times=30, accel_x=1e-3, accel_y=0.0)
        windows = alg._compute_time_windows(td, test_orbit_ephemeris=ephem)
        # High curvature should trigger windowing
        assert len(windows) > 1

    def test_auto_window_with_straight_ephemeris(self):
        """Auto window with zero curvature should return single window."""
        alg = self._make_base(window_duration=None)
        td = _make_detections(300, arc_days=29.0)
        ephem = _make_ephemeris(n_times=30, accel_x=0.0, accel_y=0.0)
        windows = alg._compute_time_windows(td, test_orbit_ephemeris=ephem)
        assert len(windows) == 1


class TestSliceDetectionsToWindow:

    def test_slices_by_time(self):
        """Verify detections are filtered to the window time range."""
        td = _make_detections(100, mjd0=59000.0, arc_days=29.0)
        sliced = VelocityGridBase._slice_detections_to_window(td, 59005.0, 59012.0)

        mjd = sliced.coordinates.time.rescale("utc").mjd().to_numpy(zero_copy_only=False)
        assert len(sliced) > 0
        assert mjd.min() >= 59005.0
        assert mjd.max() <= 59012.0

    def test_full_range_returns_all(self):
        """Window spanning full arc returns all detections."""
        td = _make_detections(100, mjd0=59000.0, arc_days=29.0)
        sliced = VelocityGridBase._slice_detections_to_window(td, 58999.0, 59030.0)
        assert len(sliced) == len(td)

    def test_empty_window(self):
        """Window with no detections returns empty."""
        td = _make_detections(100, mjd0=59000.0, arc_days=29.0)
        sliced = VelocityGridBase._slice_detections_to_window(td, 60000.0, 60010.0)
        assert len(sliced) == 0
