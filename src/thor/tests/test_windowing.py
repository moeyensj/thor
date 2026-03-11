"""Tests for the sliding-window clustering utilities."""

import numpy as np

from thor.clustering.windowing import (
    _gnomonic_project,
    _max_linear_residual,
    compute_linearity_window,
    compute_time_windows,
)


class TestGnomonicProject:
    """Tests for the tangent-plane projection."""

    def test_center_projects_to_origin(self):
        """Tangent point should project to (0, 0)."""
        ra = np.array([45.0])
        dec = np.array([30.0])
        tx, ty = _gnomonic_project(ra, dec, 45.0, 30.0)
        assert np.isclose(tx[0], 0.0, atol=1e-10)
        assert np.isclose(ty[0], 0.0, atol=1e-10)

    def test_small_offsets(self):
        """Small offsets from center should be approximately linear."""
        ra0, dec0 = 180.0, 0.0
        # Offset by 0.01 deg in RA
        ra = np.array([180.01])
        dec = np.array([0.0])
        tx, ty = _gnomonic_project(ra, dec, ra0, dec0)
        assert np.isclose(tx[0], 0.01, atol=1e-4)
        assert np.isclose(ty[0], 0.0, atol=1e-4)

    def test_multiple_points(self):
        """Multiple points should all project."""
        ra = np.array([10.0, 10.1, 10.2, 10.3])
        dec = np.array([20.0, 20.0, 20.0, 20.0])
        tx, ty = _gnomonic_project(ra, dec, 10.15, 20.0)
        assert len(tx) == 4
        # RA offsets should be roughly symmetric around 0
        assert tx[0] < 0 and tx[-1] > 0


class TestMaxLinearResidual:
    """Tests for _max_linear_residual."""

    def test_perfectly_linear_motion(self):
        """Linear motion should have zero residual."""
        times = np.linspace(0, 10, 20)
        theta_x = 0.1 * times  # constant velocity
        theta_y = -0.05 * times
        res = _max_linear_residual(times, theta_x, theta_y, 10.0)
        assert res < 1e-10

    def test_quadratic_motion(self):
        """Quadratic motion should have nonzero residual."""
        times = np.linspace(0, 10, 20)
        # Linear + quadratic term
        theta_x = 0.1 * times + 0.001 * times**2
        theta_y = -0.05 * times
        res = _max_linear_residual(times, theta_x, theta_y, 10.0)
        assert res > 0

    def test_smaller_window_smaller_residual(self):
        """Smaller windows should have smaller residuals for curved motion."""
        times = np.linspace(0, 10, 50)
        theta_x = 0.01 * times**2  # parabolic motion
        theta_y = np.zeros_like(times)

        res_full = _max_linear_residual(times, theta_x, theta_y, 10.0)
        res_half = _max_linear_residual(times, theta_x, theta_y, 5.0)
        assert res_half < res_full

    def test_too_few_points_returns_zero(self):
        """Windows with < 3 points should not contribute to residual."""
        times = np.array([0.0, 10.0])
        theta_x = np.array([0.0, 1.0])
        theta_y = np.array([0.0, 0.0])
        res = _max_linear_residual(times, theta_x, theta_y, 10.0)
        assert res == 0.0


class TestComputeLinearityWindow:
    """Tests for compute_linearity_window."""

    def test_linear_motion_returns_full_span(self):
        """Perfectly linear motion should use the full observation span."""
        times = np.linspace(60000, 60030, 30)
        # Linear RA/Dec motion
        ra = 180.0 + 0.01 * (times - times[0])
        dec = 10.0 + 0.005 * (times - times[0])
        radius = 0.005  # 18 arcsec

        window = compute_linearity_window(ra, dec, times, radius)
        assert np.isclose(window, 30.0, atol=0.2)

    def test_curved_motion_returns_shorter_window(self):
        """Curved motion should produce a window shorter than the full span."""
        times = np.linspace(60000, 60030, 30)
        # Add significant curvature
        dt = times - times[0]
        ra = 180.0 + 0.01 * dt + 0.001 * dt**2
        dec = 10.0 + 0.005 * dt
        radius = 0.005  # 18 arcsec

        window = compute_linearity_window(ra, dec, times, radius)
        assert window < 30.0
        assert window >= 1.0  # Should be at least min_window

    def test_fewer_than_3_points_returns_full_span(self):
        """With < 3 points, should return full span (can't measure curvature)."""
        times = np.array([60000.0, 60010.0])
        ra = np.array([180.0, 180.1])
        dec = np.array([10.0, 10.05])
        radius = 0.005

        window = compute_linearity_window(ra, dec, times, radius)
        assert np.isclose(window, 10.0)

    def test_min_window_respected(self):
        """Even with extreme curvature, should not go below min_window."""
        times = np.linspace(60000, 60030, 30)
        dt = times - times[0]
        # Extreme curvature
        ra = 180.0 + 0.5 * dt**2
        dec = 10.0 + 0.5 * dt**2
        radius = 0.0001  # Tiny radius

        window = compute_linearity_window(ra, dec, times, radius, min_window=2.0)
        assert window >= 2.0

    def test_large_radius_allows_full_span(self):
        """A large radius should tolerate significant curvature."""
        times = np.linspace(60000, 60030, 30)
        dt = times - times[0]
        ra = 180.0 + 0.01 * dt + 0.0001 * dt**2
        dec = 10.0 + 0.005 * dt
        radius = 1.0  # Very large radius

        window = compute_linearity_window(ra, dec, times, radius)
        assert np.isclose(window, 30.0, atol=0.2)


class TestComputeTimeWindows:
    """Tests for compute_time_windows."""

    def test_single_window_for_short_span(self):
        """If window_size >= total span, return single window."""
        times = np.linspace(60000, 60010, 100)
        windows = compute_time_windows(times, window_size=15.0, min_arc_length=1.0)
        assert len(windows) == 1
        assert np.isclose(windows[0].t_start, 60000.0)
        assert np.isclose(windows[0].t_end, 60010.0)

    def test_multiple_windows(self):
        """Longer span should produce multiple overlapping windows."""
        times = np.linspace(60000, 60030, 300)
        windows = compute_time_windows(times, window_size=10.0, min_arc_length=1.0)
        assert len(windows) > 1

        # Windows should cover the full span
        assert windows[0].t_start <= 60000.0 + 0.1
        assert windows[-1].t_end >= 60030.0 - 0.1

    def test_windows_overlap(self):
        """Adjacent windows should overlap."""
        times = np.linspace(60000, 60030, 300)
        windows = compute_time_windows(times, window_size=10.0, min_arc_length=2.0)

        for i in range(len(windows) - 1):
            # Each window's end should be past the next window's start
            assert windows[i].t_end > windows[i + 1].t_start, (
                f"Windows {i} and {i+1} don't overlap: "
                f"[{windows[i].t_start:.2f}, {windows[i].t_end:.2f}] and "
                f"[{windows[i+1].t_start:.2f}, {windows[i+1].t_end:.2f}]"
            )

    def test_overlap_at_least_min_arc_length(self):
        """Overlap between adjacent windows should be >= min_arc_length."""
        times = np.linspace(60000, 60030, 300)
        min_arc = 3.0
        windows = compute_time_windows(times, window_size=10.0, min_arc_length=min_arc)

        for i in range(len(windows) - 1):
            overlap = windows[i].t_end - windows[i + 1].t_start
            assert overlap >= min_arc - 0.01, (
                f"Overlap {overlap:.2f} < min_arc_length {min_arc:.2f}"
            )

    def test_stride_too_small_returns_single_window(self):
        """If window is smaller than required overlap, return single window."""
        times = np.linspace(60000, 60030, 300)
        # window_size=2 but min_arc_length=5 → overlap would exceed window
        windows = compute_time_windows(times, window_size=2.0, min_arc_length=5.0)
        assert len(windows) == 1
