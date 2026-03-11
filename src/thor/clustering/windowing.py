"""
Sliding-window utilities for velocity-grid clustering.

When a test orbit's apparent motion on the sky is nonlinear over the full
observation span, the constant-velocity de-rotation used in velocity-grid
clustering breaks down. This module provides tools to:

1. Measure the linearity of the test orbit's ephemeris trajectory on the sky.
2. Compute the optimal time-window size such that the linear approximation
   error stays below the clustering radius.
3. Split observations into overlapping time windows for independent clustering.

The key idea: for each candidate window duration, we fit a linear model
(position = p0 + v * dt) to the test orbit's sky track within every possible
sub-window and measure the maximum residual. The largest window where
this residual stays below the clustering radius is the optimal window size.

Fast NEOs with high sky curvature get short windows; slow MBAs get long ones.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def _gnomonic_project(
    ra: npt.NDArray[np.float64],
    dec: npt.NDArray[np.float64],
    ra0: float,
    dec0: float,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Gnomonic (tangent-plane) projection of (RA, Dec) positions
    onto a plane centered at (ra0, dec0).

    Parameters
    ----------
    ra, dec : ndarray
        Sky positions in degrees.
    ra0, dec0 : float
        Tangent point in degrees.

    Returns
    -------
    theta_x, theta_y : ndarray
        Projected positions in degrees.
    """
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    ra0_rad = np.radians(ra0)
    dec0_rad = np.radians(dec0)

    cos_dec = np.cos(dec_rad)
    sin_dec = np.sin(dec_rad)
    cos_dec0 = np.cos(dec0_rad)
    sin_dec0 = np.sin(dec0_rad)
    cos_dra = np.cos(ra_rad - ra0_rad)

    denom = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_dra
    theta_x = np.degrees(cos_dec * np.sin(ra_rad - ra0_rad) / denom)
    theta_y = np.degrees((cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_dra) / denom)

    return theta_x, theta_y


def _max_linear_residual(
    times: npt.NDArray[np.float64],
    theta_x: npt.NDArray[np.float64],
    theta_y: npt.NDArray[np.float64],
    window_size: float,
) -> float:
    """
    Find the maximum residual from a linear fit across all sub-windows
    of the given size.

    For each starting time, selects all points within [t, t + window_size],
    fits a linear model, and records the maximum 2D residual.

    Parameters
    ----------
    times : ndarray
        Sorted MJD times.
    theta_x, theta_y : ndarray
        Projected positions (same length as times).
    window_size : float
        Window duration in days.

    Returns
    -------
    max_residual : float
        Maximum 2D residual in degrees across all sub-windows.
    """
    n = len(times)
    max_res = 0.0

    for i in range(n):
        t_start = times[i]
        t_end = t_start + window_size

        # Find points within this window using sorted order
        j = i
        while j < n and times[j] <= t_end:
            j += 1
        if j - i < 3:
            continue

        t_w = times[i:j] - t_start
        x_w = theta_x[i:j]
        y_w = theta_y[i:j]

        # Fit linear model: pos = a + b * t
        A = np.vstack([np.ones(j - i), t_w]).T
        cx, _, _, _ = np.linalg.lstsq(A, x_w, rcond=None)
        cy, _, _, _ = np.linalg.lstsq(A, y_w, rcond=None)

        res_x = x_w - A @ cx
        res_y = y_w - A @ cy
        res = np.sqrt(res_x**2 + res_y**2)
        max_res = max(max_res, float(res.max()))

    return max_res


def compute_linearity_window(
    ephemeris_ra: npt.NDArray[np.float64],
    ephemeris_dec: npt.NDArray[np.float64],
    ephemeris_times_mjd: npt.NDArray[np.float64],
    radius: float,
    min_window: float = 1.0,
    precision: float = 0.1,
) -> float:
    """
    Compute the largest time window over which the test orbit's apparent
    sky motion is linear to within the clustering radius.

    Projects the ephemeris RA/Dec track onto a common gnomonic tangent
    plane, then binary-searches for the largest window duration where
    the maximum linear-fit residual stays below the clustering radius.

    Parameters
    ----------
    ephemeris_ra : ndarray
        Test orbit RA at each observation time, in degrees.
    ephemeris_dec : ndarray
        Test orbit Dec at each observation time, in degrees.
    ephemeris_times_mjd : ndarray
        MJD times corresponding to RA/Dec positions.
    radius : float
        Clustering radius in degrees.
    min_window : float, optional
        Minimum window size in days. [Default = 1.0]
    precision : float, optional
        Binary search precision in days. [Default = 0.1]

    Returns
    -------
    window_size : float
        Optimal window duration in days. Returns the full time span
        if the entire trajectory is sufficiently linear.
    """
    # Sort by time
    sort_idx = np.argsort(ephemeris_times_mjd)
    times = ephemeris_times_mjd[sort_idx]
    ra = ephemeris_ra[sort_idx]
    dec = ephemeris_dec[sort_idx]

    dt_total = times[-1] - times[0]

    if len(times) < 3:
        logger.info(
            f"Only {len(times)} ephemeris points; cannot measure curvature. "
            f"Using full span ({dt_total:.2f} days)."
        )
        return dt_total

    # Project onto common gnomonic tangent plane centered at midpoint
    mid = len(ra) // 2
    theta_x, theta_y = _gnomonic_project(ra, dec, float(ra[mid]), float(dec[mid]))

    # Check if the full span is already linear enough
    full_res = _max_linear_residual(times, theta_x, theta_y, dt_total)
    if full_res <= radius:
        logger.info(
            f"Full span ({dt_total:.2f} days) is linear to within radius "
            f"(max residual {full_res:.6f} deg < radius {radius:.6f} deg). "
            f"No windowing needed."
        )
        return dt_total

    logger.info(
        f"Full-span max residual ({full_res:.6f} deg) exceeds clustering radius "
        f"({radius:.6f} deg). Binary-searching for optimal window..."
    )

    # Binary search: find largest window where max_residual <= radius
    lo = min_window
    hi = dt_total

    # Check that min_window is actually sufficient
    min_res = _max_linear_residual(times, theta_x, theta_y, lo)
    if min_res > radius:
        logger.warning(
            f"Even minimum window ({lo:.2f} days) has residual {min_res:.6f} deg > "
            f"radius {radius:.6f} deg. Using minimum window anyway."
        )
        return lo

    while hi - lo > precision:
        mid_val = (lo + hi) / 2.0
        res = _max_linear_residual(times, theta_x, theta_y, mid_val)
        if res <= radius:
            lo = mid_val
        else:
            hi = mid_val

    logger.info(f"Optimal linearity window: {lo:.2f} days (full span: {dt_total:.2f} days)")
    return lo


@dataclass
class TimeWindow:
    """A time window for clustering."""

    t_start: float
    """Start time in MJD."""

    t_end: float
    """End time in MJD."""


def compute_time_windows(
    times_mjd: npt.NDArray[np.float64],
    window_size: float,
    min_arc_length: float,
) -> List[TimeWindow]:
    """
    Compute overlapping time windows that cover the full observation span.

    Windows overlap by at least ``min_arc_length`` to avoid splitting
    clusters that span a window boundary.

    Parameters
    ----------
    times_mjd : ndarray
        MJD times of all observations.
    window_size : float
        Window duration in days (from compute_linearity_window).
    min_arc_length : float
        Minimum arc length in days. Used as the minimum overlap.

    Returns
    -------
    windows : list of TimeWindow
        Overlapping time windows covering the full observation span.
    """
    t_min = float(times_mjd.min())
    t_max = float(times_mjd.max())
    dt_total = t_max - t_min

    if window_size >= dt_total:
        return [TimeWindow(t_start=t_min, t_end=t_max)]

    # Overlap ensures clusters near boundaries are captured in both windows
    overlap = max(min_arc_length, window_size * 0.2)
    stride = window_size - overlap

    if stride <= 0:
        # Window is smaller than the overlap requirement; use a single window
        logger.warning(
            f"Window size ({window_size:.2f} days) is smaller than required overlap "
            f"({overlap:.2f} days). Using a single window."
        )
        return [TimeWindow(t_start=t_min, t_end=t_max)]

    windows = []
    start = t_min
    while start < t_max:
        end = min(start + window_size, t_max)
        windows.append(TimeWindow(t_start=start, t_end=end))
        if end >= t_max:
            break
        start += stride

    return windows
