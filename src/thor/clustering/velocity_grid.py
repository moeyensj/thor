"""
Shared base class and utilities for velocity-grid clustering algorithms.

Velocity-grid clustering works by sweeping over a grid of (vx, vy) velocity
hypotheses. At each grid point, observations are de-rotated by that velocity
and 2D point clustering (DBSCAN, Hotspot2D, etc.) is applied to find groups.

This module provides:
- `VelocityGridBase`: abstract base class with all shared logic
- `calculate_clustering_parameters_from_covariance`: covariance-aware
  parameter calculation for velocity grid and clustering radius
"""

import abc
import logging
import multiprocessing as mp
import time
import uuid
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import quivr as qv
import ray
from adam_core.ray_cluster import initialize_use_ray
from adam_core.utils.iter import _iterate_chunks

from ..orbit import TestOrbitEphemeris
from ..range_and_transform import TransformedDetections
from .data import ClusterMembers, Clusters, drop_duplicate_clusters
from .metrics import filter_clusters_by_length
from .tiling import compute_auto_tile_size, compute_tile_grid, extract_tile_observations
from .windowing import TimeWindow, compute_linearity_window, compute_time_windows

logger = logging.getLogger(__name__)


def _estimate_astrometric_precision(
    transformed_detections: TransformedDetections,
    fallback: float = 1.0 / 3600,
) -> float:
    """Estimate astrometric precision from observation covariances.

    Returns fallback (default 1 arcsec) if covariances are NaN or unavailable.
    """
    try:
        covs = transformed_detections.coordinates.covariance.to_matrix()
        pos_var = covs[:, 0, 0]
        finite_mask = np.isfinite(pos_var) & (pos_var > 0)
        if np.any(finite_mask):
            return float(np.sqrt(np.median(pos_var[finite_mask])))
    except Exception:
        pass
    return fallback


def calculate_clustering_parameters_from_covariance(
    test_orbit_ephemeris: TestOrbitEphemeris,
    transformed_detections: Union[TransformedDetections, ray.ObjectRef],
    mahalanobis_distance: float = 3.0,
    velocity_bin_separation: float = 2.0,
    radius_multiplier: float = 5.0,
    density_multiplier: float = 2.5,
    min_radius: float = 0.01 / 3600,
    max_radius: float = 0.05,
    min_bins: int = 10,
    max_bins: int = 1000,
    whiten: bool = False,
    astrometric_precision: Optional[float] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, dict]:
    """
    Calculate clustering parameters (velocity grid and radius) from test orbit
    ephemeris covariances in the co-rotating gnomonic frame.

    In the co-rotating frame, the test orbit has zero velocity. The velocity
    grid is centered at (0, 0) and extends to the specified Mahalanobis distance
    based on velocity uncertainties.

    The clustering radius is computed from two competing bounds:
    - Combined lower bound: radius_multiplier x sqrt(sigma_astro^2 + sigma_orbital^2),
      where sigma_orbital is the max positional sigma from covariance eigenvalues and
      sigma_astro is the astrometric precision. This ensures the radius captures both
      astrometric scatter and orbital uncertainty.
    - Upper bound: density_multiplier x characteristic separation, where the
      density is computed from the total number of observations divided by the
      observation footprint area. This caps the radius to limit contamination
      from background detections. The upper bound is applied subject to an
      astrometric floor of radius_multiplier x sigma_astro.

    The ephemeris is automatically filtered to only include times within the
    observation time range, ensuring parameters are calculated from relevant
    covariances.

    Parameters
    ----------
    test_orbit_ephemeris : TestOrbitEphemeris
        Test orbit ephemeris with gnomonic coordinates containing covariances.
    transformed_detections : TransformedDetections or ray.ObjectRef
        Transformed detections. Used to determine the time range for filtering ephemeris.
    mahalanobis_distance : float, optional
        Mahalanobis distance threshold for velocity grid and covariance area.
        [Default = 3.0]
    velocity_bin_separation : float, optional
        Separation between adjacent velocity bins in units of clustering radius.
        [Default = 2.0]
    radius_multiplier : float, optional
        Multiplier on the maximum positional sigma for the radius lower bound.
        [Default = 5.0]
    density_multiplier : float, optional
        Multiplier on the characteristic inter-source separation for the radius
        upper bound. [Default = 2.5]
    min_radius : float, optional
        Minimum radius in degrees. [Default = 0.01/3600]
    max_radius : float, optional
        Maximum radius in degrees. [Default = 0.05]
    min_bins : int, optional
        Minimum number of bins per dimension. [Default = 10]
    max_bins : int, optional
        Maximum number of bins per dimension. [Default = 1000]
    whiten : bool, optional
        If True, also report clustering parameters in whitened (sigma) units.
        [Default = False]
    astrometric_precision : float, optional
        Astrometric precision in degrees. If None, estimated from observation
        covariances (falling back to 1 arcsec if covariances are unavailable).
        [Default = None]

    Returns
    -------
    vx : np.ndarray
        X-velocity grid values (flattened), centered at 0.
    vy : np.ndarray
        Y-velocity grid values (flattened), centered at 0.
    radius : float
        Effective clustering radius in degrees.
    metadata : dict
        Dictionary with computed parameters for logging.
    """
    logger.info("Calculating clustering parameters from test orbit ephemeris covariances...")

    # Square the Mahalanobis distance for calculations
    mahalanobis_distance_sq = mahalanobis_distance**2

    # Get transformed detections from Ray if needed
    if isinstance(transformed_detections, ray.ObjectRef):
        transformed_detections = ray.get(transformed_detections)

    # Filter ephemeris to only include times that match observation times
    obs_times_mjd = (
        transformed_detections.coordinates.time.rescale("utc").mjd().to_numpy(zero_copy_only=False)
    )
    obs_time_min = obs_times_mjd.min()
    obs_time_max = obs_times_mjd.max()

    ephemeris_gnomonic = test_orbit_ephemeris.gnomonic
    ephem_times_mjd = ephemeris_gnomonic.time.rescale("utc").mjd().to_numpy(zero_copy_only=False)

    # Filter to ephemeris points within observation time range
    time_mask = (ephem_times_mjd >= obs_time_min) & (ephem_times_mjd <= obs_time_max)
    if not np.any(time_mask):
        raise ValueError(
            f"No ephemeris points found in observation time range "
            f"[{obs_time_min:.2f}, {obs_time_max:.2f}] MJD"
        )

    ephemeris_gnomonic = ephemeris_gnomonic.apply_mask(time_mask)
    logger.info(
        f"Filtered ephemeris to {len(ephemeris_gnomonic)}/{len(test_orbit_ephemeris.gnomonic)} points "
        f"spanning observation time range [{obs_time_min:.2f}, {obs_time_max:.2f}] MJD"
    )

    n_obs = len(transformed_detections)  # Total observations across all times
    n_times = len(ephemeris_gnomonic)  # Number of unique observation times

    # Get time span
    times = ephemeris_gnomonic.time.mjd().to_numpy(zero_copy_only=False)
    dt_arc = np.max(times) - np.min(times)

    # Extract covariances from ephemeris
    covariances = ephemeris_gnomonic.covariance.to_matrix()

    # Check if covariances are valid
    if np.all(np.isnan(covariances)):
        raise ValueError("All covariance values are NaN. Cannot compute clustering parameters.")

    # Calculate mean covariance
    mean_cov = np.nanmean(covariances, axis=0)

    # Check if mean covariance is valid
    if np.any(np.isnan(mean_cov)) or np.any(~np.isfinite(mean_cov)):
        raise ValueError("Mean covariance matrix contains NaN or infinite values.")

    # Extract position and velocity covariances
    pos_cov = mean_cov[0:2, 0:2]  # theta_x, theta_y
    vel_cov = mean_cov[2:4, 2:4]  # vtheta_x, vtheta_y

    # Pre-compute scale factors for optional whitening metadata
    pos_scales = np.sqrt(np.maximum(np.diag(pos_cov), 1e-18))
    vel_scales = np.sqrt(np.maximum(np.diag(vel_cov), 1e-18))
    pos_scale_rms = np.sqrt(np.mean(pos_scales**2))

    # === Calculate effective clustering radius ===

    # --- Lower bound: positional scatter after de-rotation ---
    pos_covariances = covariances[:, 0:2, 0:2]

    # Filter to valid (finite) covariance matrices
    valid_mask = np.all(np.isfinite(pos_covariances), axis=(1, 2))
    if not np.any(valid_mask):
        raise ValueError("No valid positional covariance matrices found.")

    valid_pos_covs = pos_covariances[valid_mask]
    all_eigvals = np.linalg.eigvalsh(valid_pos_covs)  # (n_valid, 2)
    all_eigvals = np.maximum(all_eigvals, 1e-18)
    max_sigma_pos = np.sqrt(np.max(all_eigvals))
    sigma_orbital = max_sigma_pos

    # --- Astrometric precision ---
    if astrometric_precision is not None:
        sigma_astro = astrometric_precision
    else:
        sigma_astro = _estimate_astrometric_precision(transformed_detections)

    # Combined radius: quadrature sum of astrometric and orbital uncertainties
    radius_combined = radius_multiplier * np.sqrt(sigma_astro**2 + sigma_orbital**2)
    # Astrometric floor: never shrink below the pure astrometric term
    astro_floor = radius_multiplier * sigma_astro

    # --- Upper bound: detection density within the observation footprint ---
    obs_x = transformed_detections.coordinates.theta_x.to_numpy(zero_copy_only=False)
    obs_y = transformed_detections.coordinates.theta_y.to_numpy(zero_copy_only=False)
    obs_finite = np.isfinite(obs_x) & np.isfinite(obs_y)
    obs_x = obs_x[obs_finite]
    obs_y = obs_y[obs_finite]

    obs_extent_x = np.ptp(obs_x) if len(obs_x) > 1 else 0.0
    obs_extent_y = np.ptp(obs_y) if len(obs_y) > 1 else 0.0
    obs_area = obs_extent_x * obs_extent_y

    if obs_area > 0:
        density = n_obs / obs_area
        r_sep = 1.0 / np.sqrt(density) if density > 0 else np.inf
        r_upper = density_multiplier * r_sep
    else:
        density = 0.0
        r_sep = np.inf
        r_upper = np.inf

    # --- Final radius: combined bound, density cap (respecting astro floor), then hard clip ---
    if np.isfinite(r_upper):
        radius = max(min(radius_combined, r_upper), astro_floor)
    else:
        radius = radius_combined
    radius = np.clip(radius, min_radius, max_radius)

    logger.info(f"Effective clustering radius: {radius:.6f} deg ({radius * 3600:.3f} arcsec)")
    logger.info(
        f"  - sigma_orbital (max eigval): {sigma_orbital:.6f} deg ({sigma_orbital * 3600:.3f} arcsec)"
    )
    logger.info(f"  - sigma_astro: {sigma_astro:.6f} deg ({sigma_astro * 3600:.3f} arcsec)")
    logger.info(f"  - radius_combined: {radius_combined:.6f} deg")
    logger.info(
        f"  - Observation footprint: {obs_extent_x:.6f} x {obs_extent_y:.6f} deg ({obs_area:.6e} deg^2)"
    )
    logger.info(f"  - Observation density: {density:.2f} obs/deg^2")
    logger.info(f"  - Characteristic separation (r_sep): {r_sep:.6f} deg")
    logger.info(f"  - Upper bound (r_upper): {r_upper:.6f} deg")
    logger.info(f"  - Total observations: {n_obs} across {n_times} times")
    logger.info(f"  - Time span: {dt_arc:.3f} days")

    # === Calculate velocity grid ===
    # Extract standard deviations
    sigma_vx = np.sqrt(vel_cov[0, 0])
    sigma_vy = np.sqrt(vel_cov[1, 1])

    # --- Enforce minimum velocity sigma ---
    if dt_arc > 0 and mahalanobis_distance > 0:
        min_sigma_v = min_bins * velocity_bin_separation * radius / (2 * mahalanobis_distance * dt_arc)
    else:
        min_sigma_v = 0.0

    sigma_vx_orig = sigma_vx
    sigma_vy_orig = sigma_vy
    sigma_vx = max(sigma_vx, min_sigma_v)
    sigma_vy = max(sigma_vy, min_sigma_v)

    # If either sigma was inflated, scale vel_cov so the Mahalanobis
    # ellipse filtering uses the wider values (preserves correlation).
    if sigma_vx > sigma_vx_orig or sigma_vy > sigma_vy_orig:
        scale_x = sigma_vx / sigma_vx_orig if sigma_vx_orig > 0 else 1.0
        scale_y = sigma_vy / sigma_vy_orig if sigma_vy_orig > 0 else 1.0
        vel_cov = vel_cov.copy()
        vel_cov[0, 0] *= scale_x**2
        vel_cov[1, 1] *= scale_y**2
        vel_cov[0, 1] *= scale_x * scale_y
        vel_cov[1, 0] *= scale_x * scale_y

        logger.info(
            f"Velocity sigma inflated to minimum: "
            f"σ_vx {sigma_vx_orig:.6e} → {sigma_vx:.6e}, "
            f"σ_vy {sigma_vy_orig:.6e} → {sigma_vy:.6e} deg/day "
            f"(min_sigma_v={min_sigma_v:.6e})"
        )

    if np.isnan(sigma_vx) or np.isnan(sigma_vy) or sigma_vx <= 0 or sigma_vy <= 0:
        raise ValueError(f"Invalid velocity sigmas: σ_vx={sigma_vx}, σ_vy={sigma_vy}")

    logger.info(f"Velocity uncertainties: σ_vx={sigma_vx:.6f}, σ_vy={sigma_vy:.6f} deg/day")

    # Calculate rectangular bounds at Mahalanobis distance centered at zero
    vx_min = -mahalanobis_distance * sigma_vx
    vx_max = mahalanobis_distance * sigma_vx
    vy_min = -mahalanobis_distance * sigma_vy
    vy_max = mahalanobis_distance * sigma_vy

    logger.info(
        f"Velocity grid edges ({mahalanobis_distance:.1f}-sigma): "
        f"vx=[{vx_min:.6f}, {vx_max:.6f}], vy=[{vy_min:.6f}, {vy_max:.6f}] deg/day"
    )

    # Calculate number of bins
    n_vx_bins = int(np.ceil((vx_max - vx_min) * dt_arc / (velocity_bin_separation * radius)))
    n_vy_bins = int(np.ceil((vy_max - vy_min) * dt_arc / (velocity_bin_separation * radius)))

    # Apply limits
    n_vx_bins = np.clip(n_vx_bins, min_bins, max_bins)
    n_vy_bins = np.clip(n_vy_bins, min_bins, max_bins)

    # Log the derived velocity bin spacing
    dv_x = (vx_max - vx_min) / n_vx_bins if n_vx_bins > 0 else 0
    dv_y = (vy_max - vy_min) / n_vy_bins if n_vy_bins > 0 else 0
    dx_max = dv_x * dt_arc
    dy_max = dv_y * dt_arc
    logger.info(f"Velocity grid bins: vx_bins={n_vx_bins}, vy_bins={n_vy_bins}")
    logger.info(f"Velocity bin spacing: dv_x={dv_x:.6f}, dv_y={dv_y:.6f} deg/day")
    logger.info(f"Position offset at dt_max={dt_arc:.3f} days: dx={dx_max:.6f}, dy={dy_max:.6f} deg")
    logger.info(f"Bin separation in radii: x={dx_max/radius:.2f}, y={dy_max/radius:.2f}")

    # Create rectangular velocity grid
    vx_grid = np.linspace(vx_min, vx_max, n_vx_bins)
    vy_grid = np.linspace(vy_min, vy_max, n_vy_bins)
    # Include the zero velocity point
    if not np.any(np.isclose(vx_grid, 0.0, atol=1e-6)):
        vx_grid = np.sort(np.append(vx_grid, 0.0))
    if not np.any(np.isclose(vy_grid, 0.0, atol=1e-6)):
        vy_grid = np.sort(np.append(vy_grid, 0.0))

    vxx, vyy = np.meshgrid(vx_grid, vy_grid)

    # Flatten the grid
    vxx_flat = vxx.flatten()
    vyy_flat = vyy.flatten()

    # Filter to elliptical region using Mahalanobis distance
    try:
        vel_cov_inv = np.linalg.inv(vel_cov)
        velocity_vectors = np.stack([vxx_flat, vyy_flat], axis=1)
        mahalanobis_sq = np.sum(velocity_vectors @ vel_cov_inv * velocity_vectors, axis=1)
        velocity_mask = mahalanobis_sq <= mahalanobis_distance_sq

        n_total = len(velocity_mask)
        n_inside = np.sum(velocity_mask)
        logger.info(
            f"Velocity grid points: {n_inside}/{n_total} inside "
            f"{mahalanobis_distance:.1f}-sigma ellipse ({100*n_inside/n_total:.1f}%)"
        )

        # Apply mask
        vxx_flat = vxx_flat[velocity_mask]
        vyy_flat = vyy_flat[velocity_mask]
    except np.linalg.LinAlgError:
        logger.warning("Could not invert velocity covariance matrix. Using all grid points.")

    # Optional whitened representations for metadata/consumers
    radius_whitened = radius / pos_scale_rms if whiten and pos_scale_rms > 0 else None
    vx_whitened = vxx_flat / vel_scales[0] if whiten and vel_scales[0] > 0 else None
    vy_whitened = vyy_flat / vel_scales[1] if whiten and vel_scales[1] > 0 else None

    # Prepare metadata
    metadata = {
        "n_obs": n_obs,
        "n_times": n_times,
        "dt_arc": dt_arc,
        "sigma_vx": sigma_vx,
        "sigma_vy": sigma_vy,
        "vx_min": vx_min,
        "vx_max": vx_max,
        "vy_min": vy_min,
        "vy_max": vy_max,
        "n_vx_bins": n_vx_bins,
        "n_vy_bins": n_vy_bins,
        "n_velocity_points": len(vxx_flat),
        "dv_x": dv_x,
        "dv_y": dv_y,
        "sigma_astro_deg": float(sigma_astro),
        "sigma_orbital_deg": float(sigma_orbital),
        "max_sigma_pos": float(max_sigma_pos),
        "radius_multiplier": radius_multiplier,
        "radius_combined": float(radius_combined),
        "obs_area": obs_area,
        "obs_extent_x": obs_extent_x,
        "obs_extent_y": obs_extent_y,
        "density": density,
        "density_multiplier": density_multiplier,
        "r_sep": r_sep,
        "r_upper": r_upper,
        "radius": radius,
        "min_radius": min_radius,
        "max_radius": max_radius,
        "mahalanobis_distance": mahalanobis_distance,
        "whiten": whiten,
        "pos_scales_deg": pos_scales.tolist(),
        "vel_scales_deg_per_day": vel_scales.tolist(),
        "radius_whitened": radius_whitened,
        "vx_whitened_range": (
            (float(np.min(vx_whitened)), float(np.max(vx_whitened)))
            if whiten and vx_whitened is not None and len(vx_whitened) > 0
            else None
        ),
        "vy_whitened_range": (
            (float(np.min(vy_whitened)), float(np.max(vy_whitened)))
            if whiten and vy_whitened is not None and len(vy_whitened) > 0
            else None
        ),
    }

    return vxx_flat, vyy_flat, radius, metadata


# --- Shared velocity-grid worker functions ---


def _cluster_velocity(
    obs_ids: npt.ArrayLike,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    dt: npt.NDArray[np.float64],
    nights: npt.NDArray[np.int64],
    vx: float,
    vy: float,
    radius: float,
    min_obs: int,
    min_arc_length: float,
    min_nights: int,
    point_cluster_fn: Callable[
        [npt.NDArray[np.float64], float, int],
        List[npt.NDArray[np.int64]],
    ],
    alg_name: str = "clustering",
    tracklet_member_obs_ids: Optional[dict] = None,
) -> Tuple[Clusters, ClusterMembers]:
    """
    Cluster THOR projection at a single velocity hypothesis.

    De-rotates observations by (vx, vy), then applies the given 2D point
    clustering function to find groups.

    Parameters
    ----------
    obs_ids : array-like
        Observation IDs (or tracklet IDs when tracklets are used).
    x, y : ndarray
        Gnomonic coordinates in degrees (or tracklet centroids).
    dt : ndarray
        Time offsets from first observation in days.
    nights : ndarray
        Observing night indices.
    vx, vy : float
        Velocity hypothesis in deg/day.
    radius : float
        Clustering radius (eps).
    min_obs : int
        Minimum cluster size.
    min_arc_length : float
        Minimum arc length in days.
    min_nights : int
        Minimum number of unique nights.
    point_cluster_fn : callable
        2D clustering function with signature ``(points, eps, min_samples) -> list[ndarray]``.
    alg_name : str
        Algorithm name for log messages.
    tracklet_member_obs_ids : dict, optional
        When clustering on tracklet centroids, a mapping from tracklet_id
        to list of obs_ids. If provided, cluster membership is expanded
        from tracklets back to individual observations.

    Returns
    -------
    clusters : Clusters
    cluster_members : ClusterMembers
    """
    logger.debug(f"cluster: vx={vx} vy={vy} n_obs={len(obs_ids)}")
    xx = x - vx * dt
    yy = y - vy * dt

    # Drop NaNs before clustering
    finite_mask = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(dt)
    if not np.all(finite_mask):
        n_drop = np.size(finite_mask) - np.count_nonzero(finite_mask)
        logger.warning(f"Dropping {n_drop} observations with NaN coordinates before clustering.")
        xx = xx[finite_mask]
        yy = yy[finite_mask]
        dt = dt[finite_mask]
        nights = nights[finite_mask]
        obs_ids = obs_ids[finite_mask]

    if len(xx) < min_obs:
        return Clusters.empty(), ClusterMembers.empty()

    # Spatial tiling pre-filter
    x_extent = xx.max() - xx.min() if len(xx) > 0 else 0
    y_extent = yy.max() - yy.min() if len(yy) > 0 else 0
    tile_size = compute_auto_tile_size(radius, len(xx), x_extent, y_extent)
    border_width = 2.0 * radius

    tiles = compute_tile_grid(xx, yy, tile_size, border_width, min_obs)

    all_raw_clusters = []
    if len(tiles) == 1 and tiles[0].n_obs_core == len(xx):
        # Single tile covering everything - skip tiling overhead
        X = np.stack((xx, yy), 1)
        all_raw_clusters = point_cluster_fn(X, radius, min_obs)
    else:
        logger.info(
            f"{alg_name} tiling: {len(tiles)} active tiles "
            f"(tile_size={tile_size:.6f}, border={border_width:.6f})"
        )
        for tile in tiles:
            tile_idx = extract_tile_observations(tile, xx, yy)
            if len(tile_idx) < min_obs:
                continue
            X_tile = np.stack((xx[tile_idx], yy[tile_idx]), 1)
            tile_clusters = point_cluster_fn(X_tile, radius, min_obs)
            # Remap local tile indices to global indices
            for cluster_local_idx in tile_clusters:
                all_raw_clusters.append(tile_idx[cluster_local_idx])

    clusters = all_raw_clusters
    if clusters:
        sizes = [len(c) for c in clusters]
        logger.info(
            f"{alg_name} found {len(clusters)} raw clusters "
            f"(sizes: min={min(sizes)}, max={max(sizes)}, median={np.median(sizes):.0f})"
        )
    clusters, arc_lengths = filter_clusters_by_length(
        clusters,
        dt,
        nights,
        min_obs,
        min_arc_length,
        min_nights,
    )

    if len(clusters) == 0:
        return Clusters.empty(), ClusterMembers.empty()

    cluster_ids = []
    cluster_num_obs = []
    cluster_members_cluster_ids = []
    cluster_members_obs_ids = []
    for cluster in clusters:
        cid = uuid.uuid4().hex
        ids_in_cluster = obs_ids[cluster]

        if tracklet_member_obs_ids is not None:
            # Expand tracklet IDs to observation IDs
            expanded = []
            for tid in ids_in_cluster:
                expanded.extend(tracklet_member_obs_ids[tid])
            obs_ids_i = np.array(expanded)
        else:
            obs_ids_i = ids_in_cluster

        num_obs = len(obs_ids_i)

        cluster_ids.append(cid)
        cluster_num_obs.append(num_obs)
        cluster_members_cluster_ids.append(np.full(num_obs, cid))
        cluster_members_obs_ids.append(obs_ids_i)

    clusters_table = Clusters.from_kwargs(
        cluster_id=cluster_ids,
        vtheta_x=np.full(len(cluster_ids), vx),
        vtheta_y=np.full(len(cluster_ids), vy),
        arc_length=arc_lengths,
        num_obs=cluster_num_obs,
    )

    cluster_members_table = ClusterMembers.from_kwargs(
        cluster_id=np.concatenate(cluster_members_cluster_ids).tolist(),
        obs_id=np.concatenate(cluster_members_obs_ids).tolist(),
    )

    return clusters_table, cluster_members_table


def _cluster_velocity_find_worker(
    vx: npt.NDArray[np.float64],
    vy: npt.NDArray[np.float64],
    transformed_detections: TransformedDetections,
    radius: float,
    min_obs: int,
    min_arc_length: float,
    min_nights: int,
    point_cluster_fn: Callable,
    alg_name: str = "clustering",
    tracklets: Optional[object] = None,
    tracklet_members: Optional[object] = None,
) -> Tuple[Clusters, ClusterMembers]:
    """
    Worker that clusters a batch of velocity hypotheses and deduplicates.

    Parameters
    ----------
    vx, vy : ndarray
        Arrays of velocity hypotheses.
    transformed_detections : TransformedDetections
        Observations in the co-moving gnomonic frame.
    radius, min_obs, min_arc_length, min_nights :
        Clustering parameters.
    point_cluster_fn : callable
        2D clustering function.
    alg_name : str
        Algorithm name for log messages.
    tracklets : Tracklets, optional
        Pre-formed tracklets. When provided, clustering operates on
        tracklet centroids.
    tracklet_members : TrackletMembers, optional
        Mapping from tracklet_id to obs_id.

    Returns
    -------
    clusters : Clusters
    cluster_members : ClusterMembers
    """
    time_start = time.perf_counter()

    # Determine whether to cluster on tracklet centroids or raw observations
    tracklet_member_obs_ids = None
    if tracklets is not None and tracklet_members is not None and len(tracklets) > 0:
        # Cluster on tracklet centroids
        point_ids = tracklets.tracklet_id.to_numpy(zero_copy_only=False)
        nights = tracklets.night.to_numpy(zero_copy_only=False)
        x = tracklets.theta_x.to_numpy(zero_copy_only=False)
        y = tracklets.theta_y.to_numpy(zero_copy_only=False)
        mjd = tracklets.time.rescale("utc").mjd().to_numpy(zero_copy_only=False)

        # Build tracklet_id -> [obs_id, ...] lookup
        tracklet_member_obs_ids = {}
        tm_tids = tracklet_members.tracklet_id.to_numpy(zero_copy_only=False)
        tm_oids = tracklet_members.obs_id.to_numpy(zero_copy_only=False)
        for tid, oid in zip(tm_tids, tm_oids):
            if tid not in tracklet_member_obs_ids:
                tracklet_member_obs_ids[tid] = []
            tracklet_member_obs_ids[tid].append(oid)
    else:
        # Cluster on raw observations
        point_ids = transformed_detections.id.to_numpy(zero_copy_only=False)
        nights = transformed_detections.night.to_numpy(zero_copy_only=False)
        x = transformed_detections.coordinates.theta_x.to_numpy(zero_copy_only=False)
        y = transformed_detections.coordinates.theta_y.to_numpy(zero_copy_only=False)
        mjd = transformed_detections.coordinates.time.mjd().to_numpy(zero_copy_only=False)

    dt = mjd - mjd.min()

    all_clusters = Clusters.empty()
    all_cluster_members = ClusterMembers.empty()

    for vx_i, vy_i in zip(vx, vy):
        clusters_i, cluster_members_i = _cluster_velocity(
            point_ids,
            x,
            y,
            dt,
            nights,
            vx_i,
            vy_i,
            radius=radius,
            min_obs=min_obs,
            min_arc_length=min_arc_length,
            min_nights=min_nights,
            point_cluster_fn=point_cluster_fn,
            alg_name=alg_name,
            tracklet_member_obs_ids=tracklet_member_obs_ids,
        )
        if len(clusters_i) == 0:
            continue

        all_clusters = qv.concatenate([all_clusters, clusters_i])
        if all_clusters.fragmented():
            all_clusters = qv.defragment(all_clusters)
        all_cluster_members = qv.concatenate([all_cluster_members, cluster_members_i])
        if all_cluster_members.fragmented():
            all_cluster_members = qv.defragment(all_cluster_members)

    time_end = time.perf_counter()
    logger.info(
        f"Found {len(all_clusters)} clusters for {len(vx)} velocity combinations in {time_end - time_start:.3f}s"
    )

    time_start_drop = time.perf_counter()
    logger.info("Removing duplicate clusters...")
    all_clusters = qv.defragment(all_clusters)
    all_cluster_members = qv.defragment(all_cluster_members)
    all_clusters = all_clusters.sort_by([("cluster_id", "ascending")])
    all_cluster_members = all_cluster_members.sort_by([("cluster_id", "ascending")])

    num_clusters = len(all_clusters)
    all_clusters, all_cluster_members = drop_duplicate_clusters(all_clusters, all_cluster_members)
    logger.info(f"Removed {num_clusters - len(all_clusters)} duplicate clusters.")
    time_end_drop = time.perf_counter()
    logger.info(f"Cluster deduplication completed in {time_end_drop - time_start_drop:.3f} seconds.")

    return all_clusters, all_cluster_members


class VelocityGridBase(abc.ABC):
    """
    Abstract base class for velocity-grid clustering algorithms.

    Subclasses must implement :meth:`_point_cluster_fn` which returns the
    2D point-clustering function, and :attr:`_alg_name` for log messages.

    Parameters
    ----------
    radius : float
        Clustering radius in degrees (DBSCAN eps parameter).
    min_obs : int
        Minimum number of observations per cluster.
    min_arc_length : float
        Minimum arc length in days for a cluster to be accepted.
    min_nights : int
        Minimum number of unique nights a cluster must span.
    vx_range : list of float, optional
        [min, max] velocity range in x (deg/day).
    vy_range : list of float, optional
        [min, max] velocity range in y (deg/day).
    vx_bins : int, optional
        Number of bins for x-velocity grid.
    vy_bins : int, optional
        Number of bins for y-velocity grid.
    vx_values : np.ndarray, optional
        Pre-computed x-velocity values.
    vy_values : np.ndarray, optional
        Pre-computed y-velocity values.
    velocity_bin_separation : float
        Separation between velocity bins in units of clustering radius.
    mahalanobis_distance : float, optional
        Mahalanobis distance threshold for covariance-based parameters.
    radius_multiplier : float
        Multiplier on max positional sigma for radius lower bound.
    density_multiplier : float
        Multiplier on characteristic separation for radius upper bound.
    min_radius : float
        Minimum radius in degrees.
    max_radius : float
        Maximum radius in degrees.
    chunk_size : int
        Number of velocity grid points per worker chunk.
    max_processes : int, optional
        Maximum number of processes for parallelization.
    whiten : bool
        Whether to compute whitened parameters in metadata.
    astrometric_precision : float, optional
        Astrometric precision in degrees for radius calculation.
        If None, estimated from observation covariances.
    """

    def __init__(
        self,
        radius: float = 0.005,
        min_obs: int = 6,
        min_arc_length: float = 1.0,
        min_nights: int = 3,
        vx_range: Optional[List[float]] = None,
        vy_range: Optional[List[float]] = None,
        vx_bins: Optional[int] = None,
        vy_bins: Optional[int] = None,
        vx_values: Optional[npt.NDArray[np.float64]] = None,
        vy_values: Optional[npt.NDArray[np.float64]] = None,
        velocity_bin_separation: float = 2.0,
        mahalanobis_distance: Optional[float] = None,
        radius_multiplier: float = 5.0,
        density_multiplier: float = 2.5,
        min_radius: float = 0.01 / 3600,
        max_radius: float = 0.05,
        chunk_size: int = 1000,
        max_processes: Optional[int] = 1,
        whiten: bool = False,
        astrometric_precision: Optional[float] = None,
        window_enabled: bool = True,
        window_min_days: float = 1.0,
    ):
        self.radius = radius
        self.min_obs = min_obs
        self.min_arc_length = min_arc_length
        self.min_nights = min_nights
        self.vx_range = vx_range
        self.vy_range = vy_range
        self.vx_bins = vx_bins
        self.vy_bins = vy_bins
        self.vx_values = vx_values
        self.vy_values = vy_values
        self.velocity_bin_separation = velocity_bin_separation
        self.mahalanobis_distance = mahalanobis_distance
        self.radius_multiplier = radius_multiplier
        self.density_multiplier = density_multiplier
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.chunk_size = chunk_size
        self.max_processes = max_processes
        self.whiten = whiten
        self.astrometric_precision = astrometric_precision
        self.window_enabled = window_enabled
        self.window_min_days = window_min_days

    @property
    @abc.abstractmethod
    def _alg_name(self) -> str:
        """Human-readable algorithm name for log messages."""
        ...

    @abc.abstractmethod
    def _point_cluster_fn(self) -> Callable:
        """Return the 2D point clustering function ``(points, eps, min_samples) -> list[ndarray]``."""
        ...

    @abc.abstractmethod
    def _make_ray_remote(self) -> ray.remote_function.RemoteFunction:
        """Return a Ray remote wrapper for the worker that uses this algorithm's point_cluster_fn."""
        ...

    def find_clusters(
        self,
        transformed_detections: TransformedDetections,
        test_orbit_ephemeris: Optional[TestOrbitEphemeris] = None,
        tracklets: Optional[object] = None,
        tracklet_members: Optional[object] = None,
    ) -> Tuple[Clusters, ClusterMembers]:
        """
        Find clusters using velocity-grid sweep + 2D point clustering.

        Parameters
        ----------
        transformed_detections : TransformedDetections
            Observations in the test orbit's co-moving gnomonic frame.
        test_orbit_ephemeris : TestOrbitEphemeris, optional
            If provided, clustering parameters are calculated from covariances.
        tracklets : Tracklets, optional
            Pre-formed tracklets. When provided, clustering operates on
            tracklet centroids instead of individual observations.
        tracklet_members : TrackletMembers, optional
            Mapping from tracklet_id to obs_id.

        Returns
        -------
        clusters : Clusters
            Unfitted clusters found.
        cluster_members : ClusterMembers
            Members of each cluster.
        """
        time_start_cluster = time.perf_counter()
        logger.info(f"Running velocity space clustering ({self._alg_name})...")

        if isinstance(transformed_detections, str):
            transformed_detections = TransformedDetections.from_parquet(transformed_detections)
            logger.info("Loaded transformed detections from parquet path.")
        elif isinstance(transformed_detections, ray.ObjectRef):
            transformed_detections = ray.get(transformed_detections)
            logger.info("Retrieved observations from the object store.")
        if isinstance(test_orbit_ephemeris, str):
            test_orbit_ephemeris = TestOrbitEphemeris.from_parquet(test_orbit_ephemeris)
            logger.info("Loaded test orbit ephemeris from parquet path.")
        elif isinstance(test_orbit_ephemeris, ray.ObjectRef):
            test_orbit_ephemeris = ray.get(test_orbit_ephemeris)
            logger.info("Retrieved test orbit ephemeris from the object store.")

        if len(transformed_detections) == 0:
            logger.info("No observations to cluster; returning empty clusters.")
            logger.info(f"Clustering completed in {time.perf_counter() - time_start_cluster:.3f} seconds.")
            return Clusters.empty(), ClusterMembers.empty()

        # Determine velocity grid and radius
        radius = self.radius
        vxx, vyy = self._resolve_velocity_grid(transformed_detections, test_orbit_ephemeris)
        if self._resolved_radius is not None:
            radius = self._resolved_radius

        logger.info("Max sample distance: {}".format(radius))
        logger.info("Minimum samples: {}".format(self.min_obs))

        # Early exit checks
        exit_early = False
        if len(transformed_detections) > 0:
            unique_times = transformed_detections.coordinates.time.unique()
            num_unique_times = len(unique_times)
            if num_unique_times < self.min_obs:
                logger.info(
                    "Number of unique times is less than the minimum number of observations required."
                )
                exit_early = True

            time_range = unique_times.max().mjd()[0].as_py() - unique_times.min().mjd()[0].as_py()
            if time_range < self.min_arc_length:
                logger.info("Time range of transformed detections is less than the minimum arc length.")
                exit_early = True
        else:
            logger.info("No transformed detections to cluster.")
            exit_early = True

        if exit_early:
            time_end_cluster = time.perf_counter()
            logger.info("Found 0 clusters. Minimum requirements for clustering not met.")
            logger.info(f"Clustering completed in {time_end_cluster - time_start_cluster:.3f} seconds.")
            return Clusters.empty(), ClusterMembers.empty()

        if tracklets is not None and tracklet_members is not None and len(tracklets) > 0:
            n_tracklets = len(tracklets)
            n_multi = int(np.sum(np.array(tracklets.num_obs.to_pylist()) > 1))
            logger.info(
                f"Using {n_tracklets} tracklets ({n_multi} multi-obs) "
                f"instead of {len(transformed_detections)} raw observations."
            )

        # Compute sliding windows from ephemeris linearity
        windows = self._compute_windows(test_orbit_ephemeris, transformed_detections, radius)

        if len(windows) == 1:
            # Single window (full span) — no splitting needed
            all_clusters, all_cluster_members = self._run_velocity_sweep(
                vxx,
                vyy,
                transformed_detections,
                radius,
                tracklets=tracklets,
                tracklet_members=tracklet_members,
            )
        else:
            logger.info(f"Splitting observations into {len(windows)} time windows " f"for velocity sweep.")
            all_clusters = Clusters.empty()
            all_cluster_members = ClusterMembers.empty()

            obs_times_mjd = (
                transformed_detections.coordinates.time.rescale("utc").mjd().to_numpy(zero_copy_only=False)
            )

            for i, window in enumerate(windows):
                mask = (obs_times_mjd >= window.t_start) & (obs_times_mjd <= window.t_end)
                n_obs_window = int(np.sum(mask))
                if n_obs_window < self.min_obs:
                    logger.info(
                        f"Window {i+1}/{len(windows)} [{window.t_start:.2f}, {window.t_end:.2f}]: "
                        f"{n_obs_window} obs < min_obs ({self.min_obs}), skipping."
                    )
                    continue

                window_detections = transformed_detections.apply_mask(mask)
                logger.info(
                    f"Window {i+1}/{len(windows)} [{window.t_start:.2f}, {window.t_end:.2f}]: "
                    f"{n_obs_window} observations, "
                    f"{window.t_end - window.t_start:.2f} days."
                )

                w_clusters, w_members = self._run_velocity_sweep(
                    vxx,
                    vyy,
                    window_detections,
                    radius,
                    tracklets=tracklets,
                    tracklet_members=tracklet_members,
                )

                if len(w_clusters) > 0:
                    all_clusters = qv.concatenate([all_clusters, w_clusters])
                    if all_clusters.fragmented():
                        all_clusters = qv.defragment(all_clusters)
                    all_cluster_members = qv.concatenate([all_cluster_members, w_members])
                    if all_cluster_members.fragmented():
                        all_cluster_members = qv.defragment(all_cluster_members)

        num_clusters = len(all_clusters)
        if num_clusters == 0:
            time_end_cluster = time.perf_counter()
            logger.info(f"Found {len(all_clusters)} clusters, exiting early.")
            logger.info(f"Clustering completed in {time_end_cluster - time_start_cluster:.3f} seconds.")
            return Clusters.empty(), ClusterMembers.empty()

        # Drop duplicate clusters
        time_start_drop = time.perf_counter()
        logger.info("Removing duplicate clusters...")
        all_clusters = qv.defragment(all_clusters)
        all_cluster_members = qv.defragment(all_cluster_members)
        all_clusters = all_clusters.sort_by([("cluster_id", "ascending")])
        all_cluster_members = all_cluster_members.sort_by([("cluster_id", "ascending")])

        all_clusters, all_cluster_members = drop_duplicate_clusters(all_clusters, all_cluster_members)
        logger.info(f"Removed {num_clusters - len(all_clusters)} duplicate clusters.")
        time_end_drop = time.perf_counter()
        logger.info(f"Cluster deduplication completed in {time_end_drop - time_start_drop:.3f} seconds.")

        time_end_cluster = time.perf_counter()
        logger.info(f"Found {len(all_clusters)} clusters.")
        logger.info(f"Clustering completed in {time_end_cluster - time_start_cluster:.3f} seconds.")

        return all_clusters, all_cluster_members

    def _resolve_velocity_grid(
        self,
        transformed_detections: TransformedDetections,
        test_orbit_ephemeris: Optional[TestOrbitEphemeris],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Resolve velocity grid from config, returning (vxx, vyy) arrays."""
        self._resolved_radius = None

        if self.vx_values is not None and self.vy_values is not None:
            if len(self.vx_values) != len(self.vy_values):
                raise ValueError(
                    f"vx_values and vy_values must have same length. "
                    f"Got {len(self.vx_values)} and {len(self.vy_values)}."
                )
            logger.info(f"Using pre-computed velocity grid with {len(self.vx_values)} points.")
            return self.vx_values, self.vy_values

        if test_orbit_ephemeris is not None:
            logger.info("Calculating clustering parameters from test orbit covariances...")
            try:
                vxx, vyy, radius, metadata = calculate_clustering_parameters_from_covariance(
                    test_orbit_ephemeris,
                    transformed_detections,
                    mahalanobis_distance=(
                        self.mahalanobis_distance if self.mahalanobis_distance is not None else 3.0
                    ),
                    velocity_bin_separation=self.velocity_bin_separation,
                    radius_multiplier=self.radius_multiplier,
                    density_multiplier=self.density_multiplier,
                    min_radius=self.min_radius,
                    max_radius=self.max_radius,
                    whiten=self.whiten,
                    astrometric_precision=self.astrometric_precision,
                )
                self._resolved_radius = radius
                logger.info(
                    f"Covariance-informed clustering: radius={radius:.6f}°, "
                    f"vx_grid={len(vxx)} points, vy_grid={len(vyy)} points"
                )
                return vxx, vyy
            except Exception as e:
                logger.warning(f"Failed to calculate covariance parameters: {e}. Falling back to defaults.")

        # Fall back to range/bins
        vx_range = self.vx_range if self.vx_range is not None else [-0.1, 0.1]
        vy_range = self.vy_range if self.vy_range is not None else [-0.1, 0.1]
        vx_bins = self.vx_bins if self.vx_bins is not None else 100
        vy_bins = self.vy_bins if self.vy_bins is not None else 100

        vx = np.linspace(*vx_range, num=vx_bins)
        vy = np.linspace(*vy_range, num=vy_bins)
        vxx, vyy = np.meshgrid(vx, vy)
        vxx = vxx.flatten()
        vyy = vyy.flatten()

        logger.debug("X velocity range: {}".format(vx_range))
        logger.debug("X velocity bins: {}".format(vx_bins))
        logger.debug("Y velocity range: {}".format(vy_range))
        logger.debug("Y velocity bins: {}".format(vy_bins))
        logger.info(f"Generated velocity grid with {len(vxx)} points.")
        return vxx, vyy

    def _compute_windows(
        self,
        test_orbit_ephemeris: Optional[TestOrbitEphemeris],
        transformed_detections: TransformedDetections,
        radius: float,
    ) -> List[TimeWindow]:
        """
        Compute time windows based on ephemeris linearity.

        If the test orbit's sky motion is linear to within the clustering
        radius over the full observation span, returns a single window.
        Otherwise, splits into overlapping windows sized so the linear
        approximation error stays below the radius.

        Parameters
        ----------
        test_orbit_ephemeris : TestOrbitEphemeris or None
            If None, returns a single window spanning all observations.
        transformed_detections : TransformedDetections
            Observations (used for time range).
        radius : float
            Clustering radius in degrees.

        Returns
        -------
        windows : list of TimeWindow
            Time windows for clustering.
        """
        obs_times_mjd = (
            transformed_detections.coordinates.time.rescale("utc").mjd().to_numpy(zero_copy_only=False)
        )
        t_min = float(obs_times_mjd.min())
        t_max = float(obs_times_mjd.max())

        if not self.window_enabled:
            return [TimeWindow(t_start=t_min, t_end=t_max)]

        if test_orbit_ephemeris is None or len(test_orbit_ephemeris) < 3:
            return [TimeWindow(t_start=t_min, t_end=t_max)]

        # Extract ephemeris RA/Dec
        ephem = test_orbit_ephemeris.ephemeris
        ephem_ra = ephem.coordinates.lon.to_numpy(zero_copy_only=False)
        ephem_dec = ephem.coordinates.lat.to_numpy(zero_copy_only=False)
        ephem_times = ephem.coordinates.time.rescale("utc").mjd().to_numpy(zero_copy_only=False)

        # Filter to observation time range
        time_mask = (ephem_times >= t_min) & (ephem_times <= t_max)
        if np.sum(time_mask) < 3:
            return [TimeWindow(t_start=t_min, t_end=t_max)]

        ephem_ra = ephem_ra[time_mask]
        ephem_dec = ephem_dec[time_mask]
        ephem_times = ephem_times[time_mask]

        window_size = compute_linearity_window(
            ephem_ra,
            ephem_dec,
            ephem_times,
            radius=radius,
            min_window=max(self.min_arc_length * 2.0, self.window_min_days),
        )

        return compute_time_windows(obs_times_mjd, window_size, self.min_arc_length)

    def _run_velocity_sweep(
        self,
        vxx: npt.NDArray[np.float64],
        vyy: npt.NDArray[np.float64],
        transformed_detections: TransformedDetections,
        radius: float,
        tracklets: Optional[object] = None,
        tracklet_members: Optional[object] = None,
    ) -> Tuple[Clusters, ClusterMembers]:
        """Execute velocity sweep across all grid points."""
        all_clusters = Clusters.empty()
        all_cluster_members = ClusterMembers.empty()

        max_processes = self.max_processes
        if max_processes is None:
            max_processes = mp.cpu_count()

        use_ray = initialize_use_ray(num_cpus=max_processes)
        chunk_size = self.chunk_size

        # Adjust chunk_size to keep all workers busy
        total_velocity_points = len(vxx)
        if use_ray:
            num_workers = int(ray.cluster_resources().get("CPU", max_processes))
        else:
            num_workers = max_processes

        if num_workers > 0 and total_velocity_points > 0:
            chunk_size = min(chunk_size, int(np.ceil(total_velocity_points / num_workers)))
            chunk_size = max(chunk_size, 1)

        logger.info(
            f"Distributing {total_velocity_points} velocity grid points "
            f"across chunks of size {chunk_size} "
            f"({int(np.ceil(total_velocity_points / chunk_size))} chunks, "
            f"{num_workers} workers)."
        )

        if use_ray:
            if isinstance(transformed_detections, ray.ObjectRef):
                transformed_ref = transformed_detections
            else:
                transformed_ref = ray.put(transformed_detections)
                logger.info("Placed transformed detections in the object store.")

            tracklets_ref = None
            tracklet_members_ref = None
            if tracklets is not None and tracklet_members is not None:
                tracklets_ref = ray.put(tracklets)
                tracklet_members_ref = ray.put(tracklet_members)
                logger.info("Placed tracklets in the object store.")

            remote_fn = self._make_ray_remote()

            futures = []
            for vxi_chunk, vyi_chunk in zip(
                _iterate_chunks(vxx, chunk_size), _iterate_chunks(vyy, chunk_size)
            ):
                futures.append(
                    remote_fn.remote(
                        vxi_chunk,
                        vyi_chunk,
                        transformed_ref,
                        radius=radius,
                        min_obs=self.min_obs,
                        min_arc_length=self.min_arc_length,
                        min_nights=self.min_nights,
                        tracklets=tracklets_ref,
                        tracklet_members=tracklet_members_ref,
                    )
                )

                if len(futures) >= max_processes * 1.5:
                    finished, futures = ray.wait(futures, num_returns=1)
                    clusters_chunk, cluster_members_chunk = ray.get(finished[0])
                    all_clusters = qv.concatenate([all_clusters, clusters_chunk])
                    if all_clusters.fragmented():
                        all_clusters = qv.defragment(all_clusters)
                    all_cluster_members = qv.concatenate([all_cluster_members, cluster_members_chunk])
                    if all_cluster_members.fragmented():
                        all_cluster_members = qv.defragment(all_cluster_members)

            while futures:
                finished, futures = ray.wait(futures, num_returns=1)
                clusters_chunk, cluster_members_chunk = ray.get(finished[0])
                all_clusters = qv.concatenate([all_clusters, clusters_chunk])
                if all_clusters.fragmented():
                    all_clusters = qv.defragment(all_clusters)
                all_cluster_members = qv.concatenate([all_cluster_members, cluster_members_chunk])
                if all_cluster_members.fragmented():
                    all_cluster_members = qv.defragment(all_cluster_members)
        else:
            point_cluster_fn = self._point_cluster_fn()
            for vxi_chunk, vyi_chunk in zip(
                _iterate_chunks(vxx, chunk_size), _iterate_chunks(vyy, chunk_size)
            ):
                clusters_i, cluster_members_i = _cluster_velocity_find_worker(
                    vxi_chunk,
                    vyi_chunk,
                    transformed_detections,
                    radius=radius,
                    min_obs=self.min_obs,
                    min_arc_length=self.min_arc_length,
                    min_nights=self.min_nights,
                    point_cluster_fn=point_cluster_fn,
                    alg_name=self._alg_name,
                    tracklets=tracklets,
                    tracklet_members=tracklet_members,
                )

                all_clusters = qv.concatenate([all_clusters, clusters_i])
                if all_clusters.fragmented():
                    all_clusters = qv.defragment(all_clusters)
                all_cluster_members = qv.concatenate([all_cluster_members, cluster_members_i])
                if all_cluster_members.fragmented():
                    all_cluster_members = qv.defragment(all_cluster_members)

        return all_clusters, all_cluster_members
