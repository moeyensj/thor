import logging
import time
import uuid
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import ray
from scipy.spatial import cKDTree

from ..orbit import TestOrbitEphemeris
from ..range_and_transform import TransformedDetections
from .data import ClusterMembers, Clusters, drop_duplicate_clusters
from .metrics import filter_clusters_by_length
from .velocity_grid import calculate_clustering_parameters_from_covariance

logger = logging.getLogger(__name__)


class HoughLineClustering:
    """
    Clustering algorithm that finds linear trajectories in (x, y, t) space
    via a Hough-style accumulator over velocity space.

    Instead of sweeping a velocity grid and running 2D clustering at each
    point, this algorithm has each observation vote into a 2D (vx, vy)
    accumulator. Peaks in the accumulator identify candidate velocities,
    and observations consistent with those velocities are grouped into
    clusters.

    Parameters
    ----------
    radius : float
        Clustering radius in degrees.
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
        Number of bins for x-velocity accumulator.
    vy_bins : int, optional
        Number of bins for y-velocity accumulator.
    vx_values : np.ndarray, optional
        Pre-computed x-velocity values (unused, kept for interface consistency).
    vy_values : np.ndarray, optional
        Pre-computed y-velocity values (unused, kept for interface consistency).
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
        Not used directly, kept for interface consistency.
    max_processes : int, optional
        Not used directly, kept for interface consistency.
    whiten : bool
        Whether to compute whitened parameters in metadata.
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
        min_radius: float = 1 / 3600,
        max_radius: float = 0.05,
        chunk_size: int = 1000,
        max_processes: Optional[int] = 1,
        whiten: bool = False,
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

    def _resolve_velocity_grid(
        self,
        transformed_detections: TransformedDetections,
        test_orbit_ephemeris,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """
        Resolve velocity grid bounds and radius.

        Returns
        -------
        vx_edges : ndarray
            1D array of vx bin edges.
        vy_edges : ndarray
            1D array of vy bin edges.
        radius : float
            Clustering radius.
        """
        radius = self.radius

        if test_orbit_ephemeris is not None:
            logger.info("Calculating clustering parameters from test orbit covariances...")
            try:
                vxx, vyy, cov_radius, metadata = calculate_clustering_parameters_from_covariance(
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
                )
                radius = cov_radius
                # Derive grid edges from the covariance-computed velocity points
                vx_min, vx_max = vxx.min(), vxx.max()
                vy_min, vy_max = vyy.min(), vyy.max()
                n_vx = max(int(np.sqrt(len(vxx))), 10)
                n_vy = max(int(np.sqrt(len(vyy))), 10)
                vx_edges = np.linspace(vx_min, vx_max, n_vx)
                vy_edges = np.linspace(vy_min, vy_max, n_vy)
                return vx_edges, vy_edges, radius
            except Exception as e:
                logger.warning(f"Failed to calculate covariance parameters: {e}. Falling back to defaults.")

        vx_range = self.vx_range if self.vx_range is not None else [-0.1, 0.1]
        vy_range = self.vy_range if self.vy_range is not None else [-0.1, 0.1]
        vx_bins = self.vx_bins if self.vx_bins is not None else 300
        vy_bins = self.vy_bins if self.vy_bins is not None else 300

        vx_edges = np.linspace(vx_range[0], vx_range[1], vx_bins)
        vy_edges = np.linspace(vy_range[0], vy_range[1], vy_bins)
        return vx_edges, vy_edges, radius

    def find_clusters(
        self,
        transformed_detections: TransformedDetections,
        test_orbit_ephemeris=None,
    ) -> Tuple[Clusters, ClusterMembers]:
        """
        Find clusters using Hough-style line finding in (x, y, t) space.

        Parameters
        ----------
        transformed_detections : TransformedDetections
            Observations in the test orbit's co-moving gnomonic frame.
        test_orbit_ephemeris : TestOrbitEphemeris, optional
            If provided, clustering parameters are calculated from covariances.

        Returns
        -------
        clusters : Clusters
            Unfitted clusters found.
        cluster_members : ClusterMembers
            Members of each cluster.
        """
        time_start = time.perf_counter()
        logger.info("Running Hough line clustering...")

        # Handle Ray ObjectRef / parquet path inputs
        if isinstance(transformed_detections, str):
            transformed_detections = TransformedDetections.from_parquet(transformed_detections)
        elif isinstance(transformed_detections, ray.ObjectRef):
            transformed_detections = ray.get(transformed_detections)
        if isinstance(test_orbit_ephemeris, str):
            test_orbit_ephemeris = TestOrbitEphemeris.from_parquet(test_orbit_ephemeris)
        elif isinstance(test_orbit_ephemeris, ray.ObjectRef):
            test_orbit_ephemeris = ray.get(test_orbit_ephemeris)

        if len(transformed_detections) == 0:
            logger.info("No observations to cluster; returning empty clusters.")
            return Clusters.empty(), ClusterMembers.empty()

        # Early exit checks
        unique_times = transformed_detections.coordinates.time.unique()
        num_unique_times = len(unique_times)
        if num_unique_times < self.min_obs:
            logger.info("Number of unique times is less than the minimum number of observations required.")
            return Clusters.empty(), ClusterMembers.empty()

        time_range = unique_times.max().mjd()[0].as_py() - unique_times.min().mjd()[0].as_py()
        if time_range < self.min_arc_length:
            logger.info("Time range of transformed detections is less than the minimum arc length.")
            return Clusters.empty(), ClusterMembers.empty()

        # Resolve velocity grid and radius
        vx_edges, vy_edges, radius = self._resolve_velocity_grid(transformed_detections, test_orbit_ephemeris)
        logger.info(
            f"Hough accumulator: {len(vx_edges)} x {len(vy_edges)} velocity bins, radius={radius:.6f}°"
        )

        # Extract observation arrays
        obs_ids = transformed_detections.id.to_numpy(zero_copy_only=False)
        nights = transformed_detections.night.to_numpy(zero_copy_only=False)
        x = transformed_detections.coordinates.theta_x.to_numpy(zero_copy_only=False)
        y = transformed_detections.coordinates.theta_y.to_numpy(zero_copy_only=False)
        mjd = transformed_detections.coordinates.time.mjd().to_numpy(zero_copy_only=False)
        dt = mjd - mjd.min()

        # Drop NaNs
        finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(dt)
        if not np.all(finite_mask):
            n_drop = np.size(finite_mask) - np.count_nonzero(finite_mask)
            logger.warning(f"Dropping {n_drop} observations with NaN coordinates.")
            x = x[finite_mask]
            y = y[finite_mask]
            dt = dt[finite_mask]
            nights = nights[finite_mask]
            obs_ids = obs_ids[finite_mask]

        n_obs = len(x)
        if n_obs < self.min_obs:
            return Clusters.empty(), ClusterMembers.empty()

        # Build 2D Hough accumulator
        # For each pair of observations (i, j) at different times, compute the
        # velocity (vx, vy) = (dx/dt, dy/dt) and vote into the accumulator.
        # For efficiency, use a histogram approach: for each observation,
        # compute its vote for each velocity bin.
        vx_step = vx_edges[1] - vx_edges[0] if len(vx_edges) > 1 else 1.0
        vy_step = vy_edges[1] - vy_edges[0] if len(vy_edges) > 1 else 1.0
        n_vx = len(vx_edges)
        n_vy = len(vy_edges)

        # For each observation, for each velocity bin, compute the de-rotated
        # position: x0 = x_i - vx * (t_i - t_ref). Then bin the x0, y0 values.
        # Instead, we use the Hough approach: for each pair (i, j), compute
        # the implied velocity. This is O(n^2) which is too expensive for
        # large n. Instead, use the accumulator approach where each observation
        # votes across velocity space.

        # Accumulator: count how many observations land within radius of
        # each other after de-rotation at each velocity.
        # Efficient approach: for each velocity bin center, de-rotate all
        # observations and count how many pairs are within radius.
        # This is still O(n_vx * n_vy * n), same as velocity grid sweep.

        # Optimized Hough: for each pair of observations at different times,
        # compute the implied velocity and vote. For n observations, take
        # a subset of pairs to keep it tractable.
        # Use pairs with sufficient time separation.
        dt_min_pair = self.min_arc_length / 4.0  # minimum dt for a pair to vote

        all_cluster_ids = []
        all_cluster_vx = []
        all_cluster_vy = []
        all_cluster_arc_lengths = []
        all_cluster_num_obs = []
        all_member_cluster_ids = []
        all_member_obs_ids = []

        # Build accumulator from observation pairs
        accumulator = np.zeros((n_vx, n_vy), dtype=np.float64)

        # For tractability, sample pairs if n is large
        max_pairs = 500000
        pair_count = 0

        for i in range(n_obs):
            if pair_count >= max_pairs:
                break
            for j in range(i + 1, n_obs):
                dt_ij = dt[j] - dt[i]
                if abs(dt_ij) < dt_min_pair:
                    continue

                vx_ij = (x[j] - x[i]) / dt_ij
                vy_ij = (y[j] - y[i]) / dt_ij

                # Find the bin for this velocity
                ix = int((vx_ij - vx_edges[0]) / vx_step)
                iy = int((vy_ij - vy_edges[0]) / vy_step)

                if 0 <= ix < n_vx and 0 <= iy < n_vy:
                    accumulator[ix, iy] += 1

                pair_count += 1
                if pair_count >= max_pairs:
                    break

        logger.info(f"Hough accumulator built from {pair_count} observation pairs.")

        # Find peaks in the accumulator above threshold
        # A peak should have at least min_obs * (min_obs - 1) / 2 votes
        # if min_obs observations are collinear, but in practice noise
        # reduces this. Use a lower threshold.
        min_votes = max(self.min_obs - 1, 3)
        peak_mask = accumulator >= min_votes

        if not np.any(peak_mask):
            logger.info("No peaks found in Hough accumulator.")
            time_end = time.perf_counter()
            logger.info(f"Hough clustering completed in {time_end - time_start:.3f} seconds.")
            return Clusters.empty(), ClusterMembers.empty()

        peak_indices = np.argwhere(peak_mask)
        # Sort by vote count (descending) to process strongest peaks first
        peak_votes = accumulator[peak_mask]
        sort_order = np.argsort(-peak_votes)
        peak_indices = peak_indices[sort_order]

        logger.info(f"Found {len(peak_indices)} Hough peaks above threshold.")

        # For each peak velocity, extract observations that cluster
        for pi in range(len(peak_indices)):
            ix, iy = peak_indices[pi]
            vx_peak = vx_edges[ix]
            vy_peak = vy_edges[iy]

            # De-rotate all observations at this velocity
            xx = x - vx_peak * dt
            yy = y - vy_peak * dt

            # Find groups of points within radius of each other
            # Simple approach: find the densest point and grow a cluster
            points_2d = np.column_stack([xx, yy])
            tree = cKDTree(points_2d)
            neighbors = tree.query_ball_point(points_2d, radius)

            # Find points with enough neighbors
            counts = np.array([len(nb) for nb in neighbors])
            core_mask = counts >= self.min_obs

            if not np.any(core_mask):
                continue

            # Simple connected-component clustering on core points
            visited = np.zeros(n_obs, dtype=bool)
            raw_clusters: List[npt.NDArray[np.int64]] = []

            for seed in np.where(core_mask)[0]:
                if visited[seed]:
                    continue
                # BFS from seed
                cluster_set = set()
                queue = [seed]
                while queue:
                    pt = queue.pop()
                    if visited[pt]:
                        continue
                    visited[pt] = True
                    cluster_set.add(pt)
                    if core_mask[pt]:
                        for nb in neighbors[pt]:
                            if not visited[nb]:
                                queue.append(nb)

                if len(cluster_set) >= self.min_obs:
                    raw_clusters.append(np.array(sorted(cluster_set), dtype=np.int64))

            if not raw_clusters:
                continue

            # Filter by arc length / nights
            filtered_clusters, arc_lengths = filter_clusters_by_length(
                raw_clusters,
                dt,
                nights,
                self.min_obs,
                self.min_arc_length,
                self.min_nights,
            )

            for ci, cluster_indices in enumerate(filtered_clusters):
                cid = uuid.uuid4().hex
                obs_ids_c = obs_ids[cluster_indices]
                num_obs_c = len(obs_ids_c)

                all_cluster_ids.append(cid)
                all_cluster_vx.append(vx_peak)
                all_cluster_vy.append(vy_peak)
                all_cluster_arc_lengths.append(arc_lengths[ci])
                all_cluster_num_obs.append(num_obs_c)
                all_member_cluster_ids.extend([cid] * num_obs_c)
                all_member_obs_ids.extend(obs_ids_c.tolist())

        if len(all_cluster_ids) == 0:
            time_end = time.perf_counter()
            logger.info(
                f"Found 0 clusters. Hough clustering completed in {time_end - time_start:.3f} seconds."
            )
            return Clusters.empty(), ClusterMembers.empty()

        all_clusters = Clusters.from_kwargs(
            cluster_id=all_cluster_ids,
            vtheta_x=np.array(all_cluster_vx),
            vtheta_y=np.array(all_cluster_vy),
            arc_length=all_cluster_arc_lengths,
            num_obs=all_cluster_num_obs,
        )

        all_cluster_members = ClusterMembers.from_kwargs(
            cluster_id=all_member_cluster_ids,
            obs_id=all_member_obs_ids,
        )

        # Deduplicate
        all_clusters = all_clusters.sort_by([("cluster_id", "ascending")])
        all_cluster_members = all_cluster_members.sort_by([("cluster_id", "ascending")])
        num_before = len(all_clusters)
        all_clusters, all_cluster_members = drop_duplicate_clusters(all_clusters, all_cluster_members)
        logger.info(f"Removed {num_before - len(all_clusters)} duplicate clusters.")

        time_end = time.perf_counter()
        logger.info(f"Found {len(all_clusters)} clusters.")
        logger.info(f"Hough clustering completed in {time_end - time_start:.3f} seconds.")

        return all_clusters, all_cluster_members
