"""
Tracklet formation in the co-moving gnomonic frame.

Tracklets are short-arc groupings of same-night observations that are positionally
and kinematically consistent. They are formed by finding pairs of detections across
different exposures (states) on the same night whose relative velocity falls within
the test orbit's velocity uncertainty ellipse.

Pairs are found efficiently using KD-trees (one per exposure), and connected
components group transitive pairs into multi-observation tracklets. Observations
that cannot be linked form singleton tracklets so that every observation participates
in the downstream velocity-grid clustering.
"""

import logging
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from adam_core.time import Timestamp
from scipy.spatial import cKDTree

from ..orbit import TestOrbitEphemeris
from ..range_and_transform import TransformedDetections

logger = logging.getLogger(__name__)

__all__ = [
    "Tracklets",
    "TrackletMembers",
    "form_tracklets",
]


class Tracklets(qv.Table):
    """Summary table of tracklets (one row per tracklet)."""

    tracklet_id = qv.LargeStringColumn()
    night = qv.Int64Column()
    num_obs = qv.Int64Column()
    theta_x = qv.Float64Column()
    theta_y = qv.Float64Column()
    time = Timestamp.as_column()
    vtheta_x = qv.Float64Column(nullable=True)
    vtheta_y = qv.Float64Column(nullable=True)


class TrackletMembers(qv.Table):
    """Mapping from tracklet to its constituent observations."""

    tracklet_id = qv.LargeStringColumn()
    obs_id = qv.LargeStringColumn()


# ---------------------------------------------------------------------------
# Union-Find for connected components
# ---------------------------------------------------------------------------


class _UnionFind:
    """Simple union-find (disjoint set) data structure."""

    __slots__ = ("parent", "rank")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ---------------------------------------------------------------------------
# Velocity covariance extraction
# ---------------------------------------------------------------------------


def _extract_velocity_bounds(
    test_orbit_ephemeris: TestOrbitEphemeris,
    transformed_detections: TransformedDetections,
    mahalanobis_distance: float = 3.0,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """
    Extract velocity covariance from ephemeris and compute bounds.

    Returns
    -------
    vel_cov_inv : (2, 2) array
        Inverse of the mean 2x2 velocity covariance in the gnomonic frame.
    vel_sigmas : (2,) array
        (sigma_vx, sigma_vy) standard deviations.
    v_max : float
        Maximum scalar velocity for spatial pre-filtering (mahalanobis_distance
        times the larger sigma).
    """
    # Filter ephemeris to observation time range
    obs_times_mjd = (
        transformed_detections.coordinates.time.rescale("utc").mjd().to_numpy(zero_copy_only=False)
    )
    obs_time_min = obs_times_mjd.min()
    obs_time_max = obs_times_mjd.max()

    ephemeris_gnomonic = test_orbit_ephemeris.gnomonic
    ephem_times_mjd = ephemeris_gnomonic.time.rescale("utc").mjd().to_numpy(zero_copy_only=False)

    time_mask = (ephem_times_mjd >= obs_time_min) & (ephem_times_mjd <= obs_time_max)
    if not np.any(time_mask):
        raise ValueError(
            f"No ephemeris points found in observation time range "
            f"[{obs_time_min:.2f}, {obs_time_max:.2f}] MJD"
        )
    ephemeris_gnomonic = ephemeris_gnomonic.apply_mask(time_mask)

    covariances = ephemeris_gnomonic.covariance.to_matrix()
    mean_cov = np.nanmean(covariances, axis=0)
    vel_cov = mean_cov[2:4, 2:4]

    if np.any(np.isnan(vel_cov)) or np.any(~np.isfinite(vel_cov)):
        raise ValueError("Velocity covariance matrix contains NaN or infinite values.")

    vel_cov_inv = np.linalg.inv(vel_cov)
    sigma_vx = np.sqrt(vel_cov[0, 0])
    sigma_vy = np.sqrt(vel_cov[1, 1])
    v_max = mahalanobis_distance * max(sigma_vx, sigma_vy)

    return vel_cov_inv, np.array([sigma_vx, sigma_vy]), v_max


# ---------------------------------------------------------------------------
# Core tracklet formation
# ---------------------------------------------------------------------------


def form_tracklets(
    transformed_detections: TransformedDetections,
    test_orbit_ephemeris: TestOrbitEphemeris,
    min_obs: int = 2,
    max_velocity: Optional[float] = None,
    mahalanobis_distance: float = 3.0,
) -> Tuple[Tracklets, TrackletMembers]:
    """
    Form tracklets from transformed detections using KD-tree pair finding
    and velocity-covariance filtering.

    For each night, pairs of detections from different exposures are linked
    if their relative velocity falls within the test orbit's velocity
    uncertainty ellipse (Mahalanobis distance). Connected valid pairs are
    grouped into tracklets. Observations that do not participate in any
    multi-observation tracklet become singleton tracklets.

    Parameters
    ----------
    transformed_detections : TransformedDetections
        Observations in the test orbit's co-moving gnomonic frame.
    test_orbit_ephemeris : TestOrbitEphemeris
        Test orbit ephemeris with gnomonic covariances for velocity bounds.
    min_obs : int, optional
        Minimum number of observations for a multi-observation tracklet.
        Tracklets with fewer members are dissolved into singletons.
        [Default = 2]
    max_velocity : float, optional
        Hard upper bound on velocity magnitude in deg/day. If None, the
        bound is derived from the ephemeris covariance. [Default = None]
    mahalanobis_distance : float, optional
        Mahalanobis distance threshold for velocity filtering.
        [Default = 3.0]

    Returns
    -------
    tracklets : Tracklets
        Summary table with one row per tracklet (centroid position, time,
        velocity, and observation count).
    tracklet_members : TrackletMembers
        Mapping from tracklet_id to obs_id.
    """
    if len(transformed_detections) == 0:
        logger.info("No detections to form tracklets from.")
        return Tracklets.empty(), TrackletMembers.empty()

    # Extract velocity bounds from ephemeris covariance
    mahal_dist_sq = mahalanobis_distance**2
    vel_cov_inv, vel_sigmas, v_max_cov = _extract_velocity_bounds(
        test_orbit_ephemeris, transformed_detections, mahalanobis_distance
    )
    if max_velocity is not None:
        v_max = max_velocity
    else:
        v_max = v_max_cov

    logger.info(
        f"Tracklet velocity bounds: σ_vx={vel_sigmas[0]:.6f}, σ_vy={vel_sigmas[1]:.6f} deg/day, "
        f"v_max={v_max:.6f} deg/day ({mahalanobis_distance:.1f}-sigma)"
    )

    # Extract arrays once
    all_obs_ids = transformed_detections.id.to_numpy(zero_copy_only=False)
    all_nights = transformed_detections.night.to_numpy(zero_copy_only=False)
    all_state_ids = transformed_detections.state_id.to_numpy(zero_copy_only=False)
    all_x = transformed_detections.coordinates.theta_x.to_numpy(zero_copy_only=False)
    all_y = transformed_detections.coordinates.theta_y.to_numpy(zero_copy_only=False)
    all_mjd = transformed_detections.coordinates.time.rescale("utc").mjd().to_numpy(zero_copy_only=False)

    # Build a global index → obs_id mapping and work with integer indices
    n_total = len(all_obs_ids)
    uf = _UnionFind(n_total)
    linked = np.zeros(n_total, dtype=bool)

    unique_nights = np.unique(all_nights)
    total_pairs = 0

    for night in unique_nights:
        night_mask = all_nights == night
        night_indices = np.where(night_mask)[0]

        if len(night_indices) == 0:
            continue

        # Group by state_id within this night
        night_state_ids = all_state_ids[night_indices]
        unique_states = np.unique(night_state_ids)

        if len(unique_states) < 2:
            # Only one exposure this night — no pairs possible
            continue

        # Build per-state data
        state_data: Dict[str, Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]] = {}
        for sid in unique_states:
            state_mask = night_state_ids == sid
            idx = night_indices[state_mask]
            x = all_x[idx]
            y = all_y[idx]
            mjd = all_mjd[idx]

            # Filter NaNs
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(mjd)
            if not np.all(finite):
                idx = idx[finite]
                x = x[finite]
                y = y[finite]
                mjd = mjd[finite]

            if len(idx) > 0:
                state_data[sid] = (idx, x, y, mjd)

        state_keys = sorted(state_data.keys())

        # For each pair of states, find consistent pairs
        for i_s in range(len(state_keys)):
            for j_s in range(i_s + 1, len(state_keys)):
                sid_a = state_keys[i_s]
                sid_b = state_keys[j_s]
                idx_a, x_a, y_a, mjd_a = state_data[sid_a]
                idx_b, x_b, y_b, mjd_b = state_data[sid_b]

                # Use representative dt (mean time difference between states)
                dt = np.mean(mjd_b) - np.mean(mjd_a)
                if abs(dt) < 1e-10:
                    continue

                search_radius = v_max * abs(dt)

                # Build KD-tree on state B
                points_b = np.column_stack([x_b, y_b])
                tree_b = cKDTree(points_b)

                # Query for each detection in state A
                points_a = np.column_stack([x_a, y_a])
                candidates = tree_b.query_ball_point(points_a, search_radius)

                for ia, cand_list in enumerate(candidates):
                    if len(cand_list) == 0:
                        continue

                    # Compute per-detection dt for accurate velocity
                    dt_ia = mjd_b[cand_list] - mjd_a[ia]
                    # Skip any zero dt
                    valid_dt = np.abs(dt_ia) > 1e-10
                    if not np.any(valid_dt):
                        continue

                    cand_arr = np.array(cand_list)[valid_dt]
                    dt_ia = dt_ia[valid_dt]

                    # Compute velocity vectors
                    dvx = (x_b[cand_arr] - x_a[ia]) / dt_ia
                    dvy = (y_b[cand_arr] - y_a[ia]) / dt_ia

                    # Mahalanobis distance filter: v^T @ Sigma_inv @ v <= threshold
                    vel_vectors = np.column_stack([dvx, dvy])
                    mahal_sq = np.sum(vel_vectors @ vel_cov_inv * vel_vectors, axis=1)
                    accept = mahal_sq <= mahal_dist_sq

                    if max_velocity is not None:
                        v_mag = np.sqrt(dvx**2 + dvy**2)
                        accept &= v_mag <= max_velocity

                    # Union accepted pairs
                    global_a = idx_a[ia]
                    for jb in cand_arr[accept]:
                        global_b = idx_b[jb]
                        uf.union(global_a, global_b)
                        linked[global_a] = True
                        linked[global_b] = True
                        total_pairs += 1

    logger.info(f"Found {total_pairs} valid observation pairs across {len(unique_nights)} nights.")

    # Build connected components
    components: Dict[int, List[int]] = {}
    for i in range(n_total):
        root = uf.find(i)
        if root not in components:
            components[root] = []
        components[root].append(i)

    # Separate multi-obs tracklets from singletons
    tracklet_groups: List[npt.NDArray[np.int64]] = []
    singleton_indices: List[int] = []

    for indices in components.values():
        if len(indices) >= min_obs:
            tracklet_groups.append(np.array(indices, dtype=np.int64))
        else:
            singleton_indices.extend(indices)

    n_multi = len(tracklet_groups)
    n_singleton = len(singleton_indices)
    n_multi_obs = sum(len(g) for g in tracklet_groups)
    logger.info(
        f"Formed {n_multi} multi-observation tracklets ({n_multi_obs} obs) "
        f"and {n_singleton} singletons."
    )

    # Build output tables
    tracklet_ids: List[str] = []
    tracklet_nights: List[int] = []
    tracklet_num_obs: List[int] = []
    tracklet_theta_x: List[float] = []
    tracklet_theta_y: List[float] = []
    tracklet_mjds: List[float] = []
    tracklet_vx: List[Optional[float]] = []
    tracklet_vy: List[Optional[float]] = []
    member_tracklet_ids: List[str] = []
    member_obs_ids: List[str] = []

    def _add_tracklet(indices: npt.NDArray[np.int64]) -> None:
        tid = uuid.uuid4().hex
        obs_ids = all_obs_ids[indices]
        x = all_x[indices]
        y = all_y[indices]
        mjd = all_mjd[indices]
        night = all_nights[indices[0]]

        # Centroid
        cx = np.nanmean(x)
        cy = np.nanmean(y)
        ct = np.nanmean(mjd)

        # Velocity from linear fit (or finite difference for 2 obs)
        n = len(indices)
        if n >= 2:
            dt_local = mjd - ct
            if np.ptp(dt_local) > 1e-10:
                # Simple linear regression: v = Σ(dt * dx) / Σ(dt²)
                dx = x - cx
                dy = y - cy
                dt2 = np.sum(dt_local**2)
                vx_fit = float(np.sum(dt_local * dx) / dt2)
                vy_fit = float(np.sum(dt_local * dy) / dt2)
            else:
                vx_fit = None
                vy_fit = None
        else:
            vx_fit = None
            vy_fit = None

        tracklet_ids.append(tid)
        tracklet_nights.append(int(night))
        tracklet_num_obs.append(n)
        tracklet_theta_x.append(float(cx))
        tracklet_theta_y.append(float(cy))
        tracklet_mjds.append(float(ct))
        tracklet_vx.append(vx_fit)
        tracklet_vy.append(vy_fit)

        for oid in obs_ids:
            member_tracklet_ids.append(tid)
            member_obs_ids.append(str(oid))

    # Multi-observation tracklets
    for group in tracklet_groups:
        _add_tracklet(group)

    # Singleton tracklets
    for idx in singleton_indices:
        _add_tracklet(np.array([idx], dtype=np.int64))

    # Build Timestamp from mean MJDs
    tracklet_times = Timestamp.from_mjd(tracklet_mjds, scale="utc")

    tracklets = Tracklets.from_kwargs(
        tracklet_id=tracklet_ids,
        night=tracklet_nights,
        num_obs=tracklet_num_obs,
        theta_x=tracklet_theta_x,
        theta_y=tracklet_theta_y,
        time=tracklet_times,
        vtheta_x=tracklet_vx,
        vtheta_y=tracklet_vy,
    )

    tracklet_members = TrackletMembers.from_kwargs(
        tracklet_id=member_tracklet_ids,
        obs_id=member_obs_ids,
    )

    logger.info(f"Total tracklets: {len(tracklets)} ({n_multi} multi-obs + {n_singleton} singletons)")
    return tracklets, tracklet_members
