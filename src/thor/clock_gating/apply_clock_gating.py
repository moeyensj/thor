"""
Apply Kepler clock gating to observation-orbit overlaps.

This module provides functionality to filter observation-orbit overlaps
using Kepler's laws and produce K-chains for downstream processing.
"""

import uuid
from typing import Dict, Any, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from adam_core.geometry.overlap import OverlapHits
from adam_core.geometry.clock_gating import ClockGateConfig, apply_clock_gating as adam_apply_clock_gating
from adam_core.geometry.anomaly import AnomalyLabels
from adam_core.observations.rays import ObservationRays
from adam_core.orbits.polyline import OrbitsPlaneParams
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.time import Timestamp

from .kchains import Chains, ChainMembers


def apply_clock_gating(
    hits: OverlapHits,
    rays: ObservationRays,
    anomaly_labels: "AnomalyLabels",
    cfg: Dict[str, Any] | None = None,
) -> Tuple[Chains, ChainMembers]:
    """
    Apply Kepler clock gating to observation-orbit overlaps.
    
    This function filters observation-orbit overlaps using Kepler's laws
    to eliminate time-inconsistent pairings and produces K-chains for
    downstream processing.
    
    Parameters
    ----------
    hits : OverlapHits
        Geometric overlaps between observations and test orbits.
    rays : ObservationRays
        Observation rays containing timing information.
    anomaly_labels : AnomalyLabels
        Anomaly labels containing orbital elements (M, n, r) per hit.
    cfg : dict, optional
        Clock gating configuration parameters. Supports:
        - tau0_minutes: Base time tolerance in minutes (default: 10.0)
        - alpha_minutes_per_day: Time tolerance growth rate (default: 0.05)
        - use_jax: Whether to use JAX acceleration (default: True)
        
    Returns
    -------
    chains : Chains
        K-chains metadata table.
    chain_members : ChainMembers
        K-chain membership table.
        
    Notes
    -----
    When anomaly_labels is provided, applies time-consistency filtering
    based on mean anomaly progression. Otherwise, creates trivial chains
    with one chain per orbit containing all associated detections.
    """
    if len(hits) == 0 or len(anomaly_labels) == 0:
        # Return empty tables
        return (
            Chains.empty(),
            ChainMembers.empty(),
        )
    
    # Parse configuration
    if cfg is None:
        cfg = {}
    tau0_minutes = cfg.get("tau0_minutes", 10.0)
    alpha_minutes_per_day = cfg.get("alpha_minutes_per_day", 0.05)
    use_jax = cfg.get("use_jax", True)
    
    # Apply real clock gating
    chains, members = _apply_real_clock_gating(
        hits, rays, anomaly_labels, tau0_minutes, alpha_minutes_per_day, use_jax
    )

    # Optional promotion thresholds
    min_chain_size = int(cfg.get("min_chain_size", 1))
    min_chain_days = float(cfg.get("min_chain_days", 0.0))
    if len(chains) == 0 or (min_chain_size <= 1 and min_chain_days <= 0.0):
        return chains, members

    if len(chains) > 0:
        keep_mask = []
        for i in range(len(chains)):
            size = chains.size[i].as_py()
            tmin = chains.t_min[i].as_py()
            tmax = chains.t_max[i].as_py()
            keep = (size >= min_chain_size) and ((tmax - tmin) >= min_chain_days)
            keep_mask.append(keep)
        keep_idx = np.nonzero(np.array(keep_mask, dtype=bool))[0]
        if len(keep_idx) == 0:
            return Chains.empty(), ChainMembers.empty()
        # Filter chains
        chains = chains.take(keep_idx)
        # Filter members by kept chain_ids
        keep_chain_ids = set(chains.chain_id.to_pylist())
        mem_keep = [cid.as_py() in keep_chain_ids for cid in members.chain_id]
        mem_keep_idx = np.nonzero(np.array(mem_keep, dtype=bool))[0]
        members = members.take(mem_keep_idx) if len(mem_keep_idx) > 0 else ChainMembers.empty()
    return chains, members


# Removed stub gating; real gating is always applied.


def _apply_real_clock_gating(
    hits: OverlapHits, 
    rays: ObservationRays, 
    anomaly_labels: "AnomalyLabels",
    tau0_minutes: float,
    alpha_minutes_per_day: float,
    use_jax: bool,
) -> Tuple[Chains, ChainMembers]:
    """
    Real clock gating using anomaly labels and time consistency.
    
    Vectorized per-orbit neighbor checks with a small time-local window and
    fixed neighbor cap to minimize Python overhead. JAX acceleration can be
    added later once shapes are stabilized.
    """
    # Parameters for neighborhood search (configurable later)
    max_neighbors_per_node = 32
    max_time_days = None  # type: float | None

    # 1) Deduplicate anomaly labels on (orbit_id, det_id) using columnar ops
    labels = anomaly_labels.sort_by([
        ("orbit_id", "ascending"),
        ("det_id", "ascending"),
        ("time_mjd", "ascending"),
    ])
    orbit_ids = labels.orbit_id.to_numpy(zero_copy_only=False)
    det_ids = labels.det_id.to_numpy(zero_copy_only=False)
    times = labels.time_mjd.to_numpy(zero_copy_only=False).astype(float)
    M_rad = np.radians(labels.M_deg.to_numpy(zero_copy_only=False).astype(float))
    n_rad_per_day = (labels.n_deg_per_day.to_numpy(zero_copy_only=False).astype(float) * np.pi / 180.0)

    if len(orbit_ids) == 0:
        return Chains.empty(), ChainMembers.empty()

    # Keep first per (orbit_id, det_id)
    first_mask = np.ones(len(orbit_ids), dtype=bool)
    first_mask[1:] = (orbit_ids[1:] != orbit_ids[:-1]) | (det_ids[1:] != det_ids[:-1])
    if not np.all(first_mask):
        orbit_ids = orbit_ids[first_mask]
        det_ids = det_ids[first_mask]
        times = times[first_mask]
        M_rad = M_rad[first_mask]
        n_rad_per_day = n_rad_per_day[first_mask]

    # 2) Sort by (orbit_id, time_mjd) for per-orbit chronological processing
    order = np.lexsort((times, orbit_ids))
    orbit_ids = orbit_ids[order]
    det_ids = det_ids[order]
    times = times[order]
    M_rad = M_rad[order]
    n_rad_per_day = n_rad_per_day[order]

    # Prepare outputs
    chains_data = {
        "chain_id": [],
        "orbit_id": [],
        "size": [],
        "t_min": [],
        "t_max": [],
    }
    members_data = {
        "chain_id": [],
        "det_id": [],
        "time_mjd": [],
    }

    # Simple union-find
    class DSU:
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
                self.parent[ra] = rb
            elif self.rank[rb] < self.rank[ra]:
                self.parent[rb] = ra
            else:
                self.parent[rb] = ra
                self.rank[ra] += 1

    # Compute group boundaries for each orbit
    group_starts = np.flatnonzero(np.r_[True, orbit_ids[1:] != orbit_ids[:-1]])
    group_ends = np.r_[group_starts[1:], len(orbit_ids)]

    # Process per orbit group
    for g_start, g_end in zip(group_starts, group_ends):
        orbit_id = orbit_ids[g_start]
        det_ids_g = det_ids[g_start:g_end]
        times_g = times[g_start:g_end]
        M_rad_g = M_rad[g_start:g_end]
        n_rad_per_day_g = n_rad_per_day[g_start:g_end]

        dsu = DSU(len(det_ids_g))
        for i in range(len(det_ids_g)):
            # Neighborhood indices (j > i, capped)
            j_start = i + 1
            j_end = min(len(det_ids_g), i + 1 + max_neighbors_per_node)
            if j_start >= j_end:
                continue
            js = np.arange(j_start, j_end, dtype=np.int32)
            dt = times_g[js] - times_g[i]
            # Positive dt constraint and optional time window
            mask = dt > 0.0
            if max_time_days is not None:
                mask &= dt <= max_time_days
            if not np.any(mask):
                continue
            js = js[mask]
            dt = dt[mask]
            # Vectorized acceptance
            dM = ((M_rad_g[js] - M_rad_g[i]) + np.pi) % (2 * np.pi) - np.pi
            n_i = n_rad_per_day_g[i]
            # Choose k to minimize residual
            k = np.rint((n_i * dt - dM) / (2 * np.pi))
            predicted_dt = (dM + 2 * np.pi * k) / n_i
            residual_minutes = np.abs((dt - predicted_dt) * 24.0 * 60.0)
            tau_minutes = tau0_minutes + alpha_minutes_per_day * dt
            ok = residual_minutes <= tau_minutes
            if not np.any(ok):
                continue
            for j in js[ok]:
                dsu.union(i, int(j))

        # Collect components
        root_to_indices: dict[int, list[int]] = {}
        for idx in range(len(det_ids_g)):
            root = dsu.find(idx)
            root_to_indices.setdefault(root, []).append(idx)

        # Emit chains
        for indices in root_to_indices.values():
            if len(indices) == 0:
                continue
            chain_id = f"{orbit_id}_{uuid.uuid4().hex[:8]}"
            member_times = [float(times_g[k]) for k in indices]
            chains_data["chain_id"].append(chain_id)
            chains_data["orbit_id"].append(str(orbit_id))
            chains_data["size"].append(len(indices))
            chains_data["t_min"].append(min(member_times))
            chains_data["t_max"].append(max(member_times))

            for k in indices:
                members_data["chain_id"].append(chain_id)
                members_data["det_id"].append(str(det_ids_g[k]))
                members_data["time_mjd"].append(float(times_g[k]))

    chains = Chains.from_kwargs(**chains_data) if len(chains_data["chain_id"]) > 0 else Chains.empty()
    members = (
        ChainMembers.from_kwargs(**members_data)
        if len(members_data["chain_id"]) > 0
        else ChainMembers.empty()
    )

    return chains, members


def _wrap_angle_rad(delta: float) -> float:
    """
    Wrap an angle in radians to (-pi, pi].
    """
    return (delta + np.pi) % (2 * np.pi) - np.pi


def accept_time_consistency(
    M_i_rad: float,
    t_i_mjd: float,
    M_j_rad: float,
    t_j_mjd: float,
    n_rad_per_day: float,
    *,
    tau0_minutes: float = 10.0,
    alpha_minutes_per_day: float = 0.05,
) -> tuple[bool, int, float]:
    """
    Check Kepler clock time-consistency between two detections.

    Accept i→j if |Δt - (wrap(M_j - M_i) + 2π k)/n| ≤ τ(Δt), where τ(Δt) = τ0 + α Δt.

    Parameters
    ----------
    M_i_rad, M_j_rad : float
        Mean anomalies at observations i and j in radians.
    t_i_mjd, t_j_mjd : float
        Times (MJD, days) of observations i and j.
    n_rad_per_day : float
        Mean motion in radians per day.
    tau0_minutes : float
        Base tolerance in minutes.
    alpha_minutes_per_day : float
        Tolerance growth per day in minutes/day.

    Returns
    -------
    (accepted, k, residual_minutes)
        accepted: bool, k: chosen revolution integer, residual in minutes.
    """
    dt_days = t_j_mjd - t_i_mjd
    if dt_days <= 0.0 or n_rad_per_day <= 0.0:
        return False, 0, float("inf")

    dM = _wrap_angle_rad(M_j_rad - M_i_rad)
    # Choose integer k to minimize time residual
    k_float = (n_rad_per_day * dt_days - dM) / (2 * np.pi)
    k = int(np.round(k_float))
    predicted_dt_days = (dM + 2 * np.pi * k) / n_rad_per_day
    residual_days = dt_days - predicted_dt_days
    residual_minutes = abs(residual_days * 24.0 * 60.0)

    tau_minutes = tau0_minutes + alpha_minutes_per_day * dt_days
    return residual_minutes <= tau_minutes, k, residual_minutes
