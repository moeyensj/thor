"""
Backward-compatible re-export shim.

All clustering functionality has moved to :mod:`thor.clustering`.
This module re-exports the public API so that existing imports
(``from thor.clusters import …``) continue to work.
"""

# ruff: noqa: F401
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import ray

from .clustering import (
    ClusterMembers,
    Clusters,
    FittedClusterMembers,
    FittedClusters,
    VelocityGridDBSCAN,
    VelocityGridHotspot2D,
    calculate_clustering_parameters_from_covariance,
    drop_duplicate_clusters,
    filter_clusters_by_length,
    fit_clusters,
    hash_obs_ids,
)
from .clustering.algorithms import ClusteringAlgorithm
from .orbit import TestOrbitEphemeris
from .range_and_transform import TransformedDetections

logger = logging.getLogger(__name__)

# Legacy names for the old 2D-only clustering algorithm classes
DBSCANClustering = VelocityGridDBSCAN
Hotspot2DClustering = VelocityGridHotspot2D

__all__ = [
    "cluster_and_link",
    "fit_clusters",
    "calculate_clustering_parameters_from_covariance",
    "ClusteringAlgorithm",
    "DBSCANClustering",
    "Hotspot2DClustering",
    "VelocityGridDBSCAN",
    "VelocityGridHotspot2D",
    "Clusters",
    "ClusterMembers",
    "FittedClusters",
    "FittedClusterMembers",
    "drop_duplicate_clusters",
    "hash_obs_ids",
    "filter_clusters_by_length",
]


def cluster_and_link(
    observations: Union[TransformedDetections, ray.ObjectRef],
    test_orbit_ephemeris: Optional[Union[TestOrbitEphemeris, ray.ObjectRef]] = None,
    velocity_bin_separation: float = 2.0,
    min_obs: int = 5,
    min_arc_length: float = 1.0,
    min_nights: int = 3,
    mahalanobis_distance: Optional[float] = None,
    alg: Union[str, "ClusteringAlgorithm"] = "dbscan",
    radius: float = 0.005,
    radius_multiplier: float = 5.0,
    density_multiplier: float = 2.5,
    min_radius: float = 1 / 3600,
    max_radius: float = 0.05,
    vx_range: Optional[List[float]] = None,
    vy_range: Optional[List[float]] = None,
    vx_bins: Optional[int] = None,
    vy_bins: Optional[int] = None,
    vx_values: Optional[npt.NDArray[np.float64]] = None,
    vy_values: Optional[npt.NDArray[np.float64]] = None,
    chunk_size: int = 1000,
    max_processes: Optional[int] = 1,
    whiten: bool = False,
) -> Tuple[Clusters, ClusterMembers]:
    """
    Cluster and link correctly projected (after ranging and shifting)
    detections.

    This is a convenience wrapper that instantiates the appropriate
    algorithm class from the ``alg`` parameter and delegates to its
    ``find_clusters`` method.

    Parameters
    ----------
    observations : TransformedDetections or ray.ObjectRef
        Transformed detections to cluster.
    test_orbit_ephemeris : TestOrbitEphemeris or ray.ObjectRef, optional
        Test orbit ephemeris with covariances.
    velocity_bin_separation : float, optional
        Separation between velocity bins in units of clustering radius.
    min_obs : int, optional
        Minimum number of observations per cluster.
    min_arc_length : float, optional
        Minimum arc length in days for a cluster to be accepted.
    min_nights : int, optional
        Minimum number of unique nights a cluster must span.
    mahalanobis_distance : float, optional
        Mahalanobis distance threshold for covariance-based parameters.
    alg : str or ClusteringAlgorithm, optional
        Algorithm to use. Can be "dbscan", "hotspot_2d", or a
        `ClusteringAlgorithm` instance. Default: "dbscan"
    radius : float, optional
        Clustering radius in degrees. Default: 0.005.
    radius_multiplier : float, optional
        Multiplier on max positional sigma for radius lower bound.
    density_multiplier : float, optional
        Multiplier on characteristic separation for radius upper bound.
    min_radius : float, optional
        Minimum radius in degrees.
    max_radius : float, optional
        Maximum radius in degrees.
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
    chunk_size : int, optional
        Number of velocity grid points per worker chunk.
    max_processes : int, optional
        Maximum number of processes for parallelization.
    whiten : bool, optional
        Whether to compute whitened parameters in metadata.

    Returns
    -------
    clusters : Clusters
        Unfitted clusters found by the clustering algorithm.
    cluster_members : ClusterMembers
        Members of each cluster.
    """
    # If alg is already a ClusteringAlgorithm instance, use it directly
    if not isinstance(alg, str):
        return alg.find_clusters(observations, test_orbit_ephemeris)

    # Build kwargs common to both algorithms
    kwargs = dict(
        radius=radius,
        min_obs=min_obs,
        min_arc_length=min_arc_length,
        min_nights=min_nights,
        vx_range=vx_range,
        vy_range=vy_range,
        vx_bins=vx_bins,
        vy_bins=vy_bins,
        vx_values=vx_values,
        vy_values=vy_values,
        velocity_bin_separation=velocity_bin_separation,
        mahalanobis_distance=mahalanobis_distance,
        radius_multiplier=radius_multiplier,
        density_multiplier=density_multiplier,
        min_radius=min_radius,
        max_radius=max_radius,
        chunk_size=chunk_size,
        max_processes=max_processes,
        whiten=whiten,
    )

    if alg == "dbscan":
        algorithm = VelocityGridDBSCAN(**kwargs)
    elif alg == "hotspot_2d":
        algorithm = VelocityGridHotspot2D(**kwargs)
    else:
        raise NotImplementedError(f"algorithm '{alg}' is not implemented")

    return algorithm.find_clusters(observations, test_orbit_ephemeris)
