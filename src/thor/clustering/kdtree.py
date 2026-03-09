import logging
from typing import Callable, List

import numpy as np
import numpy.typing as npt
import ray
from scipy.spatial import cKDTree

from .velocity_grid import VelocityGridBase, _cluster_velocity_find_worker

logger = logging.getLogger(__name__)


def _find_clusters_kdtree(
    points: npt.NDArray[np.float64],
    eps: float,
    min_samples: int,
) -> List[npt.NDArray[np.int64]]:
    """
    Find clusters using KD-tree range queries and connected components.

    For each point, counts neighbors within ``eps``. Points with at least
    ``min_samples`` neighbors are "hot". Connected components of hot points
    (sharing neighbors) form clusters, and non-hot points within ``eps`` of
    a hot point are included in its cluster.

    Parameters
    ----------
    points : `~numpy.ndarray` (N, 2)
        2D array of point coordinates.
    eps : float
        Maximum distance for neighbor queries.
    min_samples : int
        Minimum number of neighbors (including self) for a point
        to be considered a core point.

    Returns
    -------
    clusters : list of `~numpy.ndarray`
        List of arrays, each containing the indices of points
        belonging to a cluster.
    """
    n = len(points)
    if n == 0:
        return []

    tree = cKDTree(points, leafsize=30)
    neighbors = tree.query_ball_point(points, eps)

    # Identify core points (those with >= min_samples neighbors)
    is_core = np.array([len(nb) >= min_samples for nb in neighbors], dtype=bool)

    if not np.any(is_core):
        return []

    # Union-find for connected components of core points
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Connect core points that share neighbors
    for i in range(n):
        if not is_core[i]:
            continue
        for j in neighbors[i]:
            if is_core[j]:
                union(i, j)

    # Assign border points to the nearest core point's component
    labels = np.full(n, -1, dtype=np.intp)
    for i in range(n):
        if is_core[i]:
            labels[i] = find(i)

    for i in range(n):
        if labels[i] != -1:
            continue
        for j in neighbors[i]:
            if is_core[j]:
                labels[i] = find(j)
                break

    # Extract clusters
    unique_labels = np.unique(labels[labels != -1])
    clusters = []
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        clusters.append(cluster_indices)

    return clusters


def _kdtree_find_worker(
    vx,
    vy,
    transformed_detections,
    radius=1 / 3600,
    min_obs=6,
    min_arc_length=1.5,
    min_nights=3,
    tracklets=None,
    tracklet_members=None,
):
    """Ray-serializable worker that uses KD-tree point clustering."""
    return _cluster_velocity_find_worker(
        vx,
        vy,
        transformed_detections,
        radius=radius,
        min_obs=min_obs,
        min_arc_length=min_arc_length,
        min_nights=min_nights,
        point_cluster_fn=_find_clusters_kdtree,
        alg_name="KDTree",
        tracklets=tracklets,
        tracklet_members=tracklet_members,
    )


_kdtree_find_remote = ray.remote(_kdtree_find_worker)
_kdtree_find_remote.options(num_returns=1, num_cpus=1)


class VelocityGridKDTree(VelocityGridBase):
    """
    Clustering algorithm that performs a velocity-grid sweep with KD-tree
    range-count clustering at each grid point.

    Instead of DBSCAN's iterative core-point expansion, directly queries
    a KD-tree for neighbors within ``radius``. Points with enough neighbors
    are "hot" and connected components of hot points form clusters.
    Potentially faster than DBSCAN for sparse point clouds.

    See `VelocityGridBase` for parameter documentation.
    """

    @property
    def _alg_name(self) -> str:
        return "KDTree"

    def _point_cluster_fn(self) -> Callable:
        return _find_clusters_kdtree

    def _make_ray_remote(self) -> ray.remote_function.RemoteFunction:
        return _kdtree_find_remote
