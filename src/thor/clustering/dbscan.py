import logging
from typing import Callable, List

import numpy as np
import numpy.typing as npt
import ray

from .velocity_grid import VelocityGridBase, _cluster_velocity_find_worker

# Disable GPU until the GPU-accelerated clustering codes
# are better tested and implemented
USE_GPU = False

if USE_GPU:
    from cuml.cluster import DBSCAN
else:
    from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


def _find_clusters_dbscan(
    points: npt.NDArray[np.float64],
    eps: float,
    min_samples: int,
) -> List[npt.NDArray[np.int64]]:
    """
    Find clusters using the DBSCAN algorithm.

    Parameters
    ----------
    points : `~numpy.ndarray` (N, 2)
        2D array of point coordinates.
    eps : float
        Maximum distance between two samples for them to be
        considered in the same neighborhood.
    min_samples : int
        Minimum number of samples in a neighborhood for a point
        to be considered a core point.

    Returns
    -------
    clusters : list of `~numpy.ndarray`
        List of arrays, each containing the indices of points
        belonging to a cluster.
    """
    # ball_tree algorithm appears to run about 30-40% faster based on a single
    # test orbit and (vx, vy), run on a laptop, improving from 300ms to 180ms.
    #
    # Runtime is not very sensitive to leaf_size, but 30 appears to be roughly
    # optimal, and is the default value anyway.
    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        algorithm="ball_tree",
        leaf_size=30,
    )
    db.fit(points)

    cluster_labels = np.unique(db.labels_[np.where(db.labels_ != -1)])
    clusters = []
    for label in cluster_labels:
        cluster_indices = np.where(db.labels_ == label)[0]
        clusters.append(cluster_indices)
    del db
    return clusters


def _dbscan_find_worker(
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
    """Ray-serializable worker that uses DBSCAN point clustering."""
    return _cluster_velocity_find_worker(
        vx,
        vy,
        transformed_detections,
        radius=radius,
        min_obs=min_obs,
        min_arc_length=min_arc_length,
        min_nights=min_nights,
        point_cluster_fn=_find_clusters_dbscan,
        alg_name="DBSCAN",
        tracklets=tracklets,
        tracklet_members=tracklet_members,
    )


_dbscan_find_remote = ray.remote(_dbscan_find_worker)
_dbscan_find_remote.options(num_returns=1, num_cpus=1)


class VelocityGridDBSCAN(VelocityGridBase):
    """
    Clustering algorithm that performs a velocity-grid sweep with DBSCAN
    at each grid point.

    This implements the `ClusteringAlgorithm` protocol. DBSCAN is certain
    to find all clusters with at least ``min_obs`` points that are separated
    by at most ``radius``, but is slower than Hotspot2D.

    See `VelocityGridBase` for parameter documentation.
    """

    @property
    def _alg_name(self) -> str:
        return "DBSCAN"

    def _point_cluster_fn(self) -> Callable:
        return _find_clusters_dbscan

    def _make_ray_remote(self) -> ray.remote_function.RemoteFunction:
        return _dbscan_find_remote
