import logging
from typing import Callable, List

import numpy as np
import numpy.typing as npt
import ray
from sklearn.cluster import OPTICS

from .velocity_grid import VelocityGridBase, _cluster_velocity_find_worker

logger = logging.getLogger(__name__)


def _find_clusters_optics(
    points: npt.NDArray[np.float64],
    eps: float,
    min_samples: int,
) -> List[npt.NDArray[np.int64]]:
    """
    Find clusters using the OPTICS algorithm with xi steepness extraction.

    Unlike DBSCAN which uses a single fixed-radius threshold, OPTICS
    computes a reachability ordering of all points and then extracts
    clusters by detecting steep drops/rises in the reachability plot
    (the "xi" method). This allows it to find clusters at multiple
    density scales simultaneously — e.g. a tight bright-object cluster
    embedded in a looser faint-object cluster.

    ``max_eps`` is set to ``eps`` to limit the neighborhood search
    radius (same computational bound as DBSCAN), but the actual cluster
    boundaries are determined by the reachability structure, not by a
    hard distance cut.

    Parameters
    ----------
    points : `~numpy.ndarray` (N, 2)
        2D array of point coordinates.
    eps : float
        Maximum neighborhood radius (limits the search, not the
        cluster extraction).
    min_samples : int
        Minimum number of samples in a neighborhood for a point
        to be considered a core point.

    Returns
    -------
    clusters : list of `~numpy.ndarray`
        List of arrays, each containing the indices of points
        belonging to a cluster.
    """
    op = OPTICS(
        max_eps=eps,
        min_samples=min_samples,
        cluster_method="xi",
        xi=0.05,
        algorithm="ball_tree",
        leaf_size=30,
    )
    op.fit(points)

    cluster_labels = np.unique(op.labels_[op.labels_ != -1])
    clusters = []
    for label in cluster_labels:
        cluster_indices = np.where(op.labels_ == label)[0]
        clusters.append(cluster_indices)
    del op
    return clusters


def _optics_find_worker(
    vx, vy, transformed_detections, radius=1 / 3600, min_obs=6, min_arc_length=1.5, min_nights=3
):
    """Ray-serializable worker that uses OPTICS point clustering."""
    return _cluster_velocity_find_worker(
        vx,
        vy,
        transformed_detections,
        radius=radius,
        min_obs=min_obs,
        min_arc_length=min_arc_length,
        min_nights=min_nights,
        point_cluster_fn=_find_clusters_optics,
        alg_name="OPTICS",
    )


_optics_find_remote = ray.remote(_optics_find_worker)
_optics_find_remote.options(num_returns=1, num_cpus=1)


class VelocityGridOPTICS(VelocityGridBase):
    """
    Clustering algorithm that performs a velocity-grid sweep with OPTICS
    at each grid point.

    OPTICS (Ordering Points To Identify Clustering Structure) is a
    density-based algorithm that can find clusters at varying density
    scales. It uses xi-steepness extraction to detect clusters from
    the reachability plot, unlike DBSCAN which applies a single
    fixed-radius threshold.

    See `VelocityGridBase` for parameter documentation.
    """

    @property
    def _alg_name(self) -> str:
        return "OPTICS"

    def _point_cluster_fn(self) -> Callable:
        return _find_clusters_optics

    def _make_ray_remote(self) -> ray.remote_function.RemoteFunction:
        return _optics_find_remote
