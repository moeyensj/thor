import logging
from typing import Callable, List

import numba
import numpy as np
import numpy.typing as npt
import ray

from .velocity_grid import VelocityGridBase, _cluster_velocity_find_worker

logger = logging.getLogger(__name__)


# --- Hotspot2D numba internals ---


@numba.njit
def _enforce_shape(points):
    """Ensure that datapoints are in a shape of (2, N)."""
    if points.shape[0] != 2:
        return points.T
    return points


@numba.njit
def _make_points_nonzero(points):
    # Scale up to nonzero
    #
    # Careful: numba.njit(parallel=True) would give wrong result, since min()
    # would get re-evaluated!
    return points - points.min()


@numba.njit(parallel=False)
def _quantize_points(points, eps):
    """Quantize points to be scaled in units of eps."""
    return (points / eps).astype(np.int32)


@numba.njit
def _sort_order_2d(points):
    """
    Return the indices that would sort points by x and then y. Points must be
    integers.

    This is about twice as fast as np.lexsort(points), and can be numba-d. It
    works by transforming the 2D sequence of int pairs into a 1D sequence of
    integers, and then sorting that 1D sequence.
    """
    scale = points.max() + 1
    return np.argsort(points[0, :] * scale + points[1, :])


@numba.njit
def _find_runs(sorted_points, min_samples, expected_n_clusters=16):
    """
    Find all subsequences of at least min_samples length with the same x, y
    values in a sorted dataset of (x, y) values.

    Returns a 2D array of the (x, y) values that match. The array may have
    duplicated values.
    """
    result = np.empty((2, expected_n_clusters * 2), dtype="float64")
    n_hit = 0

    n_consecutive = 1
    prev_x = np.nan
    prev_y = np.nan

    for i in range(sorted_points.shape[1]):
        if sorted_points[0, i] == prev_x and sorted_points[1, i] == prev_y:
            n_consecutive += 1
            if n_consecutive == min_samples:
                if n_hit == result.shape[1]:
                    result = _extend_2d_array(result, result.shape[1] * 2)
                result[0, n_hit] = sorted_points[0, i]
                result[1, n_hit] = sorted_points[1, i]
                n_hit = n_hit + 1
        else:
            prev_x = sorted_points[0, i]
            prev_y = sorted_points[1, i]
            n_consecutive = 1
    return result[:, :n_hit]


@numba.njit
def _extend_2d_array(src, new_size):
    dst = np.empty((2, new_size), dtype=src.dtype)
    for i in range(src.shape[1]):
        dst[0, i] = src[0, i]
        dst[1, i] = src[1, i]
    return dst


@numba.njit
def _label_clusters(runs, points_quantized):
    """
    Produce a 1D array of integers which label each X-Y in points_quantized with
    a cluster.
    """
    labels = np.full(points_quantized.shape[1], -1, np.int64)
    for i in range(points_quantized.shape[1]):
        for j in range(runs.shape[1]):
            if runs[0, j] == points_quantized[0, i] and runs[1, j] == points_quantized[1, i]:
                labels[i] = j
                break
    return labels


@numba.njit
def _hotspot_2d_inner(points, eps, min_samples):
    """
    Core work of the hotspot2d algorithm: quantize the points, sort them, find
    runs in the sorted list, and label each point with an ID from the runs.
    """
    points_quantized = _quantize_points(_make_points_nonzero(points), 2 * eps)
    sort_order = _sort_order_2d(points_quantized)
    sorted_points = points_quantized[:, sort_order]
    runs = _find_runs(sorted_points, min_samples)
    cluster_labels = _label_clusters(runs, points_quantized)
    return cluster_labels


@numba.njit(parallel=False)
def _adjust_labels(labels, new_minimum):
    """
    Given a bunch of integer labels, adjust the labels to start at new_minimum.
    """
    labels[labels != -1] = labels[labels != -1] + new_minimum


@numba.njit
def _build_label_aliases(labels1, labels2, labels3, labels4, n):
    label_aliases = {}
    for i in range(n):
        # Prefer names from labels1, then labels2, then labels3, then labels4.
        if labels1[i] != -1:
            label = labels1[i]
            if labels2[i] != -1:
                label_aliases[labels2[i]] = label
            if labels3[i] != -1:
                label_aliases[labels3[i]] = label
            if labels4[i] != -1:
                label_aliases[labels4[i]] = label
        elif labels2[i] != -1:
            label = labels2[i]
            if labels3[i] != -1:
                label_aliases[labels3[i]] = label
            if labels4[i] != -1:
                label_aliases[labels4[i]] = label
        elif labels3[i] != -1:
            label = labels3[i]
            if labels4[i] != -1:
                label_aliases[labels4[i]] = label
    return label_aliases


@numba.njit
def _hotspot_multilabel(points, eps, min_samples):
    """
    Run the hotspot2d algorithm 4 times. Each time, the input points are
    adjusted a bit, offsetting them by 'eps' in the X, then Y, then X and Y
    directions. This helps deal with edge effects in the binning of datapoints.
    """
    n = points.shape[1]

    # Find and label runs in the dataset.
    labels1 = _hotspot_2d_inner(points, eps, min_samples)

    # Repeat, but with X+eps, Y
    for i in range(n):
        points[0, i] = points[0, i] + eps
    labels2 = _hotspot_2d_inner(points, eps, min_samples)
    # Adjust labels so they don't collide with those of labels1.
    _adjust_labels(labels2, labels1.max() + 1)

    # Repeat, but with X+eps, Y+eps.
    for i in range(n):
        points[1, i] = points[1, i] + eps
    labels3 = _hotspot_2d_inner(points, eps, min_samples)
    _adjust_labels(labels3, labels2.max() + 1)

    # Repeat, but with X, Y+eps
    for i in range(n):
        points[0, i] = points[0, i] - eps
    labels4 = _hotspot_2d_inner(points, eps, min_samples)
    _adjust_labels(labels4, labels3.max() + 1)

    # Make an empty array which will store the cluster IDs of each point.
    final_labels = np.full(n, -1, dtype=labels1.dtype)

    # Many of the label arrays we built will have the same clusters, but with
    # different integer labels. Build a mapping to standardize things.
    label_aliases = _build_label_aliases(labels1, labels2, labels3, labels4, n)

    # Apply labels.
    for i in range(n):
        if labels1[i] != -1:
            final_labels[i] = labels1[i]
        elif labels2[i] != -1:
            final_labels[i] = label_aliases.get(labels2[i], labels2[i])
        elif labels3[i] != -1:
            final_labels[i] = label_aliases.get(labels3[i], labels3[i])
        elif labels4[i] != -1:
            final_labels[i] = label_aliases.get(labels4[i], labels4[i])

    return final_labels


def _find_clusters_hotspots_2d(
    points: npt.NDArray[np.float64],
    eps: float,
    min_samples: int,
) -> List[npt.NDArray[np.int64]]:
    """
    Find clusters using the Hotspot2D algorithm.

    This algorithm works by quantizing points into a grid, with a resolution of
    2*eps. It searches the quantized points to find values that appear at least
    min_samples times in the dataset.

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
    points = points.copy()
    points = _enforce_shape(points)

    clusters = []
    cluster_labels = _hotspot_multilabel(points, eps, min_samples)
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        if label == -1:
            continue
        cluster_indices = np.where(cluster_labels == label)[0]
        clusters.append(cluster_indices)

    return clusters


# --- Ray worker and class ---


def _hotspot2d_find_worker(
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
    """Ray-serializable worker that uses Hotspot2D point clustering."""
    return _cluster_velocity_find_worker(
        vx,
        vy,
        transformed_detections,
        radius=radius,
        min_obs=min_obs,
        min_arc_length=min_arc_length,
        min_nights=min_nights,
        point_cluster_fn=_find_clusters_hotspots_2d,
        alg_name="Hotspot2D",
        tracklets=tracklets,
        tracklet_members=tracklet_members,
    )


_hotspot2d_find_remote = ray.remote(_hotspot2d_find_worker)
_hotspot2d_find_remote.options(num_returns=1, num_cpus=1)


class VelocityGridHotspot2D(VelocityGridBase):
    """
    Clustering algorithm that performs a velocity-grid sweep with Hotspot2D
    at each grid point.

    This implements the `ClusteringAlgorithm` protocol. Hotspot2D is much
    faster (perhaps 10-20x) than DBSCAN, but may miss some clusters,
    particularly when points are spaced a distance of 'eps' apart.

    See `VelocityGridBase` for parameter documentation.
    """

    @property
    def _alg_name(self) -> str:
        return "Hotspot2D"

    def _point_cluster_fn(self) -> Callable:
        return _find_clusters_hotspots_2d

    def _make_ray_remote(self) -> ray.remote_function.RemoteFunction:
        return _hotspot2d_find_remote
