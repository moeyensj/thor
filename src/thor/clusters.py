import hashlib
import logging
import multiprocessing as mp
import time
import uuid
from typing import List, Literal, Optional, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_core.coordinates import Origin
from adam_core.coordinates.residuals import Residuals
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp
from adam_core.utils.iter import _iterate_chunks

from .orbit import TestOrbitEphemeris
from .projections import GnomonicCoordinates
from .range_and_transform import TransformedDetections

# Disable GPU until the GPU-accelerated clustering codes
# are better tested and implemented
USE_GPU = False

if USE_GPU:
    from cuml.cluster import DBSCAN
else:
    from sklearn.cluster import DBSCAN


__all__ = [
    "cluster_and_link",
    "calculate_clustering_parameters_from_covariance",
    "Clusters",
    "ClusterMembers",
    "FittedClusters",
    "FittedClusterMembers",
]

logger = logging.getLogger(__name__)


def hash_obs_ids(obs_ids: List[str]) -> str:
    """
    Create unique strings for each set unique set of observation IDs

    We use hashes rather than original string in order to save memory.
    """
    return hashlib.md5("".join(sorted(set(obs_ids))).encode()).hexdigest()


def drop_duplicate_clusters(
    clusters: "FittedClusters",
    cluster_members: "FittedClusterMembers",
) -> Tuple["FittedClusters", "FittedClusterMembers"]:
    """
    Drop clusters that have identical sets of observation IDs.

    Parameters
    ----------
    clusters: `~thor.clusters.Clusters`
        A table of clusters. Must be sorted by cluster_id.
    cluster_members: `~thor.clusters.ClusterMembers`
        A table of cluster members. Must be sorted by cluster_id.

    Returns
    -------
    `~thor.clusters.Clusters`, `~thor.clusters.ClusterMembers`
        A table of clusters with duplicate clusters removed.
        The cluster members belonging to those clusters.
    """
    if isinstance(clusters, ray.ObjectRef):
        clusters = ray.get(clusters)
    if isinstance(cluster_members, ray.ObjectRef):
        cluster_members = ray.get(cluster_members)

    if len(clusters) == 0 or len(cluster_members) == 0:
        return FittedClusters.empty(), FittedClusterMembers.empty()

    # Ensure clusters and cluster members are sorted by cluster id
    # by spot checking the first few and last few rows are
    # in sorted order
    assert clusters.cluster_id[:3].to_pylist() == sorted(
        clusters.cluster_id[:3].to_pylist()
    ), "clusters must be sorted by cluster_id"  # noqa: E501
    assert clusters.cluster_id[-3:].to_pylist() == sorted(
        clusters.cluster_id[-3:].to_pylist()
    ), "clusters must be sorted by cluster_id"  # noqa: E501
    assert cluster_members.cluster_id[:3].to_pylist() == sorted(
        cluster_members.cluster_id[:3].to_pylist()
    ), "cluster_members must be sorted by cluster_id"  # noqa: E501
    assert cluster_members.cluster_id[-3:].to_pylist() == sorted(
        cluster_members.cluster_id[-3:].to_pylist()
    ), "cluster_members must be sorted by cluster_id"  # noqa: E501

    # We used to use a group by in pyarrow here,
    # but found the memory accumulationw as too high.
    # A simple loop that accumulates the distinct obs ids
    # for each cluster is more memory efficient.
    logger.info("Accumulating cluster observation IDs into single strings.")
    obs_ids_per_cluster: Union[List[str], pa.Array] = []
    current_obs_ids: List[str] = []
    current_cluster_id = None
    for member in cluster_members:
        cluster_id = member.cluster_id.to_pylist()[0]
        obs_id = member.obs_id.to_pylist()[0]
        if cluster_id != current_cluster_id:
            if current_cluster_id is not None:
                obs_ids_per_cluster.append(hash_obs_ids(current_obs_ids))
            current_cluster_id = cluster_id
            current_obs_ids = []
        current_obs_ids.append(obs_id)
    obs_ids_per_cluster.append(hash_obs_ids(current_obs_ids))

    logger.info("Grouping by unique observation sets.")
    obs_ids_per_cluster = pa.table(
        {
            "index": pa.array(np.arange(0, len(obs_ids_per_cluster))),
            "obs_ids": obs_ids_per_cluster,
        }
    )

    obs_ids_per_cluster = obs_ids_per_cluster.combine_chunks()
    obs_ids_per_cluster = obs_ids_per_cluster.group_by(["obs_ids"], use_threads=False)

    logger.info("Taking first index of each unique observation set.")
    indices = obs_ids_per_cluster.aggregate([("index", "first")])["index_first"]
    del obs_ids_per_cluster
    indices = indices.combine_chunks()

    logger.info("Taking clusters that belong to unique observation sets.")
    clusters = clusters.take(indices)

    logger.info("Taking cluster members that belong to unique clusters.")
    cluster_members = cluster_members.apply_mask(pc.is_in(cluster_members.cluster_id, clusters.cluster_id))
    return clusters, cluster_members


class Clusters(qv.Table):
    cluster_id = qv.LargeStringColumn(default=lambda: uuid.uuid4().hex)
    vtheta_x = qv.Float64Column()
    vtheta_y = qv.Float64Column()
    arc_length = qv.Float64Column()
    num_obs = qv.Int64Column()


class ClusterMembers(qv.Table):
    cluster_id = qv.LargeStringColumn()
    obs_id = qv.LargeStringColumn()


class FittedClusters(qv.Table):
    cluster_id = qv.LargeStringColumn(default=lambda: uuid.uuid4().hex)
    test_orbit_id = qv.LargeStringColumn(nullable=True)
    time = Timestamp.as_column()
    theta_x0 = qv.Float64Column()
    theta_y0 = qv.Float64Column()
    vtheta_x = qv.Float64Column()
    vtheta_y = qv.Float64Column()
    atheta_x = qv.Float64Column()
    atheta_y = qv.Float64Column()
    origin = Origin.as_column()
    frame = qv.StringAttribute(default="testorbit")
    arc_length = qv.Float64Column()
    num_obs = qv.Int64Column()
    chi2 = qv.Float64Column()
    rchi2 = qv.Float64Column()

    def evaluate(self, times: Timestamp) -> Tuple[pa.Array, GnomonicCoordinates]:
        """
        Propagate cluster positions to given times using the fitted polynomial model.

        The polynomial motion model is:
            theta_x(t) = 0.5 * atheta_x * dt² + vtheta_x * dt + theta_x0
            theta_y(t) = 0.5 * atheta_y * dt² + vtheta_y * dt + theta_y0

        where dt = t - t0 (in days).

        Parameters
        ----------
        times : Timestamp
            Times to propagate to.

        Returns
        -------
        cluster_ids : pa.Array
            Cluster IDs for each time.
        coords : GnomonicCoordinates
            Gnomonic coordinates for each time.

        Examples
        --------
        >>> # Evaluate clusters to given times
        >>> cluster_ids, coords = clusters.evaluate(times)
        """
        times_stacked = qv.concatenate([times for i in range(len(self))])
        origin_stacked = pa.concat_arrays(
            [pa.repeat(self.origin.code[i], len(times)) for i in range(len(self))]
        )
        epochs_mjd_stacked = np.repeat(
            self.time.rescale("tdb").mjd().to_numpy(zero_copy_only=False), len(times)
        )
        x0_stacked = np.repeat(self.theta_x0.to_numpy(zero_copy_only=False), len(times))
        y0_stacked = np.repeat(self.theta_y0.to_numpy(zero_copy_only=False), len(times))
        ax_stacked = np.repeat(self.atheta_x.to_numpy(zero_copy_only=False), len(times))
        ay_stacked = np.repeat(self.atheta_y.to_numpy(zero_copy_only=False), len(times))
        vx_stacked = np.repeat(self.vtheta_x.to_numpy(zero_copy_only=False), len(times))
        vy_stacked = np.repeat(self.vtheta_y.to_numpy(zero_copy_only=False), len(times))

        dt = times_stacked.rescale("tdb").mjd().to_numpy(zero_copy_only=False) - epochs_mjd_stacked
        x = 0.5 * ax_stacked * dt**2 + vx_stacked * dt + x0_stacked
        y = 0.5 * ay_stacked * dt**2 + vy_stacked * dt + y0_stacked
        cluster_ids = np.repeat(self.cluster_id.to_numpy(zero_copy_only=False), len(times))

        coords = GnomonicCoordinates.from_kwargs(
            theta_x=x,
            theta_y=y,
            time=times_stacked,
            origin=Origin.from_kwargs(code=origin_stacked),
            frame="testorbit",
        )
        return cluster_ids, coords


class FittedClusterMembers(qv.Table):
    cluster_id = qv.LargeStringColumn()
    obs_id = qv.LargeStringColumn()
    test_orbit_id = qv.LargeStringColumn(nullable=True)
    residuals = Residuals.as_column()


def find_clusters(points, eps, min_samples, alg="hotspot_2d"):
    """
    Find all clusters in a 2-dimensional array of datapoints.

    Parameters
    ----------
    points: `~numpy.ndarray' (N x N)
        A 2-dimensional grid of (x, y) points to be clustered.
    eps: float
        The minimum distance between two points to be
        used to establish that they are in the same cluster.
    min_samples: into
        The minumum number of points in a cluster.
    alg: str
        Algorithm to use. Can be "dbscan" or "hotspot_2d".

    Returns
    -------
    list of numpy.array
        A list of clusters. Each cluster is an array of indexes into points,
        indicating that the points are members of a cluster together.

    Notes
    -----

    The algorithm chosen can have a big impact on performance and accuracy.

    alg="dbscan" uses the DBSCAN algorithm of Ester et. al. It's relatively slow
    but works with high accuracy; it is certain to find all clusters with at
    least min_samples points that are separated by at most eps.

    alg="hotspot_2d" is much faster (perhaps 10-20x faster) than dbscan, but it
    may miss some clusters, particularly when points are spaced a distance of 'eps'
    apart.
    """
    if alg == "dbscan":
        return _find_clusters_dbscan(points, eps, min_samples)
    elif alg == "hotspot_2d":
        return _find_clusters_hotspots_2d(points, eps, min_samples)
    else:
        raise NotImplementedError(f"algorithm '{alg}' is not implemented")


def filter_clusters_by_length(clusters, dt, nights, min_samples, min_arc_length, min_nights):
    """
    Filter cluster results on the conditions that they span at least
    min_arc_length in the time dimension, that each point in the cluster
    is from a different dt value, and that they span at least min_nights.

    Parameters
    -----------
    clusters: `list of numpy.ndarray'
        A list of clusters. Each cluster should be an array of indexes
        of observations that are members of the same cluster. The indexes
        are into the 'dt' and 'nights' arrays.
    dt: `~numpy.ndarray' (N)
        Change in time from the 0th exposure in units of MJD.
    nights: `~numpy.ndarray' (N)
        Observing night for each observation.
    min_samples: int
        Minimum size for a cluster to be included.
    min_arc_length: float
        Minimum arc length in units of days for a cluster to be accepted.
    min_nights: int
        Minimum number of unique nights a cluster must span.

    Returns
    -------
    list of numpy.ndarray
        The original clusters list, filtered down.
    """
    filtered_clusters = []
    arc_lengths = []
    for cluster in clusters:
        dt_in_cluster = dt[cluster]
        nights_in_cluster = nights[cluster]
        num_obs = len(dt_in_cluster)
        arc_length = dt_in_cluster.max() - dt_in_cluster.min()
        num_nights = len(np.unique(nights_in_cluster))

        if (
            (num_obs == len(np.unique(dt_in_cluster)))
            and ((num_obs >= min_samples))
            and (arc_length >= min_arc_length)
            and (num_nights >= min_nights)
        ):
            filtered_clusters.append(cluster)
            arc_lengths.append(arc_length)

    return filtered_clusters, arc_lengths


def _find_clusters_hotspots_2d(points, eps, min_samples):
    """
    This algorithm works by quantizing points into a grid, with a resolution of
    2*eps. It searches the quantized points to find values that appear at least
    min_samples times in the dataset.

    The search of the grid is done by sorting the points, and then stepping
    through the points looking for consecutive runs of the same (x, y) value
    repeated at least min_samples times.

    This is repeated, with the X values offset by +eps, which addresses edge
    effects at the right boundaries of the 2*eps-sized grid windows.

    Similarly, it is repeated with the Y value offset by +eps (dealing with the
    bottom boundary) and with both X and Y offset (dealing with the corner
    boundary).
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


@numba.njit
def _hotspot_2d_inner(points, eps, min_samples):
    """
    This function holds the core work of the hotspot2d algorithm: quantize the
    points, sort them, find runs in the sorted list, and label each point with
    an ID from the runs that have been found.
    """
    points_quantized = _quantize_points(_make_points_nonzero(points), 2 * eps)
    sort_order = _sort_order_2d(points_quantized)
    sorted_points = points_quantized[:, sort_order]
    runs = _find_runs(sorted_points, min_samples)
    cluster_labels = _label_clusters(runs, points_quantized)
    return cluster_labels


@numba.njit
def _hotspot_multilabel(points, eps, min_samples):
    """
    Run the hotspot2d algorithm 4 times. Each time, the input points are
    adjusted a bit, offsetting them by 'eps' in the X, then Y, then X and Y
    directions. This helps deal with edge effects in the binning of datapoints.

    This code is wordy and repetitive, just in order to make things simple for
    numba's compiler, which has a big impact on performance.
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
    Find all subssequences of at least min_samples length with the same x, y
    values in a sorted dataset of (x, y) values.

    Returns a 2D array of the (x, y) values that match. The array may have
    duplicated values.

    expected_n_clusters is a tuning parameter: an array is preallocated
    proportional to this value. If the guess is too low, we will allocate more
    memory than necessary. If it's too high, we will have to spend time growing
    that array.
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


def _find_clusters_dbscan(points, eps, min_samples):
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


def cluster_velocity(
    obs_ids: npt.ArrayLike,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    dt: npt.NDArray[np.float64],
    nights: npt.NDArray[np.int64],
    vx: float,
    vy: float,
    radius: float = 1 / 3600,
    min_obs: int = 6,
    min_arc_length: float = 1.5,
    min_nights: int = 3,
    alg: Literal["hotspot_2d", "dbscan"] = "dbscan",
) -> Tuple[Clusters, ClusterMembers]:
    """
    Clusters THOR projection with different velocities
    in the projection plane using `~scipy.cluster.DBSCAN`.

    Parameters
    ----------
    obs_ids : `~numpy.ndarray' (N)
        Observation IDs.
    x : `~numpy.ndarray' (N)
        Projection space x coordinate in degrees or radians.
    y : `~numpy.ndarray' (N)
        Projection space y coordinate in degrees or radians.
    dt : `~numpy.ndarray' (N)
        Change in time from 0th exposure in units of MJD.
    nights : `~numpy.ndarray' (N)
        Observing night for each observation.
    vx : float
        Projection space x velocity in units of degrees or radians per day in MJD.
    vy : float
        Projection space y velocity in units of degrees or radians per day in MJD.
    radius : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
        See: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html
    min_obs : int, optional
        The number of samples (or total weight) in a neighborhood for a
        point to be considered as a core point. This includes the point itself.
        See: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html
    min_arc_length : float, optional
        Minimum arc length in units of days for a cluster to be accepted.
    min_nights : int, optional
        Minimum number of unique nights a cluster must span.
    alg : str, optional
        Algorithm to use ("dbscan" or "hotspot_2d").

    Returns
    -------
    clusters : Clusters
        Clusters found.
    cluster_members : lusterMembers
        Cluster members.
    """
    logger.debug(f"cluster: vx={vx} vy={vy} n_obs={len(obs_ids)}")
    xx = x - vx * dt
    yy = y - vy * dt

    # Drop NaNs before clustering to satisfy DBSCAN input requirements
    # TODO: We should figure out the geometry that causes the gnomonic transform to produce NaNs. Working
    # theory is an other corner case for orbits interior to the observer.
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

    X = np.stack((xx, yy), 1)

    clusters = find_clusters(X, radius, min_obs, alg=alg)
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
    else:
        cluster_ids = []
        cluster_num_obs = []
        cluster_members_cluster_ids = []
        cluster_members_obs_ids = []
        for cluster in clusters:
            id = uuid.uuid4().hex
            obs_ids_i = obs_ids[cluster]
            num_obs = len(obs_ids_i)

            cluster_ids.append(id)
            cluster_num_obs.append(num_obs)
            cluster_members_cluster_ids.append(np.full(num_obs, id))
            cluster_members_obs_ids.append(obs_ids_i)

        clusters = Clusters.from_kwargs(
            cluster_id=cluster_ids,
            vtheta_x=np.full(len(cluster_ids), vx),
            vtheta_y=np.full(len(cluster_ids), vy),
            arc_length=arc_lengths,
            num_obs=cluster_num_obs,
        )

        cluster_members = ClusterMembers.from_kwargs(
            cluster_id=np.concatenate(cluster_members_cluster_ids).tolist(),
            obs_id=np.concatenate(cluster_members_obs_ids).tolist(),
        )

    return clusters, cluster_members


def cluster_velocity_worker(
    vx: npt.NDArray[np.float64],
    vy: npt.NDArray[np.float64],
    transformed_detections: TransformedDetections,
    test_orbit_ephemeris: Optional[TestOrbitEphemeris] = None,
    mahalanobis_distance: Optional[float] = None,
    radius: float = 1 / 3600,
    min_obs: int = 6,
    min_arc_length: float = 1.5,
    min_nights: int = 3,
    rchi2_threshold: float = 1e4,
    alg: Literal["hotspot_2d", "dbscan"] = "dbscan",
) -> Tuple[FittedClusters, FittedClusterMembers]:
    """
    Helper function for parallelizing cluster_velocity. This function takes a
    batch or chunk of velocities and returns the clusters and cluster members
    for that batch.

    """
    time_start = time.perf_counter()

    obs_ids = transformed_detections.id.to_numpy(zero_copy_only=False)
    nights = transformed_detections.night.to_numpy(zero_copy_only=False)
    x = transformed_detections.coordinates.theta_x.to_numpy(zero_copy_only=False)
    y = transformed_detections.coordinates.theta_y.to_numpy(zero_copy_only=False)
    mjd = transformed_detections.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    dt = mjd - mjd.min()

    fitted_clusters = FittedClusters.empty()
    fitted_cluster_members = FittedClusterMembers.empty()

    for vx_i, vy_i in zip(vx, vy):
        clusters_i, cluster_members_i = cluster_velocity(
            obs_ids,
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
            alg=alg,
        )
        if len(clusters_i) == 0:
            continue

        # Fit each cluster found for this velocity
        fitted_cluster_i, fitted_cluster_members_i = fit_cluster_worker(
            clusters_i, cluster_members_i, transformed_detections, clusters_i.cluster_id.to_pylist()
        )

        # Filter out clusters with rchi2 greater than the threshold
        fitted_cluster_i = fitted_cluster_i.apply_mask(pc.less_equal(fitted_cluster_i.rchi2, rchi2_threshold))
        fitted_cluster_members_i = fitted_cluster_members_i.apply_mask(
            pc.is_in(fitted_cluster_members_i.cluster_id, fitted_cluster_i.cluster_id)
        )

        fitted_clusters = qv.concatenate([fitted_clusters, fitted_cluster_i])
        if fitted_clusters.fragmented():
            fitted_clusters = qv.defragment(fitted_clusters)
        fitted_cluster_members = qv.concatenate([fitted_cluster_members, fitted_cluster_members_i])
        if fitted_cluster_members.fragmented():
            fitted_cluster_members = qv.defragment(fitted_cluster_members)

    time_end = time.perf_counter()
    logger.info(
        f"Found {len(fitted_clusters)} clusters for {len(vx)} velocity combinations in {time_end - time_start:.3f}s"
    )

    time_start_drop = time.perf_counter()
    logger.info("Removing duplicate clusters...")
    fitted_clusters = qv.defragment(fitted_clusters)
    fitted_cluster_members = qv.defragment(fitted_cluster_members)
    fitted_clusters = fitted_clusters.sort_by([("cluster_id", "ascending")])
    fitted_cluster_members = fitted_cluster_members.sort_by([("cluster_id", "ascending")])

    num_clusters = len(fitted_clusters)
    fitted_clusters, fitted_cluster_members = drop_duplicate_clusters(fitted_clusters, fitted_cluster_members)
    logger.info(f"Removed {num_clusters - len(fitted_clusters)} duplicate clusters.")
    time_end_drop = time.perf_counter()
    logger.info(f"Cluster deduplication completed in {time_end_drop - time_start_drop:.3f} seconds.")

    return fitted_clusters, fitted_cluster_members


cluster_velocity_remote = ray.remote(cluster_velocity_worker)
cluster_velocity_remote.options(
    num_returns=1,
    num_cpus=1,
)


def calculate_clustering_parameters_from_covariance(
    test_orbit_ephemeris: TestOrbitEphemeris,
    transformed_detections: Union[TransformedDetections, ray.ObjectRef],
    mahalanobis_distance: float = 3.0,
    velocity_bin_separation: float = 2.0,
    min_radius: float = 1 / 3600,
    min_bins: int = 10,
    max_bins: int = 1000,
    whiten: bool = False,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, dict]:
    """
    Calculate clustering parameters (velocity grid and radius) from test orbit
    ephemeris covariances in the co-rotating gnomonic frame.

    In the co-rotating frame, the test orbit has zero velocity. The velocity
    grid is centered at (0, 0) and extends to the specified Mahalanobis distance
    based on velocity uncertainties.

    The clustering radius is computed from the observation density, calculated
    as the total number of observations divided by the minimum area of the
    positional covariance ellipse.

    The ephemeris is automatically filtered to only include times within the
    observation time range, ensuring parameters are calculated from relevant
    covariances.

    Parameters
    ----------
    test_orbit_ephemeris : TestOrbitEphemeris
        Test orbit ephemeris with gnomonic coordinates containing covariances.
        The gnomonic coordinates should be in a co-rotating frame centered on
        the test orbit's motion. Will be filtered to observation time range.
    transformed_detections : TransformedDetections or ray.ObjectRef
        Transformed detections (observations in the co-rotating gnomonic frame).
        Can be either the Quivr table or a Ray object reference. Used to determine
        the time range for filtering ephemeris.
    mahalanobis_distance : float, optional
        Mahalanobis distance threshold for velocity grid and covariance area.
        For a 3-sigma ellipse in 2D, set this to 3.0.
        For a 2-sigma ellipse in 2D, set this to 2.0.
        [Default = 3.0]
    velocity_bin_separation : float, optional
        Separation between adjacent velocity bins in units of clustering radius.
        At maximum time offset (dt_arc), adjacent bins will shift observations by
        velocity_bin_separation × radius in position space.
        Higher values → coarser grid → fewer duplicates. Lower values → finer grid → more duplicates.
        Recommended: 2.0 (minimal overlap), 1.0 (bins touch), 0.5 (significant overlap).
        [Default = 2.0]
    min_radius : float, optional
        Minimum radius in degrees.
        [Default = 5/3600]
    min_bins : int, optional
        Minimum number of bins per dimension.
        [Default = 10]
    max_bins : int, optional
        Maximum number of bins per dimension.
    whiten : bool, optional
        If True, also report clustering parameters in whitened (sigma) units
        derived from the positional/velocity covariances. Default is False.
        [Default = 1000]

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

    Raises
    ------
    ValueError
        If covariances are invalid or parameters cannot be computed.
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
    # cross_cov = mean_cov[0:2, 2:4]  # cross terms

    # Pre-compute scale factors for optional whitening metadata
    pos_scales = np.sqrt(np.maximum(np.diag(pos_cov), 1e-18))
    vel_scales = np.sqrt(np.maximum(np.diag(vel_cov), 1e-18))
    pos_scale_rms = np.sqrt(np.mean(pos_scales**2))

    # === Calculate effective clustering radius from observation density ===

    # Find minimum positional covariance area at n_sigma across all times
    # Area of ellipse at n_sigma is: A = π * n_sigma^2 * sqrt(det(pos_cov))
    pos_covariances = covariances[:, 0:2, 0:2]

    # Calculate determinant for each positional covariance
    # det([[a, b], [c, d]]) = ad - bc
    dets = (
        pos_covariances[:, 0, 0] * pos_covariances[:, 1, 1]
        - pos_covariances[:, 0, 1] * pos_covariances[:, 1, 0]
    )

    # Filter out invalid determinants
    valid_dets = dets[np.isfinite(dets) & (dets > 0)]

    if len(valid_dets) == 0:
        raise ValueError("No valid positional covariance determinants found.")

    # Use maximum positional covariance area across times (conservative density)
    max_det = np.max(valid_dets)
    cov_area = np.pi * mahalanobis_distance_sq * np.sqrt(max_det)

    # Background density and characteristic separation
    density = n_obs / cov_area  # obs / deg^2

    # Work in whitened units when requested
    if whiten and pos_scale_rms > 0:
        area_scale = pos_scale_rms**2
        density_sigma = density * area_scale  # obs / sigma^2
        r_sep_sigma = 1.0 / np.sqrt(density_sigma) if density_sigma > 0 else np.inf
        radius_cand_sigma = np.sqrt(1.0 / (np.pi * density_sigma)) if density_sigma > 0 else np.inf
        radius_cand_sigma = max(radius_cand_sigma, min_radius / pos_scale_rms)

        c_max = 2.5
        k_astro = 1.2
        k_bin = 1.0

        epsilon_max_sigma = c_max * r_sep_sigma if np.isfinite(r_sep_sigma) else radius_cand_sigma
        epsilon_astro_sigma = k_astro * 1.0  # sigma units
        epsilon_bin_sigma = k_bin * velocity_bin_separation * radius_cand_sigma
        epsilon_min_sigma = np.sqrt(epsilon_astro_sigma**2 + epsilon_bin_sigma**2)

        radius_sigma = min(
            epsilon_max_sigma, max(epsilon_min_sigma, radius_cand_sigma, min_radius / pos_scale_rms)
        )
        radius = radius_sigma * pos_scale_rms
        r_sep = r_sep_sigma * pos_scale_rms if np.isfinite(r_sep_sigma) else np.inf
        radius_cand = radius_cand_sigma * pos_scale_rms
        epsilon_max = epsilon_max_sigma * pos_scale_rms if np.isfinite(epsilon_max_sigma) else radius_cand
        epsilon_min = epsilon_min_sigma * pos_scale_rms
        epsilon_astro = epsilon_astro_sigma * pos_scale_rms
        epsilon_bin = epsilon_bin_sigma * pos_scale_rms
    else:
        # Separation scale in deg; use sqrt for dimensional consistency
        r_sep = 1.0 / np.sqrt(density) if density > 0 else np.inf

        # Candidate radius from density (legacy heuristic)
        radius_cand = np.sqrt(1.0 / (np.pi * density)) if density > 0 else np.inf
        radius_cand = max(radius_cand, min_radius)

        # Upper and lower bounds
        c_max = 2.5  # cap factor for background separation
        k_astro = 1.2  # scale for astrometric scatter
        k_bin = 1.0  # scale for velocity binning smear

        epsilon_max = c_max * r_sep if np.isfinite(r_sep) else radius_cand

        # Astrometric scatter term (deg); in whitened space sigma~1
        epsilon_astro = k_astro * pos_scale_rms

        # Velocity binning smear using current candidate radius
        epsilon_bin = k_bin * velocity_bin_separation * radius_cand

        epsilon_min = np.sqrt(epsilon_astro**2 + epsilon_bin**2)

        # Final radius clamp
        radius = min(epsilon_max, max(epsilon_min, radius_cand, min_radius))

    logger.info(f"Effective clustering radius: {radius:.6f} deg ({radius*3600:.3f} arcsec)")
    logger.info(f"  - Covariance area used (max, {mahalanobis_distance:.1f}-sigma): {cov_area:.6e} deg^2")
    logger.info(f"  - Observation density: {density:.2f} obs/deg^2")
    logger.info(f"  - Characteristic separation: {r_sep:.6f} deg")
    logger.info(f"  - Radius bounds: epsilon_min={epsilon_min:.6f}, epsilon_max={epsilon_max:.6f}")
    logger.info(f"  - Total observations: {n_obs} across {n_times} times")
    logger.info(f"  - Time span: {dt_arc:.3f} days")

    # === Calculate velocity grid ===
    # Extract standard deviations
    sigma_vx = np.sqrt(vel_cov[0, 0])
    sigma_vy = np.sqrt(vel_cov[1, 1])

    if np.isnan(sigma_vx) or np.isnan(sigma_vy) or sigma_vx <= 0 or sigma_vy <= 0:
        raise ValueError(f"Invalid velocity sigmas: σ_vx={sigma_vx}, σ_vy={sigma_vy}")

    logger.info(f"Velocity uncertainties: σ_vx={sigma_vx:.6f}, σ_vy={sigma_vy:.6f} deg/day")

    # Calculate rectangular bounds at Mahalanobis distance centered at zero
    vx_min = -mahalanobis_distance * sigma_vx
    vx_max = mahalanobis_distance * sigma_vx
    vy_min = -mahalanobis_distance * sigma_vy
    vy_max = mahalanobis_distance * sigma_vy

    logger.info(
        f"Velocity grid edges ({mahalanobis_distance:.1f}-sigma): vx=[{vx_min:.6f}, {vx_max:.6f}], vy=[{vy_min:.6f}, {vy_max:.6f}] deg/day"
    )

    # Calculate number of bins
    # Key insight: At maximum time offset (dt_arc), a velocity difference dv produces
    # a position difference of dv * dt_arc. For proper DBSCAN clustering with radius r,
    # we want adjacent velocity bins to be separated by velocity_bin_separation × radius
    # in position space at maximum time.
    #
    # Therefore:
    #   dv_bin * dt_arc = velocity_bin_separation * radius
    #   dv_bin = velocity_bin_separation * radius / dt_arc
    #   n_bins = velocity_range / dv_bin = velocity_range * dt_arc / (velocity_bin_separation * radius)
    #
    # The velocity_bin_separation parameter controls spacing between velocity bins:
    # - Higher values → coarser grid → fewer bins → fewer duplicate clusters
    # - Lower values → finer grid → more bins → more duplicate clusters
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
    # Create rectangular velocity grid and include the zero velocity point
    if not np.any(np.isclose(vx_grid, 0.0, atol=1e-6)):
        vx_grid = np.sort(np.append(vx_grid, 0.0))
    if not np.any(np.isclose(vy_grid, 0.0, atol=1e-6)):
        vy_grid = np.sort(np.append(vy_grid, 0.0))

    vxx, vyy = np.meshgrid(vx_grid, vy_grid)

    # Flatten the grid
    vxx_flat = vxx.flatten()
    vyy_flat = vyy.flatten()

    # Filter to elliptical region using Mahalanobis distance
    # Since we're centered at zero, the Mahalanobis distance simplifies
    try:
        vel_cov_inv = np.linalg.inv(vel_cov)
        velocity_vectors = np.stack([vxx_flat, vyy_flat], axis=1)
        mahalanobis_sq = np.sum(velocity_vectors @ vel_cov_inv * velocity_vectors, axis=1)
        velocity_mask = mahalanobis_sq <= mahalanobis_distance_sq

        n_total = len(velocity_mask)
        n_inside = np.sum(velocity_mask)
        logger.info(
            f"Velocity grid points: {n_inside}/{n_total} inside {mahalanobis_distance:.1f}-sigma ellipse ({100*n_inside/n_total:.1f}%)"
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
        "cov_area": cov_area,
        "density": density,
        "r_sep": r_sep,
        "radius_candidate": radius_cand,
        "radius": radius,
        "mahalanobis_distance": mahalanobis_distance,
        "epsilon_min": epsilon_min,
        "epsilon_max": epsilon_max,
        "epsilon_astro": epsilon_astro,
        "epsilon_bin": epsilon_bin,
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


def cluster_and_link(
    observations: Union[TransformedDetections, ray.ObjectRef],
    test_orbit_ephemeris: Optional[Union[TestOrbitEphemeris, ray.ObjectRef]] = None,
    velocity_bin_separation: float = 2.0,
    min_obs: int = 5,
    min_arc_length: float = 1.0,
    min_nights: int = 3,
    rchi2_threshold: float = 1e4,
    mahalanobis_distance: Optional[float] = None,
    alg: Literal["hotspot_2d", "dbscan"] = "dbscan",
    radius: float = 0.005,
    vx_range: Optional[List[float]] = None,
    vy_range: Optional[List[float]] = None,
    vx_bins: Optional[int] = None,
    vy_bins: Optional[int] = None,
    vx_values: Optional[npt.NDArray[np.float64]] = None,
    vy_values: Optional[npt.NDArray[np.float64]] = None,
    chunk_size: int = 1000,
    max_processes: Optional[int] = 1,
    whiten: bool = False,
) -> Tuple[FittedClusters, FittedClusterMembers]:
    """
    Cluster and link correctly projected (after ranging and shifting)
    detections.

    Parameters
    ----------
    observations : TransformedDetections or ray.ObjectRef
        Transformed detections to cluster.
    test_orbit_ephemeris : TestOrbitEphemeris or ray.ObjectRef, optional
        Test orbit ephemeris with covariances. If provided, clustering parameters
        (vx_values, vy_values, radius) will be calculated automatically from the
        covariances, overriding manual parameter specifications.
    velocity_bin_separation : float, optional
        Separation between velocity bins in units of clustering radius.
        Only used when calculating parameters from test_orbit_ephemeris.
        Higher values = coarser grid. Default: 2.0
    min_obs : int, optional
        The minimum number of samples in a neighborhood for a point to be
        considered as a core point (DBSCAN min_samples parameter).
    min_arc_length : float, optional
        Minimum arc length in days for a cluster to be accepted.
    min_nights : int, optional
        Minimum number of unique nights a cluster must span.
    rchi2_threshold : float, optional
        The maximum reduced chi-squared value for a cluster to be accepted.
    mahalanobis_distance : float, optional
        Reserved for future use. Currently not used in filtering.
    alg : {"dbscan", "hotspot_2d"}, optional
        Algorithm to use for clustering. Default: "dbscan"
    radius : float, optional
        The maximum distance (in degrees) between two samples for them to be
        considered as in the same neighborhood (DBSCAN eps parameter).
    vx_range : list of float, optional
        [min, max] velocity range in x (deg/day). Used only if test_orbit_ephemeris
        is None and vx_values is None. If None, defaults to [-0.1, 0.1].
    vy_range : list of float, optional
        [min, max] velocity range in y (deg/day). Used only if test_orbit_ephemeris
        is None and vy_values is None. If None, defaults to [-0.1, 0.1].
    vx_bins : int, optional
        Number of bins for x-velocity grid. Used only if test_orbit_ephemeris
        is None and vx_values is None. If None, defaults to 100.
    vy_bins : int, optional
        Number of bins for y-velocity grid. Used only if test_orbit_ephemeris
        is None and vy_values is None. If None, defaults to 100.
    vx_values : np.ndarray, optional
        Pre-computed x-velocity values to use for clustering. If provided,
        overrides test_orbit_ephemeris-based calculation, vx_range, and vx_bins.
    vy_values : np.ndarray, optional
        Pre-computed y-velocity values to use for clustering. If provided,
        overrides test_orbit_ephemeris-based calculation, vy_range, and vy_bins.
        Must be same length as vx_values.
    chunk_size : int, optional
        Number of velocity grid points to process in each worker chunk.
    max_processes : int, optional
        Maximum number of processes to use for parallelization.
    whiten : bool, optional
        If True, compute and expose clustering parameters in whitened (sigma)
        units in addition to the default unwhitened values. Default: False.


    Returns
    -------
    fitted_clusters : FittedClusters
        Fitted clusters with polynomial motion models.
    fitted_cluster_members : FittedClusterMembers
        Members of each fitted cluster.

    Notes
    -----
    The algorithm chosen can have a big impact on performance and accuracy.

    alg="dbscan" uses the DBSCAN algorithm of Ester et. al. It's relatively slow
    but works with high accuracy; it is certain to find all clusters with at
    least min_obs points that are separated by at most radius.

    alg="hotspot_2d" is much faster (perhaps 10-20x faster) than dbscan, but it
    may miss some clusters, particularly when points are spaced a distance of 'radius'
    apart.

    Velocity Grid Priority:
    1. If vx_values and vy_values are provided, they will be used directly.
    2. Else if test_orbit_ephemeris is provided, parameters (vx, vy, radius) are
       calculated automatically from covariances using mahalanobis_distance and
       velocity_bin_separation.
    3. Otherwise, a grid is generated from vx_range, vy_range, vx_bins, and vy_bins.

    For covariance-informed clustering, simply pass test_orbit_ephemeris. For manual
    control, you can call calculate_clustering_parameters_from_covariance() first
    and pass the results as vx_values, vy_values, and radius.
    """
    time_start_cluster = time.perf_counter()
    logger.info("Running velocity space clustering...")

    if isinstance(observations, str):
        observations = TransformedDetections.from_parquet(observations)
        logger.info("Loaded transformed detections from parquet path.")
    elif isinstance(observations, ray.ObjectRef):
        observations = ray.get(observations)
        logger.info("Retrieved observations from the object store.")
    if isinstance(test_orbit_ephemeris, str):
        test_orbit_ephemeris = TestOrbitEphemeris.from_parquet(test_orbit_ephemeris)
        logger.info("Loaded test orbit ephemeris from parquet path.")
    elif isinstance(test_orbit_ephemeris, ray.ObjectRef):
        test_orbit_ephemeris = ray.get(test_orbit_ephemeris)
        logger.info("Retrieved test orbit ephemeris from the object store.")

    if len(observations) == 0:
        logger.info("No observations to cluster; returning empty clusters.")
        logger.info(f"Clustering completed in {time.perf_counter() - time_start_cluster:.3f} seconds.")
        return FittedClusters.empty(), FittedClusterMembers.empty()

    # Determine velocity grid and radius
    if vx_values is not None and vy_values is not None:
        # Use pre-computed velocity values
        if len(vx_values) != len(vy_values):
            raise ValueError(
                f"vx_values and vy_values must have same length. Got {len(vx_values)} and {len(vy_values)}."
            )
        vxx = vx_values
        vyy = vy_values
        logger.info(f"Using pre-computed velocity grid with {len(vxx)} points.")
    elif test_orbit_ephemeris is not None:
        # Calculate from test orbit covariances
        logger.info("Calculating clustering parameters from test orbit covariances...")
        try:
            vxx, vyy, radius, metadata = calculate_clustering_parameters_from_covariance(
                test_orbit_ephemeris,
                observations,
                mahalanobis_distance=mahalanobis_distance if mahalanobis_distance is not None else 3.0,
                velocity_bin_separation=velocity_bin_separation,
                min_radius=1 / 3600,
                whiten=whiten,
            )
            logger.info(
                f"Covariance-informed clustering: radius={radius:.6f}°, "
                f"vx_grid={len(vxx)} points, vy_grid={len(vyy)} points"
            )
        except Exception as e:
            logger.warning(f"Failed to calculate covariance parameters: {e}. Falling back to defaults.")
            # Fall back to range/bins
            if vx_range is None:
                vx_range = [-0.1, 0.1]
            if vy_range is None:
                vy_range = [-0.1, 0.1]
            if vx_bins is None:
                vx_bins = 100
            if vy_bins is None:
                vy_bins = 100

            vx = np.linspace(*vx_range, num=vx_bins)
            vy = np.linspace(*vy_range, num=vy_bins)
            vxx, vyy = np.meshgrid(vx, vy)
            vxx = vxx.flatten()
            vyy = vyy.flatten()
            logger.info(f"Generated velocity grid with {len(vxx)} points.")
    else:
        # Generate velocity grid from range and bins
        if vx_range is None:
            vx_range = [-0.1, 0.1]
        if vy_range is None:
            vy_range = [-0.1, 0.1]
        if vx_bins is None:
            vx_bins = 100
        if vy_bins is None:
            vy_bins = 100

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

    logger.info("Max sample distance: {}".format(radius))
    logger.info("Minimum samples: {}".format(min_obs))

    exit_early = False
    if len(observations) > 0:
        # Calculate the unique times
        unique_times = observations.coordinates.time.unique()

        # Check that there are enough unique times to cluster
        num_unique_times = len(unique_times)
        if num_unique_times < min_obs:
            logger.info("Number of unique times is less than the minimum number of observations required.")
            exit_early = True

        # Calculate the time range and make sure it is greater than the minimum arc length
        time_range = unique_times.max().mjd()[0].as_py() - unique_times.min().mjd()[0].as_py()
        if time_range < min_arc_length:
            logger.info("Time range of transformed detections is less than the minimum arc length.")
            exit_early = True

    else:
        # If there are no transformed detections, exit early
        logger.info("No transformed detections to cluster.")
        exit_early = True

    # If any of the above conditions are met then we exit early
    if exit_early:
        time_end_cluster = time.perf_counter()
        logger.info("Found 0 clusters. Minimum requirements for clustering not met.")
        logger.info(f"Clustering completed in {time_end_cluster - time_start_cluster:.3f} seconds.")
        return Clusters.empty(), ClusterMembers.empty()

    # Accumulate fitted clusters
    fitted_clusters = FittedClusters.empty()
    fitted_cluster_members = FittedClusterMembers.empty()

    if max_processes is None:
        max_processes = mp.cpu_count()

    use_ray = initialize_use_ray(num_cpus=max_processes)
    if use_ray:
        # Put transformed detections in the Ray object store
        if isinstance(observations, ray.ObjectRef):
            transformed_ref = observations
        else:
            transformed_ref = ray.put(observations)
            logger.info("Placed transformed detections in the object store.")

        futures = []
        for vxi_chunk, vyi_chunk in zip(_iterate_chunks(vxx, chunk_size), _iterate_chunks(vyy, chunk_size)):

            futures.append(
                cluster_velocity_remote.remote(
                    vxi_chunk,
                    vyi_chunk,
                    transformed_ref,
                    radius=radius,
                    min_obs=min_obs,
                    min_arc_length=min_arc_length,
                    min_nights=min_nights,
                    rchi2_threshold=rchi2_threshold,
                    alg=alg,
                    test_orbit_ephemeris=test_orbit_ephemeris,
                    mahalanobis_distance=mahalanobis_distance,
                )
            )

            if len(futures) >= max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                fitted_clusters_chunk, fitted_cluster_members_chunk = ray.get(finished[0])
                fitted_clusters = qv.concatenate([fitted_clusters, fitted_clusters_chunk])
                if fitted_clusters.fragmented():
                    fitted_clusters = qv.defragment(fitted_clusters)

                fitted_cluster_members = qv.concatenate(
                    [fitted_cluster_members, fitted_cluster_members_chunk]
                )
                if fitted_cluster_members.fragmented():
                    fitted_cluster_members = qv.defragment(fitted_cluster_members)

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            fitted_clusters_chunk, fitted_cluster_members_chunk = ray.get(finished[0])
            fitted_clusters = qv.concatenate([fitted_clusters, fitted_clusters_chunk])
            if fitted_clusters.fragmented():
                fitted_clusters = qv.defragment(fitted_clusters)

            fitted_cluster_members = qv.concatenate([fitted_cluster_members, fitted_cluster_members_chunk])
            if fitted_cluster_members.fragmented():
                fitted_cluster_members = qv.defragment(fitted_cluster_members)

    else:
        for vxi_chunk, vyi_chunk in zip(_iterate_chunks(vxx, chunk_size), _iterate_chunks(vyy, chunk_size)):
            fitted_clusters_i, fitted_cluster_members_i = cluster_velocity_worker(
                vxi_chunk,
                vyi_chunk,
                observations,
                radius=radius,
                min_obs=min_obs,
                min_arc_length=min_arc_length,
                min_nights=min_nights,
                rchi2_threshold=rchi2_threshold,
                alg=alg,
                test_orbit_ephemeris=test_orbit_ephemeris,
                mahalanobis_distance=mahalanobis_distance,
            )

            fitted_clusters = qv.concatenate([fitted_clusters, fitted_clusters_i])
            if fitted_clusters.fragmented():
                fitted_clusters = qv.defragment(fitted_clusters)

            fitted_cluster_members = qv.concatenate([fitted_cluster_members, fitted_cluster_members_i])
            if fitted_cluster_members.fragmented():
                fitted_cluster_members = qv.defragment(fitted_cluster_members)

    num_clusters = len(fitted_clusters)
    if num_clusters == 0:
        time_end_cluster = time.perf_counter()
        logger.info(f"Found {len(fitted_clusters)} clusters, exiting early.")
        logger.info(f"Clustering completed in {time_end_cluster - time_start_cluster:.3f} seconds.")
        return FittedClusters.empty(), FittedClusterMembers.empty()

    # Ensure clusters, cluster_members are defragmented and sorted
    # prior to dropping duplicates. We do this here so that
    # we don't sort inside the function and make a whole new copy
    # while the old one stays referenced in memory

    # Drop duplicate clusters
    time_start_drop = time.perf_counter()
    logger.info("Removing duplicate clusters...")
    fitted_clusters = qv.defragment(fitted_clusters)
    fitted_cluster_members = qv.defragment(fitted_cluster_members)
    fitted_clusters = fitted_clusters.sort_by([("cluster_id", "ascending")])
    fitted_cluster_members = fitted_cluster_members.sort_by([("cluster_id", "ascending")])

    fitted_clusters, fitted_cluster_members = drop_duplicate_clusters(fitted_clusters, fitted_cluster_members)
    logger.info(f"Removed {num_clusters - len(fitted_clusters)} duplicate clusters.")
    time_end_drop = time.perf_counter()
    logger.info(f"Cluster deduplication completed in {time_end_drop - time_start_drop:.3f} seconds.")

    time_end_cluster = time.perf_counter()
    logger.info(f"Found {len(fitted_clusters)} clusters.")
    logger.info(f"Clustering completed in {time_end_cluster - time_start_cluster:.3f} seconds.")

    return fitted_clusters, fitted_cluster_members


def fit_cluster(
    cluster: Clusters, cluster_members: ClusterMembers, transformed_detections: TransformedDetections
) -> Tuple[FittedClusters, FittedClusterMembers]:
    """
    Fit a cluster with a 2nd order polynomial motion model in theta_x and theta_y.

    Parameters
    ----------
    cluster : `~thor.clusters.Clusters`
        Cluster.
    cluster_members : `~thor.clusters.ClusterMembers`
        Cluster members.
    transformed_detections : `~thor.transformed_detections.TransformedDetections`
        Transformed detections.

    Returns
    -------
    fitted_cluster : `~thor.clusters.FittedClusters`
        Fitted cluster.
    fitted_cluster_members : `~thor.clusters.FittedClusterMembers`
        Fitted cluster members.
    """
    try:
        cluster_detections = transformed_detections.apply_mask(
            pc.is_in(transformed_detections.id, cluster_members.obs_id)
        )
        cluster_detections = cluster_detections.sort_by(["coordinates.time.days", "coordinates.time.nanos"])

        gnomonic_coords = cluster_detections.coordinates
        theta_x = gnomonic_coords.theta_x.to_numpy(zero_copy_only=False)
        theta_y = gnomonic_coords.theta_y.to_numpy(zero_copy_only=False)
        time = gnomonic_coords.time.mjd().to_numpy(zero_copy_only=False)

        # Use relative time from the first observation to avoid numerical issues
        # and make x0, y0 represent position at the first observation
        t0 = time[0]
        dt = time - t0  # days since first observation

        # Fit a 2nd order polynomial to the data as a function of relative time
        # theta(dt) = theta0 + v*dt + 0.5*a*dt²
        coords = np.empty((len(dt), 2))
        coords[:, 0] = theta_x
        coords[:, 1] = theta_y
        coeffs = np.polyfit(dt, coords, 2)

        # coeffs[0] is the quadratic coefficient in theta(dt) = c2*dt² + c1*dt + c0
        # Store atheta as physical acceleration, i.e., atheta = 2 * c2
        ax = 2.0 * coeffs[0, 0]  # deg/day²
        ay = 2.0 * coeffs[0, 1]
        vx = coeffs[1, 0]  # deg/day
        vy = coeffs[1, 1]
        x0 = coeffs[2, 0]  # deg (position at first observation)
        y0 = coeffs[2, 1]

        x_pred = np.polyval(coeffs[:, 0], dt)
        y_pred = np.polyval(coeffs[:, 1], dt)

        gnomonic_pred = GnomonicCoordinates.from_kwargs(
            time=gnomonic_coords.time,
            theta_x=x_pred,
            theta_y=y_pred,
            origin=gnomonic_coords.origin,
            frame=gnomonic_coords.frame,
        )

        residuals = Residuals.calculate(gnomonic_coords, gnomonic_pred, custom_coordinates=True)

        # Get test_orbit_id from the cluster detections (all detections in a cluster share the same test_orbit_id)
        test_orbit_id = cluster_detections.test_orbit_id[0].as_py()

        fitted_cluster = FittedClusters.from_kwargs(
            cluster_id=cluster.cluster_id,
            test_orbit_id=[test_orbit_id],
            time=gnomonic_coords.time[0],
            theta_x0=[x0],
            theta_y0=[y0],
            vtheta_x=[vx],
            vtheta_y=[vy],
            atheta_x=[ax],
            atheta_y=[ay],
            origin=gnomonic_coords.origin[0],
            frame=gnomonic_coords.frame,
            arc_length=[pc.subtract(pc.max(time), pc.min(time))],
            num_obs=[len(time)],
            chi2=[pc.sum(residuals.chi2)],
            rchi2=[pc.divide(pc.sum(residuals.chi2), pc.sum(residuals.dof).as_py() - 6)],
        )

        fitted_cluster_members = FittedClusterMembers.from_kwargs(
            cluster_id=cluster_members.cluster_id,
            obs_id=cluster_members.obs_id,
            test_orbit_id=pa.repeat(test_orbit_id, len(cluster_members)),
            residuals=residuals,
        )

        return fitted_cluster, fitted_cluster_members

    except np.linalg.LinAlgError:
        cluster_id = cluster.cluster_id[0].as_py()
        logger.warning(
            f"Failed to fit cluster {cluster_id}: Singular matrix (degenerate observations). Skipping cluster."
        )
        return FittedClusters.empty(), FittedClusterMembers.empty()
    except Exception as e:
        cluster_id = cluster.cluster_id[0].as_py()
        logger.warning(f"Failed to fit cluster {cluster_id}: {e}. Skipping cluster.")
        return FittedClusters.empty(), FittedClusterMembers.empty()


def fit_cluster_worker(
    clusters: Union[Clusters, ray.ObjectRef],
    cluster_members: Union[ClusterMembers, ray.ObjectRef],
    transformed_detections: Union[TransformedDetections, ray.ObjectRef],
    cluster_ids: List[str],
) -> Tuple[FittedClusters, FittedClusterMembers]:
    """
    Worker function for fitting a single cluster (used by Ray).

    This function selects the cluster and its members by cluster_id,
    then calls fit_cluster.

    Parameters can be either the actual tables or Ray object references.
    """
    fitted_clusters = FittedClusters.empty()
    fitted_cluster_members = FittedClusterMembers.empty()

    for cluster_id in cluster_ids:
        cluster_i = clusters.select("cluster_id", cluster_id)
        cluster_members_i = cluster_members.select("cluster_id", cluster_id)

        fitted_cluster_i, fitted_cluster_members_i = fit_cluster(
            cluster_i, cluster_members_i, transformed_detections
        )
        fitted_clusters = qv.concatenate([fitted_clusters, fitted_cluster_i])
        fitted_cluster_members = qv.concatenate([fitted_cluster_members, fitted_cluster_members_i])

    return fitted_clusters, fitted_cluster_members


fit_cluster_worker_remote = ray.remote(fit_cluster_worker)
fit_cluster_worker_remote = fit_cluster_worker_remote.options(
    num_returns=1,
    num_cpus=1,
)


def fit_clusters(
    clusters: Clusters,
    cluster_members: ClusterMembers,
    transformed_detections: TransformedDetections,
    chunk_size: int = 1000,
    max_processes: Optional[int] = 1,
) -> Tuple[FittedClusters, FittedClusterMembers]:
    """
    Fit a set of clusters with a 2nd order polynomial motion model in theta_x and theta_y.

    Parameters
    ----------
    clusters : `~thor.clusters.Clusters`
        Clusters.
    cluster_members : `~thor.clusters.ClusterMembers`
        Cluster members.
    transformed_detections : `~thor.transformed_detections.TransformedDetections`
        Transformed detections.
    chunk_size : int, optional
        Chunk size to use for parallelization. When using ray,
        chunks are distributed to multiple workers for parallel processing.
        When not using ray, chunks are processed sequentially.
    max_processes : int, optional
        Maximum number of processes to use for parallelization.
        [Default = 1]

    Returns
    -------
    fitted_clusters : `~thor.clusters.FittedClusters`
        Fitted clusters.
    fitted_cluster_members : `~thor.clusters.FittedClusterMembers`
        Fitted cluster members.
    """

    fitted_clusters = FittedClusters.empty()
    fitted_cluster_members = FittedClusterMembers.empty()
    if len(clusters) == 0:
        return fitted_clusters, fitted_cluster_members

    use_ray = initialize_use_ray(num_cpus=max_processes)
    cluster_ids = clusters.cluster_id.to_pylist()
    if use_ray:
        # Put tables in Ray object store to avoid repeated serialization
        clusters_ref = ray.put(clusters)
        cluster_members_ref = ray.put(cluster_members)
        transformed_detections_ref = ray.put(transformed_detections)

        futures = []
        for cluster_id_chunk in _iterate_chunks(cluster_ids, chunk_size):
            futures.append(
                fit_cluster_worker_remote.remote(
                    clusters_ref, cluster_members_ref, transformed_detections_ref, cluster_id_chunk
                )
            )

            if len(futures) >= max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                fitted_clusters_chunk, fitted_cluster_members_chunk = ray.get(finished[0])
                fitted_clusters = qv.concatenate([fitted_clusters, fitted_clusters_chunk])
                if fitted_clusters.fragmented():
                    fitted_clusters = qv.defragment(fitted_clusters)
                fitted_cluster_members = qv.concatenate(
                    [fitted_cluster_members, fitted_cluster_members_chunk]
                )
                if fitted_cluster_members.fragmented():
                    fitted_cluster_members = qv.defragment(fitted_cluster_members)

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            fitted_clusters_chunk, fitted_cluster_members_chunk = ray.get(finished[0])
            fitted_clusters = qv.concatenate([fitted_clusters, fitted_clusters_chunk])
            fitted_cluster_members = qv.concatenate([fitted_cluster_members, fitted_cluster_members_chunk])
            if fitted_clusters.fragmented():
                fitted_clusters = qv.defragment(fitted_clusters)
            if fitted_cluster_members.fragmented():
                fitted_cluster_members = qv.defragment(fitted_cluster_members)
    else:
        for cluster_id in cluster_ids:
            cluster_i = clusters.select("cluster_id", cluster_id)
            cluster_members_i = cluster_members.select("cluster_id", cluster_id)
            fitted_cluster_i, fitted_cluster_members_i = fit_cluster(
                cluster_i, cluster_members_i, transformed_detections
            )
            fitted_clusters = qv.concatenate([fitted_clusters, fitted_cluster_i])
            fitted_cluster_members = qv.concatenate([fitted_cluster_members, fitted_cluster_members_i])

    return fitted_clusters, fitted_cluster_members
