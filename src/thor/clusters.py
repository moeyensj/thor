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
from adam_core.propagator import _iterate_chunks
from adam_core.ray_cluster import initialize_use_ray

from .range_and_transform import TransformedDetections
from .utils.linkages import sort_by_id_and_time

# Disable GPU until the GPU-accelerated clustering codes
# are better tested and implemented
USE_GPU = False

if USE_GPU:
    from cuml.cluster import DBSCAN
else:
    from sklearn.cluster import DBSCAN


__all__ = [
    "cluster_and_link",
    "Clusters",
    "ClusterMembers",
]

logger = logging.getLogger(__name__)


def hash_obs_ids(obs_ids: List[str]) -> str:
    """
    Create unique strings for each set unique set of observation IDs

    We use hashes rather than original string in order to save memory.
    """
    return hashlib.md5("".join(sorted(set(obs_ids))).encode()).hexdigest()


def drop_duplicate_clusters(
    clusters: "Clusters",
    cluster_members: "ClusterMembers",
) -> Tuple["Clusters", "ClusterMembers"]:
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


def filter_clusters_by_length(clusters, dt, min_samples, min_arc_length):
    """
    Filter cluster results on the conditions that they span at least
    min_arc_length in the time dimension, and that each point in the cluster
    is from a different dt value.

    Parameters
    -----------
    clusters: `list of numpy.ndarray'

        A list of clusters. Each cluster should be an array of indexes
        of observations that are members of the same cluster. The indexes
        are into the 'dt' array.

    dt: `~numpy.ndarray' (N)
        Change in time from the 0th exposure in units of MJD.

    min_samples: int
        Minimum size for a cluster to be included.

    min_arc_length: float
        Minimum arc length in units of days for a cluster to be accepted.

    Returns
    -------
    list of numpy.ndarray

        The original clusters list, filtered down.
    """
    filtered_clusters = []
    arc_lengths = []
    for cluster in clusters:
        dt_in_cluster = dt[cluster]
        num_obs = len(dt_in_cluster)
        arc_length = dt_in_cluster.max() - dt_in_cluster.min()
        if (
            (num_obs == len(np.unique(dt_in_cluster)))
            and ((num_obs >= min_samples))
            and (arc_length >= min_arc_length)
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
    vx: float,
    vy: float,
    radius: float = 0.005,
    min_obs: int = 5,
    min_arc_length: float = 1.0,
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
    vx : float
        Projection space x velocity in units of degrees or radians per day in MJD.
    vy : float
        Projection space y velocity in units of degrees or radians per day in MJD.
    radius : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
        See: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html
        [Default = 0.005]
    min_obs : int, optional
        The number of samples (or total weight) in a neighborhood for a
        point to be considered as a core point. This includes the point itself.
        See: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html
        [Default = 5]
    min_arc_length : float, optional
        Minimum arc length in units of days for a cluster to be accepted.

    Returns
    -------
    list
        If clusters are found, will return a list of numpy arrays containing the
        observation IDs for each cluster. If no clusters are found, will return np.NaN.
    """
    logger.debug(f"cluster: vx={vx} vy={vy} n_obs={len(obs_ids)}")
    xx = x - vx * dt
    yy = y - vy * dt

    X = np.stack((xx, yy), 1)

    clusters = find_clusters(X, radius, min_obs, alg=alg)
    clusters, arc_lengths = filter_clusters_by_length(
        clusters,
        dt,
        min_obs,
        min_arc_length,
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
    obs_ids: npt.ArrayLike,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    dt: npt.NDArray[np.float64],
    radius: float = 0.005,
    min_obs: int = 5,
    min_arc_length: float = 1.0,
    alg: Literal["hotspot_2d", "dbscan"] = "dbscan",
) -> Tuple[Clusters, ClusterMembers]:
    """
    Helper function for parallelizing cluster_velocity. This function takes a
    batch or chunk of velocities and returns the clusters and cluster members
    for that batch.

    """
    clusters = Clusters.empty()
    cluster_members = ClusterMembers.empty()
    for vx_i, vy_i in zip(vx, vy):
        clusters_i, cluster_members_i = cluster_velocity(
            obs_ids,
            x,
            y,
            dt,
            vx_i,
            vy_i,
            radius=radius,
            min_obs=min_obs,
            min_arc_length=min_arc_length,
            alg=alg,
        )
        clusters = qv.concatenate([clusters, clusters_i])
        if clusters.fragmented():
            clusters = qv.defragment(clusters)

        cluster_members = qv.concatenate([cluster_members, cluster_members_i])
        if cluster_members.fragmented():
            cluster_members = qv.defragment(cluster_members)

    return clusters, cluster_members


cluster_velocity_remote = ray.remote(cluster_velocity_worker)
cluster_velocity_remote.options(
    num_returns=1,
    num_cpus=1,
)


def cluster_and_link(
    observations: Union[TransformedDetections, ray.ObjectRef],
    vx_range: List[float] = [-0.1, 0.1],
    vy_range: List[float] = [-0.1, 0.1],
    vx_bins: int = 100,
    vy_bins: int = 100,
    radius: float = 0.005,
    min_obs: int = 5,
    min_arc_length: float = 1.0,
    alg: Literal["hotspot_2d", "dbscan"] = "dbscan",
    chunk_size: int = 1000,
    max_processes: Optional[int] = 1,
) -> Tuple[Clusters, ClusterMembers]:
    """
    Cluster and link correctly projected (after ranging and shifting)
    detections.

    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing post-range and shift observations.
    vx_range : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in x.
        [Default = [-0.1, 0.1]]
    vy_range : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in y.
        [Default = [-0.1, 0.1]]
    vx_bins : int, optional
        Length of x-velocity grid between vx_range[0]
        and vx_range[-1].
        [Default = 100]
    vy_bins: int, optional
        Length of y-velocity grid between vy_range[0]
        and vy_range[-1].
        [Default = 100]
    radius : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
        See: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html
        [Default = 0.005]
    min_obs : int, optional
        The number of samples (or total weight) in a neighborhood for a
        point to be considered as a core point. This includes the point itself.
        See: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html
        [Default = 5]
    alg: str
        Algorithm to use. Can be "dbscan" or "hotspot_2d".
    num_jobs : int, optional
        Number of jobs to launch.

    Returns
    -------
    clusters : `~pandas.DataFrame`
        DataFrame with the cluster ID, the number of observations, and the x and y velocity.
    cluster_members : `~pandas.DataFrame`
        DataFrame containing the cluster ID and the observation IDs of its members.

    Notes
    -----
    The algorithm chosen can have a big impact on performance and accuracy.

    alg="dbscan" uses the DBSCAN algorithm of Ester et. al. It's relatively slow
    but works with high accuracy; it is certain to find all clusters with at
    least min_obs points that are separated by at most radius.

    alg="hotspot_2d" is much faster (perhaps 10-20x faster) than dbscan, but it
    may miss some clusters, particularly when points are spaced a distance of 'radius'
    apart.
    """
    time_start_cluster = time.perf_counter()
    logger.info("Running velocity space clustering...")

    vx = np.linspace(*vx_range, num=vx_bins)
    vy = np.linspace(*vy_range, num=vy_bins)
    vxx, vyy = np.meshgrid(vx, vy)
    vxx = vxx.flatten()
    vyy = vyy.flatten()

    logger.debug("X velocity range: {}".format(vx_range))
    logger.debug("X velocity bins: {}".format(vx_bins))
    logger.debug("Y velocity range: {}".format(vy_range))
    logger.debug("Y velocity bins: {}".format(vy_bins))
    logger.debug("Velocity grid size: {}".format(vx_bins))
    logger.info("Max sample distance: {}".format(radius))
    logger.info("Minimum samples: {}".format(min_obs))

    if isinstance(observations, ray.ObjectRef):
        observations = ray.get(observations)
        logger.info("Retrieved observations from the object store.")

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

    clusters = Clusters.empty()
    cluster_members = ClusterMembers.empty()

    # Extract useful quantities
    obs_ids = observations.id.to_numpy(zero_copy_only=False)
    theta_x = observations.coordinates.theta_x.to_numpy(zero_copy_only=False)
    theta_y = observations.coordinates.theta_y.to_numpy(zero_copy_only=False)
    mjd = observations.coordinates.time.mjd().to_numpy(zero_copy_only=False)

    # Select detections in first exposure
    first = np.where(mjd == mjd.min())[0]
    mjd0 = mjd[first][0]
    dt = mjd - mjd0

    if max_processes is None:
        max_processes = mp.cpu_count()

    use_ray = initialize_use_ray(num_cpus=max_processes)
    if use_ray:
        # Put all arrays (which can be large) in ray's
        # local object store ahead of time
        obs_ids_ref = ray.put(obs_ids)
        theta_x_ref = ray.put(theta_x)
        theta_y_ref = ray.put(theta_y)
        dt_ref = ray.put(dt)
        refs_to_free = [obs_ids_ref, theta_x_ref, theta_y_ref, dt_ref]
        logger.info("Placed gnomonic coordinate arrays in the object store.")
        # TODO: transformed detections are already in the object store so we might
        # want to instead pass references to those rather than extract arrays
        # from them and put them in the object store again.
        futures = []
        for vxi_chunk, vyi_chunk in zip(_iterate_chunks(vxx, chunk_size), _iterate_chunks(vyy, chunk_size)):

            futures.append(
                cluster_velocity_remote.remote(
                    vxi_chunk,
                    vyi_chunk,
                    obs_ids_ref,
                    theta_x_ref,
                    theta_y_ref,
                    dt_ref,
                    radius=radius,
                    min_obs=min_obs,
                    min_arc_length=min_arc_length,
                    alg=alg,
                )
            )

            if len(futures) >= max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                clusters_chunk, cluster_members_chunk = ray.get(finished[0])
                clusters = qv.concatenate([clusters, clusters_chunk])
                if clusters.fragmented():
                    clusters = qv.defragment(clusters)

                cluster_members = qv.concatenate([cluster_members, cluster_members_chunk])
                if cluster_members.fragmented():
                    cluster_members = qv.defragment(cluster_members)

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            clusters_chunk, cluster_members_chunk = ray.get(finished[0])
            clusters = qv.concatenate([clusters, clusters_chunk])
            if clusters.fragmented():
                clusters = qv.defragment(clusters)

            cluster_members = qv.concatenate([cluster_members, cluster_members_chunk])
            if cluster_members.fragmented():
                cluster_members = qv.defragment(cluster_members)

        ray.internal.free(refs_to_free)
        logger.info(f"Removed {len(refs_to_free)} references from the object store.")

    else:
        for vxi_chunk, vyi_chunk in zip(_iterate_chunks(vxx, chunk_size), _iterate_chunks(vyy, chunk_size)):
            clusters_i, cluster_members_i = cluster_velocity_worker(
                vxi_chunk,
                vyi_chunk,
                obs_ids,
                theta_x,
                theta_y,
                dt,
                radius=radius,
                min_obs=min_obs,
                min_arc_length=min_arc_length,
                alg=alg,
            )

            clusters = qv.concatenate([clusters, clusters_i])
            if clusters.fragmented():
                clusters = qv.defragment(clusters)

            cluster_members = qv.concatenate([cluster_members, cluster_members_i])
            if cluster_members.fragmented():
                cluster_members = qv.defragment(cluster_members)

    num_clusters = len(clusters)
    if num_clusters == 0:
        time_end_cluster = time.perf_counter()
        logger.info(f"Found {len(clusters)} clusters, exiting early.")
        logger.info(f"Clustering completed in {time_end_cluster - time_start_cluster:.3f} seconds.")
        return clusters, cluster_members

    # Ensure clusters, cluster_members are defragmented and sorted
    # prior to dropping duplicates. We do this here so that
    # we don't sort inside the function and make a whole new copy
    # while the old one stays referenced in memory

    # Drop duplicate clusters
    time_start_drop = time.perf_counter()
    logger.info("Removing duplicate clusters...")
    clusters = qv.defragment(clusters)
    cluster_members = qv.defragment(cluster_members)
    clusters = clusters.sort_by([("cluster_id", "ascending")])
    cluster_members = cluster_members.sort_by([("cluster_id", "ascending")])

    clusters, cluster_members = drop_duplicate_clusters(clusters, cluster_members)
    logger.info(f"Removed {num_clusters - len(clusters)} duplicate clusters.")
    time_end_drop = time.perf_counter()
    logger.info(f"Cluster deduplication completed in {time_end_drop - time_start_drop:.3f} seconds.")

    # Sort clusters by cluster ID and observation time
    clusters, cluster_members = sort_by_id_and_time(clusters, cluster_members, observations, "cluster_id")

    time_end_cluster = time.perf_counter()
    logger.info(f"Found {len(clusters)} clusters.")
    logger.info(f"Clustering completed in {time_end_cluster - time_start_cluster:.3f} seconds.")

    return clusters, cluster_members
