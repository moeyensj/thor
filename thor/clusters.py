import numba
import numpy as np

from .config import Config
USE_GPU = Config.USE_GPU

if USE_GPU:
    import cudf
    from cuml.cluster import DBSCAN
else:
    from sklearn.cluster import DBSCAN



def find_clusters(points, eps, min_samples, alg="dbscan"):
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
    may miss some clusters.
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
    for cluster in clusters:
        dt_in_cluster = dt[cluster]
        num_obs = len(dt_in_cluster)
        arc_length = dt_in_cluster.max() - dt_in_cluster.min()
        if ((num_obs == len(np.unique(dt_in_cluster)))
            and ((num_obs >= min_samples))
            and (arc_length >= min_arc_length)):
            filtered_clusters.append(cluster)
    return filtered_clusters


def _find_clusters_hotspots_2d(points, eps, min_samples):
    """
    This algorithm works by quantizing points into a grid, with a resolution of
    2*eps. It searches the quantized points to find values that appear at least
    min_samples times in the dataset.

    The search of the grid is done by sorting the points, and then stepping
    through the points looking for consecutive runsof the same (x, y) value
    repeated at least min_samples times.
    """
    points = _enforce_shape(points)
    points_quantized = _quantize_points(points, 2*eps)  # 0.9ms
    sort_order = np.lexsort(points_quantized)           # 8.3ms
    sorted_points = points_quantized[:, sort_order]     # 0.8ms

    xy_hits = _find_runs(sorted_points, min_samples)    # 0.2ms

    if xy_hits.shape[1] == 0:
        return []

    cluster_labels = _label_clusters(xy_hits, points_quantized)  # 9.4ms
    unique_labels = np.unique(cluster_labels)           # 0.6ms
    unique_labels = unique_labels[unique_labels != -1]
    clusters = []
    for label in unique_labels:                         # 5.3ms
        cluster_indices = np.where(cluster_labels == label)[0]
        clusters.append(cluster_indices)
    return clusters


@numba.jit(nopython=True)
def _enforce_shape(points):
    """Ensure that datapoints are in a shape of (2, N)."""
    if points.shape[0] != 2:
        return points.T
    return points


@numba.jit(nopython=True, parallel=True)
def _quantize_points(points, eps):
    """Quantize points down to the nearest value of eps."""
    return points - points % eps


@numba.jit(nopython=True)
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
    result = np.empty((2, expected_n_clusters*2), dtype="float64")
    n_hit = 0

    n_consecutive = 1
    prev_x = np.nan
    prev_y = np.nan

    for i in range(sorted_points.shape[1]):
        if sorted_points[0, i] == prev_x and sorted_points[1, i] == prev_y:
            n_consecutive += 1
            if n_consecutive == min_samples:
                if n_hit == result.shape[1]:
                    result = _extend_2d_array(result, result.shape[1]*2)
                result[0, n_hit] = sorted_points[0, i]
                result[1, n_hit] = sorted_points[1, i]
                n_hit = n_hit + 1
        else:
            prev_x = sorted_points[0, i]
            prev_y = sorted_points[1, i]
            n_consecutive = 1
    return result[:, :n_hit]


@numba.jit(nopython=True)
def _extend_2d_array(src, new_size):
    dst = np.empty((2, new_size), dtype=src.dtype)
    for i in range(src.shape[1]):
        dst[0, i] = src[0, i]
        dst[1, i] = src[1, i]
    return dst


@numba.jit(nopython=True)
def _label_clusters(xy_hits, points_quantized):
    """
    Produce a 1D array of integers which label each X-Y in points_quantized with a cluster.
    """

    # A plain old for loop is probably fine here because xy_hits is likely to be
    # small.
    labels = np.zeros(points_quantized.shape[1], np.int64)
    for i in range(points_quantized.shape[1]):
        for j in range(xy_hits.shape[1]):
            if xy_hits[0, j] == points_quantized[0, i] and xy_hits[1, j] == points_quantized[1, i]:
                labels[i] = j+1
                break
    return labels - 1


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
