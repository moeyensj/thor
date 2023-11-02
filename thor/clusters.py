import numba
import numpy as np

# Disable GPU until the GPU-accelerated clustering codes
# are better tested and implemented
USE_GPU = False

if USE_GPU:
    import cudf
    from cuml.cluster import DBSCAN
else:
    from sklearn.cluster import DBSCAN


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
            if (
                runs[0, j] == points_quantized[0, i]
                and runs[1, j] == points_quantized[1, i]
            ):
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
