import logging
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def filter_clusters_by_length(
    clusters: List[npt.NDArray[np.int64]],
    dt: npt.NDArray[np.float64],
    nights: npt.NDArray[np.int64],
    min_samples: int,
    min_arc_length: float,
    min_nights: int,
) -> Tuple[List[npt.NDArray[np.int64]], List[float]]:
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
    list of float
        Arc lengths for the filtered clusters.
    """
    filtered_clusters = []
    arc_lengths = []

    # Diagnostic counters
    n_raw = len(clusters)
    reject_unique_time = 0
    reject_min_obs = 0
    reject_arc_length = 0
    reject_nights = 0

    for cluster in clusters:
        dt_in_cluster = dt[cluster]
        nights_in_cluster = nights[cluster]
        num_obs = len(dt_in_cluster)
        n_unique_dt = len(np.unique(dt_in_cluster))
        arc_length = dt_in_cluster.max() - dt_in_cluster.min()
        num_nights = len(np.unique(nights_in_cluster))

        if num_obs != n_unique_dt:
            reject_unique_time += 1
        elif num_obs < min_samples:
            reject_min_obs += 1
        elif arc_length < min_arc_length:
            reject_arc_length += 1
        elif num_nights < min_nights:
            reject_nights += 1
        else:
            filtered_clusters.append(cluster)
            arc_lengths.append(arc_length)

    if n_raw > 0 and len(filtered_clusters) == 0:
        logger.info(
            f"All {n_raw} clusters rejected: "
            f"unique_time={reject_unique_time}, min_obs={reject_min_obs}, "
            f"arc_length={reject_arc_length}, min_nights={reject_nights}"
        )
    elif n_raw > 0:
        logger.info(
            f"Kept {len(filtered_clusters)}/{n_raw} clusters "
            f"(rejected: unique_time={reject_unique_time}, min_obs={reject_min_obs}, "
            f"arc_length={reject_arc_length}, min_nights={reject_nights})"
        )

    return filtered_clusters, arc_lengths
