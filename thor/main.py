import os
from typing import List, Literal, Optional, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import logging
import time
import uuid

import numpy as np
import numpy.typing as npt
import quivr as qv
import ray

from .clusters import filter_clusters_by_length, find_clusters
from .range_and_transform import TransformedDetections

logger = logging.getLogger("thor")

__all__ = [
    "cluster_and_link",
]


class Clusters(qv.Table):
    cluster_id = qv.StringColumn(default=lambda: uuid.uuid4().hex)
    vtheta_x = qv.Float64Column()
    vtheta_y = qv.Float64Column()
    arc_length = qv.Float64Column()
    num_obs = qv.Int64Column()


class ClusterMembers(qv.Table):
    cluster_id = qv.StringColumn()
    obs_id = qv.StringColumn()


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
            cluster_id=np.concatenate(cluster_members_cluster_ids),
            obs_id=np.concatenate(cluster_members_obs_ids),
        )

    return clusters, cluster_members


def cluster_velocity_worker(
    vx,
    vy,
    obs_ids=None,
    x=None,
    y=None,
    dt=None,
    radius=None,
    min_obs=None,
    min_arc_length=None,
    alg=None,
):
    """
    Helper function to multiprocess clustering.

    """
    clusters, cluster_members = cluster_velocity(
        obs_ids,
        x,
        y,
        dt,
        vx,
        vy,
        radius=radius,
        min_obs=min_obs,
        min_arc_length=min_arc_length,
        alg=alg,
    )
    return clusters, cluster_members


cluster_velocity_remote = ray.remote(cluster_velocity_worker)
cluster_velocity_remote.options(
    num_returns=1,
    num_cpus=1,
)


def cluster_and_link(
    observations: TransformedDetections,
    vx_range: List[float] = [-0.1, 0.1],
    vy_range: List[float] = [-0.1, 0.1],
    vx_bins: int = 100,
    vy_bins: int = 100,
    radius: float = 0.005,
    min_obs: int = 5,
    min_arc_length: float = 1.0,
    alg: Literal["hotspot_2d", "dbscan"] = "dbscan",
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
    parallel_backend : str, optional
        Which parallelization backend to use {'ray', 'mp', 'cf'}.
        Defaults to using Python's concurrent futures module ('cf').

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
    time_start_cluster = time.time()
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

    clusters_list = []
    cluster_members_list = []
    if len(observations) > 0:
        # Extract useful quantities
        obs_ids = observations.id.to_numpy(zero_copy_only=False)
        theta_x = observations.coordinates.theta_x.to_numpy(zero_copy_only=False)
        theta_y = observations.coordinates.theta_y.to_numpy(zero_copy_only=False)
        mjd = observations.coordinates.time.mjd().to_numpy(zero_copy_only=False)

        # Select detections in first exposure
        first = np.where(mjd == mjd.min())[0]
        mjd0 = mjd[first][0]
        dt = mjd - mjd0

        if max_processes is None or max_processes > 1:

            if not ray.is_initialized():
                ray.init(address="auto")

            # Put all arrays (which can be large) in ray's
            # local object store ahead of time
            obs_ids_oid = ray.put(obs_ids)
            theta_x_oid = ray.put(theta_x)
            theta_y_oid = ray.put(theta_y)
            dt_oid = ray.put(dt)

            futures = []
            for vxi, vyi in zip(vxx, vyy):
                futures.append(
                    cluster_velocity_remote.remote(
                        vxi,
                        vyi,
                        obs_ids=obs_ids_oid,
                        x=theta_x_oid,
                        y=theta_y_oid,
                        dt=dt_oid,
                        radius=radius,
                        min_obs=min_obs,
                        min_arc_length=min_arc_length,
                        alg=alg,
                    )
                )

            while futures:
                finished, futures = ray.wait(futures, num_returns=1)
                result = ray.get(finished[0])
                clusters_list.append(result[0])
                cluster_members_list.append(result[1])

        else:

            for vxi, vyi in zip(vxx, vyy):
                clusters_i, cluster_members_i = cluster_velocity_worker(
                    vxi,
                    vyi,
                    obs_ids=obs_ids,
                    x=theta_x,
                    y=theta_y,
                    dt=dt,
                    radius=radius,
                    min_obs=min_obs,
                    min_arc_length=min_arc_length,
                    alg=alg,
                )
                clusters_list.append(clusters_i)
                cluster_members_list.append(cluster_members_i)

    clusters = qv.concatenate(clusters_list)
    cluster_members = qv.concatenate(cluster_members_list)

    time_end_cluster = time.time()
    logger.info("Found {} clusters.".format(len(clusters)))
    logger.info(
        "Clustering completed in {:.3f} seconds.".format(
            time_end_cluster - time_start_cluster
        )
    )

    return clusters, cluster_members
