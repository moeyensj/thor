import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import concurrent.futures as cf
import logging
import multiprocessing as mp
import time
import uuid
from functools import partial

import numpy as np
import pandas as pd
from astropy.time import Time

from .cell import Cell
from .clusters import filter_clusters_by_length, find_clusters
from .orbit import TestOrbit
from .orbits import generateEphemeris
from .utils import _checkParallel, _initWorker

logger = logging.getLogger("thor")

__all__ = [
    "rangeAndShift_worker",
    "rangeAndShift",
    "clusterVelocity",
    "clusterVelocity_worker",
    "clusterAndLink",
]


def rangeAndShift_worker(observations, ephemeris, cell_area=10):

    assert len(observations["mjd_utc"].unique()) == 1
    assert len(ephemeris["mjd_utc"].unique()) == 1
    assert observations["mjd_utc"].unique()[0] == ephemeris["mjd_utc"].unique()[0]
    observation_time = observations["mjd_utc"].unique()[0]

    # Create Cell centered on the sky-plane location of the
    # test orbit
    cell = Cell(
        ephemeris[["RA_deg", "Dec_deg"]].values[0],
        observation_time,
        area=cell_area,
    )

    # Grab observations within cell
    cell.getObservations(observations)

    if len(cell.observations) != 0:

        # Create test orbit with state of orbit at visit time
        test_orbit = TestOrbit(
            ephemeris[["obj_x", "obj_y", "obj_z", "obj_vx", "obj_vy", "obj_vz"]].values[
                0
            ],
            observation_time,
        )

        # Prepare rotation matrices
        test_orbit.prepare()

        # Apply rotation matrices and transform observations into the orbit's
        # frame of motion.
        test_orbit.applyToObservations(cell.observations)

        projected_observations = cell.observations

    else:

        projected_observations = pd.DataFrame()

    return projected_observations


def clusterVelocity(
    obs_ids,
    x,
    y,
    dt,
    vx,
    vy,
    eps=0.005,
    min_obs=5,
    min_arc_length=1.0,
    alg="hotspot_2d",
):
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
    vx : `~numpy.ndarray' (N)
        Projection space x velocity in units of degrees or radians per day in MJD.
    vy : `~numpy.ndarray' (N)
        Projection space y velocity in units of degrees or radians per day in MJD.
    eps : float, optional
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

    clusters = find_clusters(X, eps, min_obs, alg=alg)
    clusters = filter_clusters_by_length(
        clusters,
        dt,
        min_obs,
        min_arc_length,
    )

    cluster_ids = []
    for cluster in clusters:
        cluster_ids.append(obs_ids[cluster])

    if len(cluster_ids) == 0:
        cluster_ids = np.NaN

    return cluster_ids


def clusterVelocity_worker(
    vx,
    vy,
    obs_ids=None,
    x=None,
    y=None,
    dt=None,
    eps=None,
    min_obs=None,
    min_arc_length=None,
    alg=None,
):
    """
    Helper function to multiprocess clustering.

    """
    cluster_ids = clusterVelocity(
        obs_ids,
        x,
        y,
        dt,
        vx,
        vy,
        eps=eps,
        min_obs=min_obs,
        min_arc_length=min_arc_length,
        alg=alg,
    )
    return cluster_ids


def rangeAndShift(
    observations,
    orbit,
    cell_area=10,
    backend="PYOORB",
    backend_kwargs={},
    num_jobs=1,
    parallel_backend="cf",
):
    """Propagate the orbit to all observation times in observations. At each epoch gather a circular region of observations of size cell_area
    centered about the location of the orbit on the sky-plane. Transform and project each of the gathered observations into
    the frame of motion of the test orbit.

    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing preprocessed observations.
            Should contain the following columns:
                obs_id : observation IDs
                RA_deg : Right Ascension in degrees.
                Dec_deg : Declination in degrees.
                RA_sigma_deg : 1-sigma uncertainty for Right Ascension in degrees.
                Dec_sigma_deg : 1-sigma uncertainty for Declination in degrees.
                observatory_code : MPC observatory code
    orbit : `~numpy.ndarray` (6)
        Orbit to propagate. If backend is 'THOR', then these orbits must be expressed
        as heliocentric ecliptic cartesian elements. If backend is 'PYOORB' orbits may be
        expressed in keplerian, cometary or cartesian elements.
    cell_area : float, optional
        Cell's area in units of square degrees.
        [Default = 10]
    backend : {'THOR', 'PYOORB'}, optional
        Which backend to use.
    backend_kwargs : dict, optional
        Settings and additional parameters to pass to selected
        backend.
    num_jobs : int, optional
        Number of jobs to launch.
    parallel_backend : str, optional
        Which parallelization backend to use {'ray', 'mp', 'cf'}.
        Defaults to using Python's concurrent futures module ('cf').

    Returns
    -------
    projected_observations : {`~pandas.DataFrame`, -1}
        Observations dataframe (from cell.observations) with columns containing
        projected coordinates.

    """
    time_start = time.time()
    logger.info("Running range and shift...")
    logger.info("Assuming r = {} au".format(orbit.cartesian[0, :3]))
    logger.info("Assuming v = {} au per day".format(orbit.cartesian[0, 3:]))

    # Build observers dictionary: keys are observatory codes with exposure times (as astropy.time objects)
    # as values
    observers = {}
    for code in observations["observatory_code"].unique():
        observers[code] = Time(
            observations[observations["observatory_code"].isin([code])][
                "mjd_utc"
            ].unique(),
            format="mjd",
            scale="utc",
        )

    # Propagate test orbit to all times in observations
    ephemeris = generateEphemeris(
        orbit,
        observers,
        backend=backend,
        backend_kwargs=backend_kwargs,
        chunk_size=1,
        num_jobs=1,
        parallel_backend=parallel_backend,
    )
    if backend == "FINDORB":

        observer_states = []
        for observatory_code, observation_times in observers.items():
            observer_states.append(
                getObserverState(
                    [observatory_code],
                    observation_times,
                    frame="ecliptic",
                    origin="heliocenter",
                )
            )

        observer_states = pd.concat(observer_states)
        observer_states.reset_index(inplace=True, drop=True)
        ephemeris = ephemeris.join(
            observer_states[["obs_x", "obs_y", "obs_z", "obs_vx", "obs_vy", "obs_vz"]]
        )

    velocity_cols = []
    if backend != "PYOORB":
        velocity_cols = ["obs_vx", "obs_vy", "obs_vz"]

    observations = observations.merge(
        ephemeris[
            ["mjd_utc", "observatory_code", "obs_x", "obs_y", "obs_z"] + velocity_cols
        ],
        left_on=["mjd_utc", "observatory_code"],
        right_on=["mjd_utc", "observatory_code"],
    )

    # Split the observations into a single dataframe per unique observatory code and observation time
    # Basically split the observations into groups of unique exposures
    observations_grouped = observations.groupby(by=["observatory_code", "mjd_utc"])
    observations_split = [
        observations_grouped.get_group(g) for g in observations_grouped.groups
    ]

    # Do the same for the test orbit's ephemerides
    ephemeris_grouped = ephemeris.groupby(by=["observatory_code", "mjd_utc"])
    ephemeris_split = [ephemeris_grouped.get_group(g) for g in ephemeris_grouped.groups]

    parallel, num_workers = _checkParallel(num_jobs, parallel_backend)
    if parallel:
        if parallel_backend == "ray":
            import ray

            if not ray.is_initialized():
                ray.init(address="auto")

            rangeAndShift_worker_ray = ray.remote(rangeAndShift_worker)
            rangeAndShift_worker_ray = rangeAndShift_worker_ray.options(
                num_returns=1, num_cpus=1
            )

            p = []
            for observations_i, ephemeris_i in zip(observations_split, ephemeris_split):
                p.append(
                    rangeAndShift_worker_ray.remote(
                        observations_i, ephemeris_i, cell_area=cell_area
                    )
                )
            projected_dfs = ray.get(p)

        elif parallel_backend == "mp":
            p = mp.Pool(
                processes=num_workers,
                initializer=_initWorker,
            )
            projected_dfs = p.starmap(
                partial(rangeAndShift_worker, cell_area=cell_area),
                zip(
                    observations_split,
                    ephemeris_split,
                ),
            )
            p.close()

        elif parallel_backend == "cf":
            with cf.ProcessPoolExecutor(
                max_workers=num_workers, initializer=_initWorker
            ) as executor:
                futures = []
                for observations_i, ephemeris_i in zip(
                    observations_split, ephemeris_split
                ):
                    f = executor.submit(
                        rangeAndShift_worker,
                        observations_i,
                        ephemeris_i,
                        cell_area=cell_area,
                    )
                    futures.append(f)

                projected_dfs = []
                for f in cf.as_completed(futures):
                    projected_dfs.append(f.result())

        else:
            raise ValueError("Invalid parallel_backend: {}".format(parallel_backend))

    else:
        projected_dfs = []
        for observations_i, ephemeris_i in zip(observations_split, ephemeris_split):
            projected_df = rangeAndShift_worker(
                observations_i, ephemeris_i, cell_area=cell_area
            )
            projected_dfs.append(projected_df)

    projected_observations = pd.concat(projected_dfs)
    if len(projected_observations) > 0:
        projected_observations.sort_values(
            by=["mjd_utc", "observatory_code"], inplace=True
        )
        projected_observations.reset_index(inplace=True, drop=True)
    else:
        projected_observations = pd.DataFrame(
            columns=[
                "obs_id",
                "mjd_utc",
                "RA_deg",
                "Dec_deg",
                "RA_sigma_deg",
                "Dec_sigma_deg",
                "observatory_code",
                "obs_x",
                "obs_y",
                "obs_z",
                "obj_x",
                "obj_y",
                "obj_z",
                "theta_x_deg",
                "theta_y_deg",
            ]
        )

    time_end = time.time()
    logger.info("Found {} observations.".format(len(projected_observations)))
    logger.info(
        "Range and shift completed in {:.3f} seconds.".format(time_end - time_start)
    )

    return projected_observations


def clusterAndLink(
    observations,
    vx_range=[-0.1, 0.1],
    vy_range=[-0.1, 0.1],
    vx_bins=100,
    vy_bins=100,
    vx_values=None,
    vy_values=None,
    eps=0.005,
    min_obs=5,
    min_arc_length=1.0,
    alg="dbscan",
    num_jobs=1,
    parallel_backend="cf",
):
    """
    Cluster and link correctly projected (after ranging and shifting)
    detections.

    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing post-range and shift observations.
    vx_range : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in x.
        Will not be used if vx_values are specified.
        [Default = [-0.1, 0.1]]
    vy_range : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in y.
        Will not be used if vy_values are specified.
        [Default = [-0.1, 0.1]]
    vx_bins : int, optional
        Length of x-velocity grid between vx_range[0]
        and vx_range[-1]. Will not be used if vx_values are
        specified.
        [Default = 100]
    vy_bins: int, optional
        Length of y-velocity grid between vy_range[0]
        and vy_range[-1]. Will not be used if vy_values are
        specified.
        [Default = 100]
    vx_values : {None, `~numpy.ndarray`}, optional
        Values of velocities in x at which to cluster
        and link.
        [Default = None]
    vy_values : {None, `~numpy.ndarray`}, optional
        Values of velocities in y at which to cluster
        and link.
        [Default = None]
    eps : float, optional
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
    least min_obs points that are separated by at most eps.

    alg="hotspot_2d" is much faster (perhaps 10-20x faster) than dbscan, but it
    may miss some clusters, particularly when points are spaced a distance of 'eps'
    apart.
    """
    time_start_cluster = time.time()
    logger.info("Running velocity space clustering...")

    if vx_values is None and vx_range is not None:
        vx = np.linspace(*vx_range, num=vx_bins)
    elif vx_values is None and vx_range is None:
        raise ValueError("Both vx_values and vx_range cannot be None.")
    else:
        vx = vx_values
        vx_range = [vx_values[0], vx_values[-1]]
        vx_bins = len(vx)

    if vy_values is None and vy_range is not None:
        vy = np.linspace(*vy_range, num=vy_bins)
    elif vy_values is None and vy_range is None:
        raise ValueError("Both vy_values and vy_range cannot be None.")
    else:
        vy = vy_values
        vy_range = [vy_values[0], vy_values[-1]]
        vy_bins = len(vy)

    if vx_values is None and vy_values is None:
        vxx, vyy = np.meshgrid(vx, vy)
        vxx = vxx.flatten()
        vyy = vyy.flatten()
    elif vx_values is not None and vy_values is not None:
        vxx = vx
        vyy = vy
    else:
        raise ValueError("")

    logger.debug("X velocity range: {}".format(vx_range))
    if vx_values is not None:
        logger.debug("X velocity values: {}".format(vx_bins))
    else:
        logger.debug("X velocity bins: {}".format(vx_bins))

    logger.debug("Y velocity range: {}".format(vy_range))
    if vy_values is not None:
        logger.debug("Y velocity values: {}".format(vy_bins))
    else:
        logger.debug("Y velocity bins: {}".format(vy_bins))
    if vx_values is not None:
        logger.debug("User defined x velocity values: True")
    else:
        logger.debug("User defined x velocity values: False")
    if vy_values is not None:
        logger.debug("User defined y velocity values: True")
    else:
        logger.debug("User defined y velocity values: False")

    if vx_values is None and vy_values is None:
        logger.debug("Velocity grid size: {}".format(vx_bins * vy_bins))
    else:
        logger.debug("Velocity grid size: {}".format(vx_bins))
    logger.info("Max sample distance: {}".format(eps))
    logger.info("Minimum samples: {}".format(min_obs))

    possible_clusters = []
    if len(observations) > 0:
        # Extract useful quantities
        obs_ids = observations["obs_id"].values
        theta_x = observations["theta_x_deg"].values
        theta_y = observations["theta_y_deg"].values
        mjd = observations["mjd_utc"].values

        # Select detections in first exposure
        first = np.where(mjd == mjd.min())[0]
        mjd0 = mjd[first][0]
        dt = mjd - mjd0

        parallel, num_workers = _checkParallel(num_jobs, parallel_backend)
        if parallel:
            if parallel_backend == "ray":
                import ray

                if not ray.is_initialized():
                    ray.init(address="auto")

                clusterVelocity_worker_ray = ray.remote(clusterVelocity_worker)
                clusterVelocity_worker_ray = clusterVelocity_worker_ray.options(
                    num_returns=1, num_cpus=1
                )

                # Put all arrays (which can be large) in ray's
                # local object store ahead of time
                obs_ids_oid = ray.put(obs_ids)
                theta_x_oid = ray.put(theta_x)
                theta_y_oid = ray.put(theta_y)
                dt_oid = ray.put(dt)

                p = []
                for vxi, vyi in zip(vxx, vyy):
                    p.append(
                        clusterVelocity_worker_ray.remote(
                            vxi,
                            vyi,
                            obs_ids=obs_ids_oid,
                            x=theta_x_oid,
                            y=theta_y_oid,
                            dt=dt_oid,
                            eps=eps,
                            min_obs=min_obs,
                            min_arc_length=min_arc_length,
                            alg=alg,
                        )
                    )
                possible_clusters = ray.get(p)

            elif parallel_backend == "mp":
                p = mp.Pool(processes=num_workers, initializer=_initWorker)
                possible_clusters = p.starmap(
                    partial(
                        clusterVelocity_worker,
                        obs_ids=obs_ids,
                        x=theta_x,
                        y=theta_y,
                        dt=dt,
                        eps=eps,
                        min_obs=min_obs,
                        min_arc_length=min_arc_length,
                        alg=alg,
                    ),
                    zip(vxx, vyy),
                )
                p.close()

            elif parallel_backend == "cf":
                with cf.ProcessPoolExecutor(
                    max_workers=num_workers, initializer=_initWorker
                ) as executor:
                    futures = []
                    for vxi, vyi in zip(vxx, vyy):
                        f = executor.submit(
                            clusterVelocity_worker,
                            vxi,
                            vyi,
                            obs_ids=obs_ids,
                            x=theta_x,
                            y=theta_y,
                            dt=dt,
                            eps=eps,
                            min_obs=min_obs,
                            min_arc_length=min_arc_length,
                            alg=alg,
                        )
                        futures.append(f)

                    possible_clusters = []
                    for f in cf.as_completed(futures):
                        possible_clusters.append(f.result())

            else:
                raise ValueError(
                    "Invalid parallel_backend: {}".format(parallel_backend)
                )

        else:
            possible_clusters = []
            for vxi, vyi in zip(vxx, vyy):
                possible_clusters.append(
                    clusterVelocity(
                        obs_ids,
                        theta_x,
                        theta_y,
                        dt,
                        vxi,
                        vyi,
                        eps=eps,
                        min_obs=min_obs,
                        min_arc_length=min_arc_length,
                        alg=alg,
                    )
                )

    time_end_cluster = time.time()
    logger.info(
        "Clustering completed in {:.3f} seconds.".format(
            time_end_cluster - time_start_cluster
        )
    )

    logger.info("Restructuring clusters...")
    time_start_restr = time.time()

    possible_clusters = pd.DataFrame({"clusters": possible_clusters})

    # Remove empty clusters
    possible_clusters = possible_clusters[~possible_clusters["clusters"].isna()]

    if len(possible_clusters) != 0:
        ### The following code is a little messy, its a lot of pandas dataframe manipulation.
        ### I have tried doing an overhaul wherein the clusters and cluster_members dataframe are created per
        ### velocity combination in the clusterVelocity function. However, this adds an overhead in that function
        ### of ~ 1ms. So clustering 90,000 velocities takes 90 seconds longer which on small datasets is problematic.
        ### On large datasets, the effect is not as pronounced because the below code takes a while to run due to
        ### in-memory pandas dataframe restructuring.

        # Make DataFrame with cluster velocities so we can figure out which
        # velocities yielded clusters, add names to index so we can enable the join
        cluster_velocities = pd.DataFrame({"vtheta_x": vxx, "vtheta_y": vyy})
        cluster_velocities.index.set_names("velocity_id", inplace=True)

        # Split lists of cluster ids into one column per cluster for each different velocity
        # then stack the result
        possible_clusters = pd.DataFrame(
            possible_clusters["clusters"].values.tolist(), index=possible_clusters.index
        )
        possible_clusters = pd.DataFrame(possible_clusters.stack())
        possible_clusters.rename(columns={0: "obs_ids"}, inplace=True)
        possible_clusters = pd.DataFrame(
            possible_clusters["obs_ids"].values.tolist(), index=possible_clusters.index
        )

        # Drop duplicate clusters
        possible_clusters.drop_duplicates(inplace=True)

        # Set index names
        possible_clusters.index.set_names(["velocity_id", "cluster_id"], inplace=True)

        # Reset index
        possible_clusters.reset_index("cluster_id", drop=True, inplace=True)
        possible_clusters["cluster_id"] = [
            str(uuid.uuid4().hex) for i in range(len(possible_clusters))
        ]

        # Make clusters DataFrame
        clusters = possible_clusters.join(cluster_velocities)
        clusters.reset_index(drop=True, inplace=True)
        clusters = clusters[["cluster_id", "vtheta_x", "vtheta_y"]]

        # Make cluster_members DataFrame
        cluster_members = possible_clusters.reset_index(drop=True).copy()
        cluster_members.index = cluster_members["cluster_id"]
        cluster_members.drop("cluster_id", axis=1, inplace=True)
        cluster_members = pd.DataFrame(cluster_members.stack())
        cluster_members.rename(columns={0: "obs_id"}, inplace=True)
        cluster_members.reset_index(inplace=True)
        cluster_members.drop("level_1", axis=1, inplace=True)

        # Calculate arc length and add it to the clusters dataframe
        cluster_members_time = cluster_members.merge(
            observations[["obs_id", "mjd_utc"]], on="obs_id", how="left"
        )
        clusters_time = (
            cluster_members_time.groupby(by=["cluster_id"])["mjd_utc"]
            .apply(lambda x: x.max() - x.min())
            .to_frame()
        )
        clusters_time.reset_index(inplace=True)
        clusters_time.rename(columns={"mjd_utc": "arc_length"}, inplace=True)
        clusters = clusters.merge(
            clusters_time[["cluster_id", "arc_length"]],
            on="cluster_id",
            how="left",
        )

    else:
        cluster_members = pd.DataFrame(columns=["cluster_id", "obs_id"])
        clusters = pd.DataFrame(
            columns=["cluster_id", "vtheta_x", "vtheta_y", "arc_length"]
        )

    time_end_restr = time.time()
    logger.info(
        "Restructuring completed in {:.3f} seconds.".format(
            time_end_restr - time_start_restr
        )
    )
    logger.info("Found {} clusters.".format(len(clusters)))
    logger.info(
        "Clustering and restructuring completed in {:.3f} seconds.".format(
            time_end_restr - time_start_cluster
        )
    )

    return clusters, cluster_members
