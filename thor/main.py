import os
import sys
import time
import uuid
import yaml
import logging
import shutil
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from astropy.time import Time

from .config import Config
from .clusters import USE_GPU, find_clusters, filter_clusters_by_length
from .cell import Cell
from .orbit import TestOrbit
from .orbits import Orbits
from .observatories import getObserverState
from .orbits import generateEphemeris
from .orbits import initialOrbitDetermination
from .orbits import differentialCorrection
from .orbits import mergeAndExtendOrbits
from .backend import _init_worker
from .utils import identifySubsetLinkages

USE_RAY = Config.USE_RAY
USE_GPU = Config.USE_GPU
NUM_THREADS = Config.NUM_THREADS

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

logger = logging.getLogger("thor")

__all__ = [
    "rangeAndShift_worker",
    "rangeAndShift",
    "clusterVelocity",
    "clusterVelocity_worker",
    "clusterAndLink",
    "runTHOROrbit",
    "runTHOR",
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
            ephemeris[["obj_x", "obj_y", "obj_z", "obj_vx", "obj_vy", "obj_vz"]].values[0],
            observation_time
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
        min_samples=5,
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
    min_samples : int, optional
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

    clusters = find_clusters(X, eps, min_samples, alg=alg)
    clusters = filter_clusters_by_length(
        clusters, dt, min_samples, min_arc_length,
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
        min_samples=None,
        min_arc_length=None,
        alg=None
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
        min_samples=min_samples,
        min_arc_length=min_arc_length,
        alg=alg
    )
    return cluster_ids

if USE_RAY:
    import ray
    rangeAndShift_worker = ray.remote(rangeAndShift_worker)
    clusterVelocity_worker = ray.remote(clusterVelocity_worker)

def rangeAndShift(
        observations,
        orbit,
        cell_area=10,
        threads=NUM_THREADS,
        backend="PYOORB",
        backend_kwargs={},
    ):
    """
    Propagate the orbit to all observation times in observations. At each epoch gather a circular region of observations of size cell_area
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
            observations[observations["observatory_code"].isin([code])]["mjd_utc"].unique(),
            format="mjd",
            scale="utc"
        )

    # Propagate test orbit to all times in observations
    ephemeris = generateEphemeris(
        orbit,
        observers,
        backend=backend,
        backend_kwargs=backend_kwargs
    )
    if backend == "FINDORB":

        observer_states = []
        for observatory_code, observation_times in observers.items():
            observer_states.append(
                getObserverState(
                    [observatory_code],
                    observation_times,
                    frame='ecliptic',
                    origin='heliocenter',
                )
            )

        observer_states = pd.concat(observer_states)
        observer_states.reset_index(
            inplace=True,
            drop=True
        )
        ephemeris = ephemeris.join(observer_states[["obs_x", "obs_y", "obs_z", "obs_vx", "obs_vy", "obs_vz"]])

    velocity_cols = []
    if backend != "PYOORB":
        velocity_cols = ["obs_vx", "obs_vy", "obs_vz"]

    observations = observations.merge(
        ephemeris[["mjd_utc", "observatory_code", "obs_x", "obs_y", "obs_z"] + velocity_cols],
        left_on=["mjd_utc", "observatory_code"],
        right_on=["mjd_utc", "observatory_code"]
    )

    # Split the observations into a single dataframe per unique observatory code and observation time
    # Basically split the observations into groups of unique exposures
    observations_grouped = observations.groupby(by=["observatory_code", "mjd_utc"])
    observations_split = [observations_grouped.get_group(g) for g in observations_grouped.groups]

    # Do the same for the test orbit's ephemerides
    ephemeris_grouped = ephemeris.groupby(by=["observatory_code", "mjd_utc"])
    ephemeris_split = [ephemeris_grouped.get_group(g) for g in ephemeris_grouped.groups]

    if threads > 1:

        if USE_RAY:
            if not ray.is_initialized():
                ray.init(address="auto")

            p = []
            for observations_i, ephemeris_i in zip(observations_split, ephemeris_split):
                p.append(
                    rangeAndShift_worker.remote(
                        observations_i,
                        ephemeris_i,
                        cell_area=cell_area
                    )
                )
            projected_dfs = ray.get(p)

        else:
            p = mp.Pool(
                processes=threads,
                initializer=_init_worker,
            )

            projected_dfs = p.starmap(
                partial(
                    rangeAndShift_worker,
                    cell_area=cell_area
                ),
                zip(
                    observations_split,
                    ephemeris_split,
                )
            )
            p.close()

    else:
        projected_dfs = []
        for observations_i, ephemeris_i in zip(observations_split, ephemeris_split):
            projected_df = rangeAndShift_worker(
                observations_i,
                ephemeris_i,
                cell_area=cell_area
            )
            projected_dfs.append(projected_df)

    projected_observations = pd.concat(projected_dfs)
    if len(projected_observations) > 0:
        projected_observations.sort_values(by=["mjd_utc", "observatory_code"], inplace=True)
        projected_observations.reset_index(inplace=True, drop=True)
    else:
        projected_observations = pd.DataFrame(
            columns=[
                'obs_id', 'mjd_utc', 'RA_deg', 'Dec_deg', 'RA_sigma_deg',
                'Dec_sigma_deg', 'observatory_code', 'obs_x', 'obs_y', 'obs_z', 'obj_x',
                'obj_y', 'obj_z', 'theta_x_deg', 'theta_y_deg'
            ]
        )

    time_end = time.time()
    logger.info("Found {} observations.".format(len(projected_observations)))
    logger.info("Range and shift completed in {:.3f} seconds.".format(time_end - time_start))

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
        min_samples=5,
        min_arc_length=1.0,
        identify_subsets=False,
        threads=NUM_THREADS,
        alg="dbscan"
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
    threads : int, optional
        Number of threads to use.
        [Default = 12]
    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
        See: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html
        [Default = 0.005]
    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a
        point to be considered as a core point. This includes the point itself.
        See: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html
        [Default = 5]
    alg: str
        Algorithm to use. Can be "dbscan" or "hotspot_2d".

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
    least min_samples points that are separated by at most eps.

    alg="hotspot_2d" is much faster (perhaps 10-20x faster) than dbscan, but it
    may miss some clusters, particularly when points are spaced a distance of 'eps'
    apart.
    """
    time_start_cluster = time.time()
    logger.info("Running velocity space clustering...")

    # Extract useful quantities
    obs_ids = observations["obs_id"].values
    theta_x = observations["theta_x_deg"].values
    theta_y = observations["theta_y_deg"].values
    mjd = observations["mjd_utc"].values

    # Select detections in first exposure
    first = np.where(mjd == mjd.min())[0]
    mjd0 = mjd[first][0]
    dt = mjd - mjd0

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
    logger.info("Minimum samples: {}".format(min_samples))

    possible_clusters = []
    if threads > 1 and not USE_GPU:

        if USE_RAY:
            if not ray.is_initialized():
                ray.init(address="auto")

            # Put all arrays (which can be large) in ray's
            # local object store ahead of time
            obs_ids_oid = ray.put(obs_ids)
            theta_x_oid = ray.put(theta_x)
            theta_y_oid = ray.put(theta_y)
            dt_oid = ray.put(dt)

            p = []
            for vxi, vyi in zip(vxx, vyy):
                p.append(
                    clusterVelocity_worker.remote(
                        vxi,
                        vyi,
                        obs_ids=obs_ids_oid,
                        x=theta_x_oid,
                        y=theta_y_oid,
                        dt=dt_oid,
                        eps=eps,
                        min_samples=min_samples,
                        min_arc_length=min_arc_length,
                        alg=alg
                    )
                )
            possible_clusters = ray.get(p)

        else:

            p = mp.Pool(threads, _init_worker)
            possible_clusters = p.starmap(
                partial(
                    clusterVelocity_worker,
                    obs_ids=obs_ids,
                    x=theta_x,
                    y=theta_y,
                    dt=dt,
                    eps=eps,
                    min_samples=min_samples,
                    min_arc_length=min_arc_length,
                    alg=alg
                ),
                zip(vxx, vyy)
            )
            p.close()
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
                    min_samples=min_samples,
                    min_arc_length=min_arc_length,
                    alg=alg
                )
            )
    time_end_cluster = time.time()
    logger.info("Clustering completed in {:.3f} seconds.".format(time_end_cluster - time_start_cluster))

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
            possible_clusters["clusters"].values.tolist(),
            index=possible_clusters.index
        )
        possible_clusters = pd.DataFrame(possible_clusters.stack())
        possible_clusters.rename(
            columns={0: "obs_ids"},
            inplace=True
        )
        possible_clusters = pd.DataFrame(possible_clusters["obs_ids"].values.tolist(), index=possible_clusters.index)

        # Drop duplicate clusters
        possible_clusters.drop_duplicates(inplace=True)

        # Set index names
        possible_clusters.index.set_names(["velocity_id", "cluster_id"], inplace=True)

        # Reset index
        possible_clusters.reset_index(
            "cluster_id",
            drop=True,
            inplace=True
        )
        possible_clusters["cluster_id"] = [str(uuid.uuid4().hex) for i in range(len(possible_clusters))]

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
            observations[["obs_id", "mjd_utc"]],
            on="obs_id",
            how="left"
        )
        clusters_time = cluster_members_time.groupby(
            by=["cluster_id"])["mjd_utc"].apply(lambda x: x.max() - x.min()).to_frame()
        clusters_time.reset_index(
            inplace=True
        )
        clusters_time.rename(
            columns={"mjd_utc" : "arc_length"},
            inplace=True
        )
        clusters = clusters.merge(
            clusters_time[["cluster_id", "arc_length"]],
            on="cluster_id",
            how="left",
        )

    else:
        cluster_members = pd.DataFrame(columns=["cluster_id", "obs_id"])
        clusters = pd.DataFrame(columns=["cluster_id", "vtheta_x", "vtheta_y", "arc_length"])


    time_end_restr = time.time()
    logger.info("Restructuring completed in {:.3f} seconds.".format(time_end_restr - time_start_restr))

    if identify_subsets == True:
        logger.info("Identifying subsets...")
        clusters, cluster_members = identifySubsetLinkages(
            clusters,
            cluster_members,
            linkage_id_col="cluster_id"
        )
        logger.info("Done. {} subset clusters identified.".format(len(clusters[~clusters["subset_of"].isna()])))

    logger.info("Found {} clusters.".format(len(clusters)))
    logger.info("Clustering and restructuring completed in {:.3f} seconds.".format(time_end_restr - time_start_cluster))

    return clusters, cluster_members

def runTHOROrbit(
        preprocessed_observations,
        orbit,
        range_shift_config=Config.RANGE_SHIFT_CONFIG,
        cluster_link_config=Config.CLUSTER_LINK_CONFIG,
        iod_config=Config.IOD_CONFIG,
        od_config=Config.OD_CONFIG,
        odp_config=Config.ODP_CONFIG,
        out_dir=None,
        if_exists="continue",
        logging_level=logger.info
    ):
    logger = logging.getLogger("thor")
    logger.setLevel(logging_level)

    RUN_CONFIG = {
        "RANGE_SHIFT_CONFIG" : range_shift_config,
        "CLUSTER_LINK_CONFIG" : cluster_link_config,
        "IOD_CONFIG" : iod_config,
        "OD_CONFIG" : od_config,
        "ODP_CONFIG" : odp_config
    }
    status = {
        "rangeAndShift" : False,
        "clusterAndLink" : False,
        "initialOrbitDetermination" : False,
        "differentialCorrection" : False,
        "mergeAndExtendOrbits" : False,
        "complete" : False
    }

    continue_ = False
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            logger.debug("Created {} directory.".format(out_dir))

        else:
            if if_exists == "continue":
                logger.warning("{} directory already exists, attempting to continue previous run.".format(out_dir))
                continue_ = True
            elif if_exists == "erase":
                logger.warning("{} directory already exists, removing previous results.".format(out_dir))
                shutil.rmtree(out_dir)
                os.mkdir(out_dir)
                logger.debug("Created {} directory.".format(out_dir))
            else:
                err = (
                    "if_exists should be one of {'continue', 'erase'}."
                )
                raise ValueError(err)

        file_handler = logging.FileHandler(
            os.path.join(out_dir, "thor.log"),
            encoding="utf-8",
            delay=False
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(levelname)s] [%(thread)s] %(message)s (%(filename)s, %(funcName)s, %(lineno)d)',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        # The primary files which will be used to determine if the run
        # can be continued from a previous state and, if so, from where
        # to continue the run
        config_file = os.path.join(out_dir, "config.yml")
        test_orbit_file = os.path.join(out_dir, "test_orbit.csv")
        status_file = os.path.join(out_dir, "status.yml")
        config_eq = False
        test_orbit_eq = False
        save_orbit = True
        save_config = True

        if continue_:

            if not os.path.exists(config_file):
                logger.warning("No previous configuration file found.")
                save_config = True
            else:
                logger.info("Previous configuration file found. Comparing settings...")

                config_file_prev = open(config_file, "r")
                config_prev = yaml.load(config_file_prev, Loader=yaml.FullLoader)

                if config_prev != RUN_CONFIG:
                    logger.warning("Previous configuration does not match current configuration. Processing will not continue from previous state.")
                else:
                    config_eq = True
                    save_config = False
                    logger.info("Previous configuration matches current configuration.")

            if not os.path.exists(test_orbit_file):
                logger.warning("No previous test orbit file found.")
                save_orbit = True
            else:
                logger.info("Previous test orbit file found.")

                test_orbit_prev = Orbits.from_csv(
                    test_orbit_file,
                )
                if test_orbit_prev != orbit:
                    logger.warning("Previous test orbit does not match current test orbit.")
                else:
                    test_orbit_eq = True
                    save_orbit = False
                    logger.info("Previous test orbit matches current test orbit.")

            if not os.path.exists(status_file):
                logger.warning("No previous status file found.")
            else:
                if test_orbit_eq and config_eq:
                    with open(status_file, "r") as status_in:
                        status = yaml.load(status_in, Loader=yaml.FullLoader)
                    logger.info("Previous status file found.")

        if save_config:
            with open(config_file, "w") as config_out:
                yaml.safe_dump(RUN_CONFIG, config_out)
            logger.debug("Saved config.yml.")

        if save_orbit:
            orbit.to_csv(
                test_orbit_file
            )
            logger.debug("Saved test_orbit.csv.")

            if status["complete"]:
                logger.info("Orbit has already finished processing.")

    if not status["complete"]:
        if not status["rangeAndShift"]:
            projected_observations = rangeAndShift(
                preprocessed_observations,
                orbit,
                **range_shift_config
            )
            if out_dir is not None:
                projected_observations.to_csv(
                    os.path.join(out_dir, "projected_observations.csv"),
                    index=False,
                    float_format="%.15e"
                )
                logger.debug("Saved projected_observations.csv.")

        else:
            logger.info("Range and shift completed previously.")
            projected_observations = pd.read_csv(
                os.path.join(out_dir, "projected_observations.csv"),
                index_col=False,
                dtype={"obs_id" : str},
                float_precision="round_trip"
            )
            logger.debug("Read projected_observations.csv.")

        status["rangeAndShift"] = True
        if out_dir is not None:
            with open(status_file, "w") as status_out:
                yaml.safe_dump(status, status_out)
            logger.debug("Updated status.yml.")

        if not status["clusterAndLink"]:
            clusters, cluster_members = clusterAndLink(
                projected_observations,
                **cluster_link_config
            )
            if out_dir is not None:
                clusters.to_csv(
                    os.path.join(out_dir, "clusters.csv"),
                    index=False,
                    float_format="%.15e"
                )
                logger.debug("Saved clusters.csv.")

                cluster_members.to_csv(
                    os.path.join(out_dir, "cluster_members.csv"),
                    index=False,
                    float_format="%.15e"
                )
                logger.debug("Saved cluster_members.csv.")
        else:
            logger.info("Clustering completed previously.")
            clusters = pd.read_csv(
                os.path.join(out_dir, "clusters.csv"),
                index_col=False
            )
            logger.debug("Read clusters.csv.")

            cluster_members = pd.read_csv(
                os.path.join(out_dir, "cluster_members.csv"),
                index_col=False,
                dtype={"obs_id" : str},
                float_precision="round_trip"
            )
            logger.debug("Read cluster_members.csv.")

        status["clusterAndLink"] = True
        if out_dir is not None:
            with open(status_file, "w") as status_out:
                yaml.safe_dump(status, status_out)
            logger.debug("Updated status.yml.")

        if not status["initialOrbitDetermination"]:
            iod_orbits, iod_orbit_members = initialOrbitDetermination(
                projected_observations,
                cluster_members,
                **iod_config
            )
            if out_dir is not None:
                Orbits.from_df(iod_orbits).to_csv(
                    os.path.join(out_dir, "iod_orbits.csv")
                )
                logger.debug("Saved iod_orbits.csv.")

                iod_orbit_members.to_csv(
                    os.path.join(out_dir, "iod_orbit_members.csv"),
                    index=False,
                    float_format="%.15e"
                )
                logger.debug("Saved iod_orbit_members.csv.")
        else:
            logger.info("Initial orbit determination completed previously.")
            iod_orbits = Orbits.from_csv(
                os.path.join(out_dir, "iod_orbits.csv"),
            ).to_df(include_units=False)
            logger.debug("Read iod_orbits.csv.")

            iod_orbit_members = pd.read_csv(
                os.path.join(out_dir, "iod_orbit_members.csv"),
                index_col=False,
                dtype={"obs_id" : str},
                float_precision="round_trip"
            )
            logger.debug("Read iod_orbit_members.csv.")

        status["initialOrbitDetermination"] = True
        if out_dir is not None:
            with open(status_file, "w") as status_out:
                yaml.safe_dump(status, status_out)
            logger.debug("Updated status.yml.")

        iod_orbits = iod_orbits[["orbit_id", "epoch", "x", "y", "z", "vx", "vy", "vz"]]
        iod_orbit_members = iod_orbit_members[iod_orbit_members["outlier"] == 0][["orbit_id", "obs_id"]]
        iod_orbits = iod_orbits[iod_orbits["orbit_id"].isin(iod_orbit_members["orbit_id"].unique())]
        for df in [iod_orbits, iod_orbit_members]:
            df.reset_index(
                inplace=True,
                drop=True
            )

        if not status["differentialCorrection"]:
            od_orbits, od_orbit_members = differentialCorrection(
                iod_orbits,
                iod_orbit_members,
                projected_observations,
                **od_config
            )
            if out_dir is not None:
                Orbits.from_df(od_orbits).to_csv(
                    os.path.join(out_dir, "od_orbits.csv")
                )
                logger.debug("Saved od_orbits.csv.")

                od_orbit_members.to_csv(
                    os.path.join(out_dir, "od_orbit_members.csv"),
                    index=False,
                    float_format="%.15e"
                )
                logger.debug("Saved od_orbit_members.csv.")
        else:
            logger.info("Differential correction completed previously.")
            od_orbits = Orbits.from_csv(
                os.path.join(out_dir, "od_orbits.csv"),
            ).to_df(include_units=False)
            logger.debug("Read od_orbits.csv.")

            od_orbit_members = pd.read_csv(
                os.path.join(out_dir, "od_orbit_members.csv"),
                index_col=False,
                dtype={"obs_id" : str},
                float_precision="round_trip"
            )
            logger.debug("Read od_orbit_members.csv.")

        status["differentialCorrection"] = True
        if out_dir is not None:
            with open(status_file, "w") as status_out:
                yaml.safe_dump(status, status_out)
            logger.debug("Updated status.yml.")

        od_orbit_members = od_orbit_members[od_orbit_members["outlier"] == 0][["orbit_id", "obs_id"]]
        od_orbits = od_orbits[od_orbits["orbit_id"].isin(od_orbit_members["orbit_id"].unique())]
        for df in [od_orbits, od_orbit_members]:
            df.reset_index(
                inplace=True,
                drop=True
            )

        if not status["mergeAndExtendOrbits"]:

            recovered_orbits, recovered_orbit_members = mergeAndExtendOrbits(
                od_orbits,
                od_orbit_members,
                projected_observations,
                **odp_config
            )
            if out_dir is not None:
                Orbits.from_df(recovered_orbits).to_csv(
                    os.path.join(out_dir, "recovered_orbits.csv")
                )
                logger.debug("Saved recovered_orbits.csv.")

                recovered_orbit_members.to_csv(
                    os.path.join(out_dir, "recovered_orbit_members.csv"),
                    index=False,
                    float_format="%.15e"
                )
                logger.debug("Saved recovered_orbit_members.csv.")
        else:
            logger.info("Orbit extension and merging completed previously.")
            recovered_orbits = Orbits.from_csv(
                os.path.join(out_dir, "recovered_orbits.csv"),
            ).to_df(include_units=False)
            logger.debug("Read recovered_orbits.csv.")

            recovered_orbit_members = pd.read_csv(
                os.path.join(out_dir, "recovered_orbit_members.csv"),
                index_col=False,
                dtype={"obs_id" : str},
                float_precision="round_trip"
            )
            logger.debug("Read recovered_orbit_members.csv.")

        status["mergeAndExtendOrbits"] = True
        status["complete"] = True
        if out_dir is not None:
            with open(status_file, "w") as status_out:
                yaml.safe_dump(status, status_out)
            logger.debug("Updated status.yml.")

    else:
        logger.info("Orbit previously completed processing.")
        recovered_orbits = Orbits.from_csv(
            os.path.join(out_dir, "recovered_orbits.csv"),
        ).to_df(include_units=False)
        logger.debug("Read recovered_orbits.csv.")

        recovered_orbit_members = pd.read_csv(
            os.path.join(out_dir, "recovered_orbit_members.csv"),
            index_col=False,
            dtype={"obs_id" : str},
            float_precision="round_trip"
        )
        logger.debug("Read recovered_orbit_members.csv.")

    logger.removeHandler(file_handler)
    return recovered_orbits, recovered_orbit_members

def runTHOR(
        preprocessed_observations,
        test_orbits,
        range_shift_config=Config.RANGE_SHIFT_CONFIG,
        cluster_link_config=Config.CLUSTER_LINK_CONFIG,
        iod_config=Config.IOD_CONFIG,
        od_config=Config.OD_CONFIG,
        odp_config=Config.ODP_CONFIG,
        out_dir=None,
        if_exists="continue",
        logging_level=logger.info
    ):
    logger.setLevel(logging_level)

    # Connect to ray cluster if enabled
    if USE_RAY and not ray.is_initialized():
        ray.init(address="auto")

    # Build the configuration directory which stores
    # the run parameters
    RUN_CONFIG = {
        "RANGE_SHIFT_CONFIG" : range_shift_config,
        "CLUSTER_LINK_CONFIG" : cluster_link_config,
        "IOD_CONFIG" : iod_config,
        "OD_CONFIG" : od_config,
        "ODP_CONFIG" : odp_config
    }
    orbits_completed = []

    continue_ = False
    if_exists_ = if_exists
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            logger.debug("Created {} directory.".format(out_dir))

        else:
            if if_exists == "continue":
                logger.warning("{} directory already exists, attempting to continue previous run.".format(out_dir))
                continue_ = True
            elif if_exists == "erase":
                logger.warning("{} directory already exists, removing previous results.".format(out_dir))
                shutil.rmtree(out_dir)
                os.mkdir(out_dir)
                logger.debug("Created {} directory.".format(out_dir))
            else:
                err = (
                    "if_exists should be one of {'continue', 'erase'}."
                )
                raise ValueError(err)

        # The primary files which will be used to determine if the run
        # can be continued from a previous state and, if so, from where
        # to continue the run
        config_file = os.path.join(out_dir, "config.yml")
        test_orbits_in_file = os.path.join(out_dir, "test_orbits_in.csv")
        status_file = os.path.join(out_dir, "status.txt")
        config_eq = False
        test_orbits_eq = False
        save_orbits = True
        save_config = True

        # Add summary file for test_orbits that tracks number of recovered orbits and number of observations
        # linked in addition to the test_orbit_id used by THOR
        test_orbits_out_file = os.path.join(out_dir, "test_orbits_out.csv")
        if continue_:
            if not os.path.exists(config_file):
                logger.warning("No previous configuration file found.")
                save_config = True
                if_exists_ = "erase"
            else:
                logger.info("Previous configuration file found. Comparing settings...")

                config_file_prev = open(config_file, "r")
                config_prev = yaml.load(config_file_prev, Loader=yaml.FullLoader)
                config_file_prev.close()

                if config_prev != RUN_CONFIG:
                    logger.warning("Previous configuration does not match current configuration. Processing will not continue from previous state.")
                    if_exists_ = "erase"
                else:
                    config_eq = True
                    save_config = False
                    logger.info("Previous configuration matches current configuration.")

            if not os.path.exists(test_orbits_in_file):
                logger.warning("No previous test orbits file found.")
                save_orbits = True
            else:
                logger.info("Previous test orbits file found.")

                test_orbits_prev = Orbits.from_csv(
                    test_orbits_in_file,
                )
                if test_orbits_prev != test_orbits:
                    logger.warning("Previous test orbits do not match current test orbits.")
                else:
                    test_orbits_eq = True
                    save_orbits = False
                    test_orbits_df = test_orbits_prev.to_df(include_units=False)
                    logger.info("Previous test orbits match current test orbits.")

            if not os.path.exists(status_file):
                logger.warning("No previous status file found.")
            else:
                if test_orbits_eq and config_eq:
                    orbits_completed = np.loadtxt(
                        os.path.join(out_dir, "status.txt"),
                        delimiter="\n",
                        dtype=str,
                        ndmin=1
                    )
                    logger.info("Previous status file found.")

        if (not test_orbits_eq or not config_eq) and continue_:
            if if_exists == "continue":
                logger.critical("Previous run cannot continue from previous state.")
                raise ValueError("Previous run cannot continue from previous state. Set if_exists to 'erase' or change/delete the output directory.")
            elif if_exists == "erase":
                shutil.rmtree(out_dir)
                os.mkdir(out_dir)
                logger.debug("Created {} directory.".format(out_dir))
            else:
                pass

        if save_config:
            with open(config_file, "w") as config_out:
                yaml.safe_dump(RUN_CONFIG, config_out)
            logger.debug("Saved config.yml.")

        if save_orbits:
            test_orbits.to_csv(
                test_orbits_in_file
            )
            logger.debug("Saved test_orbits_in.csv.")

            preprocessed_observations.to_csv(
                os.path.join(out_dir, "preprocessed_observations.csv"),
                index=False,
                float_format="%.15e"
            )
            logger.debug("Saved preprocessed_observations.csv.")

    test_orbit_dfs = []
    recovered_orbits_dfs = []
    recovered_orbit_members_dfs = []
    obs_ids_linked = []
    num_orbits = len(test_orbits)
    if num_orbits != len(orbits_completed):
        test_orbits_split = test_orbits[len(orbits_completed):].split(1)
    else:
        test_orbits_split = []

    # If orbits have previously completed, read the results and continue iterating
    # through orbits not previously completed.
    id_offset = 0
    if len(orbits_completed) > 0:
        logger.info("{}/{} orbits have previously finished processing.".format(len(orbits_completed), num_orbits))

        test_orbits_df = Orbits.from_csv(
            test_orbits_out_file,
        ).to_df(include_units=False)
        logger.debug("Read previous test_orbits_out.csv.")

        recovered_orbits = Orbits.from_csv(
            os.path.join(out_dir, "recovered_orbits.csv"),
        ).to_df(include_units=False)
        logger.debug("Read previous recovered_orbits.csv.")

        recovered_orbit_members = pd.read_csv(
            os.path.join(out_dir, "recovered_orbit_members.csv"),
            index_col=False,
            dtype={"obs_id" : str},
            float_precision="round_trip"
        )
        logger.debug("Read previous recovered_orbit_members.csv.")

        test_orbit_dfs = [test_orbits_df]
        recovered_orbits_dfs = [recovered_orbits]
        recovered_orbit_members_dfs = [recovered_orbit_members]
        obs_ids_linked = recovered_orbit_members["obs_id"].values
        id_offset = len(orbits_completed)

    if len(test_orbits_split) != 0:
        for i, orbit_i in enumerate(test_orbits_split):

            time_start = time.time()
            orbit_id = "{:08d}".format(i + id_offset)

            logger.info("Processing orbit {} ({}/{})...".format(orbit_id, i + 1 + id_offset, num_orbits))

            if out_dir is not None:
                orbit_dir = os.path.join(out_dir, "orbit_{}".format(orbit_id))
            else:
                orbit_dir = None

            linked_mask = (~preprocessed_observations["obs_id"].isin(obs_ids_linked))

            recovered_orbits_i, recovered_orbit_members_i = runTHOROrbit(
                preprocessed_observations[linked_mask],
                orbit_i,
                range_shift_config=range_shift_config,
                cluster_link_config=cluster_link_config,
                iod_config=iod_config,
                od_config=od_config,
                odp_config=odp_config,
                out_dir=orbit_dir,
                if_exists=if_exists_,
                logging_level=logging_level
            )

            time_end = time.time()

            if len(recovered_orbits_i) > 0:
                recovered_orbits_i.insert(0, "test_orbit_id", orbit_id)
                recovered_orbit_members_i.insert(0, "test_orbit_id", orbit_id)
                obs_ids_linked_i = recovered_orbit_members_i["obs_id"].unique()
                obs_ids_linked = np.concatenate([obs_ids_linked, obs_ids_linked_i])

                orbits_recovered = len(recovered_orbits_i)
                observations_linked = len(obs_ids_linked_i)
            else:
                orbits_recovered = 0
                observations_linked = 0

            test_orbit_i = orbit_i.to_df(include_units=False)
            test_orbit_i["test_orbit_id"] = orbit_id
            test_orbit_i["orbits_recovered"] = orbits_recovered
            test_orbit_i["observations_linked"] = observations_linked
            test_orbit_i["processing_time"] = time_end - time_start
            test_orbit_dfs.append(test_orbit_i)

            logger.info("Completed processing orbit {} in {:.3f} seconds.".format(orbit_id, time_end - time_start))

            recovered_orbits_dfs.append(recovered_orbits_i)
            recovered_orbit_members_dfs.append(recovered_orbit_members_i)

            test_orbits_df = pd.concat(
                test_orbit_dfs,
                ignore_index=True
            )
            recovered_orbits = pd.concat(
                recovered_orbits_dfs,
                ignore_index=True
            )
            recovered_orbit_members = pd.concat(
                recovered_orbit_members_dfs,
                ignore_index=True
            )

            if out_dir is not None:
                Orbits.from_df(test_orbits_df).to_csv(
                    test_orbits_out_file
                )
                logger.debug("Saved test_orbits_out.csv.")

                Orbits.from_df(recovered_orbits).to_csv(
                    os.path.join(out_dir, "recovered_orbits.csv")
                )
                logger.debug("Saved recovered_orbits.csv.")

                recovered_orbit_members.to_csv(
                    os.path.join(out_dir, "recovered_orbit_members.csv"),
                    index=False,
                    float_format="%.15e"
                )
                logger.debug("Saved recovered_orbit_members.csv.")

            orbits_completed = np.concatenate([orbits_completed, np.array([orbit_id])])
            if out_dir is not None:
                with open(os.path.join(out_dir, "status.txt"), "w") as status_out:
                    np.savetxt(
                        status_out,
                        orbits_completed,
                        delimiter="\n",
                        fmt="%s"
                    )
                logger.info("Saved status.txt.")


    else:

        logger.info("Run completed previously.")
        test_orbits_df = Orbits.from_csv(
                test_orbits_out_file,
        ).to_df(include_units=False)
        logger.debug("Read test_orbits_out.csv.")

        recovered_orbits = Orbits.from_csv(
            os.path.join(out_dir, "recovered_orbits.csv"),
        ).to_df(include_units=False)
        logger.debug("Read recovered_orbits.csv.")

        recovered_orbit_members = pd.read_csv(
            os.path.join(out_dir, "recovered_orbit_members.csv"),
            index_col=False,
            dtype={"obs_id" : str},
            float_precision="round_trip"
        )
        logger.debug("Read recovered_orbit_members.csv.")

    return test_orbits_df, recovered_orbits, recovered_orbit_members
