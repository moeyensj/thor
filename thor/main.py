import os
import time
import signal
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from sklearn.cluster import DBSCAN
from astropy.time import Time

from .config import Config
from .cell import Cell
from .test_orbit import TestOrbit
from .orbits.ephemeris import generateEphemeris
from .orbits.iod import iod
from .plotting import plotOrbitsFindable
from .plotting import plotOrbitsFound
from .plotting import plotOrbitsMissed
from .plotting import plotProjectionVelocitiesFindable
from .plotting import plotProjectionVelocitiesFound
from .plotting import plotProjectionVelocitiesMissed
from .plotting import _setPercentage
from .analysis import calcLinkageEfficiency
from .analysis import analyzeObservations
from .analysis import analyzeProjections
from .analysis import analyzeClusters

__all__ = ["rangeAndShift",
           "clusterVelocity",
           "_clusterVelocity",
           "clusterAndLink",
           "_initialOrbitDetermination",
           "initialOrbitDetermination",
           "runTHOR"]

def rangeAndShift(observations, 
                  orbit, 
                  t0, 
                  cell_area=10, 
                  backend="PYOORB", 
                  backend_kwargs=None, 
                  verbose=True):
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
    t0 : `~astropy.time.core.Time`
        Epoch at which orbit is defined.
    cell_area : float, optional
        Cell's area in units of square degrees. 
        [Default = 10]
    backend : {'THOR', 'PYOORB'}, optional
        Which backend to use. 
    backend_kwargs : dict, optional
        Settings and additional parameters to pass to selected 
        backend.
    verbose : bool, optional
        Print progress statements? 
        [Default = True]
        
    Returns
    -------
    projected_observations : {`~pandas.DataFrame`, -1}
        Observations dataframe (from cell.observations) with columns containing
        projected coordinates. 
    """
    time_start = time.time()
    if verbose == True:
        print("THOR: rangeAndShift")
        print("-------------------------")
        print("Running range and shift...")
        print("Assuming r = {} AU".format(orbit[:3]))
        print("Assuming v = {} AU per day".format(orbit[3:]))

    # Build observers dictionary: keys are observatory codes with exposure times (as astropy.time objects)
    # as values
    observers = {}
    for code in observations["observatory_code"].unique():
        observers[code] = Time(observations[observations["observatory_code"].isin([code])]["mjd_utc"].unique(), format="mjd", scale="utc")

    # Propagate test orbit to all times in observations
    ephemeris = generateEphemeris(
        orbit.reshape(1, -1), 
        t0, 
        observers, 
        backend=backend,     
        backend_kwargs=backend_kwargs
    )
    
    observations = observations.merge(ephemeris[["mjd_utc", "observatory_code", "obs_x", "obs_y", "obs_z"]], left_on=["mjd_utc", "observatory_code"], right_on=["mjd_utc", "observatory_code"])
    
    projected_dfs = []
    test_orbits = []
    cells = []
    
    for code, observation_times in observers.items():
        observatory_mask_observations = (observations["observatory_code"] == code)
        observatory_mask_ephemeris = (ephemeris["observatory_code"] == code)

        for t1 in observation_times:
            # Select test orbit ephemeris corresponding to visit
            ephemeris_visit = ephemeris[observatory_mask_ephemeris & (ephemeris["mjd_utc"] == t1.utc.mjd)]

            # Select observations corresponding to visit
            observations_visit = observations[observatory_mask_observations & (observations["mjd_utc"] == t1.utc.mjd)].copy()
            
            # Create Cell centered on the sky-plane location of the
            # test orbit
            cell = Cell(
                ephemeris_visit[["RA_deg", "Dec_deg"]].values[0],
                t1.utc.mjd,
                area=cell_area,
            )
            
            # Grab observations within cell
            cell.getObservations(observations_visit)

            if len(cell.observations) != 0:
                
                # Create test orbit with state of orbit at visit time
                test_orbit = TestOrbit(
                    ephemeris_visit[["obj_x", "obj_y", "obj_z", "obj_vx", "obj_vy", "obj_vz"]].values[0],
                    t1
                )

                # Prepare rotation matrices 
                test_orbit.prepare(verbose=False)

                # Apply rotation matrices and transform observations into the orbit's
                # frame of motion. 
                test_orbit.apply(cell, verbose=False)
                
                # Append results to approprate lists
                projected_dfs.append(cell.observations)
                cells.append(cell)
                test_orbits.append(test_orbit)

            else:
                continue
    
    projected_observations = pd.concat(projected_dfs)   
    projected_observations.sort_values(by=["mjd_utc", "observatory_code"], inplace=True)
    projected_observations.reset_index(inplace=True, drop=True)
    
    time_end = time.time()
    if verbose == True:
        print("Done. Final DataFrame has {} observations.".format(len(projected_observations)))
        print("Total time in seconds: {}".format(time_end - time_start))  
        print("-------------------------")
        print("")
        
    return projected_observations


def _init_worker():
    """
    Tell multiprocessing worker to ignore signals, will only
    listen to parent process. 
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    return

def clusterVelocity(obs_ids,
                    x, 
                    y, 
                    dt, 
                    vx, 
                    vy, 
                    eps=0.005, 
                    min_samples=5):
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
        
    Returns
    -------
    {list, -1}
        If clusters are found, will return a list of numpy arrays containing the 
        observation IDs for each cluster. If no clusters are found returns -1. 
    """
    xx = x - vx * dt
    yy = y - vy * dt
    X = np.vstack([xx, yy]).T  
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1).fit(X)
    
    clusters = db.labels_[np.where(db.labels_ != -1)[0]]
    cluster_ids = []
    
    if len(clusters) != 0:
        for cluster in np.unique(clusters):
            cluster_ids.append(obs_ids[np.where(db.labels_ == cluster)[0]])
    else:
        cluster_ids = np.NaN
    
    del db
    return cluster_ids
           
def _clusterVelocity(vx, vy,
                     obs_ids=None,
                     x=None,
                     y=None,
                     dt=None,
                     eps=None,
                     min_samples=None):
    """
    Helper function to multiprocess clustering.
    
    """
    return clusterVelocity(obs_ids,
                           x,
                           y,
                           dt,
                           vx,
                           vy,
                           eps=eps,
                           min_samples=min_samples) 

def clusterAndLink(observations, 
                   vx_range=[-0.1, 0.1], 
                   vy_range=[-0.1, 0.1],
                   vx_bins=100, 
                   vy_bins=100,
                   vx_values=None,
                   vy_values=None,
                   threads=12, 
                   eps=0.005, 
                   min_samples=5,
                   verbose=True):
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
    verbose : bool, optional
        Print progress statements? 
        [Default = True]
        
    Returns
    -------
    all_clusters : `~pandas.DataFrame`
        DataFrame with the cluster ID, the number of observations, and the x and y velocity. 
    cluster_members : `~pandas.DataFrame`
        DataFrame containing the cluster ID and the observation IDs of its members. 
    """ 
    # Extract useful quantities
    obs_ids = observations["obs_id"].values
    theta_x = observations["theta_x_deg"].values
    theta_y = observations["theta_y_deg"].values
    mjd = observations["mjd_utc"].values

    # Select detections in first exposure
    first = np.where(mjd == mjd.min())[0]
    theta_x0 = theta_x[first]
    theta_y0 = theta_y[first]
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

    time_start_cluster = time.time()
    if verbose == True:
        print("THOR: clusterAndLink")
        print("-------------------------")
        print("Running velocity space clustering...")
        print("X velocity range: {}".format(vx_range))
        
        if vx_values is not None:
            print("X velocity values: {}".format(vx_bins))
        else:
            print("X velocity bins: {}".format(vx_bins))
            
        print("Y velocity range: {}".format(vy_range))
        if vy_values is not None:
            print("Y velocity values: {}".format(vy_bins))
        else:
            print("Y velocity bins: {}".format(vy_bins))
        if vx_values is not None:
            print("User defined x velocity values: True")
        else: 
            print("User defined x velocity values: False")
        if vy_values is not None:
            print("User defined y velocity values: True")
        else:
            print("User defined y velocity values: False")
            
        if vx_values is None and vy_values is None:
            print("Velocity grid size: {}".format(vx_bins * vy_bins))
        else: 
            print("Velocity grid size: {}".format(vx_bins))
        print("Max sample distance: {}".format(eps))
        print("Minimum samples: {}".format(min_samples))

    
    possible_clusters = []
    if threads > 1:
        if verbose:
            print("Using {} threads...".format(threads))
        p = mp.Pool(threads, _init_worker)
        try:
            possible_clusters = p.starmap(partial(_clusterVelocity, 
                                                  obs_ids=obs_ids,
                                                  x=theta_x,
                                                  y=theta_y,
                                                  dt=dt,
                                                  eps=eps,
                                                  min_samples=min_samples),
                                                  zip(vxx.T, vyy.T))
        
        except KeyboardInterrupt:
            p.terminate()
            
        p.close()
    else:
        possible_clusters = []
        for vxi, vyi in zip(vxx, vyy):
            possible_clusters.append(clusterVelocity(
                obs_ids,
                theta_x, 
                theta_y, 
                dt, 
                vxi, 
                vyi, 
                eps=eps, 
                min_samples=min_samples)
            )
    time_end_cluster = time.time()    
    
    if verbose == True:
        print("Done. Completed in {} seconds.".format(time_end_cluster - time_start_cluster))
        print("")
        
    if verbose == True:
        print("Restructuring clusters...")
    time_start_restr = time.time()
    
    possible_clusters = pd.DataFrame({"clusters": possible_clusters})

    # Remove empty clusters
    possible_clusters = possible_clusters[~possible_clusters["clusters"].isna()]

    if len(possible_clusters) != 0:
        # Make DataFrame with cluster velocities so we can figure out which 
        # velocities yielded clusters, add names to index so we can enable the join
        cluster_velocities = pd.DataFrame({"theta_vx": vxx, "theta_vy": vyy})
        cluster_velocities.index.set_names("velocity_id", inplace=True)

        # Split lists of cluster ids into one column per cluster for each different velocity
        # then stack the result
        possible_clusters = pd.DataFrame(possible_clusters["clusters"].values.tolist(), index=possible_clusters.index)
        possible_clusters = pd.DataFrame(possible_clusters.stack())
        possible_clusters.rename(columns={0: "obs_ids"}, inplace=True)
        possible_clusters = pd.DataFrame(possible_clusters["obs_ids"].values.tolist(), index=possible_clusters.index)

        # Drop duplicate clusters
        possible_clusters.drop_duplicates(inplace=True)

        # Set index names
        possible_clusters.index.set_names(["velocity_id", "cluster_id"], inplace=True)

        # Reset index
        possible_clusters.reset_index("cluster_id", drop=True, inplace=True)
        possible_clusters["cluster_id"] = np.arange(1, len(possible_clusters) + 1)

        # Make allClusters DataFrame
        all_clusters = possible_clusters.join(cluster_velocities)
        all_clusters.reset_index(drop=True, inplace=True)
        all_clusters = all_clusters[["cluster_id", "theta_vx", "theta_vy"]]

        # Make clusterMembers DataFrame
        cluster_members = possible_clusters.reset_index(drop=True).copy()
        cluster_members.index = cluster_members["cluster_id"]
        cluster_members.drop("cluster_id", axis=1, inplace=True)
        cluster_members = pd.DataFrame(cluster_members.stack())
        cluster_members.rename(columns={0: "obs_id"}, inplace=True)
        cluster_members.reset_index(inplace=True)
        cluster_members.drop("level_1", axis=1, inplace=True)        
        
    else: 
        cluster_members = pd.DataFrame(columns=["cluster_id", "obs_id"])
        all_clusters = pd.DataFrame(columns=["cluster_id", "theta_vx", "theta_vy"])
    
    time_end_restr = time.time()
   
    if verbose == True:
        print("Done. Completed in {} seconds.".format(time_end_restr - time_start_restr))
        print("")
        print("Found {} clusters.".format(len(all_clusters)))
        print("Total time in seconds: {}".format(time_end_restr - time_start_cluster))   
        print("-------------------------")
        print("")
        
    return all_clusters, cluster_members

def _initialOrbitDetermination(
                              obs_ids, 
                              cluster_id, 
                              observations=None, 
                              observation_selection_method=None, 
                              chi2_threshold=None,
                              contamination_percentage=None,
                              iterate=None, 
                              light_time=None,
                              backend=None,
                              backend_kwargs=None):
    """
    Helper function to multiprocess clustering.
    
    """
    orbit, orbit_members, outliers = iod(
        observations[observations["obs_id"].isin(obs_ids)], 
        observation_selection_method=observation_selection_method, 
        chi2_threshold=chi2_threshold,
        contamination_percentage=contamination_percentage,
        iterate=iterate, 
        light_time=light_time,
        backend=backend,
        backend_kwargs=backend_kwargs)
    
    orbit["cluster_id"] = [cluster_id]
    
    return orbit, orbit_members, outliers


def initialOrbitDetermination(observations, 
                              cluster_members, 
                              observation_selection_method="combinations", 
                              chi2_threshold=10**3,
                              contamination_percentage=20.0,
                              iterate=False, 
                              light_time=True,
                              threads=30,
                              backend="THOR",
                              backend_kwargs=None):

    
    grouped = cluster_members.groupby(by="cluster_id")["obs_id"].apply(list)
    cluster_ids = list(grouped.index.values)
    obs_ids = grouped.values.tolist()

    orbit_dfs = []
    orbit_members_dfs = []
    outliers_list = []
    
    if threads > 1:
        p = mp.Pool(threads, _init_worker)
        try: 
            results = p.starmap(
                partial(
                    _initialOrbitDetermination,
                    observations=observations,
                    observation_selection_method=observation_selection_method, 
                    chi2_threshold=chi2_threshold,
                    contamination_percentage=contamination_percentage,
                    iterate=iterate, 
                    light_time=light_time,
                    backend=backend,
                    backend_kwargs=backend_kwargs
                ),
                zip(
                    obs_ids, 
                    cluster_ids
                )
            )
            
            results = list(zip(*results))
            orbit_dfs = results[0]
            orbit_members_dfs = results[1]
            outliers_list = results[2]
        
        except KeyboardInterrupt:
            p.terminate()
        
        p.close()

    else:
        for obs_ids_i, cluster_id_i in zip(obs_ids, cluster_ids):

            orbit, orbit_members, outliers = iod(
                observations[observations["obs_id"].isin(obs_ids_i)], 
                observation_selection_method=observation_selection_method, 
                chi2_threshold=chi2_threshold,
                contamination_percentage=contamination_percentage,
                iterate=iterate, 
                light_time=light_time,
                backend=backend,
                backend_kwargs=backend_kwargs)
            
            if len(orbit) > 0:
                orbit["cluster_id"] = [cluster_id_i]
                orbit_dfs.append(orbit)
                orbit_members_dfs.append(orbit_members)
                outliers_list.append(outliers)
                
            else:
                continue
                

    if len(orbit_dfs) > 0:
        
        orbits = pd.concat(orbit_dfs)
        orbits.dropna(inplace=True)
        orbits.reset_index(inplace=True, drop=True)
        orbits = orbits[[
            "orbit_id", 
            'cluster_id',
            "epoch_mjd_utc",
            "obj_x",
            "obj_y",
            "obj_z",
            "obj_vx",
            "obj_vy",
            "obj_vz",
            "arc_length",
            "num_obs",
            "chi2"
        ]]
        
        orbit_members = pd.concat(orbit_members_dfs)
        orbit_members.dropna(inplace=True)
        orbit_members.reset_index(inplace=True, drop=True)
        
        outliers = np.concatenate(outliers_list)
         
    else:
        orbits = pd.DataFrame(columns=[
            "orbit_id", 
            "cluster_id",
            "epoch_mjd_utc",
            "obj_x",
            "obj_y",
            "obj_z",
            "obj_vx",
            "obj_vy",
            "obj_vz",
            "arc_length",
            "num_obs",
            "chi2"
        ])
        
        orbit_members = pd.DataFrame(columns=[
            "orbit_id",
            "obs_id"
        ])

        outliers = np.array([])
    
    return orbits, orbit_members, outliers

def runTHOR(observations,
            orbits,
            knownOrbits=None,
            runDir=None,
            cell_area=10, 
            vx_range=[-0.1, 0.1], 
            vy_range=[-0.1, 0.1],
            vx_bins=100, 
            vy_bins=100,
            vx_values=None,
            vy_values=None,
            threads=30, 
            eps=0.005, 
            min_samples=5,
            contaminationThreshold=0.2,
            unknownIDs=[],
            falsePositiveIDs=[],
            verbose=True,
            columnMapping=Config.columnMapping):
    """
    Run THOR on observations using the given orbits.
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing observations.
    orbits : `~pandas.DataFrame`
        DataFrame with test orbits: need sky-plane location, epoch, heliocentric distance and cartesian velocity. 
    knownOrbits: {None, `~pandas.DataFrame`}, optional
        DataFrame with Keplerian orbital elements of known objects. Used for plotting
        purposes. 
        [Default = None]
    runDir : {None, str}, optional
        If None, intermittent files wont be saved. If string path is passed, intermittent files
        will be saved inside directory. Each orbit will have its own sub-folder, with the relevant files
        for each saved in these subfolders. 
        [Default = None]
    cell_area : float, optional
        Cell's area in units of square degrees. 
        [Default = 10]
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
    contaminationThreshold : float, optional
        Percentage (expressed between 0 and 1) of imposter observations in a cluster permitted for the 
        object to be found. 
        [Default = 0.2]
    unknownIDs : list, optional
        Values in the name column for unknown observations.
        [Default = `~thor.Config.unknownIDs`]
    falsePositiveIDs : list, optional
        Names of false positive IDs.
        [Default = `~thor.Config.falsePositiveIDs`]
    observatoryCode : str, optional
        Observatory from which to measure ephemerides.
        [Default = `~thor.Config.oorbObservatoryCode`]
    verbose : bool, optional
        Print progress statements? 
        [Default = True]
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~thor.Config.columnMapping`]
    
    Returns
    -------
    allObjects : `~pandas.DataFrame`
        Object summary DataFrame for the survey.
    summary : `~pandas.DataFrame`
        Overall summary DataFrame for the survey.
    summary_orbits : `~pandas.DataFrame`
        Overall summary DataFrame each orbit.
    """
    if verbose == True:
        print("THOR: runTHOR")
        print("-------------------------")
        print("Running THOR with {} orbits...".format(len(orbits)))
        print("")
    
    found_known_objects = []
    linked_detections = []
    summaries = []
    allObjects = []
    
    if runDir != None:
        try:
            os.mkdir(runDir)
        except:
            raise ValueError("runDir exists!")
        errFile = open(os.path.join(runDir, "errors.txt"), "a")
    
    # Analyze observations for entire survey
    allObjects_survey, summary_survey = analyzeObservations(
        observations,
        min_samples=min_samples, 
        unknownIDs=unknownIDs,
        falsePositiveIDs=falsePositiveIDs,
        verbose=True,
        columnMapping=columnMapping)
    
    # Save survey and orbit dataframes
    if runDir != None:
        allObjects_survey.to_csv(os.path.join(runDir, "allObjects_survey.txt"), sep=" ", index=False)
        summary_survey.to_csv(os.path.join(runDir, "summary_survey.txt"), sep=" ", index=False)
        orbits.to_csv(os.path.join(runDir, "orbits.txt"), sep=" ", index=False)
    
    # Plot findable orbits if known orbits are provided
    if type(knownOrbits) == pd.DataFrame:
        # Plot findable orbits (linear semi-major axis)
        try:
            fig, ax = plotOrbitsFindable(allObjects_survey, 
                                         knownOrbits, 
                                         testOrbits=orbits, 
                                         columnMapping=columnMapping)
            ax.set_xlim(0.0, 5.0)
            ax.text(_setPercentage(ax.get_xlim(), 0.02), 
                    _setPercentage(ax.get_ylim(), 0.93), 
                   "Findable Objects: {}".format(len(allObjects_survey[allObjects_survey["findable"] == 1])))

            if runDir != None:
                fig.savefig(os.path.join(runDir, "known_orbits_findable.png"))
        except:
            errFile.write("Survey: Error 1: Could not plot findable orbits.\n")

        # Plot findable orbits (log semi-major axis)
        try:
            fig, ax = plotOrbitsFindable(allObjects_survey, 
                                         knownOrbits, 
                                         testOrbits=orbits, 
                                         columnMapping=columnMapping)
            ax.set_xscale("log")
            ax.text(_setPercentage(ax.get_xlim(), 0.0001), 
                    _setPercentage(ax.get_ylim(), 0.93), 
                   "Findable Objects: {}".format(len(allObjects_survey[allObjects_survey["findable"] == 1])))

            if runDir != None:
                fig.savefig(os.path.join(runDir, "known_orbits_findable_log.png"))
        except:
            errFile.write("Survey: Error 2: Could not plot findable orbits (log).\n")
            
    # Run orbits
    for orbit_id in orbits["orbit_id"].unique():
        time_start = time.time()
        if verbose == True:
            print("THOR: runTHOR")
            print("-------------------------")
            print("Running orbit {}...".format(orbit_id))
            print("")
        
        # Make orbitDir variable
        if runDir != None:
            orbitDir = os.path.join(runDir, "orbit_{:04d}".format(orbit_id))
            os.mkdir(orbitDir)
            
        # Select orbit
        orbit = orbits[orbits["orbit_id"] == orbit_id]
       
        # Save orbit to file
        if runDir != None:
            orbit.to_csv(os.path.join(orbitDir, "orbit.txt"), sep=" ", index=False)

        # Propagate the orbit and gather all nearby detections
        projected_obs = rangeAndShift(
            observations, 
            orbit[columnMapping["RA_deg"]].values[0],
            orbit[columnMapping["Dec_deg"]].values[0],
            orbit[[columnMapping["r_au"]]].values[0], 
            orbit[[columnMapping["obj_dx/dt_au_p_day"],
                   columnMapping["obj_dy/dt_au_p_day"],
                   columnMapping["obj_dz/dt_au_p_day"]]].values[0], 
            orbit[columnMapping["exp_mjd"]].values[0],
            mjds=mjds, 
            cell_area=cell_area,
            verbose=False,
            columnMapping=columnMapping)
        
        if type(projected_obs) == int:
            if runDir!= None:
                errFile.write("Orbit ID {:04d}: Error 0: Skipping orbit.\n".format(orbit_id))
            if verbose == True:
                print("Skipping orbit {}.".format(orbit_id))
                print("-------------------------")
                print("")
            continue
        
        # Save projected observations to file
        if runDir != None:
            projected_obs.to_csv(os.path.join(orbitDir, "projected_obs.txt"), sep=" ", index=False)

        # Analyze propagated observations
        allObjects_projection, summary_projection = analyzeProjections(
            projected_obs[~projected_obs[columnMapping["obs_id"]].isin(linked_detections)],
            min_samples=min_samples,
            unknownIDs=unknownIDs,
            falsePositiveIDs=falsePositiveIDs,
            columnMapping=columnMapping)
        summary_projection["orbit_id"] = orbit_id
        allObjects_projection["orbit_id"] = np.ones(len(allObjects_projection), dtype=int) * orbit_id
        
        # Save projected observations to file
        if runDir != None:
            allObjects_projection.to_csv(os.path.join(orbitDir, "allObjects.txt"), sep=" ", index=False)
            summary_projection.to_csv(os.path.join(orbitDir, "summary.txt"), sep=" ", index=False)
        
        # Plot projection velocities of findable known objects
        try:
            fig, ax = plotProjectionVelocitiesFindable(allObjects_projection, vx_range=vx_range, vy_range=vy_range)
            if runDir != None:
                fig.savefig(os.path.join(orbitDir, "projection_findable.png"))
        except:
            errFile.write("Orbit ID {:04d}: Error 3: Could not plot projection velocities findable.\n".format(orbit_id))
            
        # Plot findable orbits if known orbits are provided
        if type(knownOrbits) == pd.DataFrame:
            # Plot findable orbits (linear semi-major axis)
            try:
                fig, ax = plotOrbitsFindable(allObjects_projection, 
                                             knownOrbits, 
                                             testOrbits=orbit, 
                                             columnMapping=columnMapping)
                ax.set_xlim(0.0, 5.0)
                ax.text(_setPercentage(ax.get_xlim(), 0.02), 
                        _setPercentage(ax.get_ylim(), 0.93), 
                       "Findable Objects: {}".format(len(allObjects_projection[allObjects_projection["findable"] == 1])))

                if runDir != None:
                    fig.savefig(os.path.join(orbitDir, "known_orbits_findable.png"))
            except:
                errFile.write("Orbit ID {:04d}: Error 4: Could not plot findable orbits.\n".format(orbit_id))

            # Plot findable orbits (log semi-major axis)
            try:
                fig, ax = plotOrbitsFindable(allObjects_projection, 
                                            knownOrbits, 
                                            testOrbits=orbit, 
                                            columnMapping=columnMapping)
                ax.set_xscale("log")
                ax.text(_setPercentage(ax.get_xlim(), 0.0001), 
                        _setPercentage(ax.get_ylim(), 0.93), 
                       "Findable Objects: {}".format(len(allObjects_projection[allObjects_projection["findable"] == 1])))

                if runDir != None:
                    fig.savefig(os.path.join(orbitDir, "known_orbits_findable_log.png"))
            except:
                errFile.write("Orbit ID {:04d}: Error 5: Could not plot findable orbits (log).\n".format(orbit_id))
                
        # Cluster and link
        allClusters_projection, clusterMembers_projection = clusterAndLink(
            projected_obs[~projected_obs[columnMapping["obs_id"]].isin(linked_detections)],
            vx_range=vx_range, 
            vy_range=vy_range,
            vx_bins=vx_bins, 
            vy_bins=vy_bins,
            vx_values=vx_values,
            vy_values=vy_values,
            threads=threads, 
            eps=eps, 
            min_samples=min_samples,
            verbose=True,
            columnMapping=columnMapping)
        
        # Save cluster files to file
        if runDir != None:
            allClusters_projection.to_csv(os.path.join(orbitDir, "allClusters.txt"), sep=" ", index=False)
            clusterMembers_projection.to_csv(os.path.join(orbitDir, "clusterMembers.txt"), sep=" ", index=False)
            
        # Analyze resulting clusters
        allClusters_projection, clusterMembers_projection, allObjects_projection, summary_projection = analyzeClusters(
            projected_obs[~projected_obs[columnMapping["obs_id"]].isin(linked_detections)],
            allClusters_projection, 
            clusterMembers_projection, 
            allObjects_projection,
            summary_projection,  
            min_samples=min_samples, 
            contaminationThreshold=contaminationThreshold, 
            unknownIDs=unknownIDs,
            falsePositiveIDs=falsePositiveIDs,
            verbose=True,
            columnMapping=columnMapping)
        
        # Save cluster files to file
        if runDir != None:
            allClusters_projection.to_csv(os.path.join(orbitDir, "allClusters.txt"), sep=" ", index=False)
            clusterMembers_projection.to_csv(os.path.join(orbitDir, "clusterMembers.txt"), sep=" ", index=False)
            allObjects_projection.to_csv(os.path.join(orbitDir, "allObjects.txt"), sep=" ", index=False)
            summary_projection.to_csv(os.path.join(orbitDir, "summary.txt"), sep=" ", index=False)
        
        # Calculate linkage efficiency for known objects
        linkage_efficiency = calcLinkageEfficiency(
            allObjects_projection, 
            vx_range=vx_range, 
            vy_range=vy_range,
            verbose=True)
        summary_projection["percent_linkage_efficiency"] = linkage_efficiency * 100

        # Plot projection velocities of found and missed known objects
        try:
            fig, ax = plotProjectionVelocitiesFound(allObjects_projection, vx_range=vx_range, vy_range=vy_range)
            if runDir != None:
                fig.savefig(os.path.join(orbitDir, "projection_found.png"))
        except:
            errFile.write("Orbit ID {:04d}: Error 6: Could not plot projection velocities found.\n".format(orbit_id))
        
        try:
            fig, ax = plotProjectionVelocitiesMissed(allObjects_projection, vx_range=vx_range, vy_range=vy_range)
            if runDir != None:
                fig.savefig(os.path.join(orbitDir, "projection_missed.png"))
        except:
            errFile.write("Orbit ID {:04d}: Error 7: Could not plot projection velocities missed.\n".format(orbit_id))


        # Grab the linked detections
        linked_detections_projection = grabLinkedDetections(projected_obs, allClusters_projection, clusterMembers_projection, columnMapping=columnMapping)
        summary_projection["num_linked_observations"] = len(linked_detections_projection)
        linked_detections = np.concatenate([linked_detections, linked_detections_projection])
        
        # Save linked detections
        if runDir != None:
            np.savetxt(os.path.join(orbitDir, "linked_detections.txt"), linked_detections_projection, fmt="%i")
            
        # Grab time to complete orbit processing
        time_end = time.time()
        duration = time_end - time_start
        summary_projection["time_seconds"] = duration
        
        # Arrange columns in projection summary dataframe
        summary_projection = summary_projection[[
            'orbit_id',
            'percent_completeness', 
            'percent_linkage_efficiency',
            'time_seconds',
            'num_unique_known_objects', 
            'num_unique_known_objects_findable',
            'num_unique_known_objects_found', 
            'num_unique_known_objects_missed',
            'num_known_object_observations', 
            'num_unknown_object_observations',
            'num_false_positive_observations', 
            'percent_known_object_observations',
            'percent_unknown_object_observations',
            'percent_false_positive_observations', 
            'num_known_object_pure_clusters',
            'num_known_object_partial_clusters', 
            'num_unknown_object_pure_clusters',
            'num_unknown_object_partial_clusters',
            'num_false_positive_pure_clusters',
            'num_false_positive_partial_clusters',
            'num_duplicate_visit_clusters',
            'num_false_clusters', 
            'num_total_clusters', 
            'num_linked_observations']]
        
        # Update tracking arrays
        found_known_objects.append(allObjects_projection[allObjects_projection["found"] == 1][columnMapping["name"]].values)
        allObjects.append(allObjects_projection)
        summaries.append(summary_projection)
        
        # Save summary dataframe
        if runDir != None:
            summary_projection.to_csv(os.path.join(orbitDir, "summary.txt"), sep=" ", index=False)
        
        if verbose == True:
            print("Finished orbit {}.".format(orbit_id))
            print("") 
            print("Total time in seconds: {}".format(duration))
            print("-------------------------")
            print("")
        
        # Plot found and missed orbits if known orbits are provided
        if type(knownOrbits) == pd.DataFrame:
            # Plot found orbits (linear semi-major axis)
            try:
                fig, ax = plotOrbitsFound(allObjects_projection, 
                                          knownOrbits, 
                                          testOrbits=orbit, 
                                          columnMapping=columnMapping)
                ax.set_xlim(0.0, 5.0)
                ax.text(_setPercentage(ax.get_xlim(), 0.02), 
                        _setPercentage(ax.get_ylim(), 0.93), 
                       "Found Objects: {}".format(len(allObjects_projection[allObjects_projection["found"] == 1])))

                if runDir != None:
                    fig.savefig(os.path.join(orbitDir, "known_orbits_found.png"))
            except:
                errFile.write("Orbit ID {:04d}: Error 8: Could not plot found orbits.\n".format(orbit_id))


            # Plot found orbits (log semi-major axis)
            try:
                fig, ax = plotOrbitsFound(allObjects_projection, 
                                          knownOrbits, 
                                          testOrbits=orbit, 
                                          columnMapping=columnMapping)
                ax.set_xscale("log")
                ax.text(_setPercentage(ax.get_xlim(), 0.0001), 
                        _setPercentage(ax.get_ylim(), 0.93), 
                       "Found Objects: {}".format(len(allObjects_projection[allObjects_projection["found"] == 1])))

                if runDir != None:
                    fig.savefig(os.path.join(orbitDir, "known_orbits_found_log.png"))
            except:
                errFile.write("Orbit ID {:04d}: Error 9: Could not plot found orbits (log).\n".format(orbit_id))

            # Plot missed orbits (linear semi-major axis)
            try:
                fig, ax = plotOrbitsMissed(allObjects_projection, 
                                           knownOrbits, 
                                           testOrbits=orbit, 
                                           columnMapping=columnMapping)
                ax.set_xlim(0.0, 5.0)
                ax.text(_setPercentage(ax.get_xlim(), 0.02), 
                        _setPercentage(ax.get_ylim(), 0.93), 
                       "Missed Objects: {}".format(len(allObjects_projection[(allObjects_projection["found"] == 0) 
                                                                         & (allObjects_projection["findable"] == 1)])))

                if runDir != None:
                    fig.savefig(os.path.join(orbitDir, "known_orbits_missed.png"))
            except:
                errFile.write("Orbit ID {:04d}: Error 10: Could not plot missed orbits.\n".format(orbit_id))

            # Plot missed orbits (log semi-major axis)
            try:
                fig, ax = plotOrbitsMissed(allObjects_projection, 
                                           knownOrbits, 
                                           testOrbits=orbit, 
                                           columnMapping=columnMapping)
                ax.set_xscale("log")
                ax.text(_setPercentage(ax.get_xlim(), 0.0001), 
                        _setPercentage(ax.get_ylim(), 0.93), 
                       "Missed Objects: {}".format(len(allObjects_projection[(allObjects_projection["found"] == 0) 
                                                                        & (allObjects_projection["findable"] == 1)])))

                if runDir != None:
                    fig.savefig(os.path.join(orbitDir, "known_orbits_missed_log.png"))
            except: 
                errFile.write("Orbit ID {:04d}: Error 11: Could not plot missed orbits (log).\n".format(orbit_id))

    # Concatenate the projection based dataframes
    allObjects = pd.concat(allObjects)
    allObjects.reset_index(inplace=True, drop=True)
    summaries_projection = pd.concat(summaries)
    summaries_projection.reset_index(inplace=True, drop=True)
    
    # Update the survey allObjects dataframe with the found known objects
    found_known_objects = np.concatenate(found_known_objects)
    found_known_objects = np.unique(found_known_objects)
    allObjects_survey.loc[allObjects_survey[columnMapping["name"]].isin(found_known_objects), "found"] = 1
    allObjects_survey.loc[allObjects_survey["found"] != 1, "found"] = 0
    
    # Add completeness to summary dataframe
    num_known_found = len(allObjects_survey[allObjects_survey["found"] == 1])
    num_known_missed =  len(allObjects_survey[(allObjects_survey["found"] == 0)
                                              & (allObjects_survey["findable"] == 1)])
    completeness =  100 * (num_known_found / len(allObjects_survey[allObjects_survey["findable"] == 1]))
    summary_survey["percent_completeness"] = completeness
    summary_survey["num_unique_known_objects_found"] = num_known_found
    summary_survey["num_unique_known_objects_missed"] = num_known_missed
    
    # Rearrange survey summary dataframe columns
    summary_survey = summary_survey[[
        'percent_completeness',
        'num_unique_known_objects', 
        'num_unique_known_objects_findable',
        'num_unique_known_objects_found',
        'num_unique_known_objects_missed',
        'num_known_object_observations', 
        'num_unknown_object_observations',
        'num_false_positive_observations', 
        'percent_known_object_observations',
        'percent_unknown_object_observations',
        'percent_false_positive_observations', 
    ]]
    
    # Save survey dataframes
    if runDir != None:
        allObjects_survey.to_csv(os.path.join(runDir, "allObjects_survey.txt"), sep=" ", index=False)
        summary_survey.to_csv(os.path.join(runDir, "summary_survey.txt"), sep=" ", index=False)
        summaries_projection.to_csv(os.path.join(runDir, "summary_orbits.txt"), sep=" ", index=False)
        np.savetxt(os.path.join(runDir, "linked_detections.txt"), linked_detections, fmt="%i")
    
    # Plot found and missed orbits if known orbits are provided
    if type(knownOrbits) == pd.DataFrame:
        
        # Plot found orbits (linear semi-major axis)
        try:
            fig, ax = plotOrbitsFound(allObjects_survey, 
                                      knownOrbits, 
                                      testOrbits=orbits, 
                                      columnMapping=columnMapping)
            ax.set_xlim(0.0, 5.0)
            ax.text(_setPercentage(ax.get_xlim(), 0.02), 
                    _setPercentage(ax.get_ylim(), 0.93), 
                   "Found Objects: {}".format(len(allObjects_survey[allObjects_survey["found"] == 1])))

            if runDir != None:
                fig.savefig(os.path.join(runDir, "known_orbits_found.png"))
        except:
            errFile.write("Survey: Error 12: Could not plot found orbits.\n")
    
        # Plot found orbits (log semi-major axis)
        try:
            fig, ax = plotOrbitsFound(allObjects_survey, 
                                      knownOrbits, 
                                      testOrbits=orbits, 
                                      columnMapping=columnMapping)
            ax.set_xscale("log")
            ax.text(_setPercentage(ax.get_xlim(), 0.0001), 
                    _setPercentage(ax.get_ylim(), 0.93), 
                   "Found Objects: {}".format(len(allObjects_survey[allObjects_survey["found"] == 1])))

            if runDir != None:
                fig.savefig(os.path.join(runDir, "known_orbits_found_log.png"))
        except:
            errFile.write("Survey: Error 13: Could not plot found orbits (log).\n")

        # Plot missed orbits (linear semi-major axis)
        try:
            fig, ax = plotOrbitsMissed(allObjects_survey, 
                                       knownOrbits, 
                                       testOrbits=orbits, 
                                       columnMapping=columnMapping)
            ax.set_xlim(0.0, 5.0)
            ax.text(_setPercentage(ax.get_xlim(), 0.02), 
                    _setPercentage(ax.get_ylim(), 0.93), 
                   "Missed Objects: {}".format(len(allObjects_survey[(allObjects_survey["found"] == 0) 
                                                                     & (allObjects_survey["findable"] == 1)])))

            if runDir != None:
                fig.savefig(os.path.join(runDir, "known_orbits_missed.png"))
        except:
            errFile.write("Survey: Error 14: Could not plot missed orbits.\n")
        
        # Plot missed orbits (log semi-major axis)
        try:
            fig, ax = plotOrbitsMissed(allObjects_survey, 
                                       knownOrbits, 
                                       testOrbits=orbits, 
                                       columnMapping=columnMapping)
            ax.set_xscale("log")
            ax.text(_setPercentage(ax.get_xlim(), 0.0001), 
                    _setPercentage(ax.get_ylim(), 0.93), 
                   "Missed Objects: {}".format(len(allObjects_survey[(allObjects_survey["found"] == 0) 
                                                                    & (allObjects_survey["findable"] == 1)])))

            if runDir != None:
                fig.savefig(os.path.join(runDir, "known_orbits_missed_log.png"))
        except:
            errFile.write("Survey: Error 15: Could not plot missed orbits (log).\n")
            
    if runDir != None:
        errFile.close()
        
    if verbose == True:
        print("THOR: runTHOR")
        print("-------------------------")
        print("Done. Finished running THOR with {} orbits.".format(len(orbits)))
        print("Completeness for known objects (%): {:1.3f}".format(completeness))
        print("")
        print("-------------------------")
            
    return allObjects_survey, summary_survey, summaries_projection
        