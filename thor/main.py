import os
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from sklearn.cluster import DBSCAN

from .config import Config
from .cell import Cell
from .particle import TestParticle
from .pyoorb import propagateTestParticle
from .data_processing import findAverageObject
from .data_processing import findExposureTimes
from .data_processing import buildCellForVisit
from .plotting import plotScatterContour

__all__ = ["rangeAndShift",
           "clusterVelocity",
           "_clusterVelocity",
           "clusterAndLink",
           "analyzeObservations",
           "analyzeProjections",
           "analyzeClusters",
           "runRangeAndShiftOnVisit",
           "runClusterAndLinkOnVisit"]

def rangeAndShift(observations,
                  cell, 
                  r, 
                  v,
                  numNights=14,
                  mjds="auto",
                  dMax=20.0,
                  includeEquatorialProjection=True,
                  saveFile=None,
                  verbose=True, 
                  columnMapping=Config.columnMapping):
    """
    Range and shift a cell.
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing observations.
    cell : `~thor.Cell`
        THOR cell. 
    r : float
        Heliocentric distance in AU.
    v : `~numpy.ndarray` (1, 3)
        Velocity vector in AU per day (ecliptic). 
    numNights : int, optional
        Number of nights from the first exposure to consider 
        for ranging and shifting. 
        [Default = 14]
    mjds : {'auto', `~numpy.ndarray` (N)}
        If mjds is 'auto', will propagate the particle to middle of each unique night in 
        obervations and look for all detections within an angular search radius (defined by a 
        maximum allowable angular speed) and extract the exposure time. Alternatively, an array
        of exposure times may be passed. 
    dMax : float, optional
        Maximum angular distance (in RA and Dec) permitted when searching for exposure times
        in degrees. 
        [Default = 20.0]
    includeEquatorialProjection : bool, optional
        Include naive shifting in equatorial coordinates without properly projecting
        to the plane of the orbit. This is useful if performance comparisons want to be made.
        [Default = True]
    saveFile : {None, str}, optional
        Path to save DataFrame to or None. 
        [Default = None]
    verbose : bool, optional
        Print progress statements? 
        [Default = True]
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~thor.Config.columnMapping`]
        
    Returns
    -------
    projected_obs : `~pandas.DataFrame`
        Observations dataframe (from cell.observations) with columns containing
        projected coordinates. 
    """
    # If initial doesn't have observations loaded,
    # get them
    if cell.observations is None:
        cell.getObservations()
        
    time_start = time.time()
    if verbose == True:
        print("THOR: rangeAndShift")
        print("-------------------------")
        print("Running range and shift...")
        print("Assuming r = {} AU".format(r))
        print("Assuming v = {} AU per day".format(v))
        
    x_e = cell.observations[[columnMapping["obs_x_au"], columnMapping["obs_y_au"], columnMapping["obs_z_au"]]].values[0]
   
    # Instantiate particle
    particle = TestParticle(cell.center, r, v, x_e, cell.mjd)
    
    # Prepare transformation matrices
    particle.prepare(verbose=verbose)
    
    if mjds == "auto":
        mjds = findExposureTimes(observations, particle.x_a, v, cell.mjd, numNights=numNights, dMax=dMax, verbose=verbose)
        
    # Apply tranformations to observations
    particle.apply(cell, verbose=verbose)
    
    # Add initial cell and particle to lists
    cells = [cell]
    particles = [particle]
    
    # Propagate test particle and generate ephemeris for all mjds
    if verbose == True:
        print("Propagating test particle...")
    ephemeris = propagateTestParticle(
        particle.elements,
        particle.mjd, 
        mjds,
        elementType="cartesian",
        mjdScale="UTC",
        H=10,
        G=0.15)
    if verbose == True:
        print("Done.")
        print("")
    
    if includeEquatorialProjection is True:
        cell.observations["theta_x_eq_deg"] = cell.observations[columnMapping["RA_deg"]] - particle.coords_eq_ang[0]
        cell.observations["theta_y_eq_deg"] = cell.observations[columnMapping["Dec_deg"]] - particle.coords_eq_ang[1]
    
    # Initialize final dataframe and add observations
    final_df = pd.DataFrame()
    final_df = pd.concat([cell.observations, final_df])
    
    if verbose == True:
        print("Reading ephemeris and gathering observations...")
        print("")
    for mjd_f in mjds:
        if verbose == True:
            print("Building particle and cell for {}".format(mjd_f))
        oldCell = cells[-1]
        oldParticle = particles[-1]
        
        # Get propagated particle
        propagated = ephemeris[ephemeris["mjd"] == mjd_f]
        
        # Get new equatorial coordinates
        new_coords_eq_ang = propagated[["RA_deg", "Dec_deg"]].values[0]
        
        # Get new barycentric distance
        new_r = propagated["r_au"].values[0]
        
        # Get new velocity in ecliptic cartesian coordinates
        new_v = propagated[["HEclObj_dX/dt_au_p_day",
                            "HEclObj_dY/dt_au_p_day",
                            "HEclObj_dZ/dt_au_p_day"]].values[0]
        
        # Get new location of observer
        new_x_e = propagated[["HEclObsy_X_au",
                              "HEclObsy_Y_au",
                              "HEclObsy_Z_au"]].values[0]
        
        # Get new mjd (same as mjd_f)
        new_mjd = mjd_f

        # Define new cell at new coordinates
        newCell = Cell(new_coords_eq_ang,
                       new_mjd,
                       area=oldCell.area,
                       shape=oldCell.shape,
                       dataframe=oldCell.dataframe)
        
        # Get the observations in that cell
        newCell.getObservations()
        
        # Define new particle at new coordinates
        newParticle = TestParticle(new_coords_eq_ang,
                                   new_r,
                                   new_v,
                                   new_x_e,
                                   new_mjd)
        
        # Prepare transformation matrices
        newParticle.prepare(verbose=verbose)
       
        # Apply tranformations to new observations
        newParticle.apply(newCell, verbose=verbose)
        
        if includeEquatorialProjection is True:
            newCell.observations["theta_x_eq_deg"] = newCell.observations[columnMapping["RA_deg"]] - newParticle.coords_eq_ang[0]
            newCell.observations["theta_y_eq_deg"] = newCell.observations[columnMapping["Dec_deg"]] - newParticle.coords_eq_ang[1]
        
        # Add observations to final dataframe
        final_df = pd.concat([newCell.observations, final_df])
    
        # Append new cell and particle to lists
        cells.append(newCell)
        particles.append(newParticle)
        
    final_df.sort_values(by=columnMapping["exp_mjd"], inplace=True)
    
    time_end = time.time()
    if verbose == True:
        print("Done. Final DataFrame has {} observations.".format(len(final_df)))
        print("Total time in seconds: {}".format(time_end - time_start))  
        
    if saveFile is not None:
        if verbose is True:
            print("Saving to {}".format(saveFile))
        final_df.to_csv(saveFile, sep=" ", index=False)
    
    if verbose == True:
        print("-------------------------")
        print("")
        
    return final_df

def clusterVelocity(obsIds,
                    x, 
                    y, 
                    dt, 
                    vx, 
                    vy, 
                    eps=0.005, 
                    minSamples=5):
    """
    Clusters THOR projection with different velocities
    in the projection plane using `~scipy.cluster.DBSCAN`.
    
    Parameters
    ----------
    obsIds : `~numpy.ndarray' (N)
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
    minSamples : int, optional
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
    db = DBSCAN(eps=eps, min_samples=minSamples).fit(X)
    
    clusters = db.labels_[np.where(db.labels_ != -1)[0]]
    cluster_ids = []
    
    if len(clusters) != 0:
        for cluster in np.unique(clusters):
            cluster_ids.append(obsIds[np.where(db.labels_ == cluster)[0]])
    else:
        cluster_ids = -1
    
    del db
    return cluster_ids
           
def _clusterVelocity(vx, vy,
                     obsIds=None,
                     x=None,
                     y=None,
                     dt=None,
                     eps=None,
                     minSamples=None):
    """
    Helper function to multiprocess clustering.
    
    """
    return clusterVelocity(obsIds,
                           x,
                           y,
                           dt,
                           vx,
                           vy,
                           eps=eps,
                           minSamples=minSamples) 

def clusterAndLink(observations, 
                   vxRange=[-0.1, 0.1], 
                   vyRange=[-0.1, 0.1],
                   vxBins=100, 
                   vyBins=100,
                   vxValues=None,
                   vyValues=None,
                   threads=12, 
                   eps=0.005, 
                   minSamples=5,
                   saveFiles=None,
                   verbose=True,
                   columnMapping=Config.columnMapping):
    """
    Cluster and link correctly projected (after ranging and shifting)
    detections.
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing post-range and shift observations.
    vxRange : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in x. 
        Will not be used if vxValues are specified. 
        [Default = [-0.1, 0.1]]
    vxRange : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in y. 
        Will not be used if vyValues are specified. 
        [Default = [-0.1, 0.1]]
    vxBins : int, optional
        Length of x-velocity grid between vxRange[0] 
        and vxRange[-1]. Will not be used if vxValues are 
        specified. 
        [Default = 100]
    vyBins: int, optional
        Length of y-velocity grid between vyRange[0] 
        and vyRange[-1]. Will not be used if vyValues are 
        specified. 
        [Default = 100]
    vxValues : {None, `~numpy.ndarray`}, optional
        Values of velocities in x at which to cluster
        and link. 
        [Default = None]
    vyValues : {None, `~numpy.ndarray`}, optional
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
    minSamples : int, optional
        The number of samples (or total weight) in a neighborhood for a 
        point to be considered as a core point. This includes the point itself.
        See: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html
        [Default = 5]
    saveFiles : {None, list}, optional
        List of paths to save DataFrames to ([allClusters, clusterMembers]) or None. 
        [Default = None]
    verbose : bool, optional
        Print progress statements? 
        [Default = True]
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~thor.Config.columnMapping`]
        
    Returns
    -------
    allClusters : `~pandas.DataFrame`
        DataFrame with the cluster ID, the number of observations, and the x and y velocity. 
    clusterMembers : `~pandas.DataFrame`
        DataFrame containing the cluster ID and the observation IDs of its members. 
    """ 
    # Extract useful quantities
    obs_ids = observations[columnMapping["obs_id"]].values
    theta_x = observations["theta_x_deg"].values
    theta_y = observations["theta_y_deg"].values
    mjd = observations[columnMapping["exp_mjd"]].values
    truth = observations[columnMapping["name"]].values

    # Select detections in first exposure
    first = np.where(mjd == mjd.min())[0]
    theta_x0 = theta_x[first]
    theta_y0 = theta_y[first]
    mjd0 = mjd[first][0]
    dt = mjd - mjd0

    # Grab remaining detections
    #remaining = np.where(mjd != mjd.min())[0]
    if vxValues is None and vxRange is not None:
        vx = np.linspace(*vxRange, num=vxBins)
    elif vxValues is None and vxRange is None:
        raise ValueError("Both vxValues and vxRange cannot be None.")
    else:
        vx = vxValues
        vxRange = [vxValues[0], vxValues[-1]]
        vxBins = len(vx)
     
    if vyValues is None and vyRange is not None:
        vy = np.linspace(*vyRange, num=vyBins)
    elif vyValues is None and vyRange is None:
        raise ValueError("Both vyValues and vyRange cannot be None.")
    else:
        vy = vyValues
        vyRange = [vyValues[0], vyValues[-1]]
        vyBins = len(vy)
        
    if vxValues is None and vyValues is None:
        vxx, vyy = np.meshgrid(vx, vy)    
        vxx = vxx.flatten()
        vyy = vyy.flatten()
    elif vxValues is not None and vyValues is not None:
        vxx = vx
        vyy = vy
    else:
        raise ValueError("")

    time_start_cluster = time.time()
    if verbose == True:
        print("THOR: clusterAndLink")
        print("-------------------------")
        print("Running velocity space clustering...")
        print("X velocity range: {}".format(vxRange))
        
        if vxValues is not None:
            print("X velocity values: {}".format(vxBins))
        else:
            print("X velocity bins: {}".format(vxBins))
            
        print("Y velocity range: {}".format(vyRange))
        if vyValues is not None:
            print("Y velocity values: {}".format(vyBins))
        else:
            print("Y velocity bins: {}".format(vyBins))
        if vxValues is not None:
            print("User defined x velocity values: True")
        else: 
            print("User defined x velocity values: False")
        if vyValues is not None:
            print("User defined y velocity values: True")
        else:
            print("User defined y velocity values: False")
            
        if vxValues is None and vyValues is None:
            print("Velocity grid size: {}".format(vxBins * vyBins))
        else: 
            print("Velocity grid size: {}".format(vxBins))
        print("Max sample distance: {}".format(eps))
        print("Minimum samples: {}".format(minSamples))

    
    possible_clusters = []
    if threads > 1:
        if verbose:
            print("Using {} threads...".format(threads))
        p = mp.Pool(threads)
        possible_clusters = p.starmap(partial(_clusterVelocity, 
                                              obsIds=obs_ids,
                                              x=theta_x,
                                              y=theta_y,
                                              dt=dt,
                                              eps=eps,
                                              minSamples=minSamples),
                                              zip(vxx.T, vyy.T))
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
                minSamples=minSamples)
            )
    time_end_cluster = time.time()    
    
    if verbose == True:
        print("Done. Completed in {} seconds.".format(time_end_cluster - time_start_cluster))
        print("")
        
    if verbose == True:
        print("Restructuring clusters...")
    
    # Clean up returned arrays and remove empty cases
    time_start_restr = time.time()
    populated_clusters = []
    populated_cluster_velocities = []
    for cluster, vxi, vyi in zip(possible_clusters, vxx, vyy):
        if type(cluster) == int:
            continue
        else:
            for c in cluster:
                populated_clusters.append(c)
                populated_cluster_velocities.append([vxi, vyi])
                
    if len(populated_clusters) == 0:
        time_end_restr = time.time()
        clusterMembers = pd.DataFrame(columns=["cluster_id", "obs_id"])
        allClusters = pd.DataFrame(columns=["cluster_id", "theta_vx", "theta_vy", "num_obs"])
        print("No clusters found.")
        if verbose == True:
            print("Total time in seconds: {}".format(time_end_restr - time_start_cluster))
        
        if saveFiles is not None:
            if verbose == True:
                print("Saving allClusters to {}".format(saveFiles[0]))
                allClusters.to_csv(saveFiles[0], sep=" ", index=False)
                print("Saving clusterMembers to {}".format(saveFiles[1]))
                clusterMembers.to_csv(saveFiles[1], sep=" ", index=False)
                
        if verbose == True:    
            print("-------------------------")
            print("")
    
        return allClusters, clusterMembers
                
    cluster_ids = np.arange(1, len(populated_clusters) + 1, dtype=int)
    num_members = np.zeros(len(populated_clusters), dtype=int)
    members_array = np.empty(0, dtype=int)
    id_array = np.empty(0, dtype=int)
    vs = np.array(populated_cluster_velocities)
    
    for i, (cluster_id, cluster) in enumerate(zip(cluster_ids, populated_clusters)):
        num_obs = len(cluster)
        num_members[i] = num_obs
        id_array_i = np.ones(num_obs, dtype=int) * cluster_id
        id_array = np.concatenate([id_array, id_array_i])
        members_array = np.concatenate([members_array, cluster])
        
    clusterMembers = pd.DataFrame({"cluster_id" : id_array, 
                                   "obs_id" : members_array})
    
    allClusters = pd.DataFrame({"cluster_id" : cluster_ids,
                                "theta_vx" : vs[:, 0],
                                "theta_vy" : vs[:, 1],
                                "num_obs" : num_members})
    
    time_end_restr = time.time()
    if verbose == True:
        print("Done. Completed in {} seconds.".format(time_end_restr - time_start_restr))
        print("")
        print("Found {} clusters.".format(len(allClusters)))
        print("Total time in seconds: {}".format(time_end_restr - time_start_cluster))
    
    if saveFiles is not None:
        if verbose == True:
            print("Saving allClusters to {}".format(saveFiles[0]))
            print("Saving clusterMembers to {}".format(saveFiles[1]))
        
        allClusters.to_csv(saveFiles[0], sep=" ", index=False)
        clusterMembers.to_csv(saveFiles[1], sep=" ", index=False)
                
    if verbose == True:    
        print("-------------------------")
        print("")
        
    return allClusters, clusterMembers

def analyzeObservations(observations,
                        minSamples=5, 
                        saveFiles=None,
                        verbose=True,
                        columnMapping=Config.columnMapping):
    """
    Count the number of objects that should be findable as a pure
    or partial cluster.
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing post-range and shift observations.
    minSamples : int, optional
        The number of samples (or total weight) in a neighborhood for a 
        point to be considered as a core point. This includes the point itself.
        See: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html
        [Default = 5]
    saveFiles : {None, list}, optional
        List of paths to save DataFrames to ([allClusters, clusterMembers, allObjects, summary]) or None. 
        [Default = None]
    verbose : bool, optional
        Print progress statements? 
        [Default = True]
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~thor.Config.columnMapping`]
    
    Returns
    -------
    allObjects : `~pandas.DataFrame`
        Object summary DataFrame.
    summary : `~pandas.DataFrame`
        Overall summary DataFrame. 
    """
    time_start = time.time()
    if verbose == True:
        print("THOR: analyzeObservations")
        print("-------------------------")
        print("Analyzing observations...")
    
    # Count number of noise detections, real object detections, the number of unique objects
    num_noise_obs = len(observations[observations[columnMapping["name"]] == "NS"])
    num_object_obs = len(observations[observations[columnMapping["name"]] != "NS"])
    unique_objects = observations[observations[columnMapping["name"]] != "NS"][columnMapping["name"]].unique()
    num_unique_objects = len(unique_objects)
    num_obs_per_object = observations[observations[columnMapping["name"]] != "NS"][columnMapping["name"]].value_counts().values
    objects_num_obs_descending = observations[observations[columnMapping["name"]] != "NS"][columnMapping["name"]].value_counts().index.values
    findable = objects_num_obs_descending[np.where(num_obs_per_object >= minSamples)[0]]
    
    # Populate allObjects DataFrame
    allObjects = pd.DataFrame(columns=[
        columnMapping["name"], 
        "num_obs", 
        "findable",
        "found"])
    
    allObjects[columnMapping["name"]] = objects_num_obs_descending
    allObjects["num_obs"] = num_obs_per_object
    allObjects.loc[allObjects[columnMapping["name"]].isin(findable), "findable"] = 1
    allObjects.loc[allObjects["findable"] != 1, ["findable"]] = 0
    num_findable = len(allObjects[allObjects["findable"] == 1])
    
    # Prepare summary DataFrame
    summary = pd.DataFrame({
        "num_observations_object": num_object_obs,
        "num_observations_noise": num_noise_obs,
        "obs_contamination" : num_noise_obs / len(observations) * 100,
        "unique_objects" : num_unique_objects,
        "unique_objects_findable" : num_findable}, index=[0]) 
    
    time_end = time.time()
    if verbose == True:
        print("Object observations: {}".format(num_object_obs))
        print("Noise observations: {}".format(num_noise_obs))
        print("Observation contamination (%): {}".format(num_noise_obs / len(observations) * 100))
        print("Unique objects: {}".format(num_unique_objects))
        print("Unique objects with at least {} detections: {}".format(minSamples, num_findable))
        print("") 
        print("Total time in seconds: {}".format(time_end - time_start))
        print("-------------------------")
        print("")
        
    return allObjects, summary

def analyzeProjections(observations,
                       minSamples=5, 
                       saveFiles=None,
                       verbose=True,
                       columnMapping=Config.columnMapping):
    """
    Count the number of objects that should be findable as a pure
    or partial cluster.
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing post-range and shift observations.
    minSamples : int, optional
        The number of samples (or total weight) in a neighborhood for a 
        point to be considered as a core point. This includes the point itself.
        See: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html
        [Default = 5]
    saveFiles : {None, list}, optional
        List of paths to save DataFrames to ([allClusters, clusterMembers, allObjects, summary]) or None. 
        [Default = None]
    verbose : bool, optional
        Print progress statements? 
        [Default = True]
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~thor.Config.columnMapping`]
    
    Returns
    -------
    allObjects : `~pandas.DataFrame`
        Object summary DataFrame.
    summary : `~pandas.DataFrame`
        Overall summary DataFrame. 
    """
    time_start = time.time()
    if verbose == True:
        print("THOR: analyzeObservations")
        print("-------------------------")
        print("Analyzing projections...")
    
    # Count number of noise detections, real object detections, the number of unique objects
    num_noise_obs = len(observations[observations[columnMapping["name"]] == "NS"])
    num_object_obs = len(observations[observations[columnMapping["name"]] != "NS"])
    unique_objects = observations[observations[columnMapping["name"]] != "NS"][columnMapping["name"]].unique()
    num_unique_objects = len(unique_objects)
    num_obs_per_object = observations[observations[columnMapping["name"]] != "NS"][columnMapping["name"]].value_counts().values
    objects_num_obs_descending = observations[observations[columnMapping["name"]] != "NS"][columnMapping["name"]].value_counts().index.values
    findable = objects_num_obs_descending[np.where(num_obs_per_object >= minSamples)[0]]
    
    # Populate allObjects DataFrame
    allObjects = pd.DataFrame(columns=[
        columnMapping["name"], 
        "num_obs", 
        "findable",
        "found_pure", 
        "found_partial",
        "found",
        "dtheta_x/dt_median",
        "dtheta_y/dt_median",
        "dtheta_x/dt_sigma",
        "dtheta_y/dt_sigma",
        "r_au_median",
        "Delta_au_median",
        "r_au_sigma",
        "Delta_au_sigma"])
    
    allObjects[columnMapping["name"]] = objects_num_obs_descending
    allObjects["num_obs"] = num_obs_per_object
    allObjects.loc[allObjects[columnMapping["name"]].isin(findable), "findable"] = 1
    num_findable = len(allObjects[allObjects["findable"] == 1])

    for obj in findable:
        dets = observations[observations[columnMapping["name"]].isin([obj])]
        dt = dets[columnMapping["exp_mjd"]].values[1:] - dets[columnMapping["exp_mjd"]].values[0]
        dx = dets["theta_x_deg"].values[1:] - dets["theta_x_deg"].values[0]
        dy = dets["theta_y_deg"].values[1:] - dets["theta_y_deg"].values[0]
        allObjects.loc[allObjects[columnMapping["name"]].isin([obj]), ["dtheta_x/dt_median"]] = np.median(dx/dt)
        allObjects.loc[allObjects[columnMapping["name"]].isin([obj]), ["dtheta_y/dt_median"]] = np.median(dy/dt)
        allObjects.loc[allObjects[columnMapping["name"]].isin([obj]), ["Delta_au_median"]] = np.median(dets[columnMapping["Delta_au"]].values)
        allObjects.loc[allObjects[columnMapping["name"]].isin([obj]), ["r_au_median"]] = np.median(dets[columnMapping["r_au"]].values)
        allObjects.loc[allObjects[columnMapping["name"]].isin([obj]), ["dtheta_x/dt_sigma"]] = np.std(dx/dt)
        allObjects.loc[allObjects[columnMapping["name"]].isin([obj]), ["dtheta_y/dt_sigma"]] = np.std(dy/dt)
        allObjects.loc[allObjects[columnMapping["name"]].isin([obj]), ["r_au_sigma"]] = np.std(dets[columnMapping["r_au"]].values)
        allObjects.loc[allObjects[columnMapping["name"]].isin([obj]), ["Delta_au_sigma"]] = np.std(dets[columnMapping["Delta_au"]].values)

    allObjects.loc[allObjects["findable"] != 1, ["findable"]] = 0
    
    # Prepare summary DataFrame
    summary = pd.DataFrame({
        "num_observations_object": num_object_obs,
        "num_observations_noise": num_noise_obs,
        "obs_contamination" : num_noise_obs / len(observations) * 100,
        "unique_objects" : num_unique_objects,
        "unique_objects_findable" : num_findable}, index=[0]) 
    
    time_end = time.time()
    if verbose == True:
        print("Object observations: {}".format(num_object_obs))
        print("Noise observations: {}".format(num_noise_obs))
        print("Observation contamination (%): {}".format(num_noise_obs / len(observations) * 100))
        print("Unique objects: {}".format(num_unique_objects))
        print("Unique objects with at least {} detections: {}".format(minSamples, num_findable))
        print("") 
        print("Total time in seconds: {}".format(time_end - time_start))
        print("-------------------------")
        print("")
        
    return allObjects, summary

    
def analyzeClusters(observations,
                    allClusters, 
                    clusterMembers,
                    allObjects, 
                    summary,
                    minSamples=5, 
                    contaminationThreshold=0.2, 
                    saveFiles=None,
                    verbose=True,
                    columnMapping=Config.columnMapping):
    """
    Analyze clusters.
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing post-range and shift observations.
    allClusters : `~pandas.DataFrame`
        DataFrame with the cluster ID, the number of observations, and the x and y velocity. 
    clusterMembers : `~pandas.DataFrame`
        DataFrame containing the cluster ID and the observation IDs of its members. 
    allObjects : `~pandas.DataFrame`
        Object summary DataFrame.
    summary : `~pandas.DataFrame`
        Overall summary DataFrame.
    minSamples : int, optional
        The number of samples (or total weight) in a neighborhood for a 
        point to be considered as a core point. This includes the point itself.
        See: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html
        [Default = 5]
    contaminationThreshold : float, optional
        Percentage (expressed between 0 and 1) of imposter observations in a cluster permitted for the 
        object to be found. 
        [Default = 0.8]
    saveFiles : {None, list}, optional
        List of paths to save DataFrames to ([allClusters, clusterMembers, allObjects, summary]) or None. 
        [Default = None]
    verbose : bool, optional
        Print progress statements? 
        [Default = True]
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~thor.Config.columnMapping`]
        
    Returns
    -------
    allClusters : `~pandas.DataFrame`
        DataFrame with the cluster ID, the number of observations, and the x and y velocity. 
    clusterMembers : `~pandas.DataFrame`
        DataFrame containing the cluster ID and the observation IDs of its members. 
    allObjects : `~pandas.DataFrame`
        Object summary DataFrame.
    summary : `~pandas.DataFrame`
        Overall summary DataFrame.
    """ 

    time_start = time.time()
    if verbose == True:
        print("THOR: analyzeClusters")
        print("-------------------------")
        print("Analyzing clusters...")

    # Add columns to classify pure, contaminated and false clusters
    # Add numbers to classify the number of members, and the linked object 
    # in the contaminated and pure case
    allClusters["pure"] = np.zeros(len(allClusters), dtype=int)
    allClusters["partial"] = np.zeros(len(allClusters), dtype=int)
    allClusters["false"] = np.zeros(len(allClusters), dtype=int)
    allClusters["num_members"] = np.ones(len(allClusters), dtype=int) * np.NaN
    allClusters["num_visits"] = np.ones(len(allClusters), dtype=int) * np.NaN
    allClusters["num_fields"] = np.ones(len(allClusters), dtype=int) * np.NaN
    allClusters["linked_object"] = np.ones(len(allClusters), dtype=int) * np.NaN

    # Count number of members per cluster
    observations_temp = observations.rename(columns={columnMapping["obs_id"]: "obs_id"})
    cluster_designation = observations_temp[["obs_id", columnMapping["name"]]].merge(clusterMembers, on="obs_id")
    cluster_designation.drop(columns="obs_id", inplace=True)
    cluster_designation.drop_duplicates(inplace=True)
    unique_ids_per_cluster = cluster_designation["cluster_id"].value_counts()
    allClusters["num_members"] = unique_ids_per_cluster.sort_index().values

    # Count number of visits per cluster
    cluster_visit = observations_temp[["obs_id", columnMapping["visit_id"]]].merge(clusterMembers, on="obs_id")
    cluster_visit.drop(columns="obs_id", inplace=True)
    cluster_visit.drop_duplicates(inplace=True)
    unique_visits_per_cluster = cluster_visit["cluster_id"].value_counts()
    allClusters["num_visits"] = unique_visits_per_cluster.sort_index().values

    # Count number of fields per cluster
    cluster_fields = observations_temp[["obs_id", columnMapping["field_id"]]].merge(clusterMembers, on="obs_id")
    cluster_fields.drop(columns="obs_id", inplace=True)
    cluster_fields.drop_duplicates(inplace=True)
    unique_fields_per_cluster = cluster_fields["cluster_id"].value_counts()
    allClusters["num_fields"] = unique_fields_per_cluster.sort_index().values

    # Isolate pure clusters
    single_member_clusters = cluster_designation[cluster_designation["cluster_id"].isin(allClusters[allClusters["num_members"] == 1]["cluster_id"])]
    allClusters.loc[allClusters["cluster_id"].isin(single_member_clusters["cluster_id"]), "linked_object"] = single_member_clusters[columnMapping["name"]].values
    allClusters.loc[(allClusters["linked_object"] != "NS") & (allClusters["linked_object"].notna()), "pure"] = 1

    # Grab all clusters that are not pure, calculate contamination and see if we can accept them
    cluster_designation = observations_temp[["obs_id", columnMapping["name"]]].merge(
        clusterMembers[~clusterMembers["cluster_id"].isin(allClusters[allClusters["pure"] == 1]["cluster_id"].values)], on="obs_id")
    cluster_designation.drop(columns="obs_id", inplace=True)
    cluster_designation = cluster_designation[["cluster_id", "designation"]].groupby(cluster_designation[["cluster_id", "designation"]].columns.tolist(), as_index=False).size()
    cluster_designation = cluster_designation.reset_index()
    cluster_designation.rename(columns={0: "num_obs"}, inplace=True)
    cluster_designation.sort_values(by=["cluster_id", "num_obs"], inplace=True)
    # Remove duplicate rows: keep row with the object with the mot detections in a cluster
    cluster_designation.drop_duplicates(subset=["cluster_id"], inplace=True, keep="last")
    cluster_designation = cluster_designation.merge(allClusters[["cluster_id", "num_obs"]], on="cluster_id")
    cluster_designation.rename(columns={"num_obs_x": "num_obs", "num_obs_y": "total_num_obs"}, inplace=True)
    cluster_designation["contamination"] = (1 - cluster_designation["num_obs"] / cluster_designation["total_num_obs"])
    partial_clusters = cluster_designation[(cluster_designation["num_obs"] >= minSamples) 
                                            & (cluster_designation["contamination"] <= contaminationThreshold)
                                            & (cluster_designation[columnMapping["name"]] != "NS")]

    allClusters.loc[allClusters["cluster_id"].isin(partial_clusters["cluster_id"]), "linked_object"] = partial_clusters[columnMapping["name"]]
    allClusters.loc[allClusters["cluster_id"].isin(partial_clusters["cluster_id"]), "partial"] = 1
    allClusters.loc[(allClusters["pure"] != 1) & (allClusters["partial"] != 1), "false"] = 1

    # Update allObjects DataFrame
    allObjects.loc[allObjects[columnMapping["name"]].isin(allClusters[allClusters["pure"] == 1]["linked_object"]), "found_pure"] = 1
    allObjects.loc[allObjects[columnMapping["name"]].isin(allClusters[allClusters["partial"] == 1]["linked_object"]), "found_partial"] = 1
    allObjects.loc[(allObjects["found_pure"] == 1) | (allObjects["found_partial"] == 1), "found"] = 1
    allObjects.fillna(value=0, inplace=True)

    num_pure = len(allClusters[allClusters["pure"] == 1])
    num_partial = len(allClusters[allClusters["partial"] == 1])
    num_false = len(allClusters[allClusters["false"] == 1])
    num_total = num_pure + num_partial + num_false
    num_duplicate_visits = len(allClusters[allClusters["num_obs"] != allClusters["num_visits"]])

    if verbose == True:
        print("Pure clusters: {}".format(num_pure))
        print("Partial clusters: {}".format(num_partial))
        print("Duplicate visit clusters: {}".format(num_duplicate_visits))
        print("False clusters: {}".format(num_false))
        print("Total clusters: {}".format(num_total))
        print("Cluster contamination (%): {}".format(num_false / num_total * 100))
    
    found = allObjects[allObjects["found"] == 1]
    missed = allObjects[(allObjects["found"] == 0) & (allObjects["findable"] == 1)]
    found_pure = len(allObjects[allObjects["found_pure"] == 1])
    found_partial = len(allObjects[allObjects["found_partial"] == 1])
    time_end = time.time()
    completeness = len(found) / (len(found) + len(missed)) * 100
    
    if verbose == True:
        print("Unique linked objects: {}".format(len(found)))
        print("Unique missed objects: {}".format(len(missed)))
        print("Completeness (%): {}".format(completeness))
        print("Done.")
        print("Total time in seconds: {}".format(time_end - time_start))
        
    summary["unique_objects_found_pure"] = found_pure
    summary["unique_objects_found_partial"] = found_partial
    summary["unique_objects_found"] = len(found)
    summary["unique_objects_missed"] = len(missed)
    summary["completeness"] = completeness
    summary["pure_clusters"] = num_pure
    summary["partial_clusters"] : num_partial
    summary["duplicate_visit_clusters"] = num_duplicate_visits
    summary["false_clusters"] = num_false
    summary["total_clusters"] = num_total
    summary["cluster_contamination"] = num_false / num_total * 100
        
    if saveFiles is not None:
        if verbose == True:
            print("Saving allClusters to {}".format(saveFiles[0]))
            print("Saving clusterMembers to {}".format(saveFiles[1]))
            print("Saving allObjects to {}".format(saveFiles[2]))
            print("Saving summary to {}".format(saveFiles[3]))
            
        allClusters.to_csv(saveFiles[0], sep=" ", index=False)
        clusterMembers.to_csv(saveFiles[1], sep=" ", index=False) 
        allObjects.to_csv(saveFiles[2], sep=" ", index=False) 
        summary.to_csv(saveFiles[3], sep=" ", index=False) 
        
    if verbose == True:    
        print("-------------------------")
        print("")

    return allClusters, clusterMembers, allObjects, summary

def runRangeAndShiftOnVisit(observations,
                            visitId,
                            r, 
                            v,
                            numNights=14, 
                            useAverageObject=True,
                            searchArea=0.5, 
                            searchShape="square",
                            cellArea=10, 
                            cellShape="square",
                            dMax=20.0,
                            saveFiles=None,
                            verbose=True,
                            columnMapping=Config.columnMapping):
    """
    Run range and shift on a visit. 
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing observations.
    visitId : int
        Visit ID. 
    r : float
        Heliocentric distance in AU.
    v : `~numpy.ndarray` (1, 3)
        Velocity vector in AU per day (ecliptic). 
    numNights : int, optional
        Number of nights from the time of the visit to consider 
        for ranging and shifting. 
        [Default = 14]
    useAverageObject : bool, optional
        Find an object in the original visit that represents
        the average and use that object's orbit. Ignores given 
        r and v. 
        [Default = False]
    searchArea : float, optional
        Area of THOR cell used to find average object in
        degrees squared.
        [Default = 0.5]
    searchShape : {'square', 'circle'}, optional
        Shape of the search cell. 
        [Default = 'square']
    cellArea : float, optional
        Area of THOR cell. Should be the same size as the visit. 
        [Default = 10]
    cellShape : {'square', 'circle'}, optional
        Shape of THOR cell. Should be the same shape as the visit.
        [Default = 'square']
    dMax : float, optional
        Maximum angular distance (in RA and Dec) permitted when searching for exposure times
        in degrees. 
        [Default = 20.0]
    verbose : bool, optional
        Print progress statements? 
        [Default = True]
    saveFiles : {None, list}, optional
        List of paths to save DataFrames to ([[projected_obs, avg_object]) or None. 
        If useAverageObject is False, then the second path will not be used. If
        there is no averageObject, no files will be saved.
        [Default = None]
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~thor.Config.columnMapping`]
        
    Returns
    -------
    projected_obs : `~pandas.DataFrame`
        Observations dataframe (from cell.observations) with columns containing
        projected coordinates. 
    avg_obj : {`~pandas.DataFrame`, int, None}
        If useAverageObject is True, will return a slice into the observations dataframe
        with the object's corresponding observation. If there are no real objects, will instead
        return -1. If useAverageObject is False, returns None. 
    """
    
    if verbose == True:
        print("THOR: runRangeAndShiftOnVisit")
        print("-------------------------")
        print("Running Thor on visit {}...".format(visitId))
        if useAverageObject != True:
            print("Assuming orbit with r = {}".format(r))
            print("Assuming orbit with v = {}".format(v))
        else:
            print("Search cell area: {} ".format(searchArea))
            print("Search cell shape: {} ".format(searchShape))
    
        print("Cell area: {} ".format(cellArea))
        print("Cell shape: {} ".format(cellShape))
        print("")

    small_cell = buildCellForVisit(observations, visitId, area=searchArea, shape=searchShape)
    small_cell.getObservations()
    if useAverageObject is True:
        avg_obj = findAverageObject(small_cell.observations)
        if avg_obj == -1:
            print("Can't run RaSCaLS on this visit.")
            print("Provide an orbit to run.")
            return avg_obj

        obj = small_cell.observations[small_cell.observations[columnMapping["name"]] == avg_obj]
        r = obj[columnMapping["r_au"]].values[0]
        v = obj[[columnMapping["obj_dx/dt_au_p_day"],
                 columnMapping["obj_dy/dt_au_p_day"],
                 columnMapping["obj_dz/dt_au_p_day"]]].values[0]
        

    cell = Cell(small_cell.center, small_cell.mjd, observations, area=cellArea, shape=cellShape)
    projected_obs = rangeAndShift(observations, 
                                  cell, 
                                  r, 
                                  v, 
                                  mjds="auto", 
                                  dMax=dMax, 
                                  numNights=numNights,
                                  verbose=verbose)
    
    if saveFiles is not None:
        if useAverageObject is True and avg_obj != -1:
            obj.to_csv(saveFiles[1], sep=" ", index=False)
        projected_obs.to_csv(saveFiles[0], sep=" ", index=False)
    
    if useAverageObject is True:
        return projected_obs, obj
    else:
        return projected_obs
    
def runClusterAndLinkOnVisit(observations, 
                             visitId,
                             orbits, 
                             avgObject, 
                             vxRange=[-0.1, 0.1], 
                             vyRange=[-0.1, 0.1],
                             vxBins=100, 
                             vyBins=100,
                             vxValues=None,
                             vyValues=None,
                             threads=12, 
                             eps=0.005, 
                             minSamples=5,
                             partialThreshold=0.8, 
                             saveDir=None,
                             verbose=True,
                             columnMapping=Config.columnMapping):
    """
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing post-range and shift observations.
    visitId : int
        Visit ID. 
    orbits : `~pandas.DataFrame`
        Orbit catalog.
    avgObject : `~pandas.DataFrame`
        A slice into the observations dataframe
        with the object's corresponding observation.
    vxRange : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in x. 
        Will not be used if vxValues are specified. 
        [Default = [-0.1, 0.1]]
    vxRange : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in y. 
        Will not be used if vyValues are specified. 
        [Default = [-0.1, 0.1]]
    vxBins : int, optional
        Length of x-velocity grid between vxRange[0] 
        and vxRange[-1]. Will not be used if vxValues are 
        specified. 
        [Default = 100]
    vyBins: int, optional
        Length of y-velocity grid between vyRange[0] 
        and vyRange[-1]. Will not be used if vyValues are 
        specified. 
        [Default = 100]
    vxValues : {None, `~numpy.ndarray`}, optional
        Values of velocities in x at which to cluster
        and link. 
        [Default = None]
    vyValues : {None, `~numpy.ndarray`}, optional
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
    minSamples : int, optional
        The number of samples (or total weight) in a neighborhood for a 
        point to be considered as a core point. This includes the point itself.
        See: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html
        [Default = 5]
    partialThreshold : float, optional
        Percentage (expressed between 0 and 1) of observations in a cluster required for the 
        object to be found. 
        [Default = 0.8]
    saveDir : {None, str}, optional
        Directory where to save outputs inluding plots. Will create
        a sub-directory inside directory. 
        [Default = None]
    verbose : bool, optional
        Print progress statements? 
        [Default = True]
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~thor.Config.columnMapping`]
    
    """
    avg_obj = avgObject[columnMapping["name"]].values[0]
    o = orbits[orbits[columnMapping["name"]] == avg_obj]
    
    if saveDir is not None:
        baseName = os.path.join(saveDir, "visitId{}_{}".format(visitId, avg_obj))
        os.makedirs(baseName)
        saveFiles1=[os.path.join(baseName, "allClusters_pre.txt"),
                   os.path.join(baseName, "clusterMembers_pre.txt")]
        saveFiles2=[os.path.join(baseName, "allClusters_post.txt"),
                   os.path.join(baseName, "clusterMembers_post.txt"),
                   os.path.join(baseName, "allObjects.txt"),
                   os.path.join(baseName, "summary.txt")]
    else:
        saveFiles1 = None
        saveFiles2 = None
    
    allClusters, clusterMembers = clusterAndLink(
        observations, 
        eps=eps, 
        minSamples=minSamples, 
        vxRange=vxRange, 
        vyRange=vyRange, 
        vxBins=vxBins, 
        vyBins=vyBins, 
        saveFiles=saveFiles1,
        vxValues=vxValues,
        vyValues=vyValues,
        threads=threads,
        verbose=verbose,
        columnMapping=columnMapping)
    
    allClusters, clusterMember, allObjects, summary = analyzeClusters(
        observations, 
        allClusters, 
        clusterMembers, 
        partialThreshold=partialThreshold, 
        minSamples=minSamples,
        saveFiles=saveFiles2,
        verbose=verbose,
        columnMapping=columnMapping)
    
    found = orbits[orbits[columnMapping["name"]].isin(allObjects[allObjects["found"] == 1][columnMapping["name"]])]
    missed = orbits[orbits[columnMapping["name"]].isin((allObjects[(allObjects["found"] == 0) & (allObjects["findable"] == 1)][columnMapping["name"]]))]
    
    fig, ax = plotScatterContour(missed, 
                                 columnMapping["a_au"],
                                 columnMapping["i_deg"],
                                 columnMapping["e"],
                                 plotCounts=False, 
                                 logCounts=True, 
                                 countLevels=4, 
                                 mask=None,
                                 xLabel="a [AU]",
                                 yLabel="i [deg]",
                                 zLabel="e",
                                 scatterKwargs={"s": 1, "vmin": 0, "vmax": 1})
    ax.text(ax.get_xlim()[-1] - 0.40 * (ax.get_xlim()[1] - ax.get_xlim()[0]), ax.get_ylim()[1] - 0.05 * ax.get_ylim()[1], "Missed objects: {}".format(len(missed)))
    ax.scatter(o["a_au"].values, o["i_deg"].values, c="r", s=20, marker="+")
    ax.set_title("Missed Orbits\nVisit: {}, Object: {}".format(visitId, avg_obj))
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if saveDir is not None:
        fig.savefig(os.path.join(baseName, "missed_aie.png"))
    
    fig, ax = plotScatterContour(found, 
                                 columnMapping["a_au"],
                                 columnMapping["i_deg"],
                                 columnMapping["e"],
                                 plotCounts=False, 
                                 logCounts=True, 
                                 countLevels=4, 
                                 mask=None,
                                 xLabel="a [AU]",
                                 yLabel="i [deg]",
                                 zLabel="e",
                                 scatterKwargs={"s": 1, "vmin": 0, "vmax": 1})        
    ax.scatter(o["a_au"].values, o["i_deg"].values, c="r", s=20, marker="+")
    ax.set_title("Recovered Orbits\nVisit: {}, Object: {}".format(visitId, avg_obj))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.text(ax.get_xlim()[-1] - 0.40 * (ax.get_xlim()[1] - ax.get_xlim()[0]), ax.get_ylim()[1] - 0.05 * ax.get_ylim()[1], "Found objects: {}".format(len(found)))
    if saveDir is not None:
        fig.savefig(os.path.join(baseName, "found_aie.png"))
    
    found_obs = observations[observations[columnMapping["name"]].isin(allObjects[allObjects["found"] == 1][columnMapping["name"]])]
    missed_obs = observations[observations[columnMapping["name"]].isin((allObjects[(allObjects["found"] == 0) & (allObjects["findable"] == 1)][columnMapping["name"]]))]
    
    fig, ax = plotScatterContour(missed_obs, 
                                 columnMapping["obj_dx/dt_au_p_day"], 
                                 columnMapping["obj_dy/dt_au_p_day"], 
                                 columnMapping["obj_dz/dt_au_p_day"],
                                 countLevels=4, 
                                 xLabel="dx/dt [AU per day]",
                                 yLabel="dy/dt [AU per day]",
                                 zLabel="dz/dt [AU per day]")
   
    ax.text(ax.get_xlim()[-1] - 0.40 * (ax.get_xlim()[1] - ax.get_xlim()[0]), ax.get_ylim()[1] - 0.05 * ax.get_ylim()[1], "Missed objects: {}".format(len(missed)))
    ax.scatter(*avgObject[[columnMapping["obj_dx/dt_au_p_day"], columnMapping["obj_dy/dt_au_p_day"]]].values.T, c="r", s=1, marker="+")
    ax.set_title("Missed Orbits\nVisit: {}, Object: {}".format(visitId, avg_obj))
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if saveDir is not None:
        fig.savefig(os.path.join(baseName, "missed_orbits.png"))
    
    fig, ax = plotScatterContour(found_obs, 
                                 columnMapping["obj_dx/dt_au_p_day"], 
                                 columnMapping["obj_dy/dt_au_p_day"], 
                                 columnMapping["obj_dz/dt_au_p_day"],
                                 countLevels=4, 
                                 xLabel="dx/dt [AU per day]",
                                 yLabel="dy/dt [AU per day]",
                                 zLabel="dz/dt [AU per day]")
   
    ax.scatter(*avgObject[[columnMapping["obj_dx/dt_au_p_day"], columnMapping["obj_dz/dt_au_p_day"]]].values.T, c="r", s=1, marker="+")
    ax.set_title("Found Orbits\nVisit: {}, Object: {}".format(visitId, avg_obj))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.text(ax.get_xlim()[-1] - 0.40 * (ax.get_xlim()[1] - ax.get_xlim()[0]), ax.get_ylim()[1] - 0.05 * ax.get_ylim()[1], "Found objects: {}".format(len(missed)))

    if saveDir is not None:
        fig.savefig(os.path.join(baseName, "found_orbits.png"))
    return
