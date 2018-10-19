import os
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from sklearn.cluster import DBSCAN
from astropy import units as u
from astropy import constants as c

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
           "runRangeAndShiftOnVisit"]

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
    db = DBSCAN(eps=eps, min_samples=minSamples, n_jobs=1).fit(X)
    
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
    vyRange : {None, list or `~numpy.ndarray` (2)}
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
    saveFiles : {None, list}, optional
        List of paths to save DataFrames to ([[projected_obs, avg_object]) or None. 
        If useAverageObject is False, then the second path will not be used. If
        there is no averageObject, no files will be saved.
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