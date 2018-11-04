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
from .data_processing import findExposureTimes
from .data_processing import grabLinkedDetections
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
           "runTHOR"]

def rangeAndShift(observations,
                  cell, 
                  r, 
                  v,
                  numNights=14,
                  mjds="auto",
                  dMax=20.0,
                  includeEquatorialProjection=True,
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
        cell.getObservations(columnMapping=columnMapping)
        
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
        mjds = findExposureTimes(observations, particle.x_a, v, cell.mjd, numNights=numNights, dMax=dMax, columnMapping=columnMapping, verbose=verbose)
        
    # Apply tranformations to observations
    particle.apply(cell, columnMapping=columnMapping, verbose=verbose)
    
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
        newCell.getObservations(columnMapping=columnMapping)
        
        # Define new particle at new coordinates
        newParticle = TestParticle(new_coords_eq_ang,
                                   new_r,
                                   new_v,
                                   new_x_e,
                                   new_mjd)
        
        # Prepare transformation matrices
        newParticle.prepare(verbose=verbose)
       
        # Apply tranformations to new observations
        newParticle.apply(newCell, verbose=verbose, columnMapping=columnMapping)
        
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
    #os.nice(3)
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
        clusterMembers = pd.DataFrame(columns=["cluster_id", columnMapping["obs_id"]])
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
                                   columnMapping["obs_id"] : members_array})
    
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
        print("-------------------------")
        print("")
        
    return allClusters, clusterMembers


def runTHOR(observations,
            orbits,
            knownOrbits=None,
            runDir=None,
            cellArea=10, 
            cellShape="circle",
            numNights=14,
            mjds="auto",
            dMax=20.0,
            includeEquatorialProjection=True,
            vxRange=[-0.1, 0.1], 
            vyRange=[-0.1, 0.1],
            vxBins=100, 
            vyBins=100,
            vxValues=None,
            vyValues=None,
            threads=30, 
            eps=0.005, 
            minSamples=5,
            contaminationThreshold=0.2,
            unknownIDs=Config.unknownIDs,
            falsePositiveIDs=Config.falsePositiveIDs,
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
    cellArea : float, optional
        Cell's area in units of square degrees. 
        [Default = 10]
    cellShape : {'square', 'circle'}, optional
        Cell's shape can be square or circle. Combined with the area parameter, will set the search 
        area when looking for observations contained within the defined cell. 
        [Default = 'square']
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
    
    # Analyze observations for entire survey
    allObjects_survey, summary_survey = analyzeObservations(
        observations,
        minSamples=minSamples, 
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
        fig, ax = plotOrbitsFindable(allObjects_survey, knownOrbits)
        ax.set_xlim(0.0, 5.0)
        ax.text(_setPercentage(ax.get_xlim(), 0.02), 
                _setPercentage(ax.get_ylim(), 0.93), 
               "Findable Objects: {}".format(len(allObjects_survey[allObjects_survey["findable"] == 1])))
        
        # If test orbits exist in known orbits, plot them
        test_orbits_in_known = knownOrbits[knownOrbits[columnMapping["name"]].isin(orbits[columnMapping["name"]].values)]
        if len(test_orbits_in_known) != 0:
            ax.scatter(*test_orbits_in_known[[columnMapping["a_au"], columnMapping["i_deg"]]].values.T, 
                       s=2, 
                       c="r",
                       label="Test Orbits")
            ax.legend(loc='upper right')
        
        if runDir != None:
            fig.savefig(os.path.join(runDir, "known_orbits_findable.png"))
        
        # Plot findable orbits (log semi-major axis)
        fig, ax = plotOrbitsFindable(allObjects_survey, knownOrbits)
        ax.set_xscale("log")
        ax.text(_setPercentage(ax.get_xlim(), 0.001), 
                _setPercentage(ax.get_ylim(), 0.93), 
               "Findable Objects: {}".format(len(allObjects_survey[allObjects_survey["findable"] == 1])))
        
        # If test orbits exist in known orbits, plot them
        test_orbits_in_known = knownOrbits[knownOrbits[columnMapping["name"]].isin(orbits[columnMapping["name"]].values)]
        if len(test_orbits_in_known) != 0:
            ax.scatter(*test_orbits_in_known[[columnMapping["a_au"], columnMapping["i_deg"]]].values.T, 
                       s=2, 
                       c="r",
                       label="Test Orbits")
            ax.legend(loc='upper right')
            
        if runDir != None:
            fig.savefig(os.path.join(runDir, "known_orbits_findable_log.png"))
            
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

        # Build a cell with the test orbit at its center
        center = orbit[[columnMapping["RA_deg"], columnMapping["Dec_deg"]]].values[0]
        mjd = orbit[columnMapping["exp_mjd"]].values[0]
        cell = Cell(center, mjd, observations, shape=cellShape, area=cellArea)
        cell.getObservations(columnMapping=columnMapping)

        # Propagate the orbit and gather all nearby detections
        projected_obs = rangeAndShift(
            observations, 
            cell, 
            orbit[[columnMapping["r_au"]]].values[0], 
            orbit[[columnMapping["obj_dx/dt_au_p_day"],
                   columnMapping["obj_dy/dt_au_p_day"],
                   columnMapping["obj_dz/dt_au_p_day"]]].values[0], 
            mjds=mjds, 
            dMax=dMax, 
            numNights=numNights,
            includeEquatorialProjection=includeEquatorialProjection,
            verbose=False,
            columnMapping=columnMapping)
        
        # Save projected observations to file
        if runDir != None:
            projected_obs.to_csv(os.path.join(orbitDir, "projected_obs.txt"), sep=" ", index=False)

        # Analyze propagated observations
        allObjects_projection, summary_projection = analyzeProjections(
            projected_obs[~projected_obs[columnMapping["obs_id"]].isin(linked_detections)],
            columnMapping=columnMapping)
        summary_projection["orbit_id"] = orbit_id
        allObjects_projection["orbit_id"] = np.ones(len(allObjects_projection), dtype=int) * orbit_id
        
        # Save projected observations to file
        if runDir != None:
            allObjects_projection.to_csv(os.path.join(orbitDir, "allObjects.txt"), sep=" ", index=False)
            summary_projection.to_csv(os.path.join(orbitDir, "summary.txt"), sep=" ", index=False)
        
        # Plot projection velocities of findable known objects
        fig, ax = plotProjectionVelocitiesFindable(allObjects_projection, vxRange=vxRange, vyRange=vyRange)
        if runDir != None:
            fig.savefig(os.path.join(orbitDir, "projection_findable.png"))

        # Cluster and link
        allClusters_projection, clusterMembers_projection = clusterAndLink(
            projected_obs[~projected_obs[columnMapping["obs_id"]].isin(linked_detections)],
            vxRange=vxRange, 
            vyRange=vyRange,
            vxBins=vxBins, 
            vyBins=vyBins,
            vxValues=vxValues,
            vyValues=vyValues,
            threads=threads, 
            eps=eps, 
            minSamples=minSamples,
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
            minSamples=minSamples, 
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
        summary_projection["linkage_efficiency"] = calcLinkageEfficiency(
            allObjects_projection, 
            vxRange=vxRange, 
            vyRange=vyRange,
            verbose=True)

        # Plot projection velocities of found and missed known objects
        fig, ax = plotProjectionVelocitiesFound(allObjects_projection, vxRange=vxRange, vyRange=vyRange)
        if runDir != None:
            fig.savefig(os.path.join(orbitDir, "projection_found.png"))
        fig, ax = plotProjectionVelocitiesMissed(allObjects_projection, vxRange=vxRange, vyRange=vyRange)
        if runDir != None:
            fig.savefig(os.path.join(orbitDir, "projection_missed.png"))

        # Grab the linked detections
        linked_detections_projection = grabLinkedDetections(projected_obs, allClusters_projection, clusterMembers_projection, columnMapping=columnMapping)
        summary_projection["num_linked_observations"] = len(linked_detections_projection)
        linked_detections = np.concatenate([linked_detections, linked_detections_projection])
        
        # Grab time to complete orbit processing
        time_end = time.time()
        duration = time_end - time_start
        summary_projection["time_seconds"] = duration
        
        # Arrange columns in projection summary dataframe
        summary_projection = summary_projection[[
            'orbit_id',
            'percent_completeness', 
            'linkage_efficiency',
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
    completeness =  num_known_found / len(allObjects_survey[allObjects_survey["findable"] == 1])
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
    
    # Plot found and missed orbits if known orbits are provided
    if type(knownOrbits) == pd.DataFrame:
        # Plot found orbits (linear semi-major axis)
        fig, ax = plotOrbitsFound(allObjects_survey, knownOrbits)
        ax.set_xlim(0.0, 5.0)
        ax.text(_setPercentage(ax.get_xlim(), 0.02), 
                _setPercentage(ax.get_ylim(), 0.93), 
               "Found Objects: {}".format(len(allObjects_survey[allObjects_survey["found"] == 1])))
        
        # If test orbits exist in known orbits, plot them
        test_orbits_in_known = knownOrbits[knownOrbits[columnMapping["name"]].isin(orbits[columnMapping["name"]].values)]
        if len(test_orbits_in_known) != 0:
            ax.scatter(*test_orbits_in_known[[columnMapping["a_au"], columnMapping["i_deg"]]].values.T, 
                       s=2, 
                       c="r",
                       label="Test Orbits")
            ax.legend(loc='upper right')
        
        if runDir != None:
            fig.savefig(os.path.join(runDir, "known_orbits_found.png"))
        
        # Plot found orbits (log semi-major axis)
        fig, ax = plotOrbitsFound(allObjects_survey, knownOrbits)
        ax.set_xscale("log")
        ax.text(_setPercentage(ax.get_xlim(), 0.001), 
                _setPercentage(ax.get_ylim(), 0.93), 
               "Found Objects: {}".format(len(allObjects_survey[allObjects_survey["found"] == 1])))
        
        # If test orbits exist in known orbits, plot them
        test_orbits_in_known = knownOrbits[knownOrbits[columnMapping["name"]].isin(orbits[columnMapping["name"]].values)]
        if len(test_orbits_in_known) != 0:
            ax.scatter(*test_orbits_in_known[[columnMapping["a_au"], columnMapping["i_deg"]]].values.T, 
                       s=2, 
                       c="r",
                       label="Test Orbits")
            ax.legend(loc='upper right')
            
        if runDir != None:
            fig.savefig(os.path.join(runDir, "known_orbits_found_log.png"))

        # Plot missed orbits (linear semi-major axis)
        fig, ax = plotOrbitsMissed(allObjects_survey, knownOrbits)
        ax.set_xlim(0.0, 5.0)
        ax.text(_setPercentage(ax.get_xlim(), 0.02), 
                _setPercentage(ax.get_ylim(), 0.93), 
               "Missed Objects: {}".format(len(allObjects_survey[(allObjects_survey["found"] == 0) 
                                                                 & (allObjects_survey["findable"] == 1)])))
        
        # If test orbits exist in known orbits, plot them
        test_orbits_in_known = knownOrbits[knownOrbits[columnMapping["name"]].isin(orbits[columnMapping["name"]].values)]
        if len(test_orbits_in_known) != 0:
            ax.scatter(*test_orbits_in_known[[columnMapping["a_au"], columnMapping["i_deg"]]].values.T, 
                       s=2, 
                       c="r",
                       label="Test Orbits")
            ax.legend(loc='upper right')
            
        if runDir != None:
            fig.savefig(os.path.join(runDir, "known_orbits_missed.png"))
        
        # Plot missed orbits (log semi-major axis)
        fig, ax = plotOrbitsMissed(allObjects_survey, knownOrbits)
        ax.set_xscale("log")
        ax.text(_setPercentage(ax.get_xlim(), 0.001), 
                _setPercentage(ax.get_ylim(), 0.93), 
               "Missed Objects: {}".format(len(allObjects_survey[(allObjects_survey["found"] == 0) 
                                                                & (allObjects_survey["findable"] == 1)])))
        
        # If test orbits exist in known orbits, plot them
        test_orbits_in_known = knownOrbits[knownOrbits[columnMapping["name"]].isin(orbits[columnMapping["name"]].values)]
        if len(test_orbits_in_known) != 0:
            ax.scatter(*test_orbits_in_known[[columnMapping["a_au"], columnMapping["i_deg"]]].values.T, 
                       s=2, 
                       c="r",
                       label="Test Orbits")
            ax.legend(loc='upper right')
    
        if runDir != None:
            fig.savefig(os.path.join(runDir, "known_orbits_missed_log.png"))
            
    if verbose == True:
        print("THOR: runTHOR")
        print("-------------------------")
        print("Done. Finished running THOR with {} orbits.".format(len(orbits)))
        print("Completeness for known objects (%): {:1.3f}".format(completeness))
        print("")
        print("-------------------------")
            
    return allObjects_survey, summary_survey, summaries_projection
        