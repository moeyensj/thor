import time
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from sklearn.cluster import DBSCAN

from .config import Config
from .cell import Cell
from .particle import TestParticle
from .oorb import propagateTestParticle
from .data_processing import findExposureTimes
from .data_processing import buildCellForVisit

__all__ = ["rangeAndShift",
           "clusterVelocity",
           "_clusterVelocity",
           "clusterAndLink",
           "analyzeClusters",
           "runRangeAndShiftOnVisit"]

def rangeAndShift(observations,
                  cell, 
                  r, 
                  v,
                  mjds="auto",
                  vMax=3.0,
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
    mjds : {'auto', `~numpy.ndarray` (N)}
        If mjds is 'auto', will propagate the particle to middle of each unique night in 
        obervations and look for all detections within an angular search radius (defined by a 
        maximum allowable angular speed) and extract the exposure time. Alternatively, an array
        of exposure times may be passed. 
    vMax : float, optional
        Maximum angular velocity (in RA and Dec) permitted when searching for exposure times
        in degrees per day. 
        [Default = 3.0]
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
        nights = observations[columnMapping["night"]].unique() 
        cell_night = cell.observations["night"].unique()[0]
        nights.sort()
        nights = nights[nights > cell_night]
        mjds = findExposureTimes(observations, particle.x_a, v, cell.mjd, nights, verbose=verbose)
        
    
    # Apply tranformations to observations
    particle.apply(cell, verbose=verbose)
    
    # Add initial cell and particle to lists
    cells = [cell]
    particles = [particle]
    
    if includeEquatorialProjection is True:
            cell.observations["theta_x_eq_deg"] = cell.observations[columnMapping["RA_deg"]] - particle.coords_eq_ang[0]
            cell.observations["theta_y_eq_deg"] = cell.observations[columnMapping["RA_deg"]] - particle.coords_eq_ang[1]
    
    # Initialize final dataframe and add observations
    final_df = pd.DataFrame()
    final_df = pd.concat([cell.observations, final_df])

    for mjd_f in mjds:
        oldCell = cells[-1]
        oldParticle = particles[-1]
        
        # Propagate particle to new mjd
        propagated = propagateTestParticle(oldParticle.x_a,
                                           oldParticle.v,
                                           oldParticle.mjd,
                                           mjd_f,
                                           verbose=verbose)
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
        new_mjd = propagated["mjd_utc"].values[0]

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
                   vxNum=100, 
                   vyNum=100, 
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
    vxRange : list or `~numpy.ndarray` (2)
        Maximum and minimum velocity range in x. 
        [Default = [-0.1, 0.1]]
    vxRange : list or `~numpy.ndarray` (2)
        Maximum and minimum velocity range in y. 
        [Default = [-0.1, 0.1]]
    vxNum : int, optional
        Length of x-velocity grid between vxRange[0] 
        and vxRange[-1].
        [Default = 100]
    vyNum : int, optional
        Length of y-velocity grid between vyRange[0] 
        and vyRange[-1].
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
    vx = np.linspace(*vxRange, num=vxNum)
    vy = np.linspace(*vyRange, num=vyNum)
    vxx, vyy = np.meshgrid(vx, vy)    
    vxx = vxx.flatten()
    vyy = vyy.flatten()

    time_start_cluster = time.time()
    if verbose == True:
        print("THOR: clusterAndLink")
        print("-------------------------")
        print("Running velocity space clustering...")
        print("X velocity range: {}".format(vxRange))
        print("X velocity bins: {}".format(vxNum))
        print("Y velocity range: {}".format(vyRange))
        print("Y velocity bins: {}".format(vyNum))
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
        print("-------------------------")
        print("")

    return allClusters, clusterMembers

def analyzeClusters(observations,
                    allClusters, 
                    clusterMembers, 
                    minSamples=5, 
                    partialThreshold=0.8, 
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
    minSamples : int, optional
        The number of samples (or total weight) in a neighborhood for a 
        point to be considered as a core point. This includes the point itself.
        See: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html
        [Default = 5]
    partialThreshold : float, optional
        Percentage (expressed between 0 and 1) of observations in a cluster required for the 
        object to be found. 
        [Default = 0.8]
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
        Summary dataframe.
    """ 

    time_start = time.time()
    if verbose == True:
        print("THOR: analyzeClusters")
        print("-------------------------")
        print("Analyzing observations...")

    # Add columns to classify pure, contaminated and false clusters
    # Add numbers to classify the number of members, and the linked object 
    # in the contaminated and pure case
    allClusters["pure"] = np.zeros(len(allClusters), dtype=int)
    allClusters["partial"] = np.zeros(len(allClusters), dtype=int)
    allClusters["false"] = np.zeros(len(allClusters), dtype=int)
    allClusters["num_members"] = np.ones(len(allClusters), dtype=int) * np.NaN
    allClusters["linked_object"] = np.ones(len(allClusters), dtype=int) * np.NaN

    # Count number of noise detections, real object detections,
    num_noise_obs = len(observations[observations[columnMapping["name"]] == "NS"])
    num_object_obs = len(observations[observations[columnMapping["name"]] != "NS"])
    num_unique = len(observations[observations[columnMapping["name"]] != "NS"][columnMapping["name"]].unique())
    num_obs_per_object = observations[observations[columnMapping["name"]] != "NS"][columnMapping["name"]].value_counts().values
    num_min_obs_pure = len(np.where(num_obs_per_object >= minSamples)[0])
    num_min_obs_partial = len(np.where(num_obs_per_object >= partialThreshold * minSamples)[0])
    objects_num_obs_descending = observations[observations[columnMapping["name"]] != "NS"][columnMapping["name"]].value_counts().index.values
    findable_pure = objects_num_obs_descending[np.where(num_obs_per_object >= minSamples)[0]]
    findable_partial = objects_num_obs_descending[np.where(num_obs_per_object >= partialThreshold * minSamples)[0]]
    
    if verbose == True:
        print("Object observations: {}".format(num_object_obs))
        print("Noise observations: {}".format(num_noise_obs))
        print("Observation contamination (%): {}".format(num_noise_obs / len(observations) * 100))
        print("Unique objects: {}".format(num_unique))
        print("Unique objects with at least {} detections: {}".format(minSamples, num_min_obs_pure))
        print("Unique objects with at least {}% of {} detections: {}".format(partialThreshold * 100, minSamples, num_min_obs_partial))
        print("")
        print("Analyzing clusters...")
        
    observations_temp = observations.rename(columns={columnMapping["obs_id"]: "obs_id"})
    cluster_designation = observations_temp[["obs_id", columnMapping["name"]]].merge(clusterMembers, on="obs_id")
    cluster_designation.drop(columns="obs_id", inplace=True)
    cluster_designation.drop_duplicates(inplace=True)
    unique_ids_per_cluster = cluster_designation["cluster_id"].value_counts()
    allClusters["num_members"] = unique_ids_per_cluster.sort_index().values

    # Isolate pure clusters
    single_member_clusters = cluster_designation[cluster_designation["cluster_id"].isin(allClusters[allClusters["num_members"] == 1]["cluster_id"])]
    allClusters.loc[allClusters["cluster_id"].isin(single_member_clusters["cluster_id"]), "linked_object"] = single_member_clusters[columnMapping["name"]].values
    allClusters.loc[(allClusters["linked_object"] != "NS") & (allClusters["linked_object"].notna()), "pure"] = 1
    allClusters.loc[(((1 - allClusters["num_members"] / allClusters["num_obs"]) >= partialThreshold)
                     & (allClusters["pure"] != 1) & (allClusters["linked_object"].isna())), "partial"] = 1
    allClusters.loc[(allClusters["pure"] != 1) & (allClusters["partial"] != 1), "false"] = 1
    
    num_pure = len(allClusters[allClusters["pure"] == 1])
    num_partial = len(allClusters[allClusters["partial"] == 1])
    num_false = len(allClusters[allClusters["false"] == 1])
    num_total = num_pure + num_partial + num_false
    
    if verbose == True:
        print("Pure clusters: {}".format(num_pure))
        print("Partial clusters: {}".format(num_partial))
        print("False clusters: {}".format(num_false))
        print("Total clusters: {}".format(num_total))
        print("Cluster contamination (%): {}".format(num_false / num_total * 100))
        
    # Isolate partial clusters and grab the object with the most detections to cound as found (partial)
    cluster_designation = observations_temp[["obs_id", columnMapping["name"]]].merge(clusterMembers, on="obs_id")
    cluster_designation.drop(columns="obs_id", inplace=True)
    partial_ids = allClusters[allClusters["partial"] == 1]["cluster_id"].values
    partial_clusters = cluster_designation[cluster_designation["cluster_id"].isin(partial_ids)]
    partial_clusters_temp = pd.DataFrame(partial_clusters.groupby(["cluster_id", columnMapping["name"]]).size()).reset_index()
    partial_clusters_temp.rename(columns={0: "count"}, inplace=True)
    partial_clusters_temp.sort_values("count", ascending=False, inplace=True)
    partial_clusters_temp.drop_duplicates(subset=["cluster_id"], keep="first", inplace=True)
    partial_clusters_temp.sort_values("cluster_id", inplace=True)
    allClusters.loc[allClusters["cluster_id"].isin(partial_clusters_temp["cluster_id"]), "linked_object"] = partial_clusters_temp[columnMapping["name"]].values
    
    # Populate allObjects DataFrame
    allObjects = pd.DataFrame(columns=[
        columnMapping["name"], 
        "num_obs", 
        "findable_pure", 
        "findable_partial", 
        "findable",
        "found_pure", 
        "found_partial",
        "found"])
    allObjects[columnMapping["name"]] = objects_num_obs_descending
    allObjects["num_obs"] = num_obs_per_object
    allObjects.loc[allObjects[columnMapping["name"]].isin(allClusters[allClusters["pure"] == 1]["linked_object"]), "found_pure"] = 1
    allObjects.loc[allObjects[columnMapping["name"]].isin(allClusters[allClusters["partial"] == 1]["linked_object"]), "found_partial"] = 1
    allObjects.loc[allObjects[columnMapping["name"]].isin(findable_partial), "findable_partial"] = 1
    allObjects.loc[allObjects[columnMapping["name"]].isin(findable_pure), "findable_pure"] = 1
    allObjects.loc[(allObjects["findable_pure"] == 1) | (allObjects["findable_partial"] == 1), "findable"] = 1
    allObjects.loc[(allObjects["found_pure"] == 1) | (allObjects["found_partial"] == 1), "found"] = 1
    allObjects.fillna(value=0, inplace=True)
    
    found = allObjects[allObjects["found"] == 1]
    missed = allObjects[(allObjects["found"] == 0) & (allObjects["findable"] == 1)]
    time_end = time.time()
    
    if verbose == True:
        print("Unique linked objects: {}".format(len(found)))
        print("Unique missed objects: {}".format(len(missed)))
        print("Completeness (%): {}".format(len(found) / (len(found) + len(missed)) * 100))
        print("Done.")
        print("Total time in seconds: {}".format(time_end - time_start))
        print("-------------------------")
        print("")

    return allClusters, clusterMembers, allObjects

def runRangeAndShiftOnVisit(observations,
                            visitId,
                            r, 
                            v,
                            searchArea=0.5, 
                            searchShape="square",
                            cellArea=10, 
                            cellShape="square",
                            useAverageObject=True,
                            verbose=True,
                            columnMapping=True):
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
            return

        obj = small_cell.observations[small_cell.observations[config.columnMapping["name"]] == avg_obj]
        r = obj[config.columnMapping["r_au"]].values[0]
        v = obj[[config.columnMapping["obj_dx/dt_au_p_day"], 
                 config.columnMapping["obj_dy/dt_au_p_day"],
                 config.columnMapping["obj_dz/dt_au_p_day"]]].values[0]
        

    cell = Cell(small_cell.center, small_cell.mjd, observations, area=cellArea, shape=cellShape)
    projected_obs = rangeAndShift(observations, cell, r, v)
    
    return projected_obs