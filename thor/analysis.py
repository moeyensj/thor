import os
import time
import numpy as np
import pandas as pd

from .config import Config

__all__ = ["calcLinkageEfficiency",
           "analyzeObservations",
           "analyzeVisit",
           "analyzeProjections",
           "analyzeClusters"]

def calcLinkageEfficiency(allObjects, 
                          vxRange=[-0.1, 0.1], 
                          vyRange=[-0.1, 0.1],
                          verbose=True):
    """
    Calculates the linkage efficiency given the velocity ranges searched. 
    
    Parameters
    ----------
    allObjects : `~pandas.DataFrame`
        The allObjects DataFrame. Requires analyzeProjections
        and analyzeClusters to have been run on the DataFrame. 
    vxRange : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in x. 
        Will not be used if vxValues are specified. 
        [Default = [-0.1, 0.1]]
    vxRange : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in y. 
        Will not be used if vyValues are specified. 
        [Default = [-0.1, 0.1]]
    verbose : bool, optional
        Print progress statements? 
        [Default = True]
    
    Returns
    -------
    efficiency : float
        The number of objects found over the number findable as a fraction. 
    """
    # Find the objects that should have been found with the defined velocity ranges
    in_zone_findable = ((allObjects["dtheta_x/dt_median"] >= vxRange[0]) 
         & (allObjects["dtheta_x/dt_median"] <= vxRange[1]) 
         & (allObjects["dtheta_y/dt_median"] <= vyRange[1]) 
         & (allObjects["dtheta_y/dt_median"] >= vyRange[0])
         & (allObjects["findable"] == 1))
    findable = len(allObjects[in_zone_findable])
    
    # Find the objects that were missed
    in_zone_missed = ((allObjects["dtheta_x/dt_median"] >= vxRange[0]) 
         & (allObjects["dtheta_x/dt_median"] <= vxRange[1]) 
         & (allObjects["dtheta_y/dt_median"] <= vyRange[1]) 
         & (allObjects["dtheta_y/dt_median"] >= vyRange[0])
         & (allObjects["found"] == 0)
         & (allObjects["findable"] == 1))
    missed = len(allObjects[in_zone_missed])
    
    # Find the objects that were found
    in_zone_found = ((allObjects["dtheta_x/dt_median"] >= vxRange[0]) 
         & (allObjects["dtheta_x/dt_median"] <= vxRange[1]) 
         & (allObjects["dtheta_y/dt_median"] <= vyRange[1]) 
         & (allObjects["dtheta_y/dt_median"] >= vyRange[0])
         & (allObjects["found"] == 1))
    found = len(allObjects[in_zone_found])
    
    efficiency = found / findable
    
    if verbose == True:
        print("Findable objects: {}".format(findable))
        print("Found objects: {}".format(found))
        print("Missed objects: {}".format(missed))
        print("Efficiency [%]: {.f2}".format(efficiency * 100))
    return efficiency

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

def analyzeVisit(observations,
                 visitId, 
                 minSamples=5, 
                 verbose=True,
                 columnMapping=Config.columnMapping):
    """
    Count the number of objects that should be findable in the 
    survey that exist in the visit. Also calculate some visit-level
    statistics. 
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing post-range and shift observations.
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
    allObjects : `~pandas.DataFrame`
        Object summary DataFrame.
    summary : `~pandas.DataFrame`
        Overall summary DataFrame. 
    """
    time_start = time.time()
    if verbose == True:
        print("THOR: analyzeVisit")
        print("-------------------------")
        print("Analyzing visit {}...".format(visitId))
        
    visit = observations[observations[columnMapping["visit_id"]] == visitId]
    
    # Count number of noise detections, real object detections, the number of unique objects
    num_noise_obs = len(visit[visit[columnMapping["name"]] == "NS"])
    num_object_obs = len(visit[visit[columnMapping["name"]] != "NS"])
    unique_objects = visit[visit[columnMapping["name"]] != "NS"][columnMapping["name"]].unique()
    num_unique_objects = len(unique_objects)
    num_obs_per_object = observations[(observations[columnMapping["name"]] != "NS") & (observations[columnMapping["name"]].isin(unique_objects))][columnMapping["name"]].value_counts().values
    objects_num_obs_descending = observations[(observations[columnMapping["name"]] != "NS") & (observations[columnMapping["name"]].isin(unique_objects))][columnMapping["name"]].value_counts().index.values
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
        "visit_id": visitId,
        "num_observations_object": num_object_obs,
        "num_observations_noise": num_noise_obs,
        "obs_contamination" : num_noise_obs / len(visit) * 100,
        "unique_objects" : num_unique_objects,
        "unique_objects_findable" : num_findable}, index=[0]) 
    
    time_end = time.time()
    if verbose == True:
        print("Object observations in visit: {}".format(num_object_obs))
        print("Noise observations in visit: {}".format(num_noise_obs))
        print("Observation contamination (%): {}".format(num_noise_obs / len(visit) * 100))
        print("Unique objects in visit: {}".format(num_unique_objects))
        print("Unique objects with at least {} detections in survey: {}".format(minSamples, num_findable))
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
        print("THOR: analyzeProjections")
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
        [Default = 0.2]
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
    cluster_designation.sort_values(by="cluster_id", inplace=True)
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
    observations_temp = observations.rename(columns={columnMapping["obs_id"]: "obs_id"})

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

    allClusters.loc[allClusters["cluster_id"].isin(partial_clusters["cluster_id"]), "linked_object"] = partial_clusters[columnMapping["name"]].values
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

