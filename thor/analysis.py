import os
import time
import numpy as np
import pandas as pd

from .config import Config

__all__ = ["__analyzeLinkages",
           "calcLinkageFindable",
           "calcLinkageFound",
           "calcLinkageMissed",
           "calcLinkageEfficiency",
           "analyzeObservations",
           "analyzeProjections",
           "analyzeClusters"]

def __analyzeLinkages(observations, 
                      linkageMembers, 
                      allLinkages=None, 
                      allTruths=None,
                      minObs=5, 
                      contaminationThreshold=0.2, 
                      columnMapping={"linkage_id": "linkage_id",
                                     "obs_id": "obs_id",
                                     "truth": "truth"}):
    """
    Did I Find It? (Future Package)
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Pandas DataFrame with at least two columns: observation IDs and the truth values
        (the object to which the observation belongs to).
    linkageMembers : `~pandas.DataFrame`
        Pandas DataFrame with at least two columns: linkage IDs and the observation 
    allLinkages : {`~pandas.DataFrame`, None}, optional
        Pandas DataFrame with one row per linkage with at least one column: linkage IDs.
        If None, allLinkages will be created.
        [Default = None]
    allTruths : {`~pandas.DataFrame`, None}, optional
        Pandas DataFrame with one row per unique truth with at least one column: truths.
        If None, allTruths will be created.
        [Default = None]
    minObs : int, optional
        The minimum number of observations belonging to one object for a linkage to be pure. 
        The minimum number of observations belonging to one object in a contaminated linkage
        (number of contaminated observations allowed is set by the contaminationThreshold)
        for the linkage to be partial. For example, if minObs is 5 then any linkage with 5 or more 
        detections belonging to a unique object, with no detections belonging to any other object will be 
        considered a pure linkage and the object is found. Likewise, if minObs is 5 and contaminationThreshold is 
        0.2 then a linkage with 10 members, where 8 belong to one object and 2 belong to other objects, will 
        be considered a partial linkage, and the object with 8 detections is considered found. 
        [Default = 5]
    contaminationThreshold : float, optional 
        Number of detections expressed as a percentage belonging to other objects in a linkage
        allowed for the object with the most detections in the linkage to be considered found. 
        [Default = 0.2]
    columnMapping : dict, optional
        The mapping of columns in observations and linkageMembers to internally used names. 
        Needs the following: "linkage_id" : ..., "truth": ... and "obs_id" : ... .
        
    Returns
    -------
    allLinkages : `~pandas.DataFrame`
        DataFrame with added pure, partial, false, contamination, num_obs, num_members, linked_truth 
    allTruths : `~pandas.DataFrame`
        DataFrame with added found_pure, found_partial, found columns. 
    """
    # If allLinkages DataFrame does not exist, create it
    if type(allLinkages) != pd.DataFrame:
        linkage_ids = linkageMembers[columnMapping["linkage_id"]].unique()
        linkage_ids.sort()
        allLinkages = pd.DataFrame({
            columnMapping["linkage_id"] : linkage_ids})
    
    # Prepare allLinkage columns
    allLinkages["num_members"] = np.ones(len(allLinkages)) * np.NaN
    allLinkages["num_obs"] = np.ones(len(allLinkages)) * np.NaN
    allLinkages["pure"] = np.zeros(len(allLinkages), dtype=int)
    allLinkages["partial"] = np.zeros(len(allLinkages), dtype=int)
    allLinkages["false"] = np.zeros(len(allLinkages), dtype=int)
    allLinkages["contamination"] = np.ones(len(allLinkages), dtype=int) * np.NaN
    allLinkages["linked_truth"] = np.ones(len(allLinkages), dtype=int) * np.NaN
    
    # Add the number of observations each linkage as 
    allLinkages["num_obs"] = linkageMembers[columnMapping["linkage_id"]].value_counts().sort_index().values

    # If allTruths DataFrame does not exist, create it
    if type(allTruths) != pd.DataFrame:
        allTruths = pd.DataFrame({
            columnMapping["truth"] : truths})
    
    # Prepare allTruths columns
    allTruths["found_pure"] = np.zeros(len(allTruths), dtype=int)
    allTruths["found_partial"] = np.zeros(len(allTruths), dtype=int)
    allTruths["found"] = np.zeros(len(allTruths), dtype=int)
        
    ### Calculate the number of unique truth's per linkage
    
    # Grab only observation IDs and truth from observations
    linkage_truth = observations[[columnMapping["obs_id"], columnMapping["truth"]]]
    
    # Merge truth from observations with linkageMembers on observation IDs
    linkage_truth = linkage_truth.merge(
        linkageMembers[[columnMapping["linkage_id"],
                        columnMapping["obs_id"]]], 
        on=columnMapping["obs_id"])
    
    # Drop observation ID column
    linkage_truth.drop(columns=columnMapping["obs_id"], inplace=True)
    
    # Drop duplicate rows, any correct linkage will now only have one row since
    # all the truth values would have been the same, any incorrect linkage
    # will now have multiple rows for each unique truth value
    linkage_truth.drop_duplicates(inplace=True)
    
    # Sort by linkage IDs and reset index
    linkage_truth.sort_values(by=columnMapping["linkage_id"], inplace=True)
    linkage_truth.reset_index(inplace=True, drop=True)
    
    # Grab the number of unique truths per linkage and update 
    # the allLinkages DataFrame with the result
    unique_truths_per_linkage = linkage_truth[columnMapping["linkage_id"]].value_counts()
    allLinkages["num_members"] = unique_truths_per_linkage.sort_index().values
    
    ### Find all the pure linkages and identify them as such
    
    # All the linkages where num_members = 1 are pure linkages
    single_member_linkages = linkage_truth[
        linkage_truth[columnMapping["linkage_id"]].isin(
            allLinkages[(allLinkages["num_members"] == 1) & (allLinkages["num_obs"] >= minObs)][columnMapping["linkage_id"]])]
    
    # Update the linked_truth field in allLinkages with the linked object
    pure_linkages = allLinkages[columnMapping["linkage_id"]].isin(single_member_linkages[columnMapping["linkage_id"]])
    allLinkages.loc[pure_linkages, "linked_truth"] = single_member_linkages[columnMapping["truth"]].values
    
    # Update the pure field in allLinkages to indicate which linkages are pure
    allLinkages.loc[(allLinkages["linked_truth"].notna()), "pure"] = 1
    
    ### Find all the partial linkages and identify them as such
    
    # Grab only observation IDs and truth from observations
    linkage_truth = observations[[columnMapping["obs_id"], columnMapping["truth"]]]

    # Merge truth from observations with linkageMembers on observation IDs
    linkage_truth = linkage_truth.merge(
        linkageMembers[[columnMapping["linkage_id"],
                        columnMapping["obs_id"]]], 
        on=columnMapping["obs_id"])

    # Remove non-pure linkages
    linkage_truth = linkage_truth[linkage_truth[columnMapping["linkage_id"]].isin(
        allLinkages[allLinkages["pure"] != 1][columnMapping["linkage_id"]])]

    # Drop observation ID column
    linkage_truth.drop(columns=columnMapping["obs_id"], inplace=True)

    # Group by linkage IDs and truths, creates a multi-level index with linkage ID
    # as the first index, then truth as the second index and as values is the count 
    # of the number of times the truth shows up in the linkage
    linkage_truth = linkage_truth.groupby(linkage_truth[[
        columnMapping["linkage_id"],
        columnMapping["truth"]]].columns.tolist(), as_index=False).size()

    # Reset the index to create a DataFrame
    linkage_truth = linkage_truth.reset_index()

    # Rename 0 column to num_obs which counts the number of observations
    # each unique truth has in each linkage
    linkage_truth.rename(columns={0: "num_obs"}, inplace=True)

    # Sort by linkage ID and num_obs so that the truth with the most observations
    # in each linkage is last for each linkage
    linkage_truth.sort_values(by=[columnMapping["linkage_id"], "num_obs"], inplace=True)

    # Drop duplicate rows, keeping only the last row 
    linkage_truth.drop_duplicates(subset=[columnMapping["linkage_id"]], inplace=True, keep="last")

    # Grab all linkages and merge truth from observations with linkageMembers on observation IDs
    linkage_truth = linkage_truth.merge(allLinkages[[columnMapping["linkage_id"], "num_obs"]], on=columnMapping["linkage_id"])

    # Rename num_obs column in allLinkages to total_num_obs
    linkage_truth.rename(columns={"num_obs_x": "num_obs", "num_obs_y": "total_num_obs"}, inplace=True)

    # Calculate contamination 
    linkage_truth["contamination"] = (1 - linkage_truth["num_obs"] / linkage_truth["total_num_obs"])
    
    # Select partial linkages: have at least the minimum observations of a single truth and have no
    # more than x% contamination
    partial_linkages = linkage_truth[(linkage_truth["num_obs"] >= minObs) 
                                   & (linkage_truth["contamination"] <= contaminationThreshold)]
    
    # Update allLinkages to indicate partial linkages, update linked_truth field
    # Set every linkage that isn't partial or pure to false
    allLinkages.loc[allLinkages[columnMapping["linkage_id"]].isin(partial_linkages[columnMapping["linkage_id"]]), "linked_truth"] = partial_linkages[columnMapping["truth"]].values
    allLinkages.loc[allLinkages[columnMapping["linkage_id"]].isin(partial_linkages[columnMapping["linkage_id"]]), "partial"] = 1
    allLinkages.loc[allLinkages[columnMapping["linkage_id"]].isin(partial_linkages[columnMapping["linkage_id"]]), "contamination"] = partial_linkages["contamination"].values
    allLinkages.loc[(allLinkages["pure"] != 1) & (allLinkages["partial"] != 1), "false"] = 1

    # Update allTruths to indicate which objects were found in pure and partial clusters, if found in either the object is found
    allTruths.loc[allTruths[columnMapping["truth"]].isin(allLinkages[allLinkages["pure"] == 1]["linked_truth"].values), "found_pure"] = 1
    allTruths.loc[allTruths[columnMapping["truth"]].isin(allLinkages[allLinkages["partial"] == 1]["linked_truth"].values), "found_partial"] = 1
    allTruths.loc[(allTruths["found_pure"] == 1) | (allTruths["found_partial"] == 1), "found"] = 1
    
    return allLinkages, allTruths

def calcLinkageFindable(allObjects, 
                        vxRange=[-0.1, 0.1], 
                        vyRange=[-0.1, 0.1]):
    """
    Returns a slice of the allObjects DataFrame with all objects
    that are findable inside the velocity ranges.
    
    Parameters
    ----------
    allObjects : `~pandas.DataFrame`
        The allObjects DataFrame. Requires analyzeProjections
        to have been run on the DataFrame. 
    vxRange : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in x. 
        Will not be used if vxValues are specified. 
        [Default = [-0.1, 0.1]]
    vxRange : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in y. 
        Will not be used if vyValues are specified. 
        [Default = [-0.1, 0.1]]
    
    Returns
    -------
    allObjects[findable] : `~pandas.DataFrame`
        The allObjects DataFrame filtered with only the
        findable objects in the velocity ranges.
    """
    in_zone_findable = ((allObjects["dtheta_x/dt_median"] >= vxRange[0]) 
         & (allObjects["dtheta_x/dt_median"] <= vxRange[1]) 
         & (allObjects["dtheta_y/dt_median"] <= vyRange[1]) 
         & (allObjects["dtheta_y/dt_median"] >= vyRange[0])
         & (allObjects["findable"] == 1))
    return allObjects[in_zone_findable]

def calcLinkageFound(allObjects, 
                     vxRange=[-0.1, 0.1], 
                     vyRange=[-0.1, 0.1]):
    """
    Returns a slice of the allObjects DataFrame with all objects
    that have been found inside the velocity ranges.
    
    Parameters
    ----------
    allObjects : `~pandas.DataFrame`
        The allObjects DataFrame. Requires analyzeClusters
        to have been run on the DataFrame. 
    vxRange : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in x. 
        Will not be used if vxValues are specified. 
        [Default = [-0.1, 0.1]]
    vxRange : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in y. 
        Will not be used if vyValues are specified. 
        [Default = [-0.1, 0.1]]
    
    Returns
    -------
    allObjects[found] : `~pandas.DataFrame`
        The allObjects DataFrame filtered with only the
        found objects inside the velocity ranges.
    """
    in_zone_found = ((allObjects["dtheta_x/dt_median"] >= vxRange[0]) 
         & (allObjects["dtheta_x/dt_median"] <= vxRange[1]) 
         & (allObjects["dtheta_y/dt_median"] <= vyRange[1]) 
         & (allObjects["dtheta_y/dt_median"] >= vyRange[0])
         & (allObjects["found"] == 1))
    return allObjects[in_zone_found]

def calcLinkageMissed(allObjects, 
                      vxRange=[-0.1, 0.1], 
                      vyRange=[-0.1, 0.1]):
    """
    Returns a slice of the allObjects DataFrame with all objects
    that were missed inside the velocity ranges.
    
    Parameters
    ----------
    allObjects : `~pandas.DataFrame`
        The allObjects DataFrame. Requires analyzeClusters
        to have been run on the DataFrame. 
    vxRange : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in x. 
        Will not be used if vxValues are specified. 
        [Default = [-0.1, 0.1]]
    vxRange : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in y. 
        Will not be used if vyValues are specified. 
        [Default = [-0.1, 0.1]]
    
    Returns
    -------
    allObjects[missed] : `~pandas.DataFrame`
        The allObjects DataFrame filtered with only
        missed objects inside the velocity ranges.
    """
    in_zone_missed = ((allObjects["dtheta_x/dt_median"] >= vxRange[0]) 
         & (allObjects["dtheta_x/dt_median"] <= vxRange[1]) 
         & (allObjects["dtheta_y/dt_median"] <= vyRange[1]) 
         & (allObjects["dtheta_y/dt_median"] >= vyRange[0])
         & (allObjects["found"] == 0)
         & (allObjects["findable"] == 1))
    return allObjects[in_zone_missed]

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
    if verbose == True:
        print("THOR: calcLinkageEfficiency")
        print("-------------------------")
        
    # Find the objects that should have been found with the defined velocity ranges
    in_zone_findable = calcLinkageFindable(allObjects, vxRange=vxRange, vyRange=vyRange)
    findable = len(in_zone_findable)
    
    # Find the objects that were missed
    in_zone_missed = calcLinkageMissed(allObjects, vxRange=vxRange, vyRange=vyRange)
    missed = len(in_zone_missed)
    
    # Find the objects that were found
    in_zone_found = calcLinkageFound(allObjects, vxRange=vxRange, vyRange=vyRange)
    found = len(in_zone_found)

    efficiency = found / findable
    
    if verbose == True:
        print("Findable objects: {}".format(findable))
        print("Found objects: {}".format(found))
        print("Missed objects: {}".format(missed))
        print("Efficiency [%]: {}".format(efficiency * 100))
    return efficiency

def analyzeObservations(observations,
                        minSamples=5, 
                        unknownIDs=Config.unknownIDs,
                        falsePositiveIDs=Config.falsePositiveIDs,
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
    num_noise_obs = len(observations[observations[columnMapping["name"]].isin(falsePositiveIDs)])
    num_unlinked_obs = len(observations[observations[columnMapping["name"]].isin(unknownIDs)])
    num_object_obs = len(observations[~observations[columnMapping["name"]].isin(unknownIDs + falsePositiveIDs)])
    unique_objects = observations[~observations[columnMapping["name"]].isin(unknownIDs + falsePositiveIDs)][columnMapping["name"]].unique()
    num_unique_objects = len(unique_objects)
    num_obs_per_object = observations[~observations[columnMapping["name"]].isin(unknownIDs + falsePositiveIDs)][columnMapping["name"]].value_counts().values
    objects_num_obs_descending = observations[~observations[columnMapping["name"]].isin(unknownIDs + falsePositiveIDs)][columnMapping["name"]].value_counts().index.values
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
    percent_known = num_object_obs / len(observations) * 100
    percent_unknown = num_unlinked_obs / len(observations) * 100
    percent_false_positive = num_noise_obs / len(observations) * 100
    
    # Prepare summary DataFrame
    summary = pd.DataFrame({
        "num_unique_known_objects" : num_unique_objects,
        "num_unique_known_objects_findable" : num_findable,
        "num_known_object_observations": num_object_obs,
        "num_unknown_object_observations": num_unlinked_obs,
        "num_false_positive_observations": num_noise_obs,
        "percent_known_object_observations": percent_known,
        "percent_unknown_object_observations": percent_unknown,
        "percent_false_positive_observations": percent_false_positive}, index=[0]) 
    
    time_end = time.time()
    if verbose == True:
        print("Known object observations: {}".format(num_object_obs))
        print("Unknown object observations: {}".format(num_unlinked_obs))
        print("False positive observations: {}".format(num_noise_obs))
        print("Percent known object observations (%): {:1.3f}".format(percent_known))
        print("Percent unknown object observations (%): {:1.3f}".format(percent_unknown))
        print("Percent false positive observations (%): {:1.3f}".format(percent_false_positive))
        print("Unique known objects: {}".format(num_unique_objects))
        print("Unique known objects with at least {} detections: {}".format(minSamples, num_findable))
        print("") 
        print("Total time in seconds: {}".format(time_end - time_start))
        print("-------------------------")
        print("")
        
    return allObjects, summary


def analyzeProjections(observations,
                       minSamples=5, 
                       unknownIDs=Config.unknownIDs,
                       falsePositiveIDs=Config.falsePositiveIDs,
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
    num_noise_obs = len(observations[observations[columnMapping["name"]].isin(falsePositiveIDs)])
    num_unlinked_obs = len(observations[observations[columnMapping["name"]].isin(unknownIDs)])
    num_object_obs = len(observations[~observations[columnMapping["name"]].isin(unknownIDs + falsePositiveIDs)])
    unique_objects = observations[~observations[columnMapping["name"]].isin(unknownIDs + falsePositiveIDs)][columnMapping["name"]].unique()
    num_unique_objects = len(unique_objects)
    num_obs_per_object = observations[~observations[columnMapping["name"]].isin(unknownIDs + falsePositiveIDs)][columnMapping["name"]].value_counts().values
    objects_num_obs_descending = observations[~observations[columnMapping["name"]].isin(unknownIDs + falsePositiveIDs)][columnMapping["name"]].value_counts().index.values
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
    allObjects.loc[allObjects["findable"] != 1, ["findable"]] = 0
    num_findable = len(allObjects[allObjects["findable"] == 1])
    percent_known = num_object_obs / len(observations) * 100
    percent_unknown = num_unlinked_obs / len(observations) * 100
    percent_false_positive = num_noise_obs / len(observations) * 100
    
    calc_r = False
    calc_Delta = False
    
    if columnMapping["r_au"] in observations.columns:
        calc_r = True
    if columnMapping["Delta_au"] in observations.columns:
        calc_Delta = True

    for obj in findable:
        dets = observations[observations[columnMapping["name"]].isin([obj])]
        dt = dets[columnMapping["exp_mjd"]].values[1:] - dets[columnMapping["exp_mjd"]].values[0]
        dx = dets["theta_x_deg"].values[1:] - dets["theta_x_deg"].values[0]
        dy = dets["theta_y_deg"].values[1:] - dets["theta_y_deg"].values[0]
        allObjects.loc[allObjects[columnMapping["name"]].isin([obj]), ["dtheta_x/dt_median"]] = np.median(dx/dt)
        allObjects.loc[allObjects[columnMapping["name"]].isin([obj]), ["dtheta_y/dt_median"]] = np.median(dy/dt)
        allObjects.loc[allObjects[columnMapping["name"]].isin([obj]), ["dtheta_x/dt_sigma"]] = np.std(dx/dt)
        allObjects.loc[allObjects[columnMapping["name"]].isin([obj]), ["dtheta_y/dt_sigma"]] = np.std(dy/dt)
        
        if calc_r == True:
            allObjects.loc[allObjects[columnMapping["name"]].isin([obj]), ["r_au_median"]] = np.median(dets[columnMapping["r_au"]].values)
            allObjects.loc[allObjects[columnMapping["name"]].isin([obj]), ["r_au_sigma"]] = np.std(dets[columnMapping["r_au"]].values)
        
        if calc_Delta == True:
            allObjects.loc[allObjects[columnMapping["name"]].isin([obj]), ["Delta_au_median"]] = np.median(dets[columnMapping["Delta_au"]].values)
            allObjects.loc[allObjects[columnMapping["name"]].isin([obj]), ["Delta_au_sigma"]] = np.std(dets[columnMapping["Delta_au"]].values)
        
    allObjects.loc[allObjects["findable"] != 1, ["findable"]] = 0
    
    # Prepare summary DataFrame
    summary = pd.DataFrame({
        "num_unique_known_objects" : num_unique_objects,
        "num_unique_known_objects_findable" : num_findable,
        "num_known_object_observations": num_object_obs,
        "num_unknown_object_observations": num_unlinked_obs,
        "num_false_positive_observations": num_noise_obs,
        "percent_known_object_observations": percent_known,
        "percent_unknown_object_observations": percent_unknown,
        "percent_false_positive_observations": percent_false_positive}, index=[0]) 
    
    time_end = time.time()
    if verbose == True:
        print("Known object observations: {}".format(num_object_obs))
        print("Unknown object observations: {}".format(num_unlinked_obs))
        print("False positive observations: {}".format(num_noise_obs))
        print("Percent known object observations (%): {:1.3f}".format(percent_known))
        print("Percent unknown object observations (%): {:1.3f}".format(percent_unknown))
        print("Percent false positive observations (%): {:1.3f}".format(percent_false_positive))
        print("Unique known objects: {}".format(num_unique_objects))
        print("Unique known objects with at least {} detections: {}".format(minSamples, num_findable))
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
                    unknownIDs=Config.unknownIDs,
                    falsePositiveIDs=Config.falsePositiveIDs,
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

    if "linked_object" in allClusters.columns:
        allClusters.drop("linked_object", axis=1, inplace=True)
        
    allClusters, allObjects = __analyzeLinkages(observations, 
                          clusterMembers, 
                          allLinkages=allClusters, 
                          allTruths=allObjects,
                          minObs=minSamples, 
                          contaminationThreshold=contaminationThreshold, 
                          columnMapping={"linkage_id": "cluster_id",
                                         "obs_id": columnMapping["obs_id"],
                                         "truth":  columnMapping["name"]})
    allClusters.rename(columns={"linked_truth": "linked_object"}, inplace=True)
    
    # Count number of visits per cluster
    cluster_visit = observations[[columnMapping["obs_id"], columnMapping["visit_id"]]].merge(clusterMembers, on=columnMapping["obs_id"])
    cluster_visit.drop(columns=columnMapping["obs_id"], inplace=True)
    cluster_visit.drop_duplicates(inplace=True)
    unique_visits_per_cluster = cluster_visit["cluster_id"].value_counts()
    allClusters["num_visits"] = unique_visits_per_cluster.sort_index().values
    
    # Cluster breakdown for known objects
    num_pure_known = len(allClusters[(allClusters["pure"] == 1) & ~allClusters["linked_object"].isin(unknownIDs + falsePositiveIDs)])
    num_partial_known = len(allClusters[(allClusters["partial"] == 1) & ~allClusters["linked_object"].isin(unknownIDs + falsePositiveIDs)])
    
    # Cluster breakdown for unknown objects
    num_pure_unknown = len(allClusters[(allClusters["pure"] == 1) & allClusters["linked_object"].isin(unknownIDs)])
    num_partial_unknown = len(allClusters[(allClusters["partial"] == 1) & allClusters["linked_object"].isin(unknownIDs)])
    
    # Cluster breakdown for false positives
    num_pure_false_positives = len(allClusters[(allClusters["pure"] == 1) & allClusters["linked_object"].isin(falsePositiveIDs)])
    num_partial_false_positives = len(allClusters[(allClusters["partial"] == 1) & allClusters["linked_object"].isin(falsePositiveIDs)])
    
    # Cluster break down for everything else
    num_false = len(allClusters[allClusters["false"] == 1])
    num_total = len(allClusters)
    num_duplicate_visits = len(allClusters[allClusters["num_obs"] != allClusters["num_visits"]])

    if verbose == True:
        print("Known object pure clusters: {}".format(num_pure_known))
        print("Known object partial clusters: {}".format(num_partial_known))
        print("Unknown object pure clusters: {}".format(num_pure_unknown))
        print("Unknown object partial clusters: {}".format(num_partial_unknown))
        print("False positive pure clusters: {}".format(num_pure_false_positives))
        print("False positive partial clusters: {}".format(num_partial_false_positives))
        print("Duplicate visit clusters: {}".format(num_duplicate_visits))
        print("False clusters: {}".format(num_false))
        print("Total clusters: {}".format(num_total))
        print("Cluster contamination (%): {:1.3f}".format(num_false / num_total * 100))
    
    known_found = allObjects[(allObjects["found"] == 1) 
                       & (~allObjects[columnMapping["name"]].isin(unknownIDs + falsePositiveIDs))]
    known_missed = allObjects[(allObjects["found"] == 0) 
                        & (allObjects["findable"] == 1) 
                        & (~allObjects[columnMapping["name"]].isin(unknownIDs + falsePositiveIDs))]
    completeness = len(known_found) / (len(known_found) + len(known_missed)) * 100
    
    summary["num_unique_known_objects_found"] = len(known_found)
    summary["num_unique_known_objects_missed"] = len(known_missed)
    summary["percent_completeness"] = completeness
    summary["num_known_object_pure_clusters"] = num_pure_known
    summary["num_known_object_partial_clusters"] = num_partial_known
    summary["num_unknown_object_pure_clusters"] = num_pure_unknown
    summary["num_unknown_object_partial_clusters"] = num_partial_unknown
    summary["num_false_positive_pure_clusters"] = num_pure_false_positives
    summary["num_false_positive_partial_clusters"] = num_partial_false_positives
    summary["num_duplicate_visit_clusters"] = num_duplicate_visits
    summary["num_false_clusters"] = num_false
    summary["num_total_clusters"] = num_total
    
    if verbose == True:
        time_end = time.time()
        print("Unique known objects linked: {}".format(len(known_found)))
        print("Unique known objects missed: {}".format(len(known_missed)))
        print("Completeness (%): {:1.3f}".format(completeness))
        print("Done.")
        print("Total time in seconds: {}".format(time_end - time_start))
        
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