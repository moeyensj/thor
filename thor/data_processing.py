import time
import numpy as np
import pandas as pd

from .config import Config
from .cell import Cell
from .orbits import propagateOrbits

__all__ = ["findExposureTimes",
           "findAverageOrbits",
           "grabLinkedDetections"]
   
def findExposureTimes(observations, 
                      r, 
                      v, 
                      mjd, 
                      numNights=14, 
                      dMax=20.0, 
                      observatoryCode=Config.oorbObservatoryCode,
                      verbose=True,
                      columnMapping=Config.columnMapping):
    
    """
    Finds the unique exposure times of all detections that fall within
    a maximum search radius (set by a maximum angular velocity) for a 
    test particle defined by r and v at mjdStart. 
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing observations.
    r : `~numpy.ndarray` (1, 3)
        Heliocentric distance in AU.
    v : `~numpy.ndarray` (1, 3)
        Velocity vector in AU per day (ecliptic).  
    mjd : float
        Epoch at which ecliptic coordinates and velocity are measured in MJD. 
    numNights : int, optional
        List of nights at which to calculate exposure times. 
        [Default = 14]
    dMax : float, optional
        Maximum angular distance (in RA and Dec) permitted when searching for exposure times
        in degrees. 
        [Default = 20.0]
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
    mjds : `~numpy.ndarray`
        Sorted unique exposure times. 
    """
    
    if verbose == True:
        print("THOR: findExposureTimes")
        print("-------------------------")
        print("Generating particle ephemeris for the middle of every night.")
        print("Finding optimal exposure times (with maximum angular distance of {} degrees)...".format(dMax))
        print("")
    
    time_start = time.time()
    # Grab exposure times and night values, insure they are sorted
    # then grab the night corresponding to exposure time closest to
    # mjd start, use that value to calculate the appropriate array
    # of nights to use and their exposure times
    times_nights = observations[observations[columnMapping["exp_mjd"]] > mjd][[columnMapping["exp_mjd"], columnMapping["night"]]]
    times_nights.sort_values(by=columnMapping["exp_mjd"], inplace=True)
    nightStart = times_nights[times_nights[columnMapping["exp_mjd"]] >= mjd][columnMapping["night"]].values[0]
    times = np.unique(times_nights[(times_nights[columnMapping["night"]] >= nightStart) 
                         & (times_nights[columnMapping["night"]] <= nightStart + numNights)][columnMapping["exp_mjd"]].values)
    
    eph = propagateOrbits([*r, *v], mjd, times, elementType="cartesian", mjdScale="UTC", observatoryCode=observatoryCode)
    eph.rename(columns={"RA_deg": "RA_deg_orbit", "Dec_deg": "Dec_deg_orbit"}, inplace=True)
    
    df = pd.merge(observations[[columnMapping["obs_id"],
                                columnMapping["exp_mjd"], 
                                columnMapping["RA_deg"],
                                columnMapping["Dec_deg"]]],
                  eph[["mjd", "RA_deg_orbit", "Dec_deg_orbit"]], 
                  how='inner', 
                  left_on=columnMapping["exp_mjd"], 
                  right_on="mjd")

    dRA = df[columnMapping["RA_deg"]] - df["RA_deg_orbit"]
    dDec = df[columnMapping["Dec_deg"]] - df["Dec_deg_orbit"]
    dec = np.mean(df[[columnMapping["Dec_deg"], "Dec_deg_orbit"]].values, axis=1)
    d = np.sqrt((dRA * np.cos(np.radians(dec)))**2 + dDec**2)
    indexes = np.where(d <= dMax)[0]
    mjds = np.unique(df["exp_mjd"].values[indexes])
    
    time_end = time.time()
    if verbose == True:
        print("Done. Found {} unique exposure times.".format(len(mjds)))
        print("Total time in seconds: {}".format(time_end - time_start))
        print("-------------------------")
        print("")
  
    return mjds


def findAverageOrbits(observations,
                      orbits,
                      dValues=None,
                      elementType="keplerian",
                      unknownIDs=Config.unknownIDs,
                      falsePositiveIDs=Config.falsePositiveIDs,
                      verbose=True,
                      columnMapping=Config.columnMapping):
    """
    Find the object with observations that represents 
    the most average in terms of cartesian velocity and the
    heliocentric distance.
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing observations.
    orbits : `~pandas.DataFrame`
        DataFrame containing orbits for each unique object in observations.
    dValues : {list (N>=2), None}, optional
        If None, will find average orbit in all of observations. If a list, will find an 
        average orbit between each value in the list. For example, passing dValues = [1.0, 2.0, 4.0] will
        mean an average orbit will be found in the following bins: (1.0 <= d < 2.0), (2.0 <= d < 4.0).
    elementType : {'keplerian', 'cartesian'}, optional
        Find average orbits using which elements. If 'keplerian' will use a-e-i for average, 
        if 'cartesian' will use r, v. 
        [Default = 'keplerian']
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
    orbits : `~pandas.DataFrame` 
        DataFrame with name, r, v, exposure time, and sky-plane location of the average orbit in each bin of r. 
    """
    if verbose == True:
        print("THOR: findAverageObject")
        print("-------------------------")
        
    if elementType == "keplerian":
        dColumn = columnMapping["a_au"]
    elif elementType == "cartesian":
        dColumn = columnMapping["r_au"]
    else:
        raise ValueError("elementType should be one of {'keplerian', 'cartesian'}")
        
    dataframe = pd.merge(orbits, observations, on=columnMapping["name"])
    dataframe.reset_index(inplace=True, drop=True)
        
    dBins = []
    if dValues != None:
        if verbose == True:
            print("Finding average orbit in {} distance bins...".format(len(dValues) - 1))
        for d_i, d_f in zip(dValues[:-1], dValues[1:]):
            dBins.append(dataframe[(dataframe[dColumn] >= d_i) & (dataframe[dColumn] < d_f)])
    else: 
        if verbose == True:
            print("Finding average orbit...")
        dBins.append(dataframe)
    
    average_orbits = []
    
    for i, obs in enumerate(dBins):
        objects = obs[~obs[columnMapping["name"]].isin(falsePositiveIDs + unknownIDs)]

        if len(objects) == 0:
            # No real objects
            if verbose == True:
                print("No real objects found.")
            
            orbit = pd.DataFrame({"orbit_id" : i + 1,
                columnMapping["r_au"] : np.NaN,
                columnMapping["obj_dx/dt_au_p_day"] : np.NaN,
                columnMapping["obj_dy/dt_au_p_day"] : np.NaN,
                columnMapping["obj_dz/dt_au_p_day"] : np.NaN,
                columnMapping["exp_mjd"] : np.NaN,
                columnMapping["RA_deg"] : np.NaN,
                columnMapping["Dec_deg"] : np.NaN,
                columnMapping["a_au"] : np.NaN,
                columnMapping["i_deg"] : np.NaN,
                columnMapping["e"] : np.NaN,
                columnMapping["name"]: np.NaN}, index=[0])
            average_orbits.append(orbit)
            continue
            
        if elementType == "cartesian":

            rv = objects[[
                columnMapping["obj_dx/dt_au_p_day"],
                columnMapping["obj_dy/dt_au_p_day"],
                columnMapping["obj_dz/dt_au_p_day"],
                columnMapping["r_au"]
            ]].values

            # Calculate the percent difference between the median of each velocity element
            # and the heliocentric distance
            percent_diff = np.abs((rv - np.median(rv, axis=0)) / np.median(rv, axis=0))
            
        else:
            aie = objects[[columnMapping["a_au"], 
                           columnMapping["i_deg"], 
                           columnMapping["e"]]].values

            # Calculate the percent difference between the median of each velocity element
            # and the heliocentric distance
            percent_diff = np.abs((aie - np.median(aie, axis=0)) / np.median(aie, axis=0))

        
        # Sum the percent differences
        summed_diff = np.sum(percent_diff, axis=1)

        # Find the minimum summed percent difference and call that 
        # the average object
        index = np.where(summed_diff == np.min(summed_diff))[0][0]
        name = obs[columnMapping["name"]].values[index]

        # Grab the objects, name and its r and v.
        obj_observations = obs[obs[columnMapping["name"]] == name]
        obj = obj_observations[[
            columnMapping["exp_mjd"],
            columnMapping["r_au"], 
            columnMapping["obj_dx/dt_au_p_day"],
            columnMapping["obj_dy/dt_au_p_day"],
            columnMapping["obj_dz/dt_au_p_day"],
            columnMapping["RA_deg"],
            columnMapping["Dec_deg"],
            columnMapping["a_au"],
            columnMapping["i_deg"],
            columnMapping["e"],
            columnMapping["name"]]].copy()
        obj["orbit_id"] = i + 1
        
        average_orbits.append(obj[["orbit_id", 
            columnMapping["r_au"], 
            columnMapping["obj_dx/dt_au_p_day"],
            columnMapping["obj_dy/dt_au_p_day"],
            columnMapping["obj_dz/dt_au_p_day"],
            columnMapping["exp_mjd"],
            columnMapping["RA_deg"],
            columnMapping["Dec_deg"],
            columnMapping["a_au"],
            columnMapping["i_deg"],
            columnMapping["e"],
            columnMapping["name"]]])
        
    average_orbits = pd.concat(average_orbits)
    average_orbits.sort_values(by=["orbit_id", columnMapping["exp_mjd"]], inplace=True)
    average_orbits.reset_index(inplace=True, drop=True)
    
    if verbose == True:    
        print("Done.")
        print("-------------------------")
        print("")
    return average_orbits


def grabLinkedDetections(observations, 
                         allClusters, 
                         clusterMembers, 
                         verbose=True,
                         columnMapping=Config.columnMapping):
    """
    Grabs linked observations from pure and partial clusters.
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing post-range and shift observations.
    allClusters : `~pandas.DataFrame`
        DataFrame with the cluster ID, the number of observations, and the x and y velocity. 
    clusterMembers : `~pandas.DataFrame`
        DataFrame containing the cluster ID and the observation IDs of its members. 
    verbose : bool, optional
        Print progress statements? 
        [Default = True]
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~thor.Config.columnMapping`]
    
    Returns
    -------
    `~numpy.ndarray` (N)
        Observation IDs that have been linked in pure and partial clusters. (Excluded imposter
        observations in partial clusters)
    """
    if verbose == True:
        print("THOR: grabLinkedDetections")
        print("-------------------------")
    pure_clusters = allClusters[allClusters["pure"] == 1]["cluster_id"].values
    pure_obs_ids_linked = clusterMembers[clusterMembers["cluster_id"].isin(pure_clusters)][columnMapping["obs_id"]].values

    partial_clusters = allClusters[allClusters["partial"] == 1]["cluster_id"].values
    cluster_designation = clusterMembers[clusterMembers["cluster_id"].isin(partial_clusters)].merge(observations[[columnMapping["obs_id"], columnMapping["name"]]], 
                                                                                                    left_on=columnMapping["obs_id"], 
                                                                                                    right_on=columnMapping["obs_id"])
    cluster_designation = cluster_designation.merge(allClusters[["cluster_id", "linked_object"]])
    partial_obs_ids_linked = cluster_designation[cluster_designation[columnMapping["name"]] == cluster_designation["linked_object"]][columnMapping["obs_id"]].values
    
    linked_obs_ids = np.concatenate([pure_obs_ids_linked, partial_obs_ids_linked])
    linked_obs_ids = np.unique(linked_obs_ids)
    if verbose == True:
        print("{} detections have been linked.".format(len(linked_obs_ids)))
        print("-------------------------")
        print("")
    return linked_obs_ids
    
    