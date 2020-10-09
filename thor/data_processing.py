import time
import numpy as np
import pandas as pd

from .config import Config
from .cell import Cell
from .orbits.propagate import propagateOrbits

__all__ = [
    "preprocessObservations",
    "findAverageOrbits",
]

def preprocessObservations(observations, column_mapping, mjd_scale="utc", attribution=False):
    """
    Create two seperate data frames: one with all observation data needed to run THOR stripped of
    object IDs and the other with known object IDs and attempts to attribute unknown observations to
    the latest catalog of known objects from the MPC.
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing at minimum a column of observation IDs, exposure times in MJD (with scale
        set by mjd_scale), RA in degrees, Dec in degrees, 1-sigma error in RA in degrees, 1-sigma error in 
        Dec in degrees and the observatory code. 
    column_mapping : dict
        Dictionary containing internal column names as keys mapped to column names in the data frame as values.
        Should include the following:
        {# Internal : # External
            "obs_id" :                    # Observation IDs 
            "mjd" :                       # MJDs (set scale with mjd_scale)
            "RA_deg" :                    # RA in degrees (topocentric J2000)
            "Dec_deg" :                   # Dec in degrees (topocentric J2000)
            "RA_sigma_deg" :              # 1-sigma error in RA in degrees 
            "Dec_sigma_deg" :             # 1-sigma error in Dec in degrees 
            "observatory_code" :          # MPC observatory code
            "obj_id" :                    # Object ID (designation) or NaN if unknown
        }
    mjd_scale : str, optional
        Time scale of the input MJD exposure times ("utc", "tdb", etc...)
    attribution : bool, optional
        Place holder boolean to trigger attribution
    
    Returns
    -------
    preprocessed_observations : `~pandas.DataFrame`
        DataFrame with observations in the format required by THOR.
    preprocessed_attributions : `~pandas.DataFrame`
        DataFrame containing truths.
    """
    obs_columns = [
        column_mapping["obs_id"],
        column_mapping["mjd"],
        column_mapping["RA_deg"],
        column_mapping["Dec_deg"],
        column_mapping["RA_sigma_deg"],
        column_mapping["Dec_sigma_deg"],
        column_mapping["observatory_code"],
    ]
    attrib_columns = [
        column_mapping["obs_id"],
        column_mapping["obj_id"]
    ]
    column_mapping_inv = {v : k for k, v in column_mapping.items()}
    
    for col in obs_columns:
        if col not in observations.columns:
            err = (
                "{} not found in observations.".format(col)
            )
            raise ValueError(err)
            
    preprocessed_observations = observations[obs_columns].copy()
    preprocessed_observations.rename(columns=column_mapping_inv, inplace=True)
    preprocessed_observations.reset_index(inplace=True, drop=True)
    preprocessed_observations["obs_id"] = preprocessed_observations["obs_id"].astype(str)

    if mjd_scale != "utc":
        observation_times = Time(preprocessed_observations[column_mapping["mjd_utc"]].values, format="mjd", scale=mjd_scale)
        preprocessed_observations["mjd_utc"] = observation_times.utc.mjd
    preprocessed_observations.rename(columns={"mjd":"mjd_utc"}, inplace=True)
        
    preprocessed_attributions = observations[attrib_columns].copy()
    preprocessed_attributions.rename(columns=column_mapping_inv, inplace=True)
    preprocessed_attributions.reset_index(inplace=True, drop=True)
    
    preprocessed_attributions["obj_id"] = preprocessed_attributions["obj_id"].astype(str)
    preprocessed_attributions["obs_id"] = preprocessed_attributions["obs_id"].astype(str)

    return preprocessed_observations, preprocessed_attributions

def findAverageOrbits(observations,
                      orbits,
                      d_values=None,
                      element_type="keplerian",
                      verbose=True,
                      column_mapping=Config.COLUMN_MAPPING):
    """
    Find the object with observations that represents 
    the most average in terms of cartesian velocity and the
    heliocentric distance. Assumes that a subset of the designations in the orbits 
    dataframe are identical to at least some of the designations in the observations 
    dataframe. No propagation is done, so the orbits need to be defined at an epoch near
    the time of observations, for example like the midpoint or start of a two-week window. 
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing observations.
    orbits : `~pandas.DataFrame`
        DataFrame containing orbits for each unique object in observations.
    d_values : {list (N>=2), None}, optional
        If None, will find average orbit in all of observations. If a list, will find an 
        average orbit between each value in the list. For example, passing dValues = [1.0, 2.0, 4.0] will
        mean an average orbit will be found in the following bins: (1.0 <= d < 2.0), (2.0 <= d < 4.0).
    element_type : {'keplerian', 'cartesian'}, optional
        Find average orbits using which elements. If 'keplerian' will use a-e-i for average, 
        if 'cartesian' will use r, v. 
        [Default = 'keplerian']
    verbose : bool, optional
        Print progress statements? 
        [Default = True]
    column_mapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~thor.Config.COLUMN_MAPPING`] 
    
    Returns
    -------
    orbits : `~pandas.DataFrame` 
        DataFrame with name, r, v, exposure time, and sky-plane location of the average orbit in each bin of r. 
    """
    if verbose == True:
        print("THOR: findAverageObject")
        print("-------------------------")
        
    if element_type == "keplerian":
        d_col = column_mapping["a_au"]
    elif element_type == "cartesian":
        d_col = column_mapping["r_au"]
    else:
        err = (
            "element_type should be one of {'keplerian', 'cartesian'}"
        )
        raise ValueError(err)
        
    dataframe = pd.merge(orbits, observations, on=column_mapping["name"])
    dataframe.reset_index(inplace=True, drop=True)
        
    d_bins = []
    if d_values != None:
        if verbose == True:
            print("Finding average orbit in {} distance bins...".format(len(d_values) - 1))
        for d_i, d_f in zip(d_values[:-1], d_values[1:]):
            d_bins.append(dataframe[(dataframe[d_col] >= d_i) & (dataframe[d_col] < d_f)])
    else: 
        if verbose == True:
            print("Finding average orbit...")
        d_bins.append(dataframe)
    
    average_orbits = []
    
    for i, obs in enumerate(d_bins):
        if len(obs) == 0:
            # No real objects
            if verbose == True:
                print("No real objects found.")
            
            orbit = pd.DataFrame({"orbit_id" : i + 1,
                column_mapping["exp_mjd"] : np.NaN,
                column_mapping["obj_x_au"] : np.NaN,
                column_mapping["obj_y_au"] : np.NaN,
                column_mapping["obj_z_au"] : np.NaN,
                column_mapping["obj_dx/dt_au_p_day"] : np.NaN,
                column_mapping["obj_dy/dt_au_p_day"] : np.NaN,
                column_mapping["obj_dz/dt_au_p_day"] : np.NaN,
                column_mapping["RA_deg"] : np.NaN,
                column_mapping["Dec_deg"] : np.NaN,
                column_mapping["r_au"] : np.NaN,
                column_mapping["a_au"] : np.NaN,
                column_mapping["i_deg"] : np.NaN,
                column_mapping["e"] : np.NaN,
                column_mapping["name"]: np.NaN}, index=[0])
            average_orbits.append(orbit)
            continue
            
        if element_type == "cartesian":

            rv = obs[[
                column_mapping["obj_dx/dt_au_p_day"],
                column_mapping["obj_dy/dt_au_p_day"],
                column_mapping["obj_dz/dt_au_p_day"],
                column_mapping["r_au"]
            ]].values

            # Calculate the percent difference between the median of each velocity element
            # and the heliocentric distance
            percent_diff = np.abs((rv - np.median(rv, axis=0)) / np.median(rv, axis=0))
            
        else:
            aie = obs[[column_mapping["a_au"], 
                       column_mapping["i_deg"], 
                       column_mapping["e"]]].values

            # Calculate the percent difference between the median of each velocity element
            # and the heliocentric distance
            percent_diff = np.abs((aie - np.median(aie, axis=0)) / np.median(aie, axis=0))

        
        # Sum the percent differences
        summed_diff = np.sum(percent_diff, axis=1)

        # Find the minimum summed percent difference and call that 
        # the average object
        index = np.where(summed_diff == np.min(summed_diff))[0][0]
        name = obs[column_mapping["name"]].values[index]

        # Grab the objects, name and its r and v.
        obj_observations = obs[obs[column_mapping["name"]] == name]
        obj = obj_observations[[
            column_mapping["exp_mjd"],
            column_mapping["obj_x_au"],
            column_mapping["obj_y_au"],
            column_mapping["obj_z_au"],
            column_mapping["obj_dx/dt_au_p_day"],
            column_mapping["obj_dy/dt_au_p_day"],
            column_mapping["obj_dz/dt_au_p_day"],
            column_mapping["RA_deg"],
            column_mapping["Dec_deg"],
            column_mapping["r_au"], 
            column_mapping["a_au"],
            column_mapping["i_deg"],
            column_mapping["e"],
            column_mapping["name"]]].copy()
        obj["orbit_id"] = i + 1
        
        average_orbits.append(obj[["orbit_id", 
            column_mapping["exp_mjd"],
            column_mapping["obj_x_au"],
            column_mapping["obj_y_au"],
            column_mapping["obj_z_au"],
            column_mapping["obj_dx/dt_au_p_day"],
            column_mapping["obj_dy/dt_au_p_day"],
            column_mapping["obj_dz/dt_au_p_day"],
            column_mapping["RA_deg"],
            column_mapping["Dec_deg"],
            column_mapping["r_au"], 
            column_mapping["a_au"],
            column_mapping["i_deg"],
            column_mapping["e"],
            column_mapping["name"]]])
        
    average_orbits = pd.concat(average_orbits)
    average_orbits.sort_values(by=["orbit_id", column_mapping["exp_mjd"]], inplace=True)
    average_orbits.reset_index(inplace=True, drop=True)
    
    if verbose == True:    
        print("Done.")
        print("-------------------------")
        print("")
    return average_orbits
