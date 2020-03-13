import os
import sys
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
from astropy.time import Time
from itertools import combinations
from functools import partial

from ...config import Config
from ...constants import Constants as c
from ..ephemeris import generateEphemeris
from .gauss import gaussIOD

__all__ = [
    "selectObservations",
    "iod"
]

MU = c.G * c.M_SUN
THOR_EPHEMERIS_KWARGS = {
    "light_time" : True, 
    "lt_tol" : 1e-10,
    "mu" : MU,
    "max_iter" : 1000, 
    "tol" : 1e-16
}

PYOORB_EPHEMERIS_KWARGS = {
    "orbit_type" : "cartesian", 
    "time_scale" : "TT", 
    "magnitude" : 20, 
    "slope" : 0.15, 
    "dynamical_model" : "N",
    "ephemeris_file" : "de430.dat"
}


def selectObservations(observations, method="combinations", column_mapping=Config.columnMapping):
    """
    Selects which three observations to use for IOD depending on the method. 
    
    Methods:
        'first+middle+last' : Grab the first, middle and last observations in time. 
        'thirds' : Grab the middle observation in the first third, second third, and final third. 
        'combinations' : Return the observation IDs corresponding to every possible combination of three observations with
            non-coinciding observation times.
    
    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Pandas DataFrame containing observations with at least a column of observation IDs and a column
        of exposure times. 
    method : {'first+middle+last', 'thirds', 'combinations'}, optional
        Which method to use to select observations. 
        [Default = 'combinations']
    column_mapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~thor.Config.columnMapping`]
        
    Returns
    -------
    obs_id : `~numpy.ndarray' (N, 3 or 0)
        An array of selected observation IDs. If three unique observations could 
        not be selected then returns an empty array. 
    """
    obs_ids = observations[column_mapping["obs_id"]].values
    if len(obs_ids) < 3:
        return np.array([])
    
    indexes = np.arange(0, len(obs_ids))
    times = observations[column_mapping["exp_mjd"]].values
    selected = np.array([])

    if method == "first+middle+last":
        selected_times = np.percentile(times, 
                                 [0, 50, 100], 
                                 interpolation="nearest")
        selected_index = np.intersect1d(times, selected_times, return_indices=True)[1]
        selected_index = np.array([selected_index])
        
    elif method == "thirds":
        selected_times = np.percentile(times, 
                                 [1/6*100, 50, 5/6*100], 
                                 interpolation="nearest")
        selected_index = np.intersect1d(times, selected_times, return_indices=True)[1]
        selected_index = np.array([selected_index])
        
    elif method == "combinations":
        # Make all possible combinations of 3 observations
        selected_index = np.array([np.array(index) for index in combinations(indexes, 3)])

        # Calculate arc length
        arc_length = times[selected_index][:, 2] - times[selected_index][:, 0]

        # Calculate distance of second observation from middle point (last + first) / 2
        time_from_mid = np.abs((times[selected_index][:, 2] + times[selected_index][:, 0])/2 - times[selected_index][:, 1])

        # Sort by descending arc length and ascending time from midpoint
        sort = np.lexsort((time_from_mid, -arc_length))
        selected_index = selected_index[sort]
        
    else:
        raise ValueError("method should be one of {'first+middle+last', 'thirds'}")
    
    # Make sure each returned combination of observation ids have at least 3 unique
    # times 
    keep = []
    for i, comb in enumerate(times[selected_index]):
        if len(np.unique(comb)) == 3:
            keep.append(i)
    keep = np.array(keep)
    
    # Return an empty array if no observations satisfy the criteria
    if len(keep) == 0:
        return np.array([])
    else:
        selected_index = selected_index[keep, :]
    
    return obs_ids[selected_index]

def iod(observations,
        observation_selection_method="combinations",
        chi2_threshold=10**3,
        contamination_percentage=20.0,
        iterate=True, 
        light_time=True,
        backend="THOR",
        backend_kwargs=None,
        column_mapping=Config.columnMapping):
    """
    Run initial orbit determination on a set of observations believed to belong to a single
    object. 

    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Data frame containing the observations of a possible linkage. 
    observation_selection_method : {'first+middle+last', 'thirds', 'combinations'}, optional
        Which method to use to select observations. 
        [Default = 'combinations']
    chi2_threshold : float, optional
        Minimum chi2 required for an initial orbit to be accepted. Note that chi2 here needs to be 
        interpreted carefully, residuals of order a few arcseconds easily contribute significantly 
        to the total chi2. 
    contamination_percentage : float, optional
        Maximum percent of observations that can flagged as outliers. 
    iterate : bool, optional
        Iterate the preliminary orbit solution using the state transition iterator. 
    light_time : bool, optional
        Correct preliminary orbit for light travel time.
    backend : {'THOR', 'PYOORB'}, optional
        Which backend to use for ephemeris generation.
    backend_kwargs : dict, optional
        Settings and additional parameters to pass to selected 
        backend.
    column_mapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~thor.Config.columnMapping`]

    Returns
    -------
    orbit : `~pandas.DataFrame` (7)
        Preliminary orbit with epoch_mjd (in UTC) and the the cartesian state vector. Also
        has a column for number of observations after outliers have been removed, arc length
        of the remaining observations and the value of chi2 for the solution.
    orbit_members : `~pandas.DataFrame` (3)
        Data frame with two columns: orbit_id and observation IDs.
    outliers : `~numpy.ndarray`, None
        Observation IDs of potential outlier detections.
    """
    # Extract column names
    obs_id_col = column_mapping["obs_id"]
    ra_col = column_mapping["RA_deg"]
    dec_col = column_mapping["Dec_deg"]
    time_col = column_mapping["exp_mjd"]
    ra_err_col = column_mapping["RA_sigma_deg"]
    dec_err_col = column_mapping["Dec_sigma_deg"]
    obs_code_col = column_mapping["observatory_code"]
    obs_x_col = column_mapping["obs_x_au"]
    obs_y_col = column_mapping["obs_y_au"]
    obs_z_col = column_mapping["obs_z_au"]
    
    # Extract observation IDs, sky-plane positions, sky-plane position uncertainties, times of observation,
    # and the location of the observer at each time
    obs_ids_all = observations[obs_id_col].values
    coords_all = observations[[ra_col, dec_col]].values
    ra = observations[ra_col].values
    dec = observations[dec_col].values
    sigma_ra = observations[ra_err_col].values
    sigma_dec = observations[dec_err_col].values

    coords_obs_all = observations[[ obs_x_col, obs_y_col, obs_z_col]].values
    times_all = observations[time_col].values
    times_all = Time(times_all, scale="utc", format="mjd")
    
    if backend == "THOR":
        if backend_kwargs == None:
            backend_kwargs = THOR_EPHEMERIS_KWARGS
        backend_kwargs["light_time"] = light_time

        observers = observations[[obs_code_col, time_col, obs_x_col, obs_y_col, obs_z_col]]
        
    elif backend == "PYOORB":
        if light_time == False:
            err = (
                "PYOORB does not support turning light time correction off."
            )
            raise ValueError(err)
        
        if backend_kwargs == None:
            backend_kwargs = PYOORB_EPHEMERIS_KWARGS

        observers = {}
        for code in observations[obs_code_col].unique():
            observers[code] = Time(
                observations[observations[obs_code_col] == code][time_col].values,
                scale="utc",
                format="mjd"
            )
    else:
        err = (
            "backend should be one of 'THOR' or 'PYOORB'"
        )
        raise ValueError(err)

    chi2_sol = 1e10
    orbit_sol = None
    obs_ids_sol = None
    remaining_ids = None
    arc_length = None
    outliers = np.array([])
    converged = False
    num_obs = len(observations)
    num_outliers = int(num_obs * contamination_percentage / 100.)
    
    orbit = pd.DataFrame(
            columns=[
                "orbit_id",
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
            ]
        )
    orbit_members = pd.DataFrame(
        columns=[
            "orbit_id",
            "obs_id"
        ]
    )

    # Select observation IDs to use for IOD
    obs_ids = selectObservations(
        observations, 
        method=observation_selection_method, 
        column_mapping=column_mapping
    )
    if len(obs_ids) == 0:
        return orbit, orbit_members, outliers

    j = 0
    while not converged:
        if j == len(obs_ids):
            break

        ids = obs_ids[j]
        mask = np.isin(obs_ids_all, ids)
       
        # Grab sky-plane positions of the selected observations, the heliocentric ecliptic position of the observer,
        # and the times at which the observations occur
        coords = coords_all[mask, :]
        coords_obs = coords_obs_all[mask, :]
        times = times_all[mask]

        # Run IOD 
        orbits_iod = gaussIOD(
            coords, 
            times.utc.mjd, 
            coords_obs, 
            light_time=light_time, 
            iterate=iterate, 
            max_iter=100, 
            tol=1e-15
        )
        if len(orbits_iod) == 0:
            j += 1
            continue
        
        # Propagate initial orbit to all observation times
        ephemeris = generateEphemeris(
            orbits_iod[:, 1:], 
            Time(orbits_iod[:, 0], scale="utc", format="mjd"),
            observers,
            backend=backend, 
            backend_kwargs=backend_kwargs
        )
        ephemeris = ephemeris[['orbit_id', 'mjd_utc', 'RA_deg', 'Dec_deg', 'obj_x', 'obj_y', 'obj_z', 'obj_vx', 'obj_vy', 'obj_vz']].values
        
        # For each unique initial orbit calculate residuals and chi-squared
        # Find the orbit which yields the lowest chi-squared
        orbit_ids = np.unique(ephemeris[:, 0])
        for i, orbit_id in enumerate(orbit_ids):
            orbit_name = np.array(["{}_v{}".format("_".join(ids.astype(str)), i+1)])
            # Select unique orbit solution
            orbit_i = ephemeris[np.where(ephemeris[:, 0] == orbit_id)]

            # Calculate residuals in RA, make sure to fix any potential wrap around errors
            pred_dec = np.radians(orbit_i[:, 3])
            residual_ra = (ra - orbit_i[:, 2]) * np.cos(pred_dec)
            residual_ra = np.where(residual_ra > 180., 360. - residual_ra, residual_ra)

            # Calculate residuals in Dec
            residual_dec = dec - orbit_i[:, 3]

            # Calculate chi2
            chi2 = ((residual_ra**2 / sigma_ra**2) 
                + (residual_dec**2 / sigma_dec**2))
            chi2_total = np.sum(chi2)

            # All chi2 values are above the threshold, continue loop
            if np.all(chi2 >= chi2_threshold):
                continue

            # If the total chi2 is less than the threshold accept the orbit
            elif chi2_total <= chi2_threshold:
                orbit_sol = orbits_iod[i:i+1, :]
                obs_ids_sol = ids
                chi2_sol = chi2_total    
                remaining_ids = obs_ids_all
                arc_length = times_all.max() - times_all.min()
                converged = True

            # Let's now test to see if we can remove some outliers, we 
            # anticipate we get to this stage if the three selected observations 
            # belonging to one object yield a good initial orbit but the presence of outlier
            # observations is skewing the sum total of the residuals and chi2
            elif num_outliers > 0:
                for o in range(num_outliers):
                    # Select i highest observations that contribute to 
                    # chi2 (and thereby the residuals)
                    remove = chi2[~mask].argsort()[-(o+1):]
                    
                    # Grab the obs_ids for these outliers
                    obs_id_outlier = obs_ids_all[~mask][remove]

                    # Subtract the outlier's chi2 contribution 
                    # from the total chi2 
                    chi2_new = chi2_total - np.sum(chi2[~mask][remove])
                    
                    # If the updated chi2 total is lower than our desired
                    # threshold, accept the soluton. If not, keep going.
                    if chi2_new <= chi2_threshold:
                        orbit_sol = orbits_iod[i:i+1, :]
                        obs_ids_sol = ids
                        chi2_sol = chi2_new
                        outliers = obs_id_outlier
                        num_obs = len(observations) - len(remove)
                        ids_mask = np.isin(obs_ids_all, outliers, invert=True)
                        arc_length = times_all[ids_mask].max() - times_all[ids_mask].min()
                        remaining_ids = obs_ids_all[ids_mask]
                        converged = True
                        break
                        
            else:
                continue

        j += 1

    if converged:
        orbit = pd.DataFrame(
            orbit_sol,
            columns=[
                "epoch_mjd_utc", 
                "obj_x", 
                "obj_y", 
                "obj_z", 
                "obj_vx", 
                "obj_vy", 
                "obj_vz"])
        orbit["arc_length"] = arc_length
        orbit["num_obs"] = num_obs
        orbit["chi2"] = chi2_sol
        orbit.insert(0, "orbit_id", orbit_name)
        
        orbit_members = pd.DataFrame(
            np.vstack([
                [orbit_name[0] for i in range(num_obs)], 
                remaining_ids]).T,
            columns=[
                "orbit_id",
                "obs_id",
            ]
        )
    
    return orbit, orbit_members, outliers
   