import sys
import numpy as np
from itertools import combinations

from ...config import Config
from ...constants import Constants as c
from .gauss import gaussIOD
from ..propagate import propagateOrbits

__all__ = ["selectObservations",
           "iod"]

MU = c.G * c.M_SUN


def selectObservations(observations, method="combinations", columnMapping=Config.columnMapping):
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
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~thor.Config.columnMapping`]
        
    Returns
    -------
    obs_id : `~numpy.ndarray' (N, 3 or 0)
        An array of selected observation IDs. If three unique observations could 
        not be selected then returns an empty array. 
    """
    obs_ids = observations[columnMapping["obs_id"]].values
    if len(obs_ids) < 3:
        return np.array([])
    
    indexes = np.arange(0, len(obs_ids))
    times = observations[columnMapping["exp_mjd"]].values
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
    
    return obs_ids[selected_index[keep, :]]


def iod(observations,
        observation_selection_method="combinations",
        iterate=True, 
        light_time=True,
        max_iter=50, 
        tol=1e-15, 
        propagatorKwargs={
            "observatoryCode" : "I11",
            "mjdScale" : "UTC",
            "dynamical_model" : "2",
        },
        mu=MU, 
        columnMapping=Config.columnMapping):

    # Extract column names
    obs_id_col = columnMapping["obs_id"]
    ra_col = columnMapping["RA_deg"]
    dec_col = columnMapping["Dec_deg"]
    time_col = columnMapping["exp_mjd"]
    ra_err_col = columnMapping["RA_sigma_deg"]
    dec_err_col = columnMapping["Dec_sigma_deg"]
    obs_x_col = columnMapping["obs_x_au"]
    obs_y_col = columnMapping["obs_y_au"]
    obs_z_col = columnMapping["obs_z_au"]
    
    # Extract observation IDs, sky-plane positions, sky-plane position uncertainties, times of observation,
    # and the location of the observer at each time
    obs_ids_all = observations[obs_id_col].values
    coords_eq_ang_all = observations[observations[obs_id_col].isin(obs_ids_all)][[ra_col, dec_col]].values
    coords_eq_ang_err_all = observations[observations[obs_id_col].isin(obs_ids_all)][[ra_err_col, dec_err_col]].values
    coords_obs_all = observations[observations[obs_id_col].isin(obs_ids_all)][[obs_x_col, obs_y_col, obs_z_col]].values
    times_all = observations[observations[obs_id_col].isin(obs_ids_all)][time_col].values
    
    # Select observation IDs to use for IOD
    obs_ids = selectObservations(observations, method=observation_selection_method, columnMapping=columnMapping)

    min_chi2 = 1e10
    best_orbit = None
    best_obs_ids = None

    for ids in obs_ids:
        # Grab sky-plane positions of the selected observations, the heliocentric ecliptic position of the observer,
        # and the times at which the observations occur
        coords_eq_ang = observations[observations[obs_id_col].isin(ids)][[ra_col, dec_col]].values
        coords_obs = observations[observations[obs_id_col].isin(ids)][[obs_x_col, obs_y_col, obs_z_col]].values
        times = observations[observations[obs_id_col].isin(ids)][time_col].values

        # Run IOD 
        orbits_iod = gaussIOD(coords_eq_ang, times, coords_obs, light_time=light_time, iterate=iterate, max_iter=max_iter, tol=tol)
        if np.all(np.isnan(orbits_iod)) == True:
            continue

        # Propagate initial orbit to all observation times
        orbits = propagateOrbits(orbits_iod[:, 1:], orbits_iod[:, 0], times_all, **propagatorKwargs)
        if np.all(orbits.values) == 0.0:
            continue
        orbits = orbits[['orbit_id', 'mjd', 'RA_deg', 'Dec_deg', 
                         'HEclObj_X_au', 'HEclObj_Y_au', 'HEclObj_Z_au',
                         'HEclObj_dX/dt_au_p_day', 'HEclObj_dY/dt_au_p_day', 'HEclObj_dZ/dt_au_p_day']].values
        
        # For each unique initial orbit calculate residuals and chi-squared
        # Find the orbit which yields the lowest chi-squared
        orbit_ids = np.unique(orbits[:, 0])
        for i, orbit_id in enumerate(orbit_ids):
            orbit = orbits[np.where(orbits[:, 0] == orbit_id)]
            
            pred_dec = np.radians(orbit[:, 3])
            residual_ra = (coords_eq_ang_all[:, 0] - orbit[:, 2]) * np.cos(pred_dec)
            residual_dec =  (coords_eq_ang_all[:, 1] - orbit[:, 3])
        
            chi2 = np.sum(residual_ra**2 / coords_eq_ang_err_all[:, 0]**2 + residual_dec**2 / coords_eq_ang_err_all[:, 1]**2) / (2 * len(residual_ra) - 6)

            if chi2 < min_chi2:
                best_orbit = orbits_iod[i, :]
                best_obs_ids = ids
                min_chi2 = chi2
                
    return best_orbit, best_obs_ids, min_chi2
        
   