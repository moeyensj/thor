import os
import sys
import time
import uuid
import numpy as np
import pandas as pd
import multiprocessing as mp
from astropy.time import Time
from itertools import combinations
from functools import partial

from ..config import Config
from ..backend import _init_worker
from ..backend import MJOLNIR
from ..backend import PYOORB
from .gauss import gaussIOD
from .residuals import calcResiduals
from .ephemeris import generateEphemeris

USE_RAY = Config.USE_RAY
NUM_THREADS = Config.NUM_THREADS

__all__ = [
    "selectObservations",
    "iod",
    "iod_worker",
    "initialOrbitDetermination"
]

def selectObservations(observations, method="combinations"):
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
        
    Returns
    -------
    obs_id : `~numpy.ndarray' (N, 3 or 0)
        An array of selected observation IDs. If three unique observations could 
        not be selected then returns an empty array. 
    """
    obs_ids = observations["obs_id"].values
    if len(obs_ids) < 3:
        return np.array([])
    
    indexes = np.arange(0, len(obs_ids))
    times = observations["mjd_utc"].values
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

def iod_worker(
        observations,
        observation_selection_method="combinations",
        chi2_threshold=10**3,
        min_obs=6,
        contamination_percentage=0.0,
        iterate=False, 
        light_time=True,
        backend="PYOORB",
        backend_kwargs={}
    ):
    
    iod_orbit, iod_orbit_members = iod(
        observations,
        observation_selection_method=observation_selection_method,
        chi2_threshold=chi2_threshold,
        min_obs=min_obs,
        contamination_percentage=contamination_percentage,
        iterate=iterate, 
        light_time=light_time,
        backend=backend,
        backend_kwargs=backend_kwargs
    )
    iod_orbit.insert(1, "cluster_id", observations["cluster_id"].unique()[0])
    
    return iod_orbit, iod_orbit_members

if USE_RAY:
    import ray
    iod_worker = ray.remote(iod_worker)

def iod(
        observations,
        observation_selection_method="combinations",
        chi2_threshold=200,
        min_obs=6,
        contamination_percentage=0.0,
        iterate=False, 
        light_time=True,
        backend="PYOORB",
        backend_kwargs={}
    ):
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
    backend : {'MJOLNIR', 'PYOORB'}, optional
        Which backend to use for ephemeris generation.
    backend_kwargs : dict, optional
        Settings and additional parameters to pass to selected 
        backend.

    Returns
    -------
    orbit : `~pandas.DataFrame` (7)
        Preliminary orbit with epoch_mjd (in UTC) and the the cartesian state vector. Also
        has a column for number of observations after outliers have been removed, arc length
        of the remaining observations and the value of chi2 for the solution.
    orbit_members : `~pandas.DataFrame` (3)
        Data frame with two columns: orbit_id and observation IDs.
    """
    # Extract column names
    obs_id_col = "obs_id"
    time_col = "mjd_utc"
    ra_col = "RA_deg"
    dec_col = "Dec_deg"
    ra_err_col = "RA_sigma_deg"
    dec_err_col = "Dec_sigma_deg"
    obs_code_col = "observatory_code"
    obs_x_col = "obs_x"
    obs_y_col = "obs_y"
    obs_z_col = "obs_z"
    
    # Extract observation IDs, sky-plane positions, sky-plane position uncertainties, times of observation,
    # and the location of the observer at each time
    obs_ids_all = observations[obs_id_col].values
    coords_all = observations[[ra_col, dec_col]].values
    sigmas_all = observations[[ra_err_col, dec_err_col]].values
    coords_obs_all = observations[[obs_x_col, obs_y_col, obs_z_col]].values
    times_all = observations[time_col].values
    times_all = Time(times_all, scale="utc", format="mjd")
    
    observers = {}
    for code in observations[obs_code_col].unique():
        observers[code] = Time(
            observations[observations[obs_code_col] == code][time_col].values,
            scale="utc",
            format="mjd"
        )

    if backend == "MJOLNIR":
        backend_kwargs["light_time"] = light_time
        
        backend = MJOLNIR(**backend_kwargs)
        #observers = observations[[obs_code_col, time_col, obs_x_col, obs_y_col, obs_z_col]]
        
    elif backend == "PYOORB":
        if light_time == False:
            err = (
                "PYOORB does not support turning light time correction off."
            )
            raise ValueError(err)
            
        backend = PYOORB(**backend_kwargs)
    else:
        err = (
            "backend should be one of 'MJOLNIR' or 'PYOORB'"
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
    chi2_threshold = num_obs * chi2_threshold
    num_outliers = int(num_obs * contamination_percentage / 100.)
    num_outliers = np.minimum(num_obs - min_obs, num_outliers)
    
    # Select observation IDs to use for IOD
    obs_ids = selectObservations(
        observations, 
        method=observation_selection_method, 
    )
    
    processable = True
    if len(obs_ids) == 0:
        processable = False

    j = 0
    while not converged and processable:
        if j == len(obs_ids):
            break

        ids = obs_ids[j]
        mask = np.isin(obs_ids_all, ids)
       
        # Grab sky-plane positions of the selected observations, the heliocentric ecliptic position of the observer,
        # and the times at which the observations occur
        coords = coords_all[mask, :]
        coords_obs = coords_obs_all[mask, :]
        sigmas = sigmas_all[mask, :]
        times = times_all[mask]

        # Run IOD 
        iod_orbits = gaussIOD(
            coords, 
            times.utc.mjd, 
            coords_obs, 
            light_time=light_time, 
            iterate=iterate, 
            max_iter=100, 
            tol=1e-15
        )
        if len(iod_orbits) == 0:
            j += 1
            continue

        # Propagate initial orbit to all observation times
        ephemeris = backend.generateEphemeris(
            iod_orbits, 
            observers,
            threads=1,
        )
        

        # For each unique initial orbit calculate residuals and chi-squared
        # Find the orbit which yields the lowest chi-squared
        orbit_ids = iod_orbits.ids
        for i, orbit_id in enumerate(orbit_ids):
            orbit_name = "{}_v{}".format("_".join(ids.astype(str)), i+1)
            orbit_name = str(uuid.uuid4().hex)
            iod_orbits.ids[i] = orbit_name

            ephemeris_orbit = ephemeris[ephemeris["orbit_id"] == orbit_id]
            
            # Calculate residuals and chi2
            residuals, stats = calcResiduals(
                coords_all,
                ephemeris_orbit[["RA_deg", "Dec_deg"]].values,
                sigmas_actual=sigmas_all,
                include_probabilistic=False
            )
            chi2 = stats[0]
            chi2_total = np.sum(chi2)

            # All chi2 values are above the threshold, continue loop
            if np.all(chi2 >= chi2_threshold) and num_outliers > 0:
                continue

            # If the total chi2 is less than the threshold accept the orbit
            elif chi2_total <= chi2_threshold:
                orbit_sol = iod_orbits[i:i+1]
                obs_ids_sol = ids
                chi2_total_sol = chi2_total    
                chi2_sol = chi2
                residuals_sol = residuals
                remaining_ids = obs_ids_all
                outliers = np.array([])
                arc_length = times_all.utc.mjd.max() - times_all.utc.mjd.min()
                converged = True
                break 

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
                        orbit_sol = iod_orbits[i:i+1]
                        obs_ids_sol = ids
                        chi2_total_sol = chi2_new
                        chi2_sol = chi2
                        residuals_sol = residuals
                        outliers = obs_id_outlier
                        num_obs = len(observations) - len(remove)
                        ids_mask = np.isin(obs_ids_all, outliers, invert=True)
                        arc_length = times_all[ids_mask].utc.mjd.max() - times_all[ids_mask].utc.mjd.min()
                        remaining_ids = obs_ids_all[ids_mask]
                        converged = True
                        break
                        
            else:
                processable = False
                break

        j += 1

    if not converged or not processable:
       
        orbit = pd.DataFrame(
            columns=[
                "orbit_id",
                "mjd_tdb",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "arc_length",
                "num_obs",
                "chi2",
            ]
        )
        
        orbit_members = pd.DataFrame(
            columns=[
                "orbit_id", 
                "obs_id", 
                "residual_ra", 
                "residual_dec", 
                "chi2",
                "gauss_sol",
                "outlier"
            ]
        )
    
    else:
        orbit = orbit_sol.todf()
        orbit["arc_length"] = arc_length
        orbit["num_obs"] = num_obs
        orbit["chi2"] = chi2_total_sol
        orbit["rchi2"] = chi2_total_sol / (2 * num_obs - 6)
        
        orbit_members = pd.DataFrame({
            "orbit_id" : [orbit_sol.ids[0] for i in range(len(obs_ids_all))],
            "obs_id" : obs_ids_all,
            "residual_ra_arcsec" : residuals_sol[:, 0] * 3600,
            "residual_dec_arcsec" : residuals_sol[:, 1] * 3600,
            "chi2" : chi2_sol,
            "gauss_sol" : np.zeros(len(obs_ids_all), dtype=int),
            "outlier" : np.zeros(len(obs_ids_all), dtype=int)
        })
        orbit_members.loc[orbit_members["obs_id"].isin(outliers), "outlier"] = 1
        orbit_members.loc[orbit_members["obs_id"].isin(obs_ids_sol), "gauss_sol"] = 1
    
    
    return orbit, orbit_members


def initialOrbitDetermination(
        observations, 
        linkage_members, 
        observation_selection_method='combinations',
        chi2_threshold=10**3,
        min_obs=6,
        contamination_percentage=20.0,
        iterate=False,
        light_time=True,
        threads=NUM_THREADS,
        backend="PYOORB",
        backend_kwargs={}   
    ):

    linked_observations = linkage_members.merge(observations, on="obs_id").copy()
    linked_observations.sort_values(
        by=["cluster_id", "mjd_utc"], 
        inplace=True
    )
    linked_observations.reset_index(
        drop=True,
        inplace=True
    )
    grouped_observations = linked_observations.groupby(by=["cluster_id"])
    observations_split = [grouped_observations.get_group(g).copy() for g in grouped_observations.groups]
    
    if threads > 1:
        num_linkages = len(observations_split)
    
        if USE_RAY:
            shutdown = False
            if not ray.is_initialized():
                ray.init(num_cpus=threads)
                shutdown = True

            p = []
            for observations_i in observations_split:
                
                p.append(
                    iod_worker.remote(
                        observations_i,
                        observation_selection_method=observation_selection_method,
                        chi2_threshold=chi2_threshold,
                        min_obs=min_obs,
                        contamination_percentage=contamination_percentage,
                        iterate=iterate, 
                        light_time=light_time,
                        backend=backend,
                        backend_kwargs=backend_kwargs
                    )
                )
            
            iod_orbits_dfs, iod_orbit_members_dfs = ray.get(p)

            if shutdown:
                ray.shutdown()
        else:
            p = mp.Pool(
                processes=threads,
                initializer=_init_worker,
            ) 

            results = p.starmap(
                partial(
                    iod_worker, 
                    observation_selection_method=observation_selection_method,
                    chi2_threshold=chi2_threshold,
                    min_obs=min_obs,
                    contamination_percentage=contamination_percentage,
                    iterate=iterate, 
                    light_time=light_time,
                    backend=backend,
                    backend_kwargs=backend_kwargs
                ),
                zip(
                    observations_split, 
                ) 
            )
            p.close()  
            
            results = list(zip(*results))
            iod_orbits_dfs = results[0]
            iod_orbit_members_dfs = results[1]

    else:
        
        iod_orbits_dfs = []
        iod_orbit_members_dfs = []
        for i, observations_i in enumerate(observations_split):
            
            iod_orbits_df, iod_orbit_members_df = iod_worker(
                observations_i,
                observation_selection_method=observation_selection_method,
                chi2_threshold=chi2_threshold,
                min_obs=min_obs,
                contamination_percentage=contamination_percentage,
                iterate=iterate,
                light_time=light_time,
                backend=backend,
                backend_kwargs=backend_kwargs
            )
            iod_orbits_dfs.append(iod_orbits_df)
            iod_orbit_members_dfs.append(iod_orbit_members_df)
        
    iod_orbits = pd.concat(iod_orbits_dfs)
    iod_orbits.reset_index(
        inplace=True,
        drop=True
    )
    
    iod_orbit_members = pd.concat(iod_orbit_members_dfs)
    iod_orbit_members.reset_index(
        inplace=True,
        drop=True
    )
    
    return iod_orbits, iod_orbit_members