import os
import sys
import time
import uuid
import logging
import numpy as np
import pandas as pd
import multiprocessing as mp
from astropy.time import Time
from itertools import combinations
from functools import partial

from ..config import Config
from ..utils import identifySubsetLinkages
from ..utils import sortLinkages
from ..backend import _init_worker
from ..backend import MJOLNIR
from ..backend import PYOORB
from .gauss import gaussIOD
from .residuals import calcResiduals
from .ephemeris import generateEphemeris

USE_RAY = Config.USE_RAY
NUM_THREADS = Config.NUM_THREADS

logger = logging.getLogger(__name__)

__all__ = [
    "selectObservations",
    "iod",
    "iod_worker",
    "initialOrbitDetermination"
]

def selectObservations(
        observations, 
        method="combinations"
    ):
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
        min_obs=6,
        min_arc_length=1.0,
        rchi2_threshold=10**3,
        contamination_percentage=0.0,
        iterate=False, 
        light_time=True,
        linkage_id_col="cluster_id",
        backend="PYOORB",
        backend_kwargs={}
    ):
    assert np.all(sorted(observations["mjd_utc"].values) == observations["mjd_utc"].values)
    
    iod_orbit, iod_orbit_members = iod(
        observations,
        observation_selection_method=observation_selection_method,
        min_obs=min_obs,
        min_arc_length=min_arc_length,
        rchi2_threshold=rchi2_threshold,
        contamination_percentage=contamination_percentage,
        iterate=iterate, 
        light_time=light_time,
        backend=backend,
        backend_kwargs=backend_kwargs
    )
    if len(iod_orbit) > 0:
        iod_orbit.insert(1, linkage_id_col, observations[linkage_id_col].unique()[0])
    
    return iod_orbit, iod_orbit_members

if USE_RAY:
    import ray
    iod_worker = ray.remote(iod_worker)

def iod(
        observations,
        min_obs=6,
        min_arc_length=1.0,
        contamination_percentage=0.0,
        rchi2_threshold=200,
        observation_selection_method="combinations",
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
        Dataframe of observations with at least the following columns:
            "obs_id" : Observation IDs [str],
            "mjd_utc" : Observation time in MJD UTC [float],
            "RA_deg" : equatorial J2000 Right Ascension in degrees [float],
            "Dec_deg" : equatorial J2000 Declination in degrees [float],
            "RA_sigma_deg" : 1-sigma uncertainty in equatorial J2000 RA [float],
            "Dec_sigma_deg" : 1 sigma uncertainty in equatorial J2000 Dec [float],
            "observatory_code" : MPC recognized observatory code [str],
            "obs_x" : Observatory's heliocentric ecliptic J2000 x-position in au [float],
            "obs_y" : Observatory's heliocentric ecliptic J2000 y-position in au [float],
            "obs_z" : Observatory's heliocentric ecliptic J2000 z-position in au [float],
            "obs_vx" [Optional] : Observatory's heliocentric ecliptic J2000 x-velocity in au per day [float],
            "obs_vy" [Optional] : Observatory's heliocentric ecliptic J2000 y-velocity in au per day [float],
            "obs_vz" [Optional] : Observatory's heliocentric ecliptic J2000 z-velocity in au per day [float]
    min_obs : int, optional
        Minimum number of observations that must remain in the linkage. For example, if min_obs is set to 6 and
        a linkage has 8 observations, at most the two worst observations will be flagged as outliers if their individual
        chi2 values exceed the chi2 threshold.
    contamination_percentage : float, optional
        Maximum percent of observations that can flagged as outliers. 
    rchi2_threshold : float, optional
        Maximum reduced chi2 required for an initial orbit to be accepted. 
    observation_selection_method : {'first+middle+last', 'thirds', 'combinations'}, optional
        Selects which three observations to use for IOD depending on the method. The avaliable methods are:
            'first+middle+last' : Grab the first, middle and last observations in time. 
            'thirds' : Grab the middle observation in the first third, second third, and final third. 
            'combinations' : Return the observation IDs corresponding to every possible combination of three observations with
                non-coinciding observation times.
    iterate : bool, optional
        Iterate the preliminary orbit solution using the state transition iterator. 
    light_time : bool, optional
        Correct preliminary orbit for light travel time.
    linkage_id_col : str, optional
        Name of linkage_id column in the linkage_members dataframe.
    backend : {'MJOLNIR', 'PYOORB'}, optional
        Which backend to use for ephemeris generation.
    backend_kwargs : dict, optional
        Settings and additional parameters to pass to selected 
        backend.

    Returns
    -------
    iod_orbits : `~pandas.DataFrame` 
        Dataframe with orbits found in linkages.
            "orbit_id" : Orbit ID, a uuid [str],
            "epoch" : Epoch at which orbit is defined in MJD TDB [float],
            "x" : Orbit's ecliptic J2000 x-position in au [float],
            "y" : Orbit's ecliptic J2000 y-position in au [float],
            "z" : Orbit's ecliptic J2000 z-position in au [float],
            "vx" : Orbit's ecliptic J2000 x-velocity in au per day [float],
            "vy" : Orbit's ecliptic J2000 y-velocity in au per day [float],
            "vz" : Orbit's ecliptic J2000 z-velocity in au per day [float],
            "arc_length" : Arc length in days [float],
            "num_obs" : Number of observations that were within the chi2 threshold 
                of the orbit.
            "chi2" : Total chi2 of the orbit calculated using the predicted location of the orbit
                on the sky compared to the consituent observations. 

    iod_orbit_members : `~pandas.DataFrame` 
        Dataframe of orbit members with the following columns:
            "orbit_id" : Orbit ID, a uuid [str],
            "obs_id" : Observation IDs [str], one ID per row. 
            "residual_ra_arcsec" : Residual (observed - expected) equatorial J2000 Right Ascension in arcseconds [float]
            "residual_dec_arcsec" : Residual (observed - expected) equatorial J2000 Declination in arcseconds [float]
            "chi2" : Observation's chi2 [float]
            "gauss_sol" : Flag to indicate which observations were used to calculate the Gauss soluton [int]
            "outlier" : Flag to indicate which observations are potential outliers (their chi2 is higher than
                the chi2 threshold) [float]
    """
    processable = True
    if len(observations) == 0:
        processable = False
    
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
    arc_length = None
    outliers = np.array([])
    converged = False
    num_obs = len(observations)
    if num_obs < min_obs:
        processable = False
    num_outliers = int(num_obs * contamination_percentage / 100.)
    num_outliers = np.maximum(np.minimum(num_obs - min_obs, num_outliers), 0)
    
    # Select observation IDs to use for IOD
    obs_ids = selectObservations(
        observations, 
        method=observation_selection_method, 
    )

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
            rchi2 = chi2_total / (2 * num_obs - 6)

            # The reduced chi2 is above the threshold and no outliers are 
            # allowed, this cannot be improved by outlier rejection 
            # so continue to the next IOD orbit
            if rchi2 > rchi2_threshold and num_outliers == 0:
                # If we have iterated through all iod orbits and no outliers
                # are allowed for this linkage then no other combination of 
                # observations will make it acceptable, so exit here.
                if (i + 1) == len(iod_orbits):
                    processable = False
                    break
                
                continue       

            # If the total reduced chi2 is less than the threshold accept the orbit
            elif rchi2 <= rchi2_threshold:
                orbit_sol = iod_orbits[i:i+1]
                obs_ids_sol = ids
                chi2_total_sol = chi2_total    
                chi2_sol = chi2
                rchi2_sol = rchi2
                residuals_sol = residuals
                outliers = np.array([])
                arc_length = times_all.utc.mjd.max() - times_all.utc.mjd.min()
                converged = True
                break 

            # Let's now test to see if we can remove some outliers, we 
            # anticipate that we get to this stage if the three selected observations 
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
                    # Then recalculate the reduced chi2
                    chi2_new = chi2_total - np.sum(chi2[~mask][remove])
                    num_obs_new = len(observations) - len(remove)
                    rchi2_new = chi2_new / (2 * num_obs_new - 6)
                    
                    ids_mask = np.isin(obs_ids_all, obs_id_outlier, invert=True)
                    arc_length = times_all[ids_mask].utc.mjd.max() - times_all[ids_mask].utc.mjd.min()
                    
                    # If the updated reduced chi2 total is lower than our desired
                    # threshold, accept the soluton. If not, keep going.
                    if rchi2_new <= rchi2_threshold and arc_length >= min_arc_length:
                        orbit_sol = iod_orbits[i:i+1]
                        obs_ids_sol = ids
                        chi2_total_sol = chi2_new
                        rchi2_sol = rchi2_new
                        residuals_sol = residuals
                        outliers = obs_id_outlier
                        num_obs = num_obs_new
                        ids_mask = np.isin(obs_ids_all, outliers, invert=True)
                        arc_length = times_all[ids_mask].utc.mjd.max() - times_all[ids_mask].utc.mjd.min()
                        chi2_sol = chi2
                        converged = True
                        break
                        
            else:
                continue

        j += 1

    if not converged or not processable:
       
        orbit = pd.DataFrame(
            columns=[
                "orbit_id",
                "epoch",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "arc_length",
                "num_obs",
                "chi2",
                "rchi2"
            ]
        )
        
        orbit_members = pd.DataFrame(
            columns=[
                "orbit_id", 
                "obs_id", 
                "residual_ra_arcsec", 
                "residual_dec_arcsec", 
                "chi2",
                "gauss_sol",
                "outlier"
            ]
        )
    
    else:
        orbit = orbit_sol.to_df(include_units=False)
        orbit["arc_length"] = arc_length
        orbit["num_obs"] = num_obs
        orbit["chi2"] = chi2_total_sol
        orbit["rchi2"] = rchi2_sol 
        
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
        min_obs=6,
        min_arc_length=1.0,
        contamination_percentage=20.0,
        rchi2_threshold=10**3,
        observation_selection_method='combinations',
        iterate=False,
        light_time=True,
        linkage_id_col="cluster_id",
        identify_subsets=True,
        threads=NUM_THREADS,
        backend="PYOORB",
        backend_kwargs={}
    ):
    """
    Run initial orbit determination on linkages found in observations.

    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Dataframe of observations with at least the following columns:
            "obs_id" : Observation IDs [str],
            "mjd_utc" : Observation time in MJD UTC [float],
            "RA_deg" : equatorial J2000 Right Ascension in degrees [float],
            "Dec_deg" : equatorial J2000 Declination in degrees [float],
            "RA_sigma_deg" : 1-sigma uncertainty in equatorial J2000 RA [float],
            "Dec_sigma_deg" : 1 sigma uncertainty in equatorial J2000 Dec [float],
            "observatory_code" : MPC recognized observatory code [str],
            "obs_x" : Observatory's heliocentric ecliptic J2000 x-position in au [float],
            "obs_y" : Observatory's heliocentric ecliptic J2000 y-position in au [float],
            "obs_z" : Observatory's heliocentric ecliptic J2000 z-position in au [float],
            "obs_vx" [Optional] : Observatory's heliocentric ecliptic J2000 x-velocity in au per day [float],
            "obs_vy" [Optional] : Observatory's heliocentric ecliptic J2000 y-velocity in au per day [float],
            "obs_vz" [Optional] : Observatory's heliocentric ecliptic J2000 z-velocity in au per day [float]
    linkage_members : `~pandas.DataFrame`
        Dataframe of linkages with at least two columns:
            "linkage_id" : Linkage ID [str],
            "obs_id" : Observation IDs [str], one ID per row. 
    observation_selection_method : {'first+middle+last', 'thirds', 'combinations'}, optional
        Selects which three observations to use for IOD depending on the method. The avaliable methods are:
            'first+middle+last' : Grab the first, middle and last observations in time. 
            'thirds' : Grab the middle observation in the first third, second third, and final third. 
            'combinations' : Return the observation IDs corresponding to every possible combination of three observations with
                non-coinciding observation times.
    min_obs : int, optional
        Minimum number of observations that must remain in the linkage. For example, if min_obs is set to 6 and
        a linkage has 8 observations, at most the two worst observations will be flagged as outliers. Only up t o
        the contamination percentage of observations of will be flagged as outliers, provided that at least min_obs
        observations remain in the linkage.
    rchi2_threshold : float, optional
        Minimum reduced chi2 for an initial orbit to be accepted. If an orbit
    contamination_percentage : float, optional
        Maximum percent of observations that can flagged as outliers. 
    iterate : bool, optional
        Iterate the preliminary orbit solution using the state transition iterator. 
    light_time : bool, optional
        Correct preliminary orbit for light travel time.
    linkage_id_col : str, optional
        Name of linkage_id column in the linkage_members dataframe.
    threads : int, optional
        Number of threads to use for multiprocessing. 
    backend : {'MJOLNIR', 'PYOORB'}, optional
        Which backend to use for ephemeris generation.
    backend_kwargs : dict, optional
        Settings and additional parameters to pass to selected 
        backend.

    Returns
    -------
    iod_orbits : `~pandas.DataFrame` 
        Dataframe with orbits found in linkages.
            "orbit_id" : Orbit ID, a uuid [str],
            "epoch" : Epoch at which orbit is defined in MJD TDB [float],
            "x" : Orbit's ecliptic J2000 x-position in au [float],
            "y" : Orbit's ecliptic J2000 y-position in au [float],
            "z" : Orbit's ecliptic J2000 z-position in au [float],
            "vx" : Orbit's ecliptic J2000 x-velocity in au per day [float],
            "vy" : Orbit's ecliptic J2000 y-velocity in au per day [float],
            "vz" : Orbit's ecliptic J2000 z-velocity in au per day [float],
            "arc_length" : Arc length in days [float],
            "num_obs" : Number of observations that were within the chi2 threshold 
                of the orbit.
            "chi2" : Total chi2 of the orbit calculated using the predicted location of the orbit
                on the sky compared to the consituent observations. 

    iod_orbit_members : `~pandas.DataFrame` 
        Dataframe of orbit members with the following columns:
            "orbit_id" : Orbit ID, a uuid [str],
            "obs_id" : Observation IDs [str], one ID per row. 
            "residual_ra_arcsec" : Residual (observed - expected) equatorial J2000 Right Ascension in arcseconds [float]
            "residual_dec_arcsec" : Residual (observed - expected) equatorial J2000 Declination in arcseconds [float]
            "chi2" : Observation's chi2 [float]
            "gauss_sol" : Flag to indicate which observations were used to calculate the Gauss soluton [int]
            "outlier" : Flag to indicate which observations are potential outliers (their chi2 is higher than
                the chi2 threshold) [float]
    """
    logger.info("Running initial orbit determination...")

    time_start = time.time()
    
   
    if len(observations) > 0 and len(linkage_members) > 0:

        if threads > 1:
            if USE_RAY:
                if not ray.is_initialized():
                    ray.init(address="auto")
            else:
                p = mp.Pool(
                    processes=threads,
                    initializer=_init_worker,
                ) 

        iod_orbits_dfs = []
        iod_orbit_members_dfs = []

        chunk_size = 100000
        linkage_ids = linkage_members[linkage_id_col].unique()
        for chunk in range(0, len(linkage_ids), chunk_size):

            linkage_ids_chunk = linkage_ids[chunk:chunk+chunk_size]
            linkage_members_chunk = linkage_members[linkage_members[linkage_id_col].isin(linkage_ids_chunk)]

            linked_observations = linkage_members_chunk.merge(observations, on="obs_id")
            linked_observations.sort_values(
                by=[linkage_id_col, "mjd_utc"], 
                inplace=True
            )
            linked_observations.reset_index(
                drop=True,
                inplace=True
            )
            grouped_observations = linked_observations.groupby(by=[linkage_id_col])
            observations_split = [grouped_observations.get_group(g).reset_index(drop=True) for g in grouped_observations.groups]

            if threads > 1:

                if USE_RAY:

                    p = []
                    for observations_i in observations_split:
                        p.append(
                            iod_worker.remote(
                                observations_i,
                                observation_selection_method=observation_selection_method,
                                rchi2_threshold=rchi2_threshold,
                                min_obs=min_obs,
                                min_arc_length=min_arc_length,
                                contamination_percentage=contamination_percentage,
                                iterate=iterate, 
                                light_time=light_time,
                                linkage_id_col=linkage_id_col,
                                backend=backend,
                                backend_kwargs=backend_kwargs
                            )
                        )

                    results = ray.get(p)
                    results = list(zip(*results))
                    iod_orbits_dfs_c = results[0]
                    iod_orbit_members_dfs_c = results[1]
                    iod_orbits_dfs += iod_orbits_dfs_c
                    iod_orbit_members_dfs += iod_orbit_members_dfs_c

                else:
                    results = p.starmap(
                        partial(
                            iod_worker, 
                            observation_selection_method=observation_selection_method,
                            rchi2_threshold=rchi2_threshold,
                            min_obs=min_obs,
                            min_arc_length=min_arc_length,
                            contamination_percentage=contamination_percentage,
                            iterate=iterate, 
                            light_time=light_time,
                            linkage_id_col=linkage_id_col,
                            backend=backend,
                            backend_kwargs=backend_kwargs
                        ),
                        zip(
                            observations_split, 
                        ) 
                    )

                    results = list(zip(*results))
                    iod_orbits_dfs_c = results[0]
                    iod_orbit_members_dfs_c = results[1]
                    iod_orbits_dfs += iod_orbits_dfs_c
                    iod_orbit_members_dfs += iod_orbit_members_dfs_c

            else:

                for i, observations_i in enumerate(observations_split):

                    iod_orbits_df, iod_orbit_members_df = iod_worker(
                        observations_i,
                        observation_selection_method=observation_selection_method,
                        rchi2_threshold=rchi2_threshold,
                        min_obs=min_obs,
                        min_arc_length=min_arc_length,
                        contamination_percentage=contamination_percentage,
                        iterate=iterate,
                        light_time=light_time,
                        linkage_id_col=linkage_id_col,
                        backend=backend,
                        backend_kwargs=backend_kwargs
                    )
                    iod_orbits_dfs.append(iod_orbits_df)
                    iod_orbit_members_dfs.append(iod_orbit_members_df)

        iod_orbits = pd.concat(
            iod_orbits_dfs,
            ignore_index=True
        )

        iod_orbit_members = pd.concat(
            iod_orbit_members_dfs, 
            ignore_index=True
        )

        if threads > 1:
            if not USE_RAY:
                p.close()

    else:
        iod_orbits = pd.DataFrame(
            columns=[
                "orbit_id",
                "epoch",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "arc_length",
                "num_obs",
                "chi2",
                "rchi2"
            ]
        )
        
        iod_orbit_members = pd.DataFrame(
            columns=[
                "orbit_id", 
                "obs_id", 
                "residual_ra_arcsec", 
                "residual_dec_arcsec", 
                "chi2",
                "gauss_sol",
                "outlier"
            ]
        )
    
    for col in ["num_obs"]:
        iod_orbits[col] = iod_orbits[col].astype(int)
    for col in ["gauss_sol", "outlier"]:
        iod_orbit_members[col] = iod_orbit_members[col].astype(int)

    logger.info("Found {} initial orbits.".format(len(iod_orbits)))

    if identify_subsets and len(iod_orbits) > 0:
        iod_orbits, iod_orbit_members = identifySubsetLinkages(
            iod_orbits, 
            iod_orbit_members, 
            linkage_id_col="orbit_id"
        )
        logger.info("{} subset orbits identified.".format(len(iod_orbits[~iod_orbits["subset_of"].isna()])))

    iod_orbits, iod_orbit_members = sortLinkages(
        iod_orbits,
        iod_orbit_members,
        observations,
        linkage_id_col="orbit_id"
    )

    time_end = time.time()
    logger.info("Initial orbit determination completed in {:.3f} seconds.".format(time_end - time_start))   

    return iod_orbits, iod_orbit_members