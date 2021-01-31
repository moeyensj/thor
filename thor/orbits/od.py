import uuid
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from astropy.time import Time

from ..config import Config
from ..backend import _init_worker
from ..backend import PYOORB
from ..backend import MJOLNIR
from .orbits import Orbits
from .residuals import calcResiduals


USE_RAY = Config.USE_RAY
NUM_THREADS = Config.NUM_THREADS


__all__ = [
    "od_worker",
    "od",
    "differentialCorrection"
]


def od_worker(
        orbit, 
        observations,
        chi2_tolerance=300,
        delta=1e-6,
        max_iter=20,
        method="central",
        fit_epoch=True,
        test_orbit=None,
        backend="PYOORB", 
        backend_kwargs={},
    ):
    assert orbit.ids[0] == observations["orbit_id"].unique()[0]
    
    od_orbit, od_orbit_members = od(
        orbit, 
        observations,
        chi2_tolerance=chi2_tolerance,
        delta=delta,
        max_iter=max_iter,
        method=method,
        fit_epoch=fit_epoch,
        test_orbit=test_orbit,
        backend=backend, 
        backend_kwargs=backend_kwargs,
    )
    return od_orbit, od_orbit_members

if USE_RAY:
    od_worker = ray.remote(od_worker)

def od(
        orbit, 
        observations,
        chi2_tolerance=300,
        delta=1e-6,
        max_iter=20,
        method="central",
        fit_epoch=True,
        test_orbit=None,
        backend="PYOORB", 
        backend_kwargs={},
    ):
    if backend == "MJOLNIR":
        backend = MJOLNIR(**backend_kwargs)
        
    elif backend == "PYOORB":
        backend = PYOORB(**backend_kwargs)
        
    else:
        err = (
            "backend should be one of 'MJOLNIR' or 'PYOORB'"
        )
        raise ValueError(err)
        
    obs_ids_all = observations["obs_id"].values
    coords = observations[["RA_deg", "Dec_deg"]].values
    coords_sigma = observations[["RA_sigma_deg", "Dec_sigma_deg"]].values
    
    observers = {}
    for observatory_code in observations["observatory_code"].unique():
        observatory_mask = observations["observatory_code"].isin([observatory_code])
        observers[observatory_code] = Time(
            observations[observatory_mask]["mjd_utc"].unique(),
            format="mjd",
            scale="utc"
        )
    
    # Calculate chi2 for residuals on the given observations
    # for the current orbit, the goal is for the orbit to improve 
    # such that the chi2 improves
    orbit_prev = Orbits(
        orbit.cartesian,
        orbit.epochs,
        ids=orbit.ids,
        orbit_type=orbit.orbit_type
    )
    
    ephemeris_prev = backend.generateEphemeris(
        orbit_prev, 
        observers,
        test_orbit=test_orbit, 
        threads=1
    )
    residuals_prev, stats_prev = calcResiduals(
        coords,
        ephemeris_prev[["RA_deg", "Dec_deg"]].values,
        sigmas_actual=coords_sigma,
        include_probabilistic=False
    )
    chi2_prev = stats_prev[0]
    chi2_total_prev = np.sum(chi2_prev)

    
    delta_prev = delta
    iterations = 0
    converged = False
    while not converged:
        
        delta_iter = delta_prev
        
        # Initialize the partials derivatives matrix
        num_obs = coords.shape[0]
        if num_obs > 6 and fit_epoch:
            num_params = 7
        else:
            num_params = 6
        
        A = np.zeros((coords.shape[1], num_params, num_obs))
        ATWA = np.zeros((num_params, num_params, num_obs))
        ATWb = np.zeros((num_params, 1, num_obs))
        
        # Generate ephemeris with current nominal orbit
        ephemeris_nom = backend.generateEphemeris(
            orbit_prev, 
            observers,
            test_orbit=test_orbit,
            threads=1
        )
        coords_nom = ephemeris_nom[["RA_deg", "Dec_deg"]].values
        
        
        # Modify each component of the state by a small delta
        d = np.zeros((1, 7))
        for i in range(7):
            
            if iterations == max_iter:
                break
            
            # zero the delta vector
            d *= 0.0
            
            # x, y, z [au]: 0, 1, 2 
            # vx, vy, vz [au per day]: 3, 4, 5
            # time [days] : 6
            if i < 3:
                delta_iter = delta_prev
                
                d[0, i] = orbit_prev.cartesian[0, i] * delta_iter
            elif i < 6:
                delta_iter = delta_prev
                
                d[0, i] = orbit_prev.cartesian[0, i] * delta_iter
            else:
                delta_iter = delta_prev/100000
                
                d[0, i] = delta_iter
                

            # Modify component i of the orbit by a small delta
            orbit_iter_p = Orbits(
                orbit_prev.cartesian + d[0, :6],
                orbit_prev.epochs + d[0, 6],
                orbit_type=orbit_prev.orbit_type
            )
            # Calculate the modified ephemerides
            ephemeris_mod_p = backend.generateEphemeris(
                orbit_iter_p, 
                observers,
                test_orbit=test_orbit, 
                threads=1
            )
            coords_mod_p = ephemeris_mod_p[["RA_deg", "Dec_deg"]].values
            
            delta_denom = d[0, i]
            if method == "central":
                
                # Modify component i of the orbit by a small delta
                orbit_iter_n = Orbits(
                    orbit_prev.cartesian - d[0, :6],
                    orbit_prev.epochs - d[0, 6],
                    orbit_type=orbit_prev.orbit_type
                )

                # Calculate the modified ephemerides
                ephemeris_mod_n = backend.generateEphemeris(
                    orbit_iter_n, 
                    observers,
                    test_orbit=test_orbit, 
                    threads=1
                )
                coords_mod_n = ephemeris_mod_n[["RA_deg", "Dec_deg"]].values
                
                delta_denom *= 2
                            
            else:
                coords_mod_n = coords_nom
                
                
            residuals_mod, _ = calcResiduals(
                coords_mod_p,
                coords_mod_n,
                sigmas_actual=None,
                include_probabilistic=False
            )            
            
            for n in range(num_obs):
                A[:, i:i+1, n] = residuals_mod[n:n+1].T / delta_denom
                

        for n in range(num_obs):
            W = np.diag(1 / coords_sigma[n]**2)
            ATWA[:, :, n] = A[:, :, n].T @ W @ A[:, :, n]
            ATWb[:, :, n] = A[:, :, n].T @ W @ residuals_prev[n:n+1].T

        ATWA = np.sum(ATWA, axis=2)
        ATWb = np.sum(ATWb, axis=2)
        
        delta_state = np.linalg.solve(ATWA, ATWb).T
        covariance_matrix = np.linalg.inv(ATWA)
       
        if num_params == 6:
            d_state = delta_state
            d_time = 0
        else:
            d_state = delta_state[0, :6]
            d_time = delta_state[0, 6]
            
        orbit_iter = Orbits(
            orbit_prev.cartesian + d_state,
            orbit_prev.epochs + d_time,
            orbit_type=orbit_prev.orbit_type,
            ids=orbit_prev.ids,
            covariances=[covariance_matrix]
        )
        
        # Generate ephemeris with current nominal orbit
        ephemeris_iter = backend.generateEphemeris(
            orbit_iter, 
            observers,
            test_orbit=test_orbit,
            threads=1
        )
        coords_iter = ephemeris_iter[["RA_deg", "Dec_deg"]].values
        
        residuals, stats = calcResiduals(
            coords,
            coords_iter,
            sigmas_actual=coords_sigma,
            include_probabilistic=False
        )
        chi2_iter = stats[0]
        chi2_total_iter = np.sum(chi2_iter)
        
        # If the new orbit has lower residuals than the previous,
        # accept the orbit and continue iterating
        if chi2_total_iter < chi2_total_prev:
            
            orbit_prev = orbit_iter
            residuals_prev = residuals
            stats_prev = stats
            chi2_prev = chi2_iter
            chi2_total_prev = chi2_total_iter
                    
        # If the new orbit does not have lower residuals, try changing 
        # delta to see if we get an improvement
        else:
            
            # Decrease delta to see if we can get an improvement next iteration
            delta_prev /= 10
            
            if delta_prev < 1e-11:
                converged = True
        
        iterations += 1
        if iterations == max_iter:
            break
       
    
    converged = True
    if not converged:
       
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
                #"arc_length",
                #"num_obs",
                "chi2",
                "num_params"
            ]
        )
        
        orbit_members = pd.DataFrame(
            columns=[
                "orbit_id", 
                "obs_id", 
                "residual_ra", 
                "residual_dec", 
                "chi2",
            ]
        )
    
    else:
        od_orbit = orbit_prev.todf()
        #orbit["arc_length"] = arc_length
        od_orbit["num_obs"] = num_obs
        od_orbit["chi2"] = chi2_total_prev
        od_orbit["num_params"] = num_params
        
        od_orbit_members = pd.DataFrame({
            "orbit_id" : [orbit_prev.ids[0] for i in range(len(obs_ids_all))],
            "obs_id" : obs_ids_all,
            "residual_ra" : residuals_prev[:, 0] * 3600,
            "residual_dec" : residuals_prev[:, 1] * 3600,
            "chi2" : chi2_prev,
            #"outlier" : np.zeros(len(obs_ids_all), dtype=int)
        })
        #orbit_members.loc[orbit_members["obs_id"].isin(outliers), "outlier"] = 1
        #orbit_members.loc[orbit_members["obs_id"].isin(obs_ids_sol), "gauss_sol"] = 1
    
    
    return od_orbit, od_orbit_members
            
def differentialCorrection(
        orbits, 
        orbit_members,
        observations, 
        chi2_tolerance=300,
        delta=1e-8,
        max_iter=20,
        method="central",
        fit_epoch=False,
        test_orbit=None,
        threads=60,
        backend="PYOORB", 
        backend_kwargs={}
    ):
    
    linked_observations = orbit_members[orbit_members["orbit_id"].isin(orbits["orbit_id"].values)].merge(observations, on="obs_id").copy()
    linked_observations.sort_values(
        by=["orbit_id", "mjd_utc"], 
        inplace=True
    )
    
    orbits_ = orbits.copy()
    orbits_.sort_values(
        by=["orbit_id"],
        inplace=True
    )
    orbits_.reset_index(
        inplace=True,       
        drop=True
    )
    
    
    grouped_observations = linked_observations.groupby(by=["orbit_id"])
    observations_split = [grouped_observations.get_group(g).copy() for g in grouped_observations.groups]
    
    orbits_initial = Orbits.fromdf(orbits_)
    orbits_split = orbits_initial.split(1)
    
    
    if threads > 1:
    
        if USE_RAY:
            shutdown = False
            if not ray.is_initialized():
                ray.init(num_cpus=threads)
                shutdown = True

            p = []
            for orbits_i, observations_i in zip(orbits_split, observations_split):
                
                p.append(
                    od_worker.remote(
                        orbits_i,
                        observations_i,
                        chi2_tolerance=chi2_tolerance,
                        delta=delta,
                        max_iter=max_iter,
                        method=method,
                        fit_epoch=fit_epoch,
                        test_orbit=test_orbit,
                        backend=backend, 
                        backend_kwargs=backend_kwargs,
                    )
                )
            
            od_orbits_dfs, od_orbit_members_dfs = ray.get(p)

            if shutdown:
                ray.shutdown()
        else:
            p = mp.Pool(
                processes=threads,
                initializer=_init_worker,
            ) 

            results = p.starmap(
                partial(
                    od_worker, 
                    chi2_tolerance=chi2_tolerance,
                    delta=delta,
                    max_iter=max_iter,
                    method=method,
                    fit_epoch=fit_epoch,
                    test_orbit=test_orbit,
                    backend=backend, 
                    backend_kwargs=backend_kwargs,
                ),
                zip(
                    orbits_split,
                    observations_split, 
                ) 
            )
            p.close()  
            
            results = list(zip(*results))
            od_orbits_dfs = results[0]
            od_orbit_members_dfs = results[1]

    else:
        
        od_orbits_dfs = []
        od_orbit_members_dfs = []
        for i, (orbits_i, observations_i) in enumerate(zip(orbits_split, observations_split)):
            
            od_orbits_df, od_orbit_members_df = od_worker(
                orbits_i, 
                observations_i,
                chi2_tolerance=chi2_tolerance,
                delta=delta,
                max_iter=max_iter,
                method=method,
                fit_epoch=fit_epoch,
                test_orbit=test_orbit,
                backend=backend, 
                backend_kwargs=backend_kwargs,
            )
            od_orbits_dfs.append(od_orbits_df)
            od_orbit_members_dfs.append(od_orbit_members_df)
        
    od_orbits = pd.concat(od_orbits_dfs)
    od_orbits.reset_index(
        inplace=True,
        drop=True
    )
    
    od_orbit_members = pd.concat(od_orbit_members_dfs)
    od_orbit_members.reset_index(
        inplace=True,
        drop=True
    )
    
    return od_orbits, od_orbit_members