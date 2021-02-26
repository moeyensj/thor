import time
import uuid
import numpy as np
import pandas as pd
import multiprocessing as mp
from astropy.time import Time
from functools import partial
from sklearn.neighbors import BallTree

from ..config import Config
from ..backend import _init_worker
from ..utils import verifyLinkages
from ..utils import mergeLinkages
from ..utils import sortLinkages
from ..utils import identifySubsetLinkages
from .orbits import Orbits
from .ephemeris import generateEphemeris
from .residuals import calcResiduals
from .od import differentialCorrection

USE_RAY = Config.USE_RAY
NUM_THREADS = Config.NUM_THREADS

__all__ = [
    "attribution_worker",
    "attributeObservations",
    "mergeAndExtendOrbits"
]

def attribution_worker(
        orbits, 
        observations, 
        eps=1/3600, 
        include_probabilistic=True, 
        backend="PYOORB", 
        backend_kwargs={}
    ):
    
    # Create observer's dictionary from observations
    observers = {}
    for observatory_code in observations["observatory_code"].unique():
        observers[observatory_code] = Time(
            observations[observations["observatory_code"].isin([observatory_code])]["mjd_utc"].unique(),
            scale="utc",
            format="mjd"
        )
    
    # Genereate ephemerides for each orbit at the observation times
    ephemeris = generateEphemeris(
        orbits, 
        observers, 
        backend=backend, 
        backend_kwargs=backend_kwargs,
        threads=1,
        chunk_size=1
    )
    
    # Group the predicted ephemerides and observations by visit / exposure
    ephemeris_grouped = ephemeris.groupby(by=["observatory_code", "mjd_utc"])
    ephemeris_visits = [ephemeris_grouped.get_group(g) for g in ephemeris_grouped.groups]
    observations_grouped = observations.groupby(by=["observatory_code", "mjd_utc"])
    observations_visits = [observations_grouped.get_group(g) for g in observations_grouped.groups]
    
    # Loop through each unique exposure and visit, find the nearest observations within
    # eps (haversine metric)
    distances = []
    orbit_ids_associated = []
    obs_ids_associated = []
    eps_rad = np.radians(eps)
    residuals = []
    stats = []
    for ephemeris_visit, observations_visit in zip(ephemeris_visits, observations_visits):
        
        assert len(ephemeris_visit["mjd_utc"].unique()) == 1
        assert len(observations_visit["mjd_utc"].unique()) == 1
        assert observations_visit["mjd_utc"].unique()[0] == ephemeris_visit["mjd_utc"].unique()[0]
        
        obs_ids = observations_visit[["obs_id"]].values
        orbit_ids = ephemeris_visit[["orbit_id"]].values
        coords = observations_visit[["RA_deg", "Dec_deg"]].values
        coords_latlon = observations_visit[["Dec_deg"]].values
        coords_predicted = ephemeris_visit[["RA_deg", "Dec_deg"]].values
        coords_sigma = observations_visit[["RA_sigma_deg", "Dec_sigma_deg"]].values
        
        # Haversine metric requires latitude first then longitude...
        coords_latlon = np.radians(observations_visit[["Dec_deg", "RA_deg"]].values)
        coords_predicted_latlon = np.radians(ephemeris_visit[["Dec_deg", "RA_deg"]].values)
        
        num_obs = len(coords_predicted)
        k = np.minimum(3, num_obs)

        # Build BallTree with a haversine metric on predicted ephemeris
        tree = BallTree(
            coords_predicted_latlon, 
            metric="haversine"
        )
        # Query tree using observed RA, Dec
        d, i = tree.query(
            coords_latlon, 
            k=k, 
            return_distance=True,
            dualtree=True,
            breadth_first=False,
            sort_results=False
        )
        
        # Select all observations with distance smaller or equal
        # to the maximum given distance
        mask = np.where(d <= eps_rad)
        
        if len(d[mask]) > 0:
            orbit_ids_associated.append(orbit_ids[i[mask]])
            obs_ids_associated.append(obs_ids[mask[0]])
            distances.append(d[mask].reshape(-1, 1))
            
            residuals_visit, stats_visit = calcResiduals(
                coords[mask[0]],
                coords_predicted[i[mask]],
                sigmas_actual=coords_sigma[mask[0]],
                include_probabilistic=True
            )
            residuals.append(residuals_visit)
            stats.append(np.vstack(stats_visit).T)
    
    if len(distances) > 0:
        distances = np.degrees(np.vstack(distances))
        orbit_ids_associated = np.vstack(orbit_ids_associated)
        obs_ids_associated = np.vstack(obs_ids_associated)
        residuals = np.vstack(residuals)
        stats = np.vstack(stats)

        attributions = {
            "orbit_id" : orbit_ids_associated[:, 0],
            "obs_id" : obs_ids_associated[:, 0],
            "distance" : distances[:, 0],
            "residual_ra_arcsec" : residuals[:, 0] * 3600,
            "residual_dec_arcsec" : residuals[:, 1] * 3600,
            "chi2" : stats[:, 0]
        }
        if include_probabilistic:
            attributions["probability"] = stats[:, 1]
            attributions["mahalanobis_distance"] = stats[:, 2]

        attributions = pd.DataFrame(attributions)
        
    else:
        columns = ["orbit_id", "obs_id", "distance", "residual_ra_arcsec", "residual_dec_arcsec", "chi2"]
        if include_probabilistic:
            columns += ["probability", "mahalanobis_distance"]
            
        attributions = pd.DataFrame(columns=columns)
    
    return attributions

if USE_RAY:
    import ray
    attribution_worker = ray.remote(attribution_worker)

def attributeObservations(
        orbits, 
        observations, 
        eps=5/3600, 
        include_probabilistic=True, 
        orbits_chunk_size=10,
        observations_chunk_size=100000,
        threads=NUM_THREADS,
        backend="PYOORB", 
        backend_kwargs={},
        verbose=True,
    ):
    time_start = time.time()
    if verbose:
        print("THOR: attributeObservations")
        print("-------------------------------")
        print("Running attribution...")
        print("Distance: {} degrees".format(eps))
        print("Probabilistic residuals: {}".format(include_probabilistic))
        
    orbits_split = orbits.split(orbits_chunk_size)
    observations_split = []
    for chunk in range(0, len(observations), observations_chunk_size):
        observations_split.append(observations.iloc[chunk:chunk + observations_chunk_size].copy())
    
    if threads > 1:
        
        if USE_RAY:
            
            shutdown = False
            if not ray.is_initialized():
                ray.init(num_cpus=threads)
                shutdown = True

            attribution_dfs = []
            for obs_i in observations_split:
                p = []
                for orbit_i in orbits_split:
                    p.append(
                        attribution_worker.remote(
                            orbit_i,
                            obs_i,
                            eps=eps, 
                            include_probabilistic=include_probabilistic, 
                            backend=backend, 
                            backend_kwargs=backend_kwargs
                        )
                    )
                    
                attribution_dfs_i = ray.get(p)
                attribution_dfs += attribution_dfs_i

            if shutdown:
                ray.shutdown()
        
        else:

            p = mp.Pool(
                processes=threads,
                initializer=_init_worker,
            ) 
            attribution_dfs = []
            for obs_i in observations_split:

                obs = [obs_i.copy() for i in range(len(orbits_split))]

                attribution_dfs_i = p.starmap(
                    partial(
                        attribution_worker, 
                        eps=eps, 
                        include_probabilistic=include_probabilistic, 
                        backend=backend, 
                        backend_kwargs=backend_kwargs
                    ),
                    zip(
                        orbits_split, 
                        obs, 
                    ) 
                )
                attribution_dfs += attribution_dfs_i

            p.close()  
            
    else:
        attribution_dfs = []
        for obs_i in observations_split:
            for orbit_i in orbits_split:
                attribution_df_i = attribution_worker(
                    orbit_i,
                    obs_i,
                    eps=eps, 
                    include_probabilistic=include_probabilistic, 
                    backend=backend, 
                    backend_kwargs=backend_kwargs
                )
                attribution_dfs.append(attribution_df_i)
                
        
    attributions = pd.concat(attribution_dfs)
    attributions.sort_values(
        by=["orbit_id", "obs_id", "distance"],
        inplace=True
    )
    attributions.reset_index(
        inplace=True,
        drop=True
    )
    
    time_end = time.time()
    if verbose:
        print("Attributed {} observations to {} orbits.".format(
            attributions["obs_id"].nunique(),
            attributions["orbit_id"].nunique()
        ))
        print("Total time in seconds: {}".format(time_end - time_start))   
        print("-------------------------------")
        print("")
        
    return attributions

def mergeAndExtendOrbits(
        orbits, 
        orbit_members, 
        observations, 
        min_obs=6,
        contamination_percentage=20.0,
        rchi2_threshold=5,
        eps=1/3600, 
        delta=1e-8,
        max_iter=20,
        method="central",
        fit_epoch=False,
        orbits_chunk_size=10,
        observations_chunk_size=100000,
        threads=60,
        backend="PYOORB", 
        backend_kwargs={},
        verbose=True
    ):
    """
    Attempt to extend an orbit's observational arc by running 
    attribution on the observations. This is an iterative process: attribution 
    is run, any observations found for each orbit are added to that orbit and differential correction is
    run. Orbits which are subset's of other orbits are removed. Iteration continues until there are no
    duplicate observation assignments.
    
    Parameters
    ----------
    
    
    
    
    """
    time_start = time.time()
    if verbose:
        print("THOR: mergeAndExtendOrbits")
        print("-------------------------------")
        print("Attempting to merge and extend orbits..")
        print("Minimum observations: {}".format(min_obs))
        
    orbits_iter, orbit_members_iter = verifyLinkages(
        orbits, 
        orbit_members, 
        observations, 
        linkage_id_col="orbit_id"
    )
    observations_iter = observations.copy()

    iterations = 0
    num_duplicate_obs_prev = 1e10
    odp_orbits_dfs = []
    odp_orbit_members_dfs = []

    if len(orbits_iter) > 0:
        converged = False

        while not converged:
            # Run attribution
            attributions = attributeObservations(
                Orbits.from_df(orbits_iter),
                observations_iter,
                eps=eps,
                include_probabilistic=True,
                threads=threads,
                orbits_chunk_size=orbits_chunk_size,
                observations_chunk_size=observations_chunk_size,
                backend=backend,
                backend_kwargs=backend_kwargs,
                verbose=False
            )

            assert np.all(np.isin(orbit_members_iter["obs_id"].unique(), observations_iter["obs_id"].unique()))

            # Append attributed observations to the orbit members 
            # dataframe and then drop any duplicate observations 
            # that might have been added to each orbit
            orbit_members_iter = pd.concat([
                orbit_members_iter, 
                attributions[["orbit_id", "obs_id", "residual_ra_arcsec", "residual_dec_arcsec", "chi2"]]]
            )
            orbit_members_iter.drop_duplicates(
                subset=["orbit_id", "obs_id"],
                keep="first",
                inplace=True
            )
            
            # Create a new orbit for each orbit that shares 
            # observations with another orbit
            merged_orbits, merged_orbit_members, merged_from = mergeLinkages(
                orbits_iter, 
                orbit_members_iter, 
                observations_iter,
                linkage_id_col="orbit_id"
            )

            if len(merged_orbits) > 0:
                orbits_iter = pd.concat([orbits_iter, merged_orbits])
                orbit_members_iter = pd.concat([orbit_members_iter, merged_orbit_members])

            orbits_iter, orbit_members_iter = sortLinkages(
                orbits_iter,
                orbit_members_iter[["orbit_id", "obs_id"]],
                observations_iter,
            )
            
            # Run differential orbit correction on all orbits
            # with the newly added observations to the orbits 
            # that had observations attributed to them
            orbits_iter, orbit_members_iter = differentialCorrection(
                orbits_iter,
                orbit_members_iter,
                observations_iter, 
                rchi2_threshold=rchi2_threshold,
                min_obs=min_obs,
                contamination_percentage=contamination_percentage,
                delta=delta, 
                method=method,
                max_iter=max_iter,
                threads=threads,
                fit_epoch=False,
                backend=backend,
                backend_kwargs=backend_kwargs,
                verbose=False
            )  
            orbit_members_iter = orbit_members_iter[orbit_members_iter["outlier"] == 0]
            orbit_members_iter.reset_index(
                inplace=True, 
                drop=True
            )
            
            # Identify any orbits that are subsets of a larger orbit
            orbits_iter, orbit_members_iter = identifySubsetLinkages(
                orbits_iter,
                orbit_members_iter,
            )
            # Keep only the orbits that are not a subset of a larger orbit
            orbits_iter = orbits_iter[orbits_iter["subset_of"].isna()]
            orbit_members_iter = orbit_members_iter[orbit_members_iter["orbit_id"].isin(orbits_iter["orbit_id"].values)]

            # If orbits were merged before OD, and some of the merged orbits survived OD then remove the orbits
            # that were used to merge into a larger orbit 
            if (len(merged_orbits) > 0):

                remaining_merged_orbits = orbits_iter[orbits_iter["orbit_id"].isin(merged_from["orbit_id"].values)]["orbit_id"].values
                orbits_to_remove = merged_from[merged_from["orbit_id"].isin(remaining_merged_orbits)]["merged_from"].unique()

                orbits_iter = orbits_iter[~orbits_iter["orbit_id"].isin(orbits_to_remove)]
                orbit_members_iter = orbit_members_iter[orbit_members_iter["orbit_id"].isin(orbits_iter["orbit_id"].values)]

            # Remove the orbits that were not improved from the pool of available orbits. Orbits that were not improved
            # are orbits that have already iterated to their best-fit solution given the observations available. These orbits
            # are unlikely to recover more observations in subsequent iterations and so can be saved for output.
            not_improved = orbits_iter[orbits_iter["improved"] == False]["orbit_id"].values
            orbits_out = orbits_iter[orbits_iter["orbit_id"].isin(not_improved)].copy()
            orbit_members_out = orbit_members_iter[orbit_members_iter["orbit_id"].isin(not_improved)].copy()
            not_improved_obs_ids = orbit_members_out["obs_id"].values

            # If some orbits that were not improved still share observations, keep the orbit with the lowest 
            # reduced chi2 in the pool of orbits but delete the others.
            obs_id_occurences = orbit_members_out["obs_id"].value_counts()
            duplicate_obs_ids = obs_id_occurences.index.values[obs_id_occurences.values > 1]

            while len(duplicate_obs_ids) > 0:
                duplicate_obs_id = duplicate_obs_ids[0]

                orbit_ids = orbit_members_out[orbit_members_out["obs_id"].isin([duplicate_obs_id])]["orbit_id"].values
                duplicate_orbits = orbits_out[orbits_out["orbit_id"].isin(orbit_ids)]
                orbit_to_keep = duplicate_orbits[duplicate_orbits["rchi2"] == duplicate_orbits["rchi2"].min()]["orbit_id"].values
                orbits_to_delete = duplicate_orbits[~duplicate_orbits["orbit_id"].isin(orbit_to_keep)]["orbit_id"].values

                orbits_out = orbits_out[~orbits_out["orbit_id"].isin(orbits_to_delete)]
                orbit_members_out = orbit_members_out[~orbit_members_out["orbit_id"].isin(orbits_to_delete)]

                orbits_iter = orbits_iter[~orbits_iter["orbit_id"].isin(orbits_to_delete)]
                orbit_members_iter = orbit_members_iter[~orbit_members_iter["orbit_id"].isin(orbits_to_delete)]

                obs_id_occurences = orbit_members_out["obs_id"].value_counts()
                duplicate_obs_ids = obs_id_occurences.index.values[obs_id_occurences.values > 1]
            
            observations_iter = observations_iter[~observations_iter["obs_id"].isin(orbit_members_out["obs_id"].values)]
            orbit_members_iter = orbit_members_iter[~orbit_members_iter["orbit_id"].isin(orbits_out["orbit_id"].values)]
            orbit_members_iter = orbit_members_iter[orbit_members_iter["obs_id"].isin(observations_iter["obs_id"].values)]
            orbits_iter = orbits_iter[orbits_iter["orbit_id"].isin(orbit_members_iter["orbit_id"].unique())]

            orbit_members_iter = orbit_members_iter[["orbit_id", "obs_id"]]

            odp_orbits_dfs.append(orbits_out)
            odp_orbit_members_dfs.append(orbit_members_out)

            iterations += 1
            if len(orbits_iter) == 0:
                converged = True

        odp_orbits = pd.concat(odp_orbits_dfs)
        odp_orbit_members = pd.concat(odp_orbit_members_dfs)

        odp_orbits.drop(
            columns=["subset_of", "improved"],
            inplace=True
        )
        odp_orbits, odp_orbit_members = sortLinkages(
            odp_orbits,
            odp_orbit_members,
            observations,
            linkage_id_col="orbit_id"
        )

    else:
        odp_orbits = pd.DataFrame(
            columns=[
                "orbit_id",
                "epoch",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "covariance",
                "arc_length",
                "num_obs",
                "chi2",
                "rchi2"
            ]
        )

        odp_orbit_members = pd.DataFrame(
            columns=[
                "orbit_id", 
                "obs_id", 
                "residual_ra_arcsec", 
                "residual_dec_arcsec", 
                "chi2",
                "outlier"
            ]
        )

    time_end = time.time()
    if verbose:
        print("Number of attribution / differential correction iterations: {}".format(iterations))
        print("Extended and/or merged {} orbits into {} orbits.".format(len(orbits), len(odp_orbits)))
        print("Total time in seconds: {}".format(time_end - time_start))   
        print("-------------------------------")
        print("")
    
    return odp_orbits, odp_orbit_members
