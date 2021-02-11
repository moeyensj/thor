import numpy as np
import pandas as pd
import multiprocessing as mp
from astropy.time import Time
from functools import partial
from sklearn.neighbors import BallTree

from ..config import Config
from ..backend import _init_worker
from .orbits import Orbits
from .ephemeris import generateEphemeris
from .residuals import calcResiduals

USE_RAY = Config.USE_RAY
NUM_THREADS = Config.NUM_THREADS

__all__ = [
    "attribution_worker",
    "attributeObservations",
    "extendOrbits"
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
        orbits_chunk_size=100,
        observations_chunk_size=100000,
        threads=NUM_THREADS,
        backend="PYOORB", 
        backend_kwargs={} 
    ):
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
   
    return attributions


def extendOrbits(
        od_orbits, 
        od_orbit_members, 
        observations, 
        min_obs=6,
        eps=1/3600, 
        include_probabilistic=True, 
        orbits_chunk_size=100,
        observations_chunk_size=100000,
        threads=NUM_THREADS,
        backend="PYOORB", 
        backend_kwargs={} 
    ):


    # Run attribution
    attributions = attributeObservations(
        Orbits.fromdf(od_orbits),
        observations,
        eps=eps,
        include_probabilistic=include_probabilistic,
        threads=threads,
        orbits_chunk_size=orbits_chunk_size,
        observations_chunk_size=observations_chunk_size,
        backend=backend,
        backend_kwargs=backend_kwargs
    )
    
    # Remove any orbits that were attributed to less than the minimum number of 
    # of observations desired for "discoverability"
    orbit_occurences = attributions["orbit_id"].value_counts()
    orbit_ids = orbit_occurences[orbit_occurences.values >= min_obs].index.values
    attributions = attributions[attributions["orbit_id"].isin(orbit_ids)]
    
    # Sort remaining orbits by chi2 value, start with the orbit with the lowest mean chi2. 
    # Then for each orbit remove the attributed observations and add them to the orbit's 
    # orbit members. 
    orbit_ids = attributions.groupby(by=["orbit_id"])["chi2"].mean()
    orbit_ids = orbit_ids[np.argsort(orbit_ids.values)].index.values
    od_orbit_members_dfs = []
    obs_ids_attributed = np.array([])
    for orbit_id in orbit_ids:

        attributed = attributions[attributions["orbit_id"] == orbit_id].copy()
        
        if len(attributed) >= min_obs:
            obs_ids_attributed = np.concatenate([obs_ids_attributed, attributed["obs_id"].values])
            od_orbit_members_dfs.append(attributed.reset_index(drop=True))
            attributions = attributions[~attributions["obs_id"].isin(obs_ids_attributed)]

    # Make new orbits and orbit member's dataframes
    od_orbit_members = pd.concat(od_orbit_members_dfs)
    od_orbit_members.reset_index(
        inplace=True,
        drop=True
    )
    od_orbits = od_orbits[od_orbits["orbit_id"].isin(od_orbit_members["orbit_id"].unique())]
    
    od_orbits = od_orbits.sort_values(by=["orbit_id"])
    od_orbit_members = od_orbit_members.merge(observations[["obs_id", "mjd_utc"]], on="obs_id", how="left")
    od_orbit_members.sort_values(
        by=["orbit_id", "mjd_utc"],
        inplace=True
    )
    od_orbits.drop(
        columns=["covariances", "arc_length", "num_obs", "chi2", "num_params"],
        inplace=True
    )
    od_orbit_members.drop(
        columns=["mjd_utc"],
        inplace=True
    )
    
    return od_orbits, od_orbit_members