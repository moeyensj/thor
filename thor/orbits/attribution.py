import numpy as np
import pandas as pd
import multiprocessing as mp
from astropy.time import Time
from functools import partial
from sklearn.neighbors import BallTree

from .backend import _init_worker
from .ephemeris import generateEphemeris
from .residuals import calcResiduals

__all__ = [
    "attributeOrbits"
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
        coords_predicted = ephemeris_visit[["RA_deg", "Dec_deg"]].values
        coords_sigma = observations_visit[["RA_sigma_deg", "Dec_sigma_deg"]].values
        
        coords_rad = np.radians(coords)
        coords_predicted_rad = np.radians(coords_predicted)
        
        num_obs = len(coords_predicted_rad)
        k = np.minimum(3, num_obs)

        # Build BallTree with a haversine metric on predicted ephemeris
        tree = BallTree(
            coords_predicted_rad, 
            metric="haversine"
        )
        # Query tree using observed RA, Dec
        d, i = tree.query(
            coords_rad, 
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
            "residual_ra" : residuals[:, 0],
            "residual_dec" : residuals[:, 1],
            "chi2" : stats[:, 0]
        }
        if include_probabilistic:
            attributions["probability"] = stats[:, 1]
            attributions["mahalanobis_distance"] = stats[:, 2]

        attributions = pd.DataFrame(attributions)
        
    else:
        columns = ["orbit_id", "obs_id", "distance", "residual_ra", "residual_dec", "chi2"]
        if include_probabilistic:
            columns += ["probability", "mahalanobis_distance"]
            
        attributions = pd.DataFrame(columns=columns)
    
    return attributions
