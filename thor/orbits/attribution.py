import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import concurrent.futures as cf
import logging
import multiprocessing as mp
import time
from functools import partial

import numpy as np
import pandas as pd
from astropy.time import Time
from sklearn.neighbors import BallTree

from ..utils import (
    _checkParallel,
    _initWorker,
    calcChunkSize,
    removeDuplicateObservations,
    sortLinkages,
    yieldChunks,
)
from .ephemeris import generateEphemeris
from .od import differentialCorrection
from .orbits import Orbits
from .residuals import calcResiduals

logger = logging.getLogger(__name__)

__all__ = ["attribution_worker", "attributeObservations", "mergeAndExtendOrbits"]


def attribution_worker(
    orbits,
    observations,
    eps=1 / 3600,
    include_probabilistic=True,
    backend="PYOORB",
    backend_kwargs={},
):

    # Create observer's dictionary from observations
    observers = {}
    for observatory_code in observations["observatory_code"].unique():
        observers[observatory_code] = Time(
            observations[observations["observatory_code"].isin([observatory_code])][
                "mjd_utc"
            ].unique(),
            scale="utc",
            format="mjd",
        )

    # Genereate ephemerides for each orbit at the observation times
    ephemeris = generateEphemeris(
        orbits,
        observers,
        backend=backend,
        backend_kwargs=backend_kwargs,
        num_jobs=1,
        chunk_size=1,
    )

    # Group the predicted ephemerides and observations by visit / exposure
    ephemeris_grouped = ephemeris.groupby(by=["observatory_code", "mjd_utc"])
    ephemeris_visits = [
        ephemeris_grouped.get_group(g) for g in ephemeris_grouped.groups
    ]
    observations_grouped = observations.groupby(by=["observatory_code", "mjd_utc"])
    observations_visits = [
        observations_grouped.get_group(g) for g in observations_grouped.groups
    ]

    # Loop through each unique exposure and visit, find the nearest observations within
    # eps (haversine metric)
    distances = []
    orbit_ids_associated = []
    obs_ids_associated = []
    obs_times_associated = []
    eps_rad = np.radians(eps)
    residuals = []
    stats = []
    for ephemeris_visit, observations_visit in zip(
        ephemeris_visits, observations_visits
    ):

        assert len(ephemeris_visit["mjd_utc"].unique()) == 1
        assert len(observations_visit["mjd_utc"].unique()) == 1
        assert (
            observations_visit["mjd_utc"].unique()[0]
            == ephemeris_visit["mjd_utc"].unique()[0]
        )

        obs_ids = observations_visit[["obs_id"]].values
        obs_times = observations_visit[["mjd_utc"]].values
        orbit_ids = ephemeris_visit[["orbit_id"]].values
        coords = observations_visit[["RA_deg", "Dec_deg"]].values
        coords_latlon = observations_visit[["Dec_deg"]].values
        coords_predicted = ephemeris_visit[["RA_deg", "Dec_deg"]].values
        coords_sigma = observations_visit[["RA_sigma_deg", "Dec_sigma_deg"]].values

        # Haversine metric requires latitude first then longitude...
        coords_latlon = np.radians(observations_visit[["Dec_deg", "RA_deg"]].values)
        coords_predicted_latlon = np.radians(
            ephemeris_visit[["Dec_deg", "RA_deg"]].values
        )

        num_obs = len(coords_predicted)
        k = np.minimum(3, num_obs)

        # Build BallTree with a haversine metric on predicted ephemeris
        tree = BallTree(coords_predicted_latlon, metric="haversine")
        # Query tree using observed RA, Dec
        d, i = tree.query(
            coords_latlon,
            k=k,
            return_distance=True,
            dualtree=True,
            breadth_first=False,
            sort_results=False,
        )

        # Select all observations with distance smaller or equal
        # to the maximum given distance
        mask = np.where(d <= eps_rad)

        if len(d[mask]) > 0:
            orbit_ids_associated.append(orbit_ids[i[mask]])
            obs_ids_associated.append(obs_ids[mask[0]])
            obs_times_associated.append(obs_times[mask[0]])
            distances.append(d[mask].reshape(-1, 1))

            residuals_visit, stats_visit = calcResiduals(
                coords[mask[0]],
                coords_predicted[i[mask]],
                sigmas_actual=coords_sigma[mask[0]],
                include_probabilistic=True,
            )
            residuals.append(residuals_visit)
            stats.append(np.vstack(stats_visit).T)

    if len(distances) > 0:
        distances = np.degrees(np.vstack(distances))
        orbit_ids_associated = np.vstack(orbit_ids_associated)
        obs_ids_associated = np.vstack(obs_ids_associated)
        obs_times_associated = np.vstack(obs_times_associated)
        residuals = np.vstack(residuals)
        stats = np.vstack(stats)

        attributions = {
            "orbit_id": orbit_ids_associated[:, 0],
            "obs_id": obs_ids_associated[:, 0],
            "mjd_utc": obs_times_associated[:, 0],
            "distance": distances[:, 0],
            "residual_ra_arcsec": residuals[:, 0] * 3600,
            "residual_dec_arcsec": residuals[:, 1] * 3600,
            "chi2": stats[:, 0],
        }
        if include_probabilistic:
            attributions["probability"] = stats[:, 1]
            attributions["mahalanobis_distance"] = stats[:, 2]

        attributions = pd.DataFrame(attributions)

    else:
        columns = [
            "orbit_id",
            "obs_id",
            "mjd_utc",
            "distance",
            "residual_ra_arcsec",
            "residual_dec_arcsec",
            "chi2",
        ]
        if include_probabilistic:
            columns += ["probability", "mahalanobis_distance"]

        attributions = pd.DataFrame(columns=columns)

    return attributions


def attributeObservations(
    orbits,
    observations,
    eps=5 / 3600,
    include_probabilistic=True,
    backend="PYOORB",
    backend_kwargs={},
    orbits_chunk_size=10,
    observations_chunk_size=100000,
    num_jobs=1,
    parallel_backend="cf",
):
    logger.info("Running observation attribution...")
    time_start = time.time()

    num_orbits = len(orbits)

    attribution_dfs = []

    parallel, num_workers = _checkParallel(num_jobs, parallel_backend)
    if num_workers > 1:

        if parallel_backend == "ray":
            import ray

            if not ray.is_initialized():
                ray.init(address="auto")

            attribution_worker_ray = ray.remote(attribution_worker)
            attribution_worker_ray = attribution_worker_ray.options(
                num_returns=1, num_cpus=1
            )

            # Send up to orbits_chunk_size orbits to each OD worker for processing
            chunk_size_ = calcChunkSize(
                num_orbits, num_workers, orbits_chunk_size, min_chunk_size=1
            )
            orbits_split = orbits.split(chunk_size_)

            obs_oids = []
            for observations_c in yieldChunks(observations, observations_chunk_size):
                obs_oids.append(ray.put(observations_c))

            p = []
            for obs_oid in obs_oids:
                for orbit_i in orbits_split:
                    p.append(
                        attribution_worker_ray.remote(
                            orbit_i,
                            obs_oid,
                            eps=eps,
                            include_probabilistic=include_probabilistic,
                            backend=backend,
                            backend_kwargs=backend_kwargs,
                        )
                    )

                attribution_dfs_i = ray.get(p)
                attribution_dfs += attribution_dfs_i

        elif parallel_backend == "mp":
            p = mp.Pool(
                processes=num_workers,
                initializer=_initWorker,
            )

            # Send up to orbits_chunk_size orbits to each OD worker for processing
            chunk_size_ = calcChunkSize(
                num_orbits, num_workers, orbits_chunk_size, min_chunk_size=1
            )
            orbits_split = orbits.split(chunk_size_)

            for observations_c in yieldChunks(observations, observations_chunk_size):

                obs = [observations_c for i in range(len(orbits_split))]
                attribution_dfs_i = p.starmap(
                    partial(
                        attribution_worker,
                        eps=eps,
                        include_probabilistic=include_probabilistic,
                        backend=backend,
                        backend_kwargs=backend_kwargs,
                    ),
                    zip(
                        orbits_split,
                        obs,
                    ),
                )
                attribution_dfs += attribution_dfs_i

            p.close()

        elif parallel_backend == "cf":
            with cf.ProcessPoolExecutor(
                max_workers=num_workers, initializer=_initWorker
            ) as executor:
                futures = []
                for observations_c in yieldChunks(
                    observations, observations_chunk_size
                ):
                    for orbit_c in orbits.split(orbits_chunk_size):
                        futures.append(
                            executor.submit(
                                attribution_worker,
                                orbit_c,
                                observations_c,
                                eps=eps,
                                include_probabilistic=include_probabilistic,
                                backend=backend,
                                backend_kwargs=backend_kwargs,
                            )
                        )
                attribution_dfs = []
                for future in cf.as_completed(futures):
                    attribution_dfs.append(future.result())

        else:
            raise ValueError(
                "Invalid parallel_backend '{}'. Must be one of 'ray', 'mp', or 'cf'.".format(
                    parallel_backend
                )
            )

    else:
        for observations_c in yieldChunks(observations, observations_chunk_size):
            for orbit_c in orbits.split(orbits_chunk_size):
                attribution_df_i = attribution_worker(
                    orbit_c,
                    observations_c,
                    eps=eps,
                    include_probabilistic=include_probabilistic,
                    backend=backend,
                    backend_kwargs=backend_kwargs,
                )
                attribution_dfs.append(attribution_df_i)

    attributions = pd.concat(attribution_dfs)
    attributions.sort_values(
        by=["orbit_id", "mjd_utc", "distance"], inplace=True, ignore_index=True
    )

    time_end = time.time()
    logger.info(
        "Attributed {} observations to {} orbits.".format(
            attributions["obs_id"].nunique(), attributions["orbit_id"].nunique()
        )
    )
    logger.info(
        "Attribution completed in {:.3f} seconds.".format(time_end - time_start)
    )
    return attributions


def mergeAndExtendOrbits(
    orbits,
    orbit_members,
    observations,
    min_obs=6,
    min_arc_length=1.0,
    contamination_percentage=20.0,
    rchi2_threshold=5,
    eps=1 / 3600,
    delta=1e-8,
    max_iter=20,
    method="central",
    fit_epoch=False,
    backend="PYOORB",
    backend_kwargs={},
    orbits_chunk_size=10,
    observations_chunk_size=100000,
    num_jobs=60,
    parallel_backend="cf",
):
    """
    Attempt to extend an orbit's observational arc by running
    attribution on the observations. This is an iterative process: attribution
    is run, any observations found for each orbit are added to that orbit and differential correction is
    run. Orbits which are subset's of other orbits are removed. Iteration continues until there are no
    duplicate observation assignments.

    Parameters
    ----------
    orbit_chunk_size : int, optional
        Number of orbits to send to each job.
    observations_chunk_size : int, optional
        Number of observations to process per batch.
    num_jobs : int, optional
        Number of jobs to launch.
    parallel_backend : str, optional
        Which parallelization backend to use {'ray', 'mp', cf}. Defaults to using Python's concurrent.futures
        module ('cf').
    """
    time_start = time.time()
    logger.info("Running orbit extension and merging...")

    if len(observations) > 0:
        orbits_iter, orbit_members_iter = sortLinkages(
            orbits, orbit_members, observations, linkage_id_col="orbit_id"
        )

    else:
        orbits_iter = orbits.copy()
        orbit_members_iter = orbit_members.copy()

    iterations = 0
    odp_orbits_dfs = []
    odp_orbit_members_dfs = []
    observations_iter = observations.copy()
    if len(orbits_iter) > 0 and len(observations_iter) > 0:
        converged = False

        while not converged:
            # Run attribution
            attributions = attributeObservations(
                Orbits.from_df(orbits_iter),
                observations_iter,
                eps=eps,
                include_probabilistic=True,
                backend=backend,
                backend_kwargs=backend_kwargs,
                orbits_chunk_size=orbits_chunk_size,
                observations_chunk_size=observations_chunk_size,
                num_jobs=num_jobs,
                parallel_backend=parallel_backend,
            )

            assert np.all(
                np.isin(
                    orbit_members_iter["obs_id"].unique(),
                    observations_iter["obs_id"].unique(),
                )
            )

            # Attributions are sorted by orbit ID, observation time and
            # angular distance. Keep only the one observation with smallest distance
            # for any orbits that have multiple observations attributed at the same observation time.
            attributions.drop_duplicates(
                subset=["orbit_id", "mjd_utc"],
                keep="first",
                inplace=True,
                ignore_index=True,
            )
            orbit_members_iter = attributions[
                [
                    "orbit_id",
                    "obs_id",
                    "residual_ra_arcsec",
                    "residual_dec_arcsec",
                    "chi2",
                ]
            ]
            orbits_iter = orbits_iter[
                orbits_iter["orbit_id"].isin(orbit_members_iter["orbit_id"].unique())
            ]

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
                min_arc_length=min_arc_length,
                contamination_percentage=contamination_percentage,
                delta=delta,
                method=method,
                max_iter=max_iter,
                fit_epoch=False,
                backend=backend,
                backend_kwargs=backend_kwargs,
                chunk_size=orbits_chunk_size,
                num_jobs=num_jobs,
                parallel_backend=parallel_backend,
            )
            orbit_members_iter = orbit_members_iter[orbit_members_iter["outlier"] == 0]
            orbit_members_iter.reset_index(inplace=True, drop=True)

            # Remove the orbits that were not improved from the pool of available orbits. Orbits that were not improved
            # are orbits that have already iterated to their best-fit solution given the observations available. These orbits
            # are unlikely to recover more observations in subsequent iterations and so can be saved for output.
            not_improved = orbits_iter[orbits_iter["improved"] == False][
                "orbit_id"
            ].values
            orbits_out = orbits_iter[orbits_iter["orbit_id"].isin(not_improved)].copy()
            orbit_members_out = orbit_members_iter[
                orbit_members_iter["orbit_id"].isin(not_improved)
            ].copy()
            not_improved_obs_ids = orbit_members_out["obs_id"].values

            # If some orbits that were not improved still share observations, keep the orbit with the lowest
            # reduced chi2 in the pool of orbits but delete the others.
            obs_id_occurences = orbit_members_out["obs_id"].value_counts()
            duplicate_obs_ids = obs_id_occurences.index.values[
                obs_id_occurences.values > 1
            ]

            logger.info(
                "There are {} observations that appear in more than one orbit.".format(
                    len(duplicate_obs_ids)
                )
            )
            orbits_out, orbit_members_out = removeDuplicateObservations(
                orbits_out,
                orbit_members_out,
                min_obs=min_obs,
                linkage_id_col="orbit_id",
                filter_cols=["num_obs", "arc_length", "r_sigma", "v_sigma"],
                ascending=[False, False, True, True],
            )

            observations_iter = observations_iter[
                ~observations_iter["obs_id"].isin(orbit_members_out["obs_id"].values)
            ]
            orbit_members_iter = orbit_members_iter[
                ~orbit_members_iter["orbit_id"].isin(orbits_out["orbit_id"].values)
            ]
            orbit_members_iter = orbit_members_iter[
                orbit_members_iter["obs_id"].isin(observations_iter["obs_id"].values)
            ]
            orbits_iter = orbits_iter[
                orbits_iter["orbit_id"].isin(orbit_members_iter["orbit_id"].unique())
            ]
            orbit_members_iter = orbit_members_iter[["orbit_id", "obs_id"]]

            odp_orbits_dfs.append(orbits_out)
            odp_orbit_members_dfs.append(orbit_members_out)

            iterations += 1
            if len(orbits_iter) == 0:
                converged = True

        odp_orbits = pd.concat(odp_orbits_dfs)
        odp_orbit_members = pd.concat(odp_orbit_members_dfs)

        odp_orbits.drop(columns=["improved"], inplace=True)
        odp_orbits, odp_orbit_members = sortLinkages(
            odp_orbits, odp_orbit_members, observations, linkage_id_col="orbit_id"
        )

    else:
        odp_orbits = pd.DataFrame(
            columns=[
                "orbit_id",
                "mjd_tdb",
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
                "rchi2",
            ]
        )

        odp_orbit_members = pd.DataFrame(
            columns=[
                "orbit_id",
                "obs_id",
                "residual_ra_arcsec",
                "residual_dec_arcsec",
                "chi2",
                "outlier",
            ]
        )

    time_end = time.time()
    logger.info(
        "Number of attribution / differential correction iterations: {}".format(
            iterations
        )
    )
    logger.info(
        "Extended and/or merged {} orbits into {} orbits.".format(
            len(orbits), len(odp_orbits)
        )
    )
    logger.info(
        "Orbit extension and merging completed in {:.3f} seconds.".format(
            time_end - time_start
        )
    )

    return odp_orbits, odp_orbit_members
