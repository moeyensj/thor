import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import concurrent.futures as cf
import copy
import logging
import multiprocessing as mp
import time
from functools import partial

import numpy as np
import pandas as pd
from adam_core.coordinates import CartesianCoordinates, CoordinateCovariances
from adam_core.orbits import Orbits
from adam_core.time import Timestamp
from astropy import units as u
from astropy.time import Time
from scipy.linalg import solve

from ..backend import PYOORB
from ..utils import (
    _checkParallel,
    _initWorker,
    calcChunkSize,
    sortLinkages,
    yieldChunks,
)
from .residuals import calcResiduals

logger = logging.getLogger(__name__)

__all__ = ["od_worker", "od", "differentialCorrection"]


def od_worker(
    orbits_list,
    observations_list,
    rchi2_threshold=100,
    min_obs=5,
    min_arc_length=1.0,
    contamination_percentage=20,
    delta=1e-6,
    max_iter=20,
    method="central",
    fit_epoch=False,
    test_orbit=None,
    backend="PYOORB",
    backend_kwargs={},
):
    od_orbits_dfs = []
    od_orbit_members_dfs = []
    for orbit, observations in zip(orbits_list, observations_list):
        try:
            assert orbit.orbit_id[0].as_py() == observations["orbit_id"].unique()[0]
            assert np.all(
                sorted(observations["mjd_utc"].values) == observations["mjd_utc"].values
            )
            assert len(np.unique(observations["mjd_utc"].values)) == len(
                observations["mjd_utc"].values
            )
        except:
            err = (
                "Invalid observations and orbit have been passed to the OD code.\n"
                "Orbit ID: {}".format(orbit.orbit_id[0].as_py())
            )
            raise ValueError(err)

        time_start = time.time()
        logger.debug(f"Differentially correcting orbit {orbit.orbit_id[0].as_py()}...")
        od_orbit, od_orbit_members = od(
            orbit,
            observations,
            rchi2_threshold=rchi2_threshold,
            min_obs=min_obs,
            min_arc_length=min_arc_length,
            contamination_percentage=contamination_percentage,
            delta=delta,
            max_iter=max_iter,
            method=method,
            fit_epoch=fit_epoch,
            test_orbit=test_orbit,
            backend=backend,
            backend_kwargs=backend_kwargs,
        )
        time_end = time.time()
        duration = time_end - time_start
        logger.debug(
            f"OD for orbit {orbit.orbit_id[0].as_py()} completed in {duration:.3f}s."
        )

        od_orbits_dfs.append(od_orbit)
        od_orbit_members_dfs.append(od_orbit_members)

    od_orbits = pd.concat(od_orbits_dfs, ignore_index=True)
    od_orbit_members = pd.concat(od_orbit_members_dfs, ignore_index=True)
    return od_orbits, od_orbit_members


def od(
    orbit,
    observations,
    rchi2_threshold=100,
    min_obs=5,
    min_arc_length=1.0,
    contamination_percentage=0.0,
    delta=1e-6,
    max_iter=20,
    method="central",
    fit_epoch=False,
    test_orbit=None,
    backend="PYOORB",
    backend_kwargs={},
):
    if backend == "PYOORB":
        backend = PYOORB(**backend_kwargs)
    else:
        err = "backend should be 'PYOORB'"
        raise ValueError(err)

    if method not in ["central", "finite"]:
        err = "method should be one of 'central' or 'finite'."
        raise ValueError(err)

    observables = ["RA_deg", "Dec_deg"]

    obs_ids_all = observations["obs_id"].values
    coords = observations[observables].values
    coords_sigma = observations[["RA_sigma_deg", "Dec_sigma_deg"]].values

    observers = {}
    for observatory_code in observations["observatory_code"].unique():
        observatory_mask = observations["observatory_code"].isin([observatory_code])
        observers[observatory_code] = Time(
            observations[observatory_mask]["mjd_utc"].unique(),
            format="mjd",
            scale="utc",
        )

    # FLAG: can we stop iterating to find a solution?
    converged = False
    # FLAG: has an orbit with reduced chi2 less than the reduced chi2 of the input orbit been found?
    improved = False
    # FLAG: has an orbit with reduced chi2 less than the rchi2_threshold been found?
    solution_found = False
    # FLAG: is this orbit processable? Does it have at least min_obs observations?
    processable = True
    # FLAG: is this the first iteration with a successful differential correction (this solution is always stored as the solution
    # which needs to be improved.. input orbits may not have been previously corrected with current set of observations so this
    # forces at least one succesful iteration to have been taken.)
    first_solution = True

    num_obs = len(observations)
    if num_obs < min_obs:
        logger.debug("This orbit has fewer than {} observations.".format(min_obs))
        processable = False
    else:
        num_outliers = int(num_obs * contamination_percentage / 100.0)
        num_outliers = np.maximum(np.minimum(num_obs - min_obs, num_outliers), 0)
        logger.debug("Maximum number of outliers allowed: {}".format(num_outliers))
        outliers_tried = 0

        # Calculate chi2 for residuals on the given observations
        # for the current orbit, the goal is for the orbit to improve
        # such that the chi2 improves
        orbit_prev_ = copy.deepcopy(orbit)

        ephemeris_prev_ = backend._generateEphemeris(orbit_prev_, observers)
        residuals_prev_, stats_prev_ = calcResiduals(
            coords,
            ephemeris_prev_[observables].values,
            sigmas_actual=coords_sigma,
            include_probabilistic=False,
        )
        num_obs_ = len(observations)
        chi2_prev_ = stats_prev_[0]
        chi2_total_prev_ = np.sum(chi2_prev_)
        rchi2_prev_ = np.sum(chi2_prev_) / (2 * num_obs - 6)

        # Save the initial orbit in case we need to reset
        # to it later
        orbit_prev = orbit_prev_
        ephemeris_prev = ephemeris_prev_
        residuals_prev = residuals_prev_
        num_obs = num_obs_
        chi2_prev = chi2_prev_
        chi2_total_prev = chi2_total_prev_
        rchi2_prev = rchi2_prev_

        ids_mask = np.array([True for i in range(num_obs)])
        times_all = ephemeris_prev["mjd_utc"].values
        obs_id_outlier = []
        delta_prev = delta
        iterations = 0

        DELTA_INCREASE_FACTOR = 5
        DELTA_DECREASE_FACTOR = 100

        max_iter_i = max_iter
        max_iter_outliers = max_iter * (num_outliers + 1)

    while not converged and processable:
        iterations += 1

        # We add 1 here because the iterations are counted as they start, not
        # as they finish. There are a lot of 'continue' statements down below that
        # will exit the current iteration if something fails which makes accounting for
        # iterations at the start of an iteration easier.
        if iterations == max_iter_outliers + 1:
            logger.debug(f"Maximum number of iterations completed.")
            break
        if iterations == max_iter_i + 1 and (
            solution_found or (num_outliers == outliers_tried)
        ):
            logger.debug(f"Maximum number of iterations completed.")
            break
        logger.debug(f"Starting iteration number: {iterations}/{max_iter_outliers}")

        # Make sure delta is well bounded
        if delta_prev < 1e-14:
            delta_prev *= DELTA_INCREASE_FACTOR
            logger.debug("Delta is too small, increasing.")
        elif delta_prev > 1e-2:
            delta_prev /= DELTA_DECREASE_FACTOR
            logger.debug("Delta is too large, decreasing.")
        else:
            pass

        delta_iter = delta_prev
        logger.debug(f"Starting iteration {iterations} with delta {delta_iter}.")

        # Initialize the partials derivatives matrix
        if num_obs > 6 and fit_epoch:
            num_params = 7
        else:
            num_params = 6

        A = np.zeros((coords.shape[1], num_params, num_obs))
        ATWA = np.zeros((num_params, num_params, num_obs))
        ATWb = np.zeros((num_params, 1, num_obs))

        # Generate ephemeris with current nominal orbit
        ephemeris_nom = backend._generateEphemeris(orbit_prev, observers)
        coords_nom = ephemeris_nom[observables].values

        # Modify each component of the state by a small delta
        d = np.zeros((1, 7))
        for i in range(num_params):

            # zero the delta vector
            d *= 0.0

            # x, y, z [au]: 0, 1, 2
            # vx, vy, vz [au per day]: 3, 4, 5
            # time [days] : 6
            if i < 3:
                delta_iter = delta_prev

                d[0, i] = orbit_prev.coordinates.values[0, i] * delta_iter
            elif i < 6:
                delta_iter = delta_prev

                d[0, i] = orbit_prev.coordinates.values[0, i] * delta_iter
            else:
                delta_iter = delta_prev / 100000

                d[0, i] = delta_iter

            # Modify component i of the orbit by a small delta
            cartesian_elements_p = orbit_prev.coordinates.values + d[0, :6]
            orbit_iter_p = Orbits.from_kwargs(
                coordinates=CartesianCoordinates.from_kwargs(
                    x=cartesian_elements_p[:, 0],
                    y=cartesian_elements_p[:, 1],
                    z=cartesian_elements_p[:, 2],
                    vx=cartesian_elements_p[:, 3],
                    vy=cartesian_elements_p[:, 4],
                    vz=cartesian_elements_p[:, 5],
                    time=orbit_prev.coordinates.time,
                    origin=orbit_prev.coordinates.origin,
                    frame=orbit_prev.coordinates.frame,
                )
            )

            # Calculate the modified ephemerides
            ephemeris_mod_p = backend._generateEphemeris(orbit_iter_p, observers)
            coords_mod_p = ephemeris_mod_p[observables].values

            delta_denom = d[0, i]
            if method == "central":

                # Modify component i of the orbit by a small delta
                cartesian_elements_n = orbit_prev.coordinates.values - d[0, :6]
                orbit_iter_n = Orbits.from_kwargs(
                    coordinates=CartesianCoordinates.from_kwargs(
                        x=cartesian_elements_n[:, 0],
                        y=cartesian_elements_n[:, 1],
                        z=cartesian_elements_n[:, 2],
                        vx=cartesian_elements_n[:, 3],
                        vy=cartesian_elements_n[:, 4],
                        vz=cartesian_elements_n[:, 5],
                        time=orbit_prev.coordinates.time,
                        origin=orbit_prev.coordinates.origin,
                        frame=orbit_prev.coordinates.frame,
                    )
                )

                # Calculate the modified ephemerides
                ephemeris_mod_n = backend._generateEphemeris(orbit_iter_n, observers)
                coords_mod_n = ephemeris_mod_n[observables].values

                delta_denom *= 2

            else:
                coords_mod_n = coords_nom

            residuals_mod, _ = calcResiduals(
                coords_mod_p,
                coords_mod_n,
                sigmas_actual=None,
                include_probabilistic=False,
            )

            for n in range(num_obs):
                try:
                    A[:, i : i + 1, n] = (
                        residuals_mod[ids_mask][n : n + 1].T / delta_denom
                    )
                except RuntimeError:
                    print(orbit_prev.orbit_id)

        for n in range(num_obs):
            W = np.diag(1 / coords_sigma[n] ** 2)
            ATWA[:, :, n] = A[:, :, n].T @ W @ A[:, :, n]
            ATWb[:, :, n] = A[:, :, n].T @ W @ residuals_prev[n : n + 1].T

        ATWA = np.sum(ATWA, axis=2)
        ATWb = np.sum(ATWb, axis=2)

        ATWA_condition = np.linalg.cond(ATWA)
        ATWb_condition = np.linalg.cond(ATWb)

        if (ATWA_condition > 1e15) or (ATWb_condition > 1e15):
            delta_prev /= DELTA_DECREASE_FACTOR
            continue
        if np.any(np.isnan(ATWA)) or np.any(np.isnan(ATWb)):
            delta_prev *= DELTA_INCREASE_FACTOR
            continue
        else:
            try:
                delta_state = solve(
                    ATWA,
                    ATWb,
                ).T
                covariance_matrix = np.linalg.inv(ATWA)
                variances = np.diag(covariance_matrix)
                if np.any(variances <= 0) or np.any(np.isnan(variances)):
                    delta_prev /= DELTA_DECREASE_FACTOR
                    logger.debug(
                        "Variances are negative, 0.0, or NaN. Discarding solution."
                    )
                    continue

                r_variances = variances[0:3]
                r_sigma = np.sqrt(np.sum(r_variances))
                r = orbit_prev.coordinates.r_mag
                if (r_sigma / r) > 1:
                    delta_prev /= DELTA_DECREASE_FACTOR
                    logger.debug(
                        "Covariance matrix is largely unconstrained. Discarding solution."
                    )
                    continue

                if np.any(np.isnan(covariance_matrix)):
                    delta_prev *= DELTA_INCREASE_FACTOR
                    logger.debug(
                        "Covariance matrix contains NaNs. Discarding solution."
                    )
                    continue

            except np.linalg.LinAlgError:
                delta_prev *= DELTA_INCREASE_FACTOR
                continue

        if num_params == 6:
            d_state = delta_state
            d_time = 0
        else:
            d_state = delta_state[0, :6]
            d_time = delta_state[0, 6]

        if np.linalg.norm(d_state[:3]) < 1e-16:
            logger.debug("Change in state is less than 1e-16 au, discarding solution.")
            delta_prev *= DELTA_DECREASE_FACTOR
            continue
        if np.linalg.norm(d_state[:3]) > 100:
            delta_prev /= DELTA_DECREASE_FACTOR
            logger.debug("Change in state is more than 100 au, discarding solution.")
            continue

        cartesian_elements = orbit_prev.coordinates.values + d_state
        orbit_iter = Orbits.from_kwargs(
            orbit_id=orbit_prev.orbit_id,
            coordinates=CartesianCoordinates.from_kwargs(
                x=cartesian_elements[:, 0],
                y=cartesian_elements[:, 1],
                z=cartesian_elements[:, 2],
                vx=cartesian_elements[:, 3],
                vy=cartesian_elements[:, 4],
                vz=cartesian_elements[:, 5],
                covariance=CoordinateCovariances.from_matrix(
                    covariance_matrix.reshape(1, 6, 6)
                ),
                time=orbit_prev.coordinates.time,
                origin=orbit_prev.coordinates.origin,
                frame=orbit_prev.coordinates.frame,
            ),
        )
        if np.linalg.norm(orbit_iter.coordinates.v_mag) > 1:
            delta_prev *= DELTA_INCREASE_FACTOR
            logger.debug("Orbit is moving extraordinarily fast, discarding solution.")
            continue

        # Generate ephemeris with current nominal orbit
        ephemeris_iter = backend._generateEphemeris(orbit_iter, observers)
        coords_iter = ephemeris_iter[observables].values

        residuals, stats = calcResiduals(
            coords, coords_iter, sigmas_actual=coords_sigma, include_probabilistic=False
        )
        chi2_iter = stats[0]
        chi2_total_iter = np.sum(chi2_iter[ids_mask])
        rchi2_iter = chi2_total_iter / (2 * num_obs - num_params)
        arc_length = times_all[ids_mask].max() - times_all[ids_mask].min()

        # If the new orbit has lower residuals than the previous,
        # accept the orbit and continue iterating until max iterations has been
        # reached. Once max iterations have been reached and the orbit still has not converged
        # to an acceptable solution, try removing an observation as an outlier and iterate again.
        if (
            (rchi2_iter < rchi2_prev) or first_solution
        ) and arc_length >= min_arc_length:

            if first_solution:
                logger.debug(
                    "Storing first successful differential correction iteration for these observations."
                )
                first_solution = False
            else:
                logger.debug("Potential improvement orbit has been found.")
            orbit_prev = orbit_iter
            residuals_prev = residuals
            chi2_prev = chi2_iter
            chi2_total_prev = chi2_total_iter
            rchi2_prev = rchi2_iter

            if rchi2_prev <= rchi2_prev_:
                improved = True

            if rchi2_prev <= rchi2_threshold:
                logger.debug("Potential solution orbit has been found.")
                solution_found = True
                converged = True

        elif (
            num_outliers > 0
            and outliers_tried <= num_outliers
            and iterations > max_iter_i
            and not solution_found
        ):

            logger.debug("Attempting to identify possible outliers.")
            # Previous fits have failed, lets reset the current best fit orbit back to its original
            # state and re-run fitting, this time removing outliers
            orbit_prev = orbit_prev_
            ephemeris_prev = ephemeris_prev_
            residuals_prev = residuals_prev_
            num_obs = num_obs_
            chi2_prev = chi2_prev_
            chi2_total_prev = chi2_total_prev_
            rchi2_prev = rchi2_prev_
            delta_prev = delta

            # Select i highest observations that contribute to
            # chi2 (and thereby the residuals)
            remove = chi2_prev.argsort()[-(outliers_tried + 1) :]

            # Grab the obs_ids for these outliers
            obs_id_outlier = obs_ids_all[remove]
            num_obs = len(observations) - len(obs_id_outlier)
            ids_mask = np.isin(obs_ids_all, obs_id_outlier, invert=True)
            arc_length = times_all[ids_mask].max() - times_all[ids_mask].min()

            logger.debug("Possible outlier(s): {}".format(obs_id_outlier))
            outliers_tried += 1
            if arc_length >= min_arc_length:
                max_iter_i = max_iter * (outliers_tried + 1)
            else:
                logger.debug(
                    "Removing the outlier will cause the arc length to go below the minimum."
                )

        # If the new orbit does not have lower residuals, try changing
        # delta to see if we get an improvement
        else:
            # logger.debug("Orbit did not improve since previous iteration, decrease delta and continue.")
            # delta_prev /= DELTA_DECREASE_FACTOR
            pass

        logger.debug(
            "Current r-chi2: {}, Previous r-chi2: {}, Max Iterations: {}, Outliers Tried: {}".format(
                rchi2_iter, rchi2_prev, max_iter_i, outliers_tried
            )
        )

    logger.debug("Solution found: {}".format(solution_found))
    logger.debug("First solution: {}".format(first_solution))

    if not solution_found or not processable or first_solution:

        od_orbit = pd.DataFrame(
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
                "r",
                "r_sigma",
                "v",
                "v_sigma",
                "arc_length",
                "num_obs",
                "num_params",
                "num_iterations",
                "chi2",
                "rchi2",
                "improved",
            ]
        )

        od_orbit_members = pd.DataFrame(
            columns=[
                "orbit_id",
                "obs_id",
                "residual_ra_arcsec",
                "residual_dec_arcsec",
                "chi2",
                "outlier",
            ]
        )

    else:
        obs_times = observations["mjd_utc"].values[ids_mask]
        od_orbit = orbit_prev.to_dataframe()
        od_orbit["r"] = orbit_prev.coordinates.r_mag
        od_orbit["r_sigma"] = orbit_prev.coordinates.sigma_r_mag
        od_orbit["v"] = orbit_prev.coordinates.v_mag
        od_orbit["v_sigma"] = orbit_prev.coordinates.sigma_v_mag
        od_orbit["arc_length"] = np.max(obs_times) - np.min(obs_times)
        od_orbit["num_obs"] = num_obs
        od_orbit["num_params"] = num_params
        od_orbit["num_iterations"] = iterations
        od_orbit["chi2"] = chi2_total_prev
        od_orbit["rchi2"] = rchi2_prev
        od_orbit["improved"] = improved

        od_orbit_members = pd.DataFrame(
            {
                "orbit_id": [
                    orbit_prev.orbit_id[0].as_py() for i in range(len(obs_ids_all))
                ],
                "obs_id": obs_ids_all,
                "residual_ra_arcsec": residuals_prev[:, 0] * 3600,
                "residual_dec_arcsec": residuals_prev[:, 1] * 3600,
                "chi2": chi2_prev,
                "outlier": np.zeros(len(obs_ids_all), dtype=int),
            }
        )
        od_orbit_members.loc[
            od_orbit_members["obs_id"].isin(obs_id_outlier), "outlier"
        ] = 1

    return od_orbit, od_orbit_members


def differentialCorrection(
    orbits,
    orbit_members,
    observations,
    min_obs=5,
    min_arc_length=1.0,
    contamination_percentage=20,
    rchi2_threshold=100,
    delta=1e-8,
    max_iter=20,
    method="central",
    fit_epoch=False,
    test_orbit=None,
    backend="PYOORB",
    backend_kwargs={},
    chunk_size=10,
    num_jobs=60,
    parallel_backend="cf",
):
    """
    Differentially correct (via finite/central differencing).

    Parameters
    ----------
    chunk_size : int, optional
        Number of orbits to send to each job.
    num_jobs : int, optional
        Number of jobs to launch.
    parallel_backend : str, optional
        Which parallelization backend to use {'ray', 'mp', 'cf'}. Defaults to using Python's concurrent.futures
        module ('cf').
    """
    logger.info("Running differential correction...")

    time_start = time.time()

    if len(orbits) > 0 and len(orbit_members) > 0:

        orbits_, orbit_members_ = sortLinkages(orbits, orbit_members, observations)

        start = time.time()
        logger.debug("Merging observations on linkage members...")
        linked_observations = orbit_members_[
            orbit_members_[["orbit_id", "obs_id"]]["orbit_id"].isin(
                orbits_["orbit_id"].values
            )
        ].merge(observations, on="obs_id", how="left")
        duration = time.time() - start
        logger.debug(f"Merging completed in {duration:.3f}s.")

        start = time.time()
        logger.debug("Grouping observations by orbit ID...")
        grouped_observations = linked_observations.groupby(by=["orbit_id"])
        logger.debug("Splitting grouped observations by orbit ID...")
        observations_split = [
            grouped_observations.get_group(g).reset_index(drop=True)
            for g in grouped_observations.groups
        ]
        duration = time.time() - start
        logger.debug(f"Grouping and splitting completed in {duration:.3f}s.")

        orbits_initial = Orbits.from_flat_dataframe(orbits_)
        orbits_split = [orbit for orbit in orbits_initial]
        num_orbits = len(orbits)

        parallel, num_workers = _checkParallel(num_jobs, parallel_backend)
        if num_workers > 1:

            if parallel_backend == "ray":
                import ray

                if not ray.is_initialized():
                    ray.init(address="auto")

                od_worker_ray = ray.remote(od_worker)
                od_worker_ray = od_worker_ray.options(num_returns=2, num_cpus=1)

                # Send up to chunk_size orbits to each OD worker for processing
                chunk_size_ = calcChunkSize(
                    num_orbits, num_workers, chunk_size, min_chunk_size=1
                )
                logger.info(
                    f"Distributing linkages in chunks of {chunk_size_} to {num_workers} ray workers."
                )

                # Put the observations and orbits into ray's local object storage ("plasma")
                orbit_oids = []
                observation_oids = []
                for orbits_i, observations_i in zip(
                    yieldChunks(orbits_split, chunk_size_),
                    yieldChunks(observations_split, chunk_size_),
                ):
                    orbit_oids.append(ray.put(orbits_i))
                    observation_oids.append(ray.put(observations_i))

                od_orbits_oids = []
                od_orbit_members_oids = []
                for orbits_oid, observations_oid in zip(orbit_oids, observation_oids):

                    od_orbits_oid, od_orbit_members_oid = od_worker_ray.remote(
                        orbits_oid,
                        observations_oid,
                        rchi2_threshold=rchi2_threshold,
                        min_obs=min_obs,
                        min_arc_length=min_arc_length,
                        contamination_percentage=contamination_percentage,
                        delta=delta,
                        max_iter=max_iter,
                        method=method,
                        fit_epoch=fit_epoch,
                        test_orbit=test_orbit,
                        backend=backend,
                        backend_kwargs=backend_kwargs,
                    )
                    od_orbits_oids.append(od_orbits_oid)
                    od_orbit_members_oids.append(od_orbit_members_oid)

                od_orbits_dfs = ray.get(od_orbits_oids)
                od_orbit_members_dfs = ray.get(od_orbit_members_oids)

            elif parallel_backend == "mp":

                chunk_size_ = calcChunkSize(
                    num_orbits, num_workers, chunk_size, min_chunk_size=1
                )
                logger.info(
                    f"Distributing linkages in chunks of {chunk_size_} to {num_workers} workers."
                )

                p = mp.Pool(
                    processes=num_workers,
                    initializer=_initWorker,
                )
                results = p.starmap(
                    partial(
                        od_worker,
                        rchi2_threshold=rchi2_threshold,
                        min_obs=min_obs,
                        min_arc_length=min_arc_length,
                        contamination_percentage=contamination_percentage,
                        delta=delta,
                        max_iter=max_iter,
                        method=method,
                        fit_epoch=fit_epoch,
                        test_orbit=test_orbit,
                        backend=backend,
                        backend_kwargs=backend_kwargs,
                    ),
                    zip(
                        yieldChunks(orbits_split, chunk_size_),
                        yieldChunks(observations_split, chunk_size_),
                    ),
                )
                p.close()

                results = list(zip(*results))
                od_orbits_dfs = results[0]
                od_orbit_members_dfs = results[1]

            elif parallel_backend == "cf":
                with cf.ProcessPoolExecutor(
                    max_workers=num_workers, initializer=_initWorker
                ) as executor:
                    futures = []
                    for orbits_i, observations_i in zip(
                        yieldChunks(orbits_split, chunk_size),
                        yieldChunks(observations_split, chunk_size),
                    ):
                        futures.append(
                            executor.submit(
                                od_worker,
                                orbits_i,
                                observations_i,
                                rchi2_threshold=rchi2_threshold,
                                min_obs=min_obs,
                                min_arc_length=min_arc_length,
                                contamination_percentage=contamination_percentage,
                                delta=delta,
                                max_iter=max_iter,
                                method=method,
                                fit_epoch=fit_epoch,
                                test_orbit=test_orbit,
                                backend=backend,
                                backend_kwargs=backend_kwargs,
                            )
                        )
                    od_orbits_dfs = []
                    od_orbit_members_dfs = []
                    for future in cf.as_completed(futures):
                        od_orbits_df, od_orbit_members_df = future.result()
                        od_orbits_dfs.append(od_orbits_df)
                        od_orbit_members_dfs.append(od_orbit_members_df)

            else:
                raise ValueError(
                    f"Unknown parallel backend: {parallel_backend}. Must be one of: 'ray', 'mp', 'cf'."
                )

        else:

            od_orbits_dfs = []
            od_orbit_members_dfs = []
            for orbits_i, observations_i in zip(
                yieldChunks(orbits_split, chunk_size),
                yieldChunks(observations_split, chunk_size),
            ):

                od_orbits_df, od_orbit_members_df = od_worker(
                    orbits_i,
                    observations_i,
                    rchi2_threshold=rchi2_threshold,
                    min_obs=min_obs,
                    min_arc_length=min_arc_length,
                    contamination_percentage=contamination_percentage,
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

        od_orbits = pd.concat(od_orbits_dfs, ignore_index=True)
        od_orbit_members = pd.concat(od_orbit_members_dfs, ignore_index=True)

        for col in ["num_obs"]:
            od_orbits[col] = od_orbits[col].astype(int)
        for col in ["outlier"]:
            od_orbit_members[col] = od_orbit_members[col].astype(int)

        od_orbits, od_orbit_members = sortLinkages(
            od_orbits, od_orbit_members, observations, linkage_id_col="orbit_id"
        )

    else:
        od_orbits = pd.DataFrame(
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
                "r",
                "r_sigma",
                "v",
                "v_sigma",
                "arc_length",
                "num_obs",
                "chi2",
                "rchi2",
            ]
        )

        od_orbit_members = pd.DataFrame(
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
    logger.info("Differentially corrected {} orbits.".format(len(od_orbits)))
    logger.info(
        "Differential correction completed in {:.3f} seconds.".format(
            time_end - time_start
        )
    )

    return od_orbits, od_orbit_members
