import logging
import multiprocessing as mp
import time
from typing import Literal, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_core.coordinates import CartesianCoordinates, CoordinateCovariances
from adam_core.coordinates.residuals import Residuals
from adam_core.orbit_determination import OrbitDeterminationObservations
from adam_core.orbits import Orbits
from adam_core.propagator import Propagator, _iterate_chunks
from adam_core.propagator.adam_pyoorb import PYOORBPropagator
from adam_core.propagator.utils import _iterate_chunk_indices
from adam_core.ray_cluster import initialize_use_ray
from scipy.linalg import solve

from ..observations.observations import Observations
from ..orbit_determination.fitted_orbits import FittedOrbitMembers, FittedOrbits
from ..orbit_determination.outliers import calculate_max_outliers

logger = logging.getLogger(__name__)

__all__ = ["differential_correction"]


def od_worker(
    orbit_ids: npt.NDArray[np.str_],
    orbits: FittedOrbits,
    orbit_members: FittedOrbitMembers,
    observations: Observations,
    rchi2_threshold: float = 100,
    min_obs: int = 5,
    min_arc_length: float = 1.0,
    contamination_percentage: float = 0.0,
    delta: float = 1e-6,
    max_iter: int = 20,
    method: Literal["central", "finite"] = "central",
    propagator: Type[Propagator] = PYOORBPropagator,
    propagator_kwargs: dict = {},
) -> Tuple[FittedOrbits, FittedOrbitMembers]:

    od_orbits = FittedOrbits.empty()
    od_orbit_members = FittedOrbitMembers.empty()
    for orbit_id in orbit_ids:
        time_start = time.time()
        logger.debug(f"Differentially correcting orbit {orbit_id}...")

        orbit = orbits.select("orbit_id", orbit_id)
        obs_ids = orbit_members.apply_mask(pc.equal(orbit_members.orbit_id, orbit_id)).obs_id
        orbit_observations = observations.apply_mask(pc.is_in(observations.id, obs_ids))

        # Sort observations by time
        orbit_observations = orbit_observations.sort_by(
            [
                "coordinates.time.days",
                "coordinates.time.nanos",
                "coordinates.origin.code",
            ]
        )

        orbit_observations = OrbitDeterminationObservations.from_kwargs(
            id=orbit_observations.id,
            coordinates=orbit_observations.coordinates,
            observers=orbit_observations.get_observers().observers.sort_by(
                ["coordinates.time.days", "coordinates.time.nanos", "code"]
            ),
        )

        od_orbit, od_orbit_orbit_members = od(
            orbit,
            orbit_observations,
            rchi2_threshold=rchi2_threshold,
            min_obs=min_obs,
            min_arc_length=min_arc_length,
            contamination_percentage=contamination_percentage,
            delta=delta,
            max_iter=max_iter,
            method=method,
            propagator=propagator,
            propagator_kwargs=propagator_kwargs,
        )
        time_end = time.time()
        duration = time_end - time_start
        logger.debug(f"OD for orbit {orbit_id} completed in {duration:.3f}s.")
        od_orbits = qv.concatenate([od_orbits, od_orbit])
        if od_orbits.fragmented():
            od_orbits = qv.defragment(od_orbits)

        od_orbit_members = qv.concatenate([od_orbit_members, od_orbit_orbit_members])
        if od_orbit_members.fragmented():
            od_orbit_members = qv.defragment(od_orbit_members)

    return od_orbits, od_orbit_members


@ray.remote
def od_worker_remote(
    orbit_ids: npt.NDArray[np.str_],
    orbit_ids_indices: Tuple[int, int],
    orbits: FittedOrbits,
    orbit_members: FittedOrbitMembers,
    observations: Observations,
    rchi2_threshold: float = 100,
    min_obs: int = 5,
    min_arc_length: float = 1.0,
    contamination_percentage: float = 0.0,
    delta: float = 1e-6,
    max_iter: int = 20,
    method: Literal["central", "finite"] = "central",
    propagator: Type[Propagator] = PYOORBPropagator,
    propagator_kwargs: dict = {},
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    orbit_ids_chunk = orbit_ids[orbit_ids_indices[0] : orbit_ids_indices[1]]
    return od_worker(
        orbit_ids_chunk,
        orbits,
        orbit_members,
        observations,
        rchi2_threshold=rchi2_threshold,
        min_obs=min_obs,
        min_arc_length=min_arc_length,
        contamination_percentage=contamination_percentage,
        delta=delta,
        max_iter=max_iter,
        method=method,
        propagator=propagator,
        propagator_kwargs=propagator_kwargs,
    )


od_worker_remote.options(num_returns=1, num_cpus=1)


def od(
    orbit: FittedOrbits,
    observations: OrbitDeterminationObservations,
    rchi2_threshold: float = 100,
    min_obs: int = 5,
    min_arc_length: float = 1.0,
    contamination_percentage: float = 0.0,
    delta: float = 1e-6,
    max_iter: int = 20,
    method: Literal["central", "finite"] = "central",
    propagator: Type[Propagator] = PYOORBPropagator,
    propagator_kwargs: dict = {},
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    # Intialize the propagator
    prop = propagator(**propagator_kwargs)

    if method not in ["central", "finite"]:
        err = "method should be one of 'central' or 'finite'."
        raise ValueError(err)

    obs_ids_all = observations.id.to_numpy(zero_copy_only=False)
    coords = observations.coordinates
    coords_sigma = coords.covariance.sigmas[:, 1:3]
    observers = observations.observers
    times_all = coords.time.mjd().to_numpy(zero_copy_only=False)

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
        max_outliers = calculate_max_outliers(num_obs, min_obs, contamination_percentage)
        logger.debug(f"Maximum number of outliers allowed: {max_outliers}")
        outliers_tried = 0

        # Calculate chi2 for residuals on the given observations
        # for the current orbit, the goal is for the orbit to improve
        # such that the chi2 improves
        orbit_prev_ = orbit.to_orbits()

        ephemeris_prev_ = prop.generate_ephemeris(orbit_prev_, observers, chunk_size=1, max_processes=1)

        # Calculate residuals and chi2
        residuals_prev_ = Residuals.calculate(
            coords,
            ephemeris_prev_.coordinates,
        )
        residuals_prev_array = np.stack(residuals_prev_.values.to_numpy(zero_copy_only=False))[:, 1:3]

        num_obs_ = len(observations)
        chi2_prev_ = residuals_prev_.chi2.to_numpy()
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
        times_all = ephemeris_prev.coordinates.time.mjd().to_numpy()
        obs_id_outlier = []
        delta_prev = delta
        iterations = 0

        DELTA_INCREASE_FACTOR = 5
        DELTA_DECREASE_FACTOR = 100

        max_iter_i = max_iter
        max_iter_outliers = max_iter * (max_outliers + 1)

    while not converged and processable:
        iterations += 1

        # We add 1 here because the iterations are counted as they start, not
        # as they finish. There are a lot of 'continue' statements down below that
        # will exit the current iteration if something fails which makes accounting for
        # iterations at the start of an iteration easier.
        if iterations == max_iter_outliers + 1:
            logger.debug("Maximum number of iterations completed.")
            break
        if iterations == max_iter_i + 1 and (solution_found or (max_outliers == outliers_tried)):
            logger.debug("Maximum number of iterations completed.")
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
        num_params = 6
        A = np.zeros((2, num_params, num_obs))
        ATWA = np.zeros((num_params, num_params, num_obs))
        ATWb = np.zeros((num_params, 1, num_obs))

        # Generate ephemeris with current nominal orbit
        ephemeris_nom = prop.generate_ephemeris(orbit_prev, observers, chunk_size=1, max_processes=1)

        # Modify each component of the state by a small delta
        d = np.zeros((1, 6))
        for i in range(num_params):
            # zero the delta vector
            d *= 0.0

            # x, y, z [au]: 0, 1, 2
            # vx, vy, vz [au per day]: 3, 4, 5
            if i < 3:
                delta_iter = delta_prev
                d[0, i] = orbit_prev.coordinates.values[0, i] * delta_iter
            elif i < 6:
                delta_iter = delta_prev
                d[0, i] = orbit_prev.coordinates.values[0, i] * delta_iter

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
            ephemeris_mod_p = prop.generate_ephemeris(orbit_iter_p, observers, chunk_size=1, max_processes=1)

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
                ephemeris_mod_n = prop.generate_ephemeris(
                    orbit_iter_n, observers, chunk_size=1, max_processes=1
                )

                delta_denom *= 2

            else:
                ephemeris_mod_n = ephemeris_nom

            residuals_mod = Residuals.calculate(
                ephemeris_mod_p.coordinates,
                ephemeris_mod_n.coordinates,
            )
            residuals_mod = np.stack(residuals_mod.values.to_numpy(zero_copy_only=False))
            residuals_mod_array = residuals_mod[:, 1:3]

            for n in range(num_obs):
                try:
                    A[:, i : i + 1, n] = residuals_mod_array[ids_mask][n : n + 1].T / delta_denom
                except RuntimeError:
                    print(orbit_prev.orbit_id)

        for n in range(num_obs):
            W = np.diag(1 / coords_sigma[n] ** 2)
            ATWA[:, :, n] = A[:, :, n].T @ W @ A[:, :, n]
            ATWb[:, :, n] = A[:, :, n].T @ W @ residuals_prev_array[n : n + 1].T

        ATWA = np.sum(ATWA, axis=2)
        ATWb = np.sum(ATWb, axis=2)

        try:
            # Calculate the condition number of both matrices
            ATWA_condition = np.linalg.cond(ATWA)
            ATWb_condition = np.linalg.cond(ATWb)
        except np.linalg.LinAlgError:
            ATWA_condition = np.nan
            ATWb_condition = np.nan
            logger.debug(f"Matrix condition calculation failed for {orbit.orbit_id[0].as_py()}")

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
                    logger.debug("Variances are negative, 0.0, or NaN. Discarding solution.")
                    continue

                r_variances = variances[0:3]
                r_sigma = np.sqrt(np.sum(r_variances))
                r = orbit_prev.coordinates.r_mag
                if (r_sigma / r) > 1:
                    delta_prev /= DELTA_DECREASE_FACTOR
                    logger.debug("Covariance matrix is largely unconstrained. Discarding solution.")
                    continue

                if np.any(np.isnan(covariance_matrix)):
                    delta_prev *= DELTA_INCREASE_FACTOR
                    logger.debug("Covariance matrix contains NaNs. Discarding solution.")
                    continue

            except np.linalg.LinAlgError:
                delta_prev *= DELTA_INCREASE_FACTOR
                continue

        if np.linalg.norm(delta_state[:3]) < 1e-16:
            logger.debug("Change in state is less than 1e-16 au, discarding solution.")
            delta_prev *= DELTA_DECREASE_FACTOR
            continue
        if np.linalg.norm(delta_state[:3]) > 100:
            delta_prev /= DELTA_DECREASE_FACTOR
            logger.debug("Change in state is more than 100 au, discarding solution.")
            continue

        cartesian_elements = orbit_prev.coordinates.values + delta_state
        orbit_iter = Orbits.from_kwargs(
            orbit_id=orbit_prev.orbit_id,
            coordinates=CartesianCoordinates.from_kwargs(
                x=cartesian_elements[:, 0],
                y=cartesian_elements[:, 1],
                z=cartesian_elements[:, 2],
                vx=cartesian_elements[:, 3],
                vy=cartesian_elements[:, 4],
                vz=cartesian_elements[:, 5],
                covariance=CoordinateCovariances.from_matrix(covariance_matrix.reshape(1, 6, 6)),
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
        ephemeris_iter = prop.generate_ephemeris(orbit_iter, observers, chunk_size=1, max_processes=1)

        residuals = Residuals.calculate(coords, ephemeris_iter.coordinates)
        chi2_iter = residuals.chi2.to_numpy()
        chi2_total_iter = np.sum(chi2_iter[ids_mask])
        rchi2_iter = chi2_total_iter / (2 * num_obs - num_params)
        arc_length = times_all[ids_mask].max() - times_all[ids_mask].min()

        # If the new orbit has lower residuals than the previous,
        # accept the orbit and continue iterating until max iterations has been
        # reached. Once max iterations have been reached and the orbit still has not converged
        # to an acceptable solution, try removing an observation as an outlier and iterate again.
        if ((rchi2_iter < rchi2_prev) or first_solution) and arc_length >= min_arc_length:
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
            max_outliers > 0
            and outliers_tried <= max_outliers
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
                logger.debug("Removing the outlier will cause the arc length to go below the minimum.")

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
        od_orbit = FittedOrbits.empty()
        od_orbit_members = FittedOrbitMembers.empty()

    else:
        obs_times = observations.coordinates.time.mjd().to_numpy()[ids_mask]
        arc_length_ = obs_times.max() - obs_times.min()
        assert arc_length == arc_length_

        od_orbit = FittedOrbits.from_kwargs(
            orbit_id=orbit_prev.orbit_id,
            object_id=orbit_prev.object_id,
            coordinates=orbit_prev.coordinates,
            arc_length=[arc_length_],
            num_obs=[num_obs],
            chi2=[chi2_total_prev],
            reduced_chi2=[rchi2_prev],
            iterations=[iterations],
            success=[improved],
            status_code=[0],
        )

        od_orbit_members = FittedOrbitMembers.from_kwargs(
            orbit_id=np.full(len(obs_ids_all), orbit_prev.orbit_id[0].as_py(), dtype="object"),
            obs_id=obs_ids_all,
            residuals=residuals_prev,
            solution=np.isin(obs_ids_all, obs_id_outlier, invert=True),
            outlier=np.isin(obs_ids_all, obs_id_outlier),
        )

    return od_orbit, od_orbit_members


def differential_correction(
    orbits: Union[FittedOrbits, ray.ObjectRef],
    orbit_members: Union[FittedOrbitMembers, ray.ObjectRef],
    observations: Union[Observations, ray.ObjectRef],
    min_obs: int = 5,
    min_arc_length: float = 1.0,
    contamination_percentage: float = 20,
    rchi2_threshold: float = 100,
    delta: float = 1e-8,
    max_iter: int = 20,
    method: Literal["central", "finite"] = "central",
    propagator: Type[Propagator] = PYOORBPropagator,
    propagator_kwargs: dict = {},
    chunk_size: int = 10,
    max_processes: Optional[int] = 1,
    orbit_ids: Optional[npt.NDArray[np.str_]] = None,
    obs_ids: Optional[npt.NDArray[np.str_]] = None,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Differentially correct (via finite/central differencing).

    Parameters
    ----------
    chunk_size : int, optional
        Number of orbits to send to each job.
    num_jobs : int, optional
        Number of jobs to launch.
    """
    time_start = time.perf_counter()
    logger.info("Running differential correction...")
    if isinstance(orbits, ray.ObjectRef):
        orbits_ref = orbits
        orbits = ray.get(orbits)
        logger.info("Retrieved orbits from the object store.")

        if orbit_ids is not None:
            orbits = orbits.apply_mask(pc.is_in(orbits.orbit_id, orbit_ids))
            logger.info("Applied mask to orbit members.")
    else:
        orbits_ref = None

    if isinstance(orbit_members, ray.ObjectRef):
        orbit_members_ref = orbit_members
        orbit_members = ray.get(orbit_members)
        logger.info("Retrieved orbit members from the object store.")

        if obs_ids is not None:
            orbit_members = orbit_members.apply_mask(pc.is_in(orbit_members.obs_id, obs_ids))
            logger.info("Applied mask to orbit members.")
        if orbit_ids is not None:
            orbit_members = orbit_members.apply_mask(pc.is_in(orbit_members.orbit_id, orbit_ids))
            logger.info("Applied mask to orbit members.")
    else:
        orbit_members_ref = None

    if isinstance(observations, ray.ObjectRef):
        observations_ref = observations
        observations = ray.get(observations)
        logger.info("Retrieved observations from the object store.")

        if obs_ids is not None:
            observations = observations.apply_mask(pc.is_in(observations.id, obs_ids))
            logger.info("Applied mask to observations.")
    else:
        observations_ref = None

    if len(orbits) == 0 or len(orbit_members) == 0:
        logger.info("Received no orbits or orbit members.")
        od_orbits = FittedOrbits.empty()
        od_orbit_members = FittedOrbitMembers.empty()
        time_end = time.perf_counter()
        logger.info(f"Differentially corrected {len(od_orbits)} orbits.")
        logger.info(f"Differential correction completed in {time_end - time_start:.3f} seconds.")
        return od_orbits, od_orbit_members

    orbit_ids = orbits.orbit_id.to_numpy(zero_copy_only=False)

    od_orbits = FittedOrbits.empty()
    od_orbit_members = FittedOrbitMembers.empty()

    if max_processes is None:
        max_processes = mp.cpu_count()

    use_ray = initialize_use_ray(num_cpus=max_processes)
    if use_ray:
        refs_to_free = []

        orbit_ids_ref = ray.put(orbit_ids)
        orbit_ids = ray.get(orbit_ids_ref)
        refs_to_free.append(orbit_ids_ref)
        logger.info("Placed orbit IDs in the object store.")

        if orbits_ref is None:
            orbits_ref = ray.put(orbits)
            orbits = ray.get(orbits_ref)
            refs_to_free.append(orbits_ref)
            logger.info("Placed orbits in the object store.")

        if orbit_members_ref is None:
            orbit_members_ref = ray.put(orbit_members)
            orbit_members = ray.get(orbit_members_ref)
            refs_to_free.append(orbit_members_ref)
            logger.info("Placed orbit members in the object store.")

        if observations_ref is None:
            observations_ref = ray.put(observations)
            refs_to_free.append(observations_ref)
            observations = ray.get(observations_ref)
            logger.info("Placed observations in the object store.")

        futures = []
        for orbit_ids_indices in _iterate_chunk_indices(orbit_ids, chunk_size):
            futures.append(
                od_worker_remote.remote(
                    orbit_ids_ref,
                    orbit_ids_indices,
                    orbits_ref,
                    orbit_members_ref,
                    observations_ref,
                    rchi2_threshold=rchi2_threshold,
                    min_obs=min_obs,
                    min_arc_length=min_arc_length,
                    contamination_percentage=contamination_percentage,
                    delta=delta,
                    max_iter=max_iter,
                    method=method,
                    propagator=propagator,
                    propagator_kwargs=propagator_kwargs,
                )
            )

            if len(futures) >= max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                od_orbits_chunk, od_orbit_members_chunk = ray.get(finished[0])
                od_orbits = qv.concatenate([od_orbits, od_orbits_chunk])
                if od_orbits.fragmented():
                    od_orbits = qv.defragment(od_orbits)
                od_orbit_members = qv.concatenate([od_orbit_members, od_orbit_members_chunk])
                if od_orbit_members.fragmented():
                    od_orbit_members = qv.defragment(od_orbit_members)

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            od_orbits_chunk, od_orbit_members_chunk = ray.get(finished[0])
            od_orbits = qv.concatenate([od_orbits, od_orbits_chunk])
            if od_orbits.fragmented():
                od_orbits = qv.defragment(od_orbits)
            od_orbit_members = qv.concatenate([od_orbit_members, od_orbit_members_chunk])
            if od_orbit_members.fragmented():
                od_orbit_members = qv.defragment(od_orbit_members)

        if len(refs_to_free) > 0:
            ray.internal.free(refs_to_free)
            logger.info(f"Removed {len(refs_to_free)} references from the object store.")

    else:
        for orbit_ids_chunk in _iterate_chunks(orbit_ids, chunk_size):
            od_orbits_chunk, od_orbit_members_chunk = od_worker(
                orbit_ids_chunk,
                orbits,
                orbit_members,
                observations,
                rchi2_threshold=rchi2_threshold,
                min_obs=min_obs,
                min_arc_length=min_arc_length,
                contamination_percentage=contamination_percentage,
                delta=delta,
                max_iter=max_iter,
                method=method,
                propagator=propagator,
                propagator_kwargs=propagator_kwargs,
            )
            od_orbits = qv.concatenate([od_orbits, od_orbits_chunk])
            if od_orbits.fragmented():
                od_orbits = qv.defragment(od_orbits)
            od_orbit_members = qv.concatenate([od_orbit_members, od_orbit_members_chunk])
            if od_orbit_members.fragmented():
                od_orbit_members = qv.defragment(od_orbit_members)

    time_end = time.perf_counter()
    logger.info(f"Differentially corrected {len(od_orbits)} orbits.")
    logger.info(f"Differential correction completed in {time_end - time_start:.3f} seconds.")

    return od_orbits, od_orbit_members
