import logging
import time
from typing import List, Literal, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_core.orbit_determination import (
    FittedOrbitMembers,
    FittedOrbits,
    OrbitDeterminationObservations,
    evaluate_orbit,
    fit_least_squares,
)
from adam_core.orbit_determination.outliers import remove_lowest_probability_observation
from adam_core.orbits import Orbits
from adam_core.propagator import PYOORB, Propagator, _iterate_chunks
from adam_core.ray_cluster import initialize_use_ray

from ..observations.observations import Observations
from ..orbit_determination.outliers import calculate_max_outliers
from ..utils.linkages import sort_by_id_and_time

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
    max_iter: int = 20,
    propagator: Type[Propagator] = PYOORB,
    propagator_kwargs: dict = {},
) -> Tuple[FittedOrbits, FittedOrbitMembers]:

    # Initialize propagator with the given kwargs for this worker
    prop = propagator(**propagator_kwargs)

    od_orbits = FittedOrbits.empty()
    od_orbit_members = FittedOrbitMembers.empty()

    for orbit_id in orbit_ids:
        time_start = time.time()
        logger.debug(f"Differentially correcting orbit {orbit_id}...")

        orbit = orbits.select("orbit_id", orbit_id)
        obs_ids = orbit_members.apply_mask(
            pc.equal(orbit_members.orbit_id, orbit_id)
        ).obs_id
        orbit_observations = observations.apply_mask(pc.is_in(observations.id, obs_ids))

        orbit_observations = OrbitDeterminationObservations.from_kwargs(
            id=orbit_observations.id,
            coordinates=orbit_observations.coordinates,
            observers=orbit_observations.get_observers().observers,
        )

        od_orbit, od_orbit_orbit_members = od(
            orbit,
            orbit_observations,
            rchi2_threshold=rchi2_threshold,
            min_obs=min_obs,
            min_arc_length=min_arc_length,
            contamination_percentage=contamination_percentage,
            max_iter=max_iter,
            propagator=prop,
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


od_worker_remote = ray.remote(od_worker)
od_worker_remote.options(num_returns=1, num_cpus=1)


def od(
    orbit: Orbits,
    observations: OrbitDeterminationObservations,
    rchi2_threshold: float = 100,
    min_obs: int = 5,
    min_arc_length: float = 1.0,
    contamination_percentage: float = 0.0,
    max_iter: int = 50,
    propagator: Propagator = PYOORB(),
) -> Tuple[FittedOrbits, FittedOrbitMembers]:

    # Evaluate the current quality of the orbit
    od_orbit_iter, od_orbit_members_iter = evaluate_orbit(
        orbit, observations, propagator, parameters=6
    )
    reduced_chi2_iter = od_orbit_iter.reduced_chi2[0].as_py()

    # Calculate maximum number of outliers permissible
    max_outliers = calculate_max_outliers(
        len(observations), min_obs, contamination_percentage
    )

    # For each possible number of outliers attempt to fit the orbit
    # If the fit is successful: meets the criteria for reduced chi2 and arc length
    # then return the fitted orbit
    # If the fit is unsuccessful: remove the observation with the lowest probability
    # and try again
    # If the number of attempts exceeds the maximum number of outliers then return
    # an empty fitted orbit
    outliers: List[str] = []
    for i in range(max_outliers + 1):
        if len(outliers) > 0:
            ignore = outliers
        else:
            ignore = None

        # Fit the orbit via least squares
        od_orbit_iter, od_orbit_members_iter = fit_least_squares(
            od_orbit_iter.to_orbits(),
            observations,
            propagator,
            ignore=ignore,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            x_scale="jac",
            max_nfev=max_iter,
        )
        # Calculate the new reduced chi2 and arc length
        reduced_chi2_iter = od_orbit_iter.reduced_chi2[0].as_py()
        arc_length_iter = od_orbit_iter.arc_length[0].as_py()

        # If the reduced chi2 and arc length meet the criteria then return the fitted orbit
        if reduced_chi2_iter <= rchi2_threshold and arc_length_iter >= min_arc_length:
            return od_orbit_iter, od_orbit_members_iter

        # If the orbit does not meet the criteria then try again but this time
        # remove the observation with the lowest probability (highest residual)
        else:
            # Remove the observation with the lowest probability
            (
                outlier,
                observations_without_outliers,
            ) = remove_lowest_probability_observation(
                od_orbit_members_iter, observations
            )
            outliers.append(outlier)

            # If the arc length of the new observations is less than the minimum arc length
            # then return an empty fitted orbit
            arc_length_without_outliers = (
                observations_without_outliers.coordinates.time.max().mjd()[0].as_py()
                - observations_without_outliers.coordinates.time.min().mjd()[0].as_py()
            )
            if arc_length_without_outliers < min_arc_length:
                return FittedOrbits.empty(), FittedOrbitMembers.empty()

    return FittedOrbits.empty(), FittedOrbitMembers.empty()


def differential_correction(
    orbits: Union[FittedOrbits, ray.ObjectRef],
    orbit_members: Union[FittedOrbitMembers, ray.ObjectRef],
    observations: Union[Observations, ray.ObjectRef],
    min_obs: int = 5,
    min_arc_length: float = 1.0,
    contamination_percentage: float = 20,
    rchi2_threshold: float = 10,
    delta: float = 1e-8,
    max_iter: int = 50,
    method: Literal["central", "finite"] = "central",
    fit_epoch: bool = False,
    propagator: Type[Propagator] = PYOORB,
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
    parallel_backend : str, optional
        Which parallelization backend to use {'ray', 'mp', 'cf'}. Defaults to using Python's concurrent.futures
        module ('cf').
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
            orbit_members = orbit_members.apply_mask(
                pc.is_in(orbit_members.obs_id, obs_ids)
            )
            logger.info("Applied mask to orbit members.")
        if orbit_ids is not None:
            orbit_members = orbit_members.apply_mask(
                pc.is_in(orbit_members.orbit_id, orbit_ids)
            )
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

    if len(orbits) > 0 and len(orbit_members) > 0:
        orbit_ids = orbits.orbit_id.to_numpy(zero_copy_only=False)

        od_orbits = FittedOrbits.empty()
        od_orbit_members = FittedOrbitMembers.empty()

        use_ray = initialize_use_ray(num_cpus=max_processes)
        if use_ray:
            refs_to_free = []
            if orbits_ref is None:
                orbits_ref = ray.put(orbits)
                refs_to_free.append(orbits_ref)
                logger.info("Placed orbits in the object store.")

            if orbit_members_ref is None:
                orbit_members_ref = ray.put(orbit_members)
                refs_to_free.append(orbit_members_ref)
                logger.info("Placed orbit members in the object store.")

            if observations_ref is None:
                observations_ref = ray.put(observations)
                refs_to_free.append(observations_ref)
                logger.info("Placed observations in the object store.")

            futures = []
            for orbit_ids_chunk in _iterate_chunks(orbit_ids, chunk_size):
                futures.append(
                    od_worker_remote.remote(
                        orbit_ids_chunk,
                        orbits_ref,
                        orbit_members_ref,
                        observations_ref,
                        rchi2_threshold=rchi2_threshold,
                        min_obs=min_obs,
                        min_arc_length=min_arc_length,
                        contamination_percentage=contamination_percentage,
                        max_iter=max_iter,
                        propagator=propagator,
                        propagator_kwargs=propagator_kwargs,
                    )
                )

            while futures:
                finished, futures = ray.wait(futures, num_returns=1)
                od_orbits_chunk, od_orbit_members_chunk = ray.get(finished[0])
                od_orbits = qv.concatenate([od_orbits, od_orbits_chunk])
                if od_orbits.fragmented():
                    od_orbits = qv.defragment(od_orbits)
                od_orbit_members = qv.concatenate(
                    [od_orbit_members, od_orbit_members_chunk]
                )
                if od_orbit_members.fragmented():
                    od_orbit_members = qv.defragment(od_orbit_members)

            if len(refs_to_free) > 0:
                ray.internal.free(refs_to_free)
                logger.info(
                    f"Removed {len(refs_to_free)} references from the object store."
                )

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
                    max_iter=max_iter,
                    propagator=propagator,
                    propagator_kwargs=propagator_kwargs,
                )
                od_orbits = qv.concatenate([od_orbits, od_orbits_chunk])
                if od_orbits.fragmented():
                    od_orbits = qv.defragment(od_orbits)
                od_orbit_members = qv.concatenate(
                    [od_orbit_members, od_orbit_members_chunk]
                )
                if od_orbit_members.fragmented():
                    od_orbit_members = qv.defragment(od_orbit_members)

        # Sort orbits by orbit ID and observation time
        od_orbits, od_orbit_members = sort_by_id_and_time(
            od_orbits, od_orbit_members, observations, "orbit_id"
        )

    else:
        logger.info("Received no orbits or orbit members.")
        od_orbits = FittedOrbits.empty()
        od_orbit_members = FittedOrbitMembers.empty()

    time_end = time.perf_counter()
    logger.info(f"Differentially corrected {len(od_orbits)} orbits.")
    logger.info(
        f"Differential correction completed in {time_end - time_start:.3f} seconds."
    )

    return od_orbits, od_orbit_members
