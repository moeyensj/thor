import logging
import multiprocessing as mp
import time
from typing import Optional, Type, Union

import quivr as qv
import ray
from adam_core.coordinates import (
    CartesianCoordinates,
    OriginCodes,
    transform_coordinates,
)
from adam_core.propagator import Propagator
from adam_core.propagator.adam_pyoorb import PYOORBPropagator
from adam_core.ray_cluster import initialize_use_ray

from .observations.observations import Observations
from .orbit import RangedPointSourceDetections, TestOrbitEphemeris, TestOrbits
from .projections import GnomonicCoordinates

__all__ = [
    "TransformedDetections",
    "range_and_transform",
]


logger = logging.getLogger(__name__)


class TransformedDetections(qv.Table):
    id = qv.LargeStringColumn()
    coordinates = GnomonicCoordinates.as_column()
    state_id = qv.LargeStringColumn()


def range_and_transform_worker(
    ranged_detections: RangedPointSourceDetections,
    observations: Observations,
    ephemeris: TestOrbitEphemeris,
    state_id: str,
) -> TransformedDetections:
    """
    Given ranged detections and their original observations, transform these to the gnomonic tangent
    plane centered on the motion of the test orbit for a single state.

    Parameters
    ----------
    ranged_detections
        Spherical coordinates that have been ranged mapped by state id
    observations
        The observations from which the ranged detections originate. These should be sorted one-to-one
        with the ranged detections
    ephemeris
        Ephemeris from which to extract the test orbit's aberrated state.
    state_id
        The ID for this particular state.

    Returns
    -------
    transformed_detections
        Detections transformed to a gnomonic tangent plane centered on the motion of the
        test orbit.
    """
    # Select the detections and ephemeris for this state id
    ranged_detections_spherical_state = ranged_detections.select("state_id", state_id)
    ephemeris_state = ephemeris.select("id", state_id)
    observations_state = observations.select("state_id", state_id)

    ranged_detections_cartesian_state = transform_coordinates(
        ranged_detections_spherical_state.coordinates,
        representation_out=CartesianCoordinates,
        frame_out="ecliptic",
        origin_out=OriginCodes.SUN,
    )

    # Transform the detections into the co-rotating frame
    return TransformedDetections.from_kwargs(
        id=observations_state.id,
        coordinates=GnomonicCoordinates.from_cartesian(
            ranged_detections_cartesian_state,
            center_cartesian=ephemeris_state.ephemeris.aberrated_coordinates,
        ),
        state_id=observations_state.state_id,
    )


range_and_transform_remote = ray.remote(range_and_transform_worker)
range_and_transform_remote = range_and_transform_remote.options(
    num_cpus=1,
    num_returns=1,
)


def range_and_transform(
    test_orbit: TestOrbits,
    observations: Union[Observations, ray.ObjectRef],
    propagator: Type[Propagator] = PYOORBPropagator,
    propagator_kwargs: dict = {},
    max_processes: Optional[int] = 1,
) -> TransformedDetections:
    """
    Range observations for a single test orbit and transform them into a
    gnomonic projection centered on the motion of the test orbit (co-rotating
    frame).

    Parameters
    ----------
    test_orbit : `~thor.orbit.TestOrbits`
        Test orbit to use to gather and transform observations.
    observations : `~thor.observations.observations.Observations`
        Observations from which range and transform the detections.
    propagator : `~adam_core.propagator.propagator.Propagator`
        Propagator to use to propagate the test orbit and generate
        ephemerides.
    max_processes : int, optional
        Maximum number of processes to use for parallelization. If
        an existing ray cluster is already running, this parameter
        will be ignored if larger than 1 or not None.

    Returns
    -------
    transformed_detections : `~thor.main.TransformedDetections`
        The transformed detections as gnomonic coordinates
        of the observations in the co-rotating frame.
    """
    time_start = time.perf_counter()
    logger.info("Running range and transform...")

    if len(test_orbit) != 1:
        raise ValueError(f"range_and_transform received {len(test_orbit)} orbits but expected 1.")

    logger.info(f"Assuming r = {test_orbit.coordinates.r[0]} au")
    logger.info(f"Assuming v = {test_orbit.coordinates.v[0]} au/d")

    if isinstance(observations, ray.ObjectRef):
        observations_ref = observations
        observations = ray.get(observations)
        logger.info("Retrieved observations from the object store.")
    else:
        observations_ref = None

    prop = propagator(**propagator_kwargs)

    if len(observations) > 0:
        # Compute the ephemeris of the test orbit (this will be cached)
        ephemeris = test_orbit.generate_ephemeris_from_observations(
            observations,
            propagator=prop,
            max_processes=max_processes,
        )

        # Assume that the heliocentric distance of all point sources in
        # the observations are the same as that of the test orbit
        ranged_detections_spherical = test_orbit.range_observations(
            observations,
            propagator=prop,
            max_processes=max_processes,
        )

        transformed_detections = TransformedDetections.empty()

        if max_processes is None:
            max_processes = mp.cpu_count()

        use_ray = initialize_use_ray(num_cpus=max_processes)
        if use_ray:
            refs_to_free = []
            if observations_ref is None:
                observations_ref = ray.put(observations)
                refs_to_free.append(observations_ref)
                logger.info("Placed observations in the object store.")

            if not isinstance(ephemeris, ray.ObjectRef):
                ephemeris_ref = ray.put(ephemeris)
                refs_to_free.append(ephemeris_ref)
                logger.info("Placed ephemeris in the object store.")
            else:
                ephemeris_ref = ephemeris

            ranged_detections_spherical_ref = ray.put(ranged_detections_spherical)

            # Get state IDs
            state_ids = observations.state_id.unique()
            futures = []
            for state_id in state_ids:
                futures.append(
                    range_and_transform_remote.remote(
                        ranged_detections_spherical_ref,
                        observations_ref,
                        ephemeris_ref,
                        state_id,
                    )
                )

                if len(futures) >= max_processes * 1.5:
                    finished, futures = ray.wait(futures, num_returns=1)
                    transformed_detections = qv.concatenate([transformed_detections, ray.get(finished[0])])
                    if transformed_detections.fragmented():
                        transformed_detections = qv.defragment(transformed_detections)

            while futures:
                finished, futures = ray.wait(futures, num_returns=1)
                transformed_detections = qv.concatenate([transformed_detections, ray.get(finished[0])])
                if transformed_detections.fragmented():
                    transformed_detections = qv.defragment(transformed_detections)

            if len(refs_to_free) > 0:
                ray.internal.free(refs_to_free)
                logger.info(f"Removed {len(refs_to_free)} references from the object store.")

        else:
            # Get state IDs
            state_ids = observations.state_id.unique()
            for state_id in state_ids:
                # mask = pc.equal(state_id, observations.state_id)

                chunk = range_and_transform_worker(
                    ranged_detections_spherical.select("state_id", state_id),
                    observations.select("state_id", state_id),
                    ephemeris.select("id", state_id),
                    state_id,
                )
                transformed_detections = qv.concatenate([transformed_detections, chunk])
                if transformed_detections.fragmented():
                    transformed_detections = qv.defragment(transformed_detections)

        transformed_detections = transformed_detections.sort_by(by=["state_id"])

    else:
        transformed_detections = TransformedDetections.empty()

    time_end = time.perf_counter()
    logger.info(f"Transformed {len(transformed_detections)} observations.")
    logger.info(f"Range and transform completed in {time_end - time_start:.3f} seconds.")
    return transformed_detections
