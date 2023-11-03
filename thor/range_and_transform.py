import logging
import time
from typing import Optional, Type, Union

import pyarrow.compute as pc
import quivr as qv
import ray
from adam_core.coordinates import (
    CartesianCoordinates,
    OriginCodes,
    transform_coordinates,
)
from adam_core.propagator import PYOORB, Propagator

from .observations.observations import Observations
from .orbit import TestOrbit, TestOrbitEphemeris
from .projections import GnomonicCoordinates

__all__ = [
    "TransformedDetections",
    "range_and_transform",
]


logger = logging.getLogger(__name__)


class TransformedDetections(qv.Table):
    id = qv.StringColumn()
    coordinates = GnomonicCoordinates.as_column()
    state_id = qv.Int64Column()


def range_and_transform_worker(
    ranged_detections: CartesianCoordinates,
    observations: Observations,
    ephemeris: TestOrbitEphemeris,
    state_id: int,
) -> TransformedDetections:
    """
    Given ranged detections and their original observations, transform these to the gnomonic tangent
    plane centered on the motion of the test orbit for a single state.

    Parameters
    ----------
    ranged_detections
        Cartesian detections ranged so that their heliocentric distance is the same as the test orbit
        for each state
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
    mask = pc.equal(state_id, observations.state_id)
    ranged_detections_state = ranged_detections.apply_mask(mask)
    ephemeris_state = ephemeris.select("id", state_id)
    observations_state = observations.select("state_id", state_id)

    # Transform the detections into the co-rotating frame
    return TransformedDetections.from_kwargs(
        id=observations_state.detections.id,
        coordinates=GnomonicCoordinates.from_cartesian(
            ranged_detections_state,
            center_cartesian=ephemeris_state.ephemeris.aberrated_coordinates,
        ),
        state_id=observations_state.state_id,
    )


range_and_transform_remote = ray.remote(range_and_transform_worker)


def range_and_transform(
    test_orbit: TestOrbit,
    observations: Union[Observations, ray.ObjectRef],
    propagator: Type[Propagator] = PYOORB,
    propagator_kwargs: dict = {},
    max_processes: Optional[int] = 1,
) -> TransformedDetections:
    """
    Range observations for a single test orbit and transform them into a
    gnomonic projection centered on the motion of the test orbit (co-rotating
    frame).

    Parameters
    ----------
    test_orbit : `~thor.orbit.TestOrbit`
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
    logger.info(f"Assuming r = {test_orbit.orbit.coordinates.r[0]} au")
    logger.info(f"Assuming v = {test_orbit.orbit.coordinates.v[0]} au/d")

    if isinstance(observations, ray.ObjectRef):
        observations = ray.get(observations)

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

        # Transform from spherical topocentric to cartesian heliocentric coordinates
        ranged_detections_cartesian = transform_coordinates(
            ranged_detections_spherical.coordinates,
            representation_out=CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SUN,
        )

        transformed_detection_list = []
        if max_processes is None or max_processes > 1:

            if not ray.is_initialized():
                logger.debug(
                    f"Ray is not initialized. Initializing with {max_processes}..."
                )
                ray.init(num_cpus=max_processes)

            if isinstance(observations, ray.ObjectRef):
                observations_ref = observations
                observations = ray.get(observations_ref)
            else:
                observations_ref = ray.put(observations)

            if isinstance(ephemeris, ray.ObjectRef):
                ephemeris_ref = ephemeris
            else:
                ephemeris_ref = ray.put(ephemeris)

            ranged_detections_cartesian_ref = ray.put(ranged_detections_cartesian)

            # Get state IDs
            state_ids = observations.state_id.unique().sort()
            futures = []
            for state_id in state_ids:
                futures.append(
                    range_and_transform_remote.remote(
                        ranged_detections_cartesian_ref,
                        observations_ref,
                        ephemeris_ref,
                        state_id,
                    )
                )

            while futures:
                finished, futures = ray.wait(futures, num_returns=1)
                transformed_detection_list.append(ray.get(finished[0]))

        else:
            # Get state IDs
            state_ids = observations.state_id.unique().sort()
            for state_id in state_ids:
                mask = pc.equal(state_id, observations.state_id)
                transformed_detection_list.append(
                    range_and_transform_worker(
                        ranged_detections_cartesian.apply_mask(mask),
                        observations.select("state_id", state_id),
                        ephemeris.select("id", state_id),
                        state_id,
                    )
                )

        transformed_detections = qv.concatenate(transformed_detection_list)
        transformed_detections = transformed_detections.sort_by(by=["state_id"])

    else:
        transformed_detections = TransformedDetections.empty()

    time_end = time.perf_counter()
    logger.info(f"Transformed {len(transformed_detections)} observations.")
    logger.info(
        f"Range and transform completed in {time_end - time_start:.3f} seconds."
    )
    return transformed_detections
