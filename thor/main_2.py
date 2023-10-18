from typing import List, Optional

import pyarrow.compute as pc
import quivr as qv
from adam_core.coordinates import (
    CartesianCoordinates,
    OriginCodes,
    transform_coordinates,
)
from adam_core.propagator import PYOORB, Propagator

from .observations import Observations
from .observations.filters import ObservationFilter, TestOrbitRadiusObservationFilter
from .orbit import TestOrbit
from .projections import GnomonicCoordinates


class TransformedDetections(qv.Table):
    id = qv.StringColumn()
    coordinates = GnomonicCoordinates.as_column()
    state_id = qv.Int64Column()


def range_and_transform(
    test_orbit: TestOrbit,
    observations: Observations,
    propagator: Propagator = PYOORB(),
    max_processes: int = 1,
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
        Maximum number of processes to use for parallelization.

    Returns
    -------
    transformed_detections : `~thor.main.TransformedDetections`
        The transformed detections as gnomonic coordinates
        of the observations in the co-rotating frame.
    """
    # Compute the ephemeris of the test orbit (this will be cached)
    ephemeris = test_orbit.generate_ephemeris_from_observations(
        observations,
        propagator=propagator,
        max_processes=max_processes,
    )

    # Assume that the heliocentric distance of all point sources in
    # the observations are the same as that of the test orbit
    ranged_detections_spherical = test_orbit.range_observations(
        observations,
        propagator=propagator,
        max_processes=max_processes,
    )

    # Transform from spherical topocentric to cartesian heliocentric coordinates
    ranged_detections_cartesian = transform_coordinates(
        ranged_detections_spherical.coordinates,
        representation_out=CartesianCoordinates,
        frame_out="ecliptic",
        origin_out=OriginCodes.SUN,
    )

    # Link the ephemeris and observations by state id
    link = qv.Linkage(
        ephemeris,
        observations,
        left_keys=ephemeris.id,
        right_keys=observations.state_id,
    )

    # Transform the detections into the co-rotating frame
    transformed_detection_list = []
    state_ids = observations.state_id.unique().sort()
    for state_id in state_ids:
        # Select the detections and ephemeris for this state id
        mask = pc.equal(state_id, observations.state_id)
        ranged_detections_cartesian_i = ranged_detections_cartesian.apply_mask(mask)
        ephemeris_i = link.select_left(state_id)
        observations_i = link.select_right(state_id)

        # Transform the detections into the co-rotating frame
        transformed_detections_i = TransformedDetections.from_kwargs(
            id=observations_i.detections.id,
            coordinates=GnomonicCoordinates.from_cartesian(
                ranged_detections_cartesian_i,
                center_cartesian=ephemeris_i.ephemeris.aberrated_coordinates,
            ),
            state_id=observations_i.state_id,
        )

        transformed_detection_list.append(transformed_detections_i)

    transformed_detections = qv.concatenate(transformed_detection_list)
    return transformed_detections


def link_test_orbit(
    test_orbit: TestOrbit,
    observations: Observations,
    filters: Optional[List[ObservationFilter]] = [
        TestOrbitRadiusObservationFilter(radius=10.0)
    ],
    propagator: Propagator = PYOORB(),
    max_processes: int = 1,
):
    """
    Find all linkages for a single test orbit.
    """
    # Apply filters to the observations
    filtered_observations = observations
    if filters is not None:
        for filter_i in filters:
            filtered_observations = filter_i.apply(filtered_observations, test_orbit)

    # Range and transform the observations
    transformed_detections = range_and_transform(
        test_orbit,
        filtered_observations,
        propagator=propagator,
        max_processes=max_processes,
    )

    # TODO: Find objects which move in straight-ish lines in the gnomonic frame.
    #
    # TODO: Run IOD against each of the discovered straight lines, and
    # filter down to plausible orbits.
    #
    # TODO: Run OD against the plausible orbits, and filter down to really good orbits.
    #
    # TODO: Perform arc extension on the really good orbits.

    return transformed_detections
