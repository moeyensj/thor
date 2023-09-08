from adam_core import propagator
from adam_core.coordinates import cartesian, transform

from . import observation_filters
from . import orbit as thor_orbit
from . import projections


def link_test_orbit(
    obs_src: observation_filters.ObservationFilter,
    test_orbit: thor_orbit.TestOrbit,
    propagator: propagator.Propagator,
    max_processes: int = 1,
):
    """
    Find all linkages for a single test orbit.
    """

    # Gather observations for the test orbit
    filtered_observations: observation_filters.Observations = obs_src.apply(
        filtered_observations
    )

    # Assume that the heliocentric distance of all point sources in
    # the observations are the same as that of the test orbit.
    #
    # FIXME: this may repeat work done in gather_observations, since
    # that might propagate the test orbit as well, depending on the
    # ObservationSource implementation. Caching?
    ranged_observations = test_orbit.range_observations(
        filtered_observations.linkage,
        max_processes=max_processes,
    )

    # Transform from spherical topocentric to cartesian heliocentric coordinates
    cartesian_heliocentric = transform.transform_coordinates(
        ranged_observations.coordinates,
        representation_out=cartesian.CartesianCoordinates,
        frame_out="ecliptic",
        origin_out="SUN",
    )

    # Project from cartesian heliocentric to gnomonic heliocentric
    # coordinates, centered on the test orbit's position.
    #
    # FIXME: this repeats propagation work done in
    # test_orbit.ranged_observation. Caching?
    test_orbit_positions = test_orbit.propagate(
        times=filtered_observations.exposures.midpoint().to_astropy(),
        propagator=propagator,
        max_processes=max_processes,
    )
    gnomonic = projections.GnomonicCoordinates.from_cartesian(
        cartesian_heliocentric,
        center_cartesian=test_orbit_positions,
    )

    # TODO: Find objects which move in straight-ish lines in the gnomonic frame.
    #
    # TODO: Run IOD against each of the discovered straight lines, and
    # filter down to plausible orbits.
    #
    # TODO: Run OD against the plausible orbits, and filter down to really good orbits.
    #
    # TODO: Perform arc extension on the really good orbits.
