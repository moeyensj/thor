import logging
import uuid
from typing import Optional, TypeVar, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from adam_core.coordinates import (
    CartesianCoordinates,
    CometaryCoordinates,
    CoordinateCovariances,
    KeplerianCoordinates,
    Origin,
    OriginCodes,
    SphericalCoordinates,
    transform_coordinates,
)
from adam_core.observers import Observers
from adam_core.orbits import Ephemeris, Orbits
from adam_core.propagator import PYOORB, Propagator
from astropy.time import Time

CoordinateType = TypeVar(
    "CoordinateType",
    bound=Union[
        CartesianCoordinates,
        SphericalCoordinates,
        KeplerianCoordinates,
        CometaryCoordinates,
    ],
)

from .observations import Observations

logger = logging.getLogger(__name__)


class RangedPointSourceDetections(qv.Table):

    id = qv.StringColumn()
    exposure_id = qv.StringColumn()
    coordinates = SphericalCoordinates.as_column()
    state_id = qv.Int64Column()


class TestOrbitEphemeris(qv.Table):

    id = qv.Int64Column()
    ephemeris = Ephemeris.as_column()
    observer = Observers.as_column()


class TestOrbit:
    def __init__(
        self,
        coordinates: CoordinateType,
        orbit_id: Optional[str] = None,
        object_id: Optional[str] = None,
    ):
        """
        Create a test orbit from a set of orbital elements.

        Parameters
        ----------
        coordinates :  `~adam_core.coordinates.cartesian.CartesianCoordinates`,
                       `~adam_core.coordinates.spherical.SphericalCoordinates`,
                       `~adam_core.coordinates.keplerian.KeplerianCoordinates`,
                       `~adam_core.coordinates.cometary.CometaryCoordinates`
            The orbital elements that define this test orbit. Can be any representation but will
            be stored internally as Cartesian elements.
        orbit_id : str, optional
            Orbit ID. If not provided, a random UUID will be generated.
        object_id : str, optional
            Object ID, if it exists.
        """
        # Test orbits should be singletons
        assert len(coordinates) == 1

        # Test orbit selection will likely occur in a non-Cartesian coordinate system
        # so we should accept any coordinate system and convert to Cartesian as the
        # most stable representation
        if not isinstance(coordinates, CartesianCoordinates):
            cartesian_coordinates = coordinates.to_cartesian()
        else:
            cartesian_coordinates = coordinates

        if orbit_id is not None:
            self.orbit_id = orbit_id
        else:
            self.orbit_id = uuid.uuid4().hex

        self.object_id = object_id

        self._orbit = Orbits.from_kwargs(
            orbit_id=[self.orbit_id],
            object_id=[self.object_id],
            coordinates=cartesian_coordinates,
        )

        self._cached_ephemeris: Optional[TestOrbitEphemeris] = None
        self._cached_observation_ids: Optional[pa.array] = None

    @classmethod
    def from_orbits(cls, orbits):
        assert len(orbits) == 1
        return cls(
            orbits.coordinates, orbits.orbit_id[0].as_py(), orbits.object_id[0].as_py()
        )

    @property
    def orbit(self):
        return self._orbit

    def _is_cache_fresh(self, observations: Observations) -> bool:
        """
        Check if the cached ephemeris is fresh. If the observation IDs are contained within the
        cached observation IDs, then the cache is fresh. Otherwise, it is stale. This permits
        observations to be filtered out without having to regenerate the ephemeris.

        Parameters
        ----------
        observations : `~thor.observations.observations.Observations`
            Observations to check against the cached ephemerides.

        Returns
        -------
        is_fresh : bool
            True if the cache is fresh, False otherwise.
        """
        if self._cached_ephemeris is None or self._cached_observation_ids is None:
            return False
        elif pc.all(
            pc.is_in(
                observations.detections.id.sort(), self._cached_observation_ids.sort()
            )
        ).as_py():
            return True
        else:
            return False

    def _cache_ephemeris(
        self, ephemeris: TestOrbitEphemeris, observations: Observations
    ):
        """
        Cache the ephemeris and observation IDs.

        Parameters
        ----------
        ephemeris : `~thor.orbit.TestOrbitEphemeris`
            States to cache.
        observations : `~thor.observations.observations.Observations`
            Observations to cache. Only observation IDs will be cached.

        Returns
        -------
        None
        """
        self._cached_ephemeris = ephemeris
        self._cached_observation_ids = observations.detections.id

    def propagate(
        self,
        times: Time,
        propagator: Propagator = PYOORB(),
        max_processes: Optional[int] = 1,
    ) -> Orbits:
        """
        Propagate this test orbit to the given times.

        Parameters
        ----------
        times : `~astropy.time.core.Time`
            Times to which to propagate the orbit.
        propagator : `~adam_core.propagator.propagator.Propagator`, optional
            Propagator to use to propagate the orbit. Defaults to PYOORB.
        num_processes : int, optional
            Number of processes to use to propagate the orbit. Defaults to 1.

        Returns
        -------
        propagated_orbit : `~adam_core.orbits.orbits.Orbits`
            The test orbit propagated to the given times.
        """
        return propagator.propagate_orbits(
            self.orbit, times, max_processes=max_processes, chunk_size=1
        )

    def generate_ephemeris(
        self,
        observers: Observers,
        propagator: Propagator = PYOORB(),
        max_processes: Optional[int] = 1,
    ) -> qv.MultiKeyLinkage[Ephemeris, Observers]:
        """
        Generate ephemeris for this test orbit at the given observers.

        Parameters
        ----------
        observers : `~adam_core.observers.Observers`
            Observers from which to generate ephemeris.
        propagator : `~adam_core.propagator.propagator.Propagator`, optional
            Propagator to use to propagate the orbit. Defaults to PYOORB.
        num_processes : int, optional
            Number of processes to use to propagate the orbit. Defaults to 1.

        Returns
        -------
        ephemeris : qv.MultiKeyLinkage[
                `~adam_core.orbits.ephemeris.Ephemeris`,
                `~adam_core.observers.observers.Observers`]
            The ephemeris of the test orbit at the given observers.
        """
        return propagator.generate_ephemeris(
            self.orbit, observers, max_processes=max_processes, chunk_size=1
        )

    def generate_ephemeris_from_observations(
        self,
        observations: Observations,
        propagator: Propagator = PYOORB(),
        max_processes: Optional[int] = 1,
    ):
        """
        For each unique time and code in the observations (a state), generate an ephemeris for
        that state and store them in a TestOrbitStates table. The observer's coordinates will also be
        stored in the table and can be referenced through out the THOR pipeline.

        These ephemerides will be cached. If the cache is fresh, the cached ephemerides will be
        returned instead of regenerating them.

        Parameters
        ----------
        observations : `~thor.observations.observations.Observations`
            Observations to compute test orbit ephemerides for.
        propagator : `~adam_core.propagator.propagator.Propagator`, optional
            Propagator to use to propagate the orbit. Defaults to PYOORB.
        num_processes : int, optional
            Number of processes to use to propagate the orbit. Defaults to 1.


        Returns
        -------
        states : `~thor.orbit.TestOrbitEphemeris`
            Table containing the ephemeris of the test orbit, its aberrated state vector, and the
            observer coordinates at each unique time of the observations.
        """
        if self._is_cache_fresh(observations):
            logger.debug(
                "Test orbit ephemeris cache is fresh. Returning cached states."
            )
            return self._cached_ephemeris

        logger.debug("Test orbit ephemeris cache is stale. Regenerating.")

        state_ids = observations.state_id.unique()
        observers = observations.get_observers()

        # Generate ephemerides for each unique state and then sort by time and code
        ephemeris = self.generate_ephemeris(
            observers, propagator=propagator, max_processes=max_processes
        ).left_table
        ephemeris = ephemeris.sort_by(
            by=[
                "coordinates.time.jd1",
                "coordinates.time.jd2",
                "coordinates.origin.code",
            ]
        )

        test_orbit_ephemeris = TestOrbitEphemeris.from_kwargs(
            id=state_ids,
            ephemeris=ephemeris,
            observer=observers,
        )
        self._cache_ephemeris(test_orbit_ephemeris, observations)
        return test_orbit_ephemeris

    def range_observations(
        self,
        observations: Observations,
        propagator: Propagator = PYOORB(),
        max_processes: Optional[int] = 1,
    ) -> RangedPointSourceDetections:
        """
        Given a set of observations, propagate this test orbit to the times of the observations and calculate the
        topocentric distance (range) assuming they lie at the same heliocentric distance as the test orbit.

        Parameters
        ----------
        observations : `~thor.observations.observations.Observations`
            Observations to range.
        propagator : `~adam_core.propagator.propagator.Propagator`, optional
            Propagator to use to propagate the orbit. Defaults to PYOORB.
        max_processes : int, optional
            Number of processes to use to propagate the orbit. Defaults to 1.

        Returns
        -------
        ranged_point_source_detections : `~thor.orbit.RangedPointSourceDetections`
            The ranged detections.
        """
        # Generate an ephemeris for each unique observation time and observatory
        # code combination
        ephemeris = self.generate_ephemeris_from_observations(
            observations, propagator=propagator, max_processes=max_processes
        )

        # Link the ephemeris to the observations
        link = qv.Linkage(
            ephemeris,
            observations,
            left_keys=ephemeris.id,
            right_keys=observations.state_id,
        )

        # Do a sorted iteration over the unique state IDs
        rpsds = []
        state_ids = observations.state_id.unique().sort()
        for state_id in state_ids:

            # Select the ephemeris and observations for this state
            ephemeris_i = link.select_left(state_id)
            observations_i = link.select_right(state_id)
            detections_i = observations_i.detections

            # Get the heliocentric distance of the object at the time of the exposure
            r_mag = ephemeris_i.ephemeris.aberrated_coordinates.r_mag[0]

            # Get the observer's heliocentric coordinates
            observer_i = ephemeris_i.observer

            # Create an array of observatory codes for the detections
            num_detections = len(observations_i)
            observatory_codes = np.repeat(
                observations_i.observatory_code[0].as_py(), num_detections
            )

            # The following can be replaced with:
            # coords = observations_i.to_spherical(observatory_codes)
            # Start replacement:
            sigma_data = np.vstack(
                [
                    pa.nulls(num_detections, pa.float64()),
                    detections_i.ra_sigma.to_numpy(zero_copy_only=False),
                    detections_i.dec_sigma.to_numpy(zero_copy_only=False),
                    pa.nulls(num_detections, pa.float64()),
                    pa.nulls(num_detections, pa.float64()),
                    pa.nulls(num_detections, pa.float64()),
                ]
            ).T
            coords = SphericalCoordinates.from_kwargs(
                lon=detections_i.ra,
                lat=detections_i.dec,
                time=detections_i.time,
                covariance=CoordinateCovariances.from_sigmas(sigma_data),
                origin=Origin.from_kwargs(code=observatory_codes),
                frame="equatorial",
            )
            # End replacement (only once
            # https://github.com/B612-Asteroid-Institute/adam_core/pull/45 is merged)

            rpsds.append(
                RangedPointSourceDetections.from_kwargs(
                    id=detections_i.id,
                    exposure_id=detections_i.exposure_id,
                    coordinates=assume_heliocentric_distance(
                        r_mag, coords, observer_i.coordinates
                    ),
                    state_id=observations_i.state_id,
                )
            )

        ranged_detections = qv.concatenate(rpsds)
        return ranged_detections


def assume_heliocentric_distance(
    r_mag: float, coords: SphericalCoordinates, origin_coords: CartesianCoordinates
) -> SphericalCoordinates:
    """
    Given a heliocentric distance, for all coordinates that do not have a topocentric distance defined (rho), calculate
    the topocentric distance assuming the coordinates are located at the given heliocentric distance.

    Parameters
    ----------
    r_mag : float
        Heliocentric distance to assume for the coordinates with missing topocentric distance. This is
        typically the same distance as the heliocentric distance of test orbit at the time
        of the coordinates.
    coords : `~adam_core.coordinates.spherical.SphericalCoordinates`
        Coordinates to assume the heliocentric distance for.
    origin_coords : `~adam_core.coordinates.cartesian.CartesianCoordinates`
        Heliocentric coordinates of the origin of the topocentric coordinates.

    Returns
    -------
    coords : `~adam_core.coordinates.spherical.SphericalCoordinates`
        Coordinates with the missing topocentric distance replaced with the calculated topocentric distance.
    """
    assert len(origin_coords) == 1
    assert np.all(origin_coords.origin == OriginCodes.SUN)

    # Extract the topocentric distance and topocentric radial velocity from the coordinates
    rho = coords.rho.to_numpy(zero_copy_only=False)
    vrho = coords.vrho.to_numpy(zero_copy_only=False)

    # Transform the coordinates to the ecliptic frame by assuming they lie on a unit sphere
    # (this assumption will only be applied to coordinates with missing rho values)
    coords_eq_unit = coords.to_unit_sphere(only_missing=True)
    coords_ec = transform_coordinates(
        coords_eq_unit, SphericalCoordinates, frame_out="ecliptic"
    )

    # Transform the coordinates to cartesian and calculate the unit vectors pointing
    # from the origin to the coordinates
    coords_ec_xyz = coords_ec.to_cartesian()
    unit_vectors = coords_ec_xyz.r_hat

    # Calculate the topocentric distance such that the heliocentric distance to the coordinate
    # is r_mag
    dotprod = np.sum(unit_vectors * origin_coords.r, axis=1)
    delta = -dotprod + np.sqrt(dotprod**2 + r_mag**2 - origin_coords.r_mag**2)

    # Where rho was not defined, replace it with the calculated topocentric distance
    coords_ec = coords_ec.set_column("rho", np.where(np.isnan(rho), delta, rho))
    coords_ec = coords_ec.set_column("vrho", vrho)

    return coords_ec
