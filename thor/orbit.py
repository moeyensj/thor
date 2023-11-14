import logging
import uuid
from typing import Optional, TypeVar, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
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
from adam_core.time import Timestamp

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


def range_observations_worker(
    observations: Observations, ephemeris: TestOrbitEphemeris, state_id: int
) -> RangedPointSourceDetections:
    """
    Range observations for a single state given the orbit's ephemeris for that state.

    Parameters
    ----------
    observations
        Observations to range.
    ephemeris
        Ephemeris from which to extract the test orbit's aberrated state (we
        use this state to get the test orbit's heliocentric distance).
    state_id
        The ID for this particular state.

    Returns
    -------
    ranged_point_source_detections
        The detections assuming they are located at the same heliocentric distance
        as the test orbit.
    """
    observations_state = observations.select("state_id", state_id)
    ephemeris_state = ephemeris.select("id", state_id)

    # Get the heliocentric position vector of the object at the time of the exposure
    r = ephemeris_state.ephemeris.aberrated_coordinates.r[0]

    # Get the observer's heliocentric coordinates
    observer_i = ephemeris_state.observer

    return RangedPointSourceDetections.from_kwargs(
        id=observations_state.id,
        exposure_id=observations_state.exposure_id,
        coordinates=assume_heliocentric_distance(
            r, observations_state.coordinates, observer_i.coordinates
        ),
        state_id=observations_state.state_id,
    )


range_observations_remote = ray.remote(range_observations_worker)


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
            pc.is_in(observations.id.sort(), self._cached_observation_ids.sort())
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
        self._cached_observation_ids = observations.id

    def propagate(
        self,
        times: Timestamp,
        propagator: Propagator = PYOORB(),
        max_processes: Optional[int] = 1,
    ) -> Orbits:
        """
        Propagate this test orbit to the given times.

        Parameters
        ----------
        times : `~adam_core.time.time.Timestamp`
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
    ) -> Ephemeris:
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
        ephemeris : `~adam_core.orbits.ephemeris.Ephemeris`
            The ephemeris of the test orbit at the given observers.
        """
        return propagator.generate_ephemeris(
            self.orbit, observers, max_processes=max_processes, chunk_size=1
        )

    def generate_ephemeris_from_observations(
        self,
        observations: Union[Observations, ray.ObjectRef],
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

        Raises
        ------
        ValueError
            If the observations are empty.
        """
        if isinstance(observations, ray.ObjectRef):
            observations = ray.get(observations)

        if len(observations) == 0:
            raise ValueError("Observations must not be empty.")

        if self._is_cache_fresh(observations):
            logger.debug(
                "Test orbit ephemeris cache is fresh. Returning cached states."
            )
            return self._cached_ephemeris

        logger.debug("Test orbit ephemeris cache is stale. Regenerating.")

        state_ids = observations.state_id.unique()
        observers_with_states = observations.get_observers()

        # Generate ephemerides for each unique state and then sort by time and code
        ephemeris = self.generate_ephemeris(
            observers_with_states.observers,
            propagator=propagator,
            max_processes=max_processes,
        )
        ephemeris = ephemeris.sort_by(
            by=[
                "coordinates.time.days",
                "coordinates.time.nanos",
                "coordinates.origin.code",
            ]
        )

        test_orbit_ephemeris = TestOrbitEphemeris.from_kwargs(
            id=observers_with_states.state_id,
            ephemeris=ephemeris,
            observer=observers_with_states.observers,
        )
        self._cache_ephemeris(test_orbit_ephemeris, observations)
        return test_orbit_ephemeris

    def range_observations(
        self,
        observations: Union[Observations, ray.ObjectRef],
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

        ranged_detections_list = []
        if max_processes is None or max_processes > 1:
            if not ray.is_initialized():
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

            # Get state IDs
            state_ids = observations.state_id.unique().sort()
            futures = []
            for state_id in state_ids:
                futures.append(
                    range_observations_remote.remote(
                        observations_ref, ephemeris_ref, state_id
                    )
                )

            while futures:
                finished, futures = ray.wait(futures, num_returns=1)
                ranged_detections_list.append(ray.get(finished[0]))

        else:
            # Get state IDs
            state_ids = observations.state_id.unique().sort()

            for state_id in state_ids:
                ranged_detections_list.append(
                    range_observations_worker(
                        observations.select("state_id", state_id),
                        ephemeris.select("id", state_id),
                        state_id,
                    )
                )

        ranged_point_source_detections = qv.concatenate(ranged_detections_list)
        return ranged_point_source_detections.sort_by(by=["state_id"])


def assume_heliocentric_distance(
    r: np.ndarray, coords: SphericalCoordinates, origin_coords: CartesianCoordinates
) -> SphericalCoordinates:
    """
    Given a heliocentric distance, for all coordinates that do not have a topocentric distance defined (rho), calculate
    the topocentric distance assuming the coordinates are located at the given heliocentric distance.

    Parameters
    ----------
    r_mag : `~numpy.ndarray` (3)
        Heliocentric position vector from which to assume each coordinate lies at the same heliocentric distance.
        In cases where the heliocentric distance is less than the heliocentric distance of the origin, the topocentric
        distance will be calculated such that the topocentric position vector is closest to the heliocentric position
        vector.
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

    r_mag = np.linalg.norm(r)

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
    sqrt = np.sqrt(dotprod**2 + r_mag**2 - origin_coords.r_mag**2)
    delta_p = -dotprod + sqrt
    delta_n = -dotprod - sqrt

    # Where rho was not defined, replace it with the calculated topocentric distance
    # By default we take the positive solution which applies for all orbits exterior to the
    # observer's orbit
    coords_ec = coords_ec.set_column("rho", np.where(np.isnan(rho), delta_p, rho))

    # For cases where the orbit is interior to the observer's orbit there are two valid solutions
    # for the topocentric distance. In this case, we take the dot product of the heliocentric position
    # vector with the calculated topocentric position vector. If the dot product is positive, then
    # that solution is closest to the heliocentric position vector and we take that solution.
    if np.any(r_mag < origin_coords.r_mag):
        coords_ec_xyz_p = coords_ec.to_cartesian()
        dotprod_p = np.sum(coords_ec_xyz_p.r * r, axis=1)
        coords_ec = coords_ec.set_column(
            "rho",
            np.where(np.isnan(rho), np.where(dotprod_p < 0, delta_n, delta_p), rho),
        )

    coords_ec = coords_ec.set_column("vrho", vrho)

    return coords_ec
