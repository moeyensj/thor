import logging
import multiprocessing as mp
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
    KeplerianCoordinates,
    OriginCodes,
    SphericalCoordinates,
    transform_coordinates,
)
from adam_core.observers import Observers
from adam_core.orbits import Ephemeris, Orbits
from adam_core.propagator import Propagator
from adam_core.propagator.adam_pyoorb import PYOORBPropagator
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp

from .observations import Observations

CoordinateType = TypeVar(
    "CoordinateType",
    bound=Union[
        CartesianCoordinates,
        SphericalCoordinates,
        KeplerianCoordinates,
        CometaryCoordinates,
    ],
)


logger = logging.getLogger(__name__)


class RangedPointSourceDetections(qv.Table):
    id = qv.LargeStringColumn()
    exposure_id = qv.LargeStringColumn()
    coordinates = SphericalCoordinates.as_column()
    state_id = qv.LargeStringColumn()


class TestOrbitEphemeris(qv.Table):
    id = qv.LargeStringColumn()
    ephemeris = Ephemeris.as_column()
    observer = Observers.as_column()


def range_observations_worker(
    observations: Observations, ephemeris: TestOrbitEphemeris, state_id: str
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
        coordinates=assume_heliocentric_distance(r, observations_state.coordinates, observer_i.coordinates),
        state_id=observations_state.state_id,
    )


range_observations_remote = ray.remote(range_observations_worker)


class TestOrbits(qv.Table):
    orbit_id = qv.LargeStringColumn(default=lambda: uuid.uuid4().hex)
    object_id = qv.LargeStringColumn(nullable=True)
    bundle_id = qv.Int64Column(nullable=True)
    coordinates = CartesianCoordinates.as_column()

    @classmethod
    def from_orbits(cls, orbits):
        return cls.from_kwargs(
            orbit_id=orbits.orbit_id,
            object_id=orbits.object_id,
            coordinates=orbits.coordinates,
        )

    def to_orbits(self):
        return Orbits.from_kwargs(
            coordinates=self.coordinates,
            orbit_id=self.orbit_id,
            object_id=self.object_id,
        )

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
        if (
            getattr(self, "_cached_ephemeris", None) is None
            and getattr(self, "_cached_observation_ids", None) is None
        ):
            self._cached_ephemeris: Optional[TestOrbitEphemeris] = None
            self._cached_observation_ids: Optional[pa.Array] = None
            return False
        elif (
            getattr(self, "_cached_ephemeris", None) is not None
            and getattr(self, "_cached_observation_ids") is not None
            and pc.all(
                pc.is_in(
                    observations.id.sort(),
                    self._cached_observation_ids.sort(),  # type: ignore
                )
            ).as_py()
        ):
            return True
        else:
            return False

    def _cache_ephemeris(self, ephemeris: TestOrbitEphemeris, observations: Observations):
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
        propagator: Propagator = PYOORBPropagator(),
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
            self.to_orbits(),
            times,
            max_processes=max_processes,
            chunk_size=1,
        )

    def generate_ephemeris(
        self,
        observers: Observers,
        propagator: Propagator = PYOORBPropagator(),
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
            self.to_orbits(),
            observers,
            max_processes=max_processes,
            chunk_size=1,
        )

    def generate_ephemeris_from_observations(
        self,
        observations: Union[Observations, ray.ObjectRef],
        propagator: Propagator = PYOORBPropagator(),
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
            logger.debug("Test orbit ephemeris cache is fresh. Returning cached states.")
            return self._cached_ephemeris

        logger.debug("Test orbit ephemeris cache is stale. Regenerating.")

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

        observers_with_states = observers_with_states.sort_by(
            by=[
                "observers.coordinates.time.days",
                "observers.coordinates.time.nanos",
                "observers.coordinates.origin.code",
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
        propagator: Propagator = PYOORBPropagator(),
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

        if max_processes is None:
            max_processes = mp.cpu_count()

        ranged_detections = RangedPointSourceDetections.empty()
        use_ray = initialize_use_ray(num_cpus=max_processes)
        if use_ray:
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
            state_ids = observations.state_id.unique()
            futures = []
            for state_id in state_ids:
                futures.append(range_observations_remote.remote(observations_ref, ephemeris_ref, state_id))

                if len(futures) >= max_processes * 1.5:
                    finished, futures = ray.wait(futures, num_returns=1)
                    ranged_detections_chunk = ray.get(finished[0])
                    ranged_detections = qv.concatenate([ranged_detections, ranged_detections_chunk])
                    if ranged_detections.fragmented():
                        ranged_detections = qv.defragment(ranged_detections)

            while futures:
                finished, futures = ray.wait(futures, num_returns=1)
                ranged_detections_chunk = ray.get(finished[0])
                ranged_detections = qv.concatenate([ranged_detections, ranged_detections_chunk])
                if ranged_detections.fragmented():
                    ranged_detections = qv.defragment(ranged_detections)

        else:
            # Get state IDs
            state_ids = observations.state_id.unique()

            for state_id in state_ids:
                ranged_detections_chunk = range_observations_worker(
                    observations.select("state_id", state_id),
                    ephemeris.select("id", state_id),
                    state_id,
                )

                ranged_detections = qv.concatenate([ranged_detections, ranged_detections_chunk])
                if ranged_detections.fragmented():
                    ranged_detections = qv.defragment(ranged_detections)

        return ranged_detections.sort_by(by=["state_id"])


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
    coords_ec = transform_coordinates(coords_eq_unit, SphericalCoordinates, frame_out="ecliptic")

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
