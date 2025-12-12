import logging
import multiprocessing as mp
import uuid
from typing import Literal, Optional, Type, TypeVar, Union

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
from adam_core.coordinates.transform import cartesian_to_frame
from adam_core.observers import Observers
from adam_core.orbits import Ephemeris, Orbits
from adam_core.propagator import Propagator
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp

from .observations import Observations
from .observations.observations import get_observation_observers
from .projections.gnomonic import GnomonicCoordinates

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
    gnomonic = GnomonicCoordinates.as_column()
    # TODO: This should be a fixed shape tensor when they can be pickled correctly...
    gnomonic_rotation_matrix = qv.FixedSizeListColumn(pa.float64(), list_size=36)


def zero_covariances(coords: CoordinateType) -> CoordinateType:
    """
    Zero out covariances that are nan, inf, or less than 1e-20 in magnitude.

    This is function is designed to be used in conjunction with transformations where we
    want to propagate covariance matrices through but not all the covariance terms are valid or defined.

    The user should careful with the downstream effects of this function.

    Parameters
    ----------
    coords : `~adam_core.coordinates.CoordinateType`
        Coordinates to zero out the covariances for.

    Returns
    -------
    coords : `~adam_core.coordinates.CoordinateType`
        Coordinates with the covariances zeroed out.
    """
    cov = coords.covariance.to_matrix()
    cov = np.where(np.isnan(cov) | np.isinf(cov) | (np.abs(cov) < 1e-20), 0, cov)
    return coords.set_column("covariance", CoordinateCovariances.from_matrix(cov))


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
    assert len(ephemeris_state) == 1

    # Get the heliocentric position vector of the object at the time of the exposure
    aberrated_coordinates = ephemeris_state.ephemeris.aberrated_coordinates
    if aberrated_coordinates.origin.code.to_pylist()[0] != "SUN":
        aberrated_coordinates = transform_coordinates(
            aberrated_coordinates,
            CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SUN,
        )

    r = aberrated_coordinates.r[0]

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
    nside = qv.Int64Column(nullable=True)
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

    def propagate(
        self,
        times: Timestamp,
        propagator_class: Type[Propagator],
        max_processes: Optional[int] = 1,
        covariance: bool = False,
        covariance_method: Literal["monte-carlo", "sigma-point", "auto"] = "monte-carlo",
        num_samples: int = 1000,
    ) -> Orbits:
        """
        Propagate this test orbit to the given times.

        Parameters
        ----------
        times : `~adam_core.time.time.Timestamp`
            Times to which to propagate the orbit.
        propagator : `~adam_core.propagator.propagator.Propagator`
            Propagator to use to propagate the orbit.
        num_processes : int, optional
            Number of processes to use to propagate the orbit. Defaults to 1.
        covariance: bool, optional
            Propagate the covariance matrices of the orbits. This is done by sampling the
            orbits from their covariance matrices and propagating each sample and for each
            sample also generating ephemerides. The covariance
            of the ephemerides is then the covariance of the samples.
        covariance_method : {'sigma-point', 'monte-carlo', 'auto'}, optional
            The method to use for sampling the covariance matrix. If 'auto' is selected then the method
            will be automatically selected based on the covariance matrix. The default is 'monte-carlo'.
        num_samples : int, optional
            The number of samples to draw when sampling with monte-carlo.

        Returns
        -------
        propagated_orbit : `~adam_core.orbits.orbits.Orbits`
            The test orbit propagated to the given times.
        """
        propagator = propagator_class()
        return propagator.propagate_orbits(
            self.to_orbits(),
            times,
            max_processes=max_processes,
            chunk_size=10,
            covariance=covariance,
            covariance_method=covariance_method,
            num_samples=num_samples,
        )

    def generate_ephemeris(
        self,
        observers: Observers,
        propagator_class: Type[Propagator],
        max_processes: Optional[int] = 1,
        covariance: bool = False,
        covariance_method: Literal["monte-carlo", "sigma-point", "auto"] = "monte-carlo",
        num_samples: int = 1000,
    ) -> Ephemeris:
        """
        Generate ephemeris for this test orbit at the given observers.

        Parameters
        ----------
        observers : `~adam_core.observers.Observers`
            Observers from which to generate ephemeris.
        propagator_class : `~adam_core.propagator.propagator.Propagator`
            Propagator to use to propagate the orbit.
        num_processes : int, optional
            Number of processes to use to propagate the orbit. Defaults to 1.
        covariance: bool, optional
            Propagate the covariance matrices of the orbits. This is done by sampling the
            orbits from their covariance matrices and propagating each sample and for each
            sample also generating ephemerides. The covariance
            of the ephemerides is then the covariance of the samples.
        covariance_method : {'sigma-point', 'monte-carlo', 'auto'}, optional
            The method to use for sampling the covariance matrix. If 'auto' is selected then the method
            will be automatically selected based on the covariance matrix. The default is 'monte-carlo'.
        num_samples : int, optional
            The number of samples to draw when sampling with monte-carlo.

        Returns
        -------
        ephemeris : `~adam_core.orbits.ephemeris.Ephemeris`
            The ephemeris of the test orbit at the given observers.
        """
        propagator = propagator_class()
        return propagator.generate_ephemeris(
            self.to_orbits(),
            observers,
            max_processes=max_processes,
            chunk_size=10,
            covariance=covariance,
            covariance_method=covariance_method,
            num_samples=num_samples,
        )

    def generate_ephemeris_from_observations(
        self,
        observations: Union[Observations, ray.ObjectRef, str],
        propagator_class: Type[Propagator],
        max_processes: Optional[int] = 1,
        covariance: bool = True,
    ):
        """
        For each unique time and code in the observations (a state), generate an ephemeris for
        that state and store them in a TestOrbitEphemeris table. The observer's coordinates will also be
        stored in the table and can be referenced throughout the THOR pipeline.

        Parameters
        ----------
        observations : `~thor.observations.observations.Observations`, ray.ObjectRef, or str
            Observations to compute test orbit ephemerides for. Can be an Observations object,
            a ray ObjectRef, or a path to a parquet file or directory containing parquet files.
            When a path is provided, only the time and observatory code columns are read for
            efficiency.
        propagator_class : `~adam_core.propagator.propagator.Propagator`
            Propagator to use to propagate the orbit.
        max_processes : int, optional
            Number of processes to use to propagate the orbit. Defaults to 1.
        covariance: bool, optional
            Propagate the covariance matrices of the orbits. This is done by sampling the
            orbits from their covariance matrices and propagating each sample and for each
            sample also generating ephemerides. The covariance
            of the ephemerides is then the covariance of the samples.

        Returns
        -------
        test_orbit_ephemeris : `~thor.orbit.TestOrbitEphemeris`
            Table containing the ephemeris of the test orbit, its aberrated state vector, and the
            observer coordinates at each unique time of the observations.
        """
        observers_with_states = get_observation_observers(observations)

        if len(observers_with_states) == 0:
            return TestOrbitEphemeris.empty()

        observers_with_states = observers_with_states.sort_by(
            by=[
                "observers.coordinates.time.days",
                "observers.coordinates.time.nanos",
                "observers.code",
            ]
        )

        # Generate ephemerides for each unique state and then sort by time and code
        ephemeris = self.generate_ephemeris(
            observers_with_states.observers,
            propagator_class=propagator_class,
            max_processes=max_processes,
            covariance=covariance,
            covariance_method="monte-carlo",
            num_samples=1000,
        )
        ephemeris = ephemeris.sort_by(
            by=[
                "coordinates.time.days",
                "coordinates.time.nanos",
                "coordinates.origin.code",
            ]
        )

        # Compute the gnomonic coordinates and rotation matrix for each state
        gnomonic, gnomonic_rotation_matrix = GnomonicCoordinates.from_cartesian(
            ephemeris.aberrated_coordinates,
            center_cartesian=ephemeris.aberrated_coordinates,
        )
        # Convert rotation matrix to FixedSizeListArray (flatten to fixed-size chunks of 36)
        gnomonic_rotation_matrix = pa.FixedSizeListArray.from_arrays(
            gnomonic_rotation_matrix.flatten(), list_size=36
        )

        test_orbit_ephemeris = TestOrbitEphemeris.from_kwargs(
            id=observers_with_states.state_id,
            ephemeris=ephemeris,
            observer=observers_with_states.observers,
            gnomonic=gnomonic,
            gnomonic_rotation_matrix=gnomonic_rotation_matrix,
        )

        return test_orbit_ephemeris

    def range_observations(
        self,
        observations: Union[Observations, ray.ObjectRef],
        propagator_class: Type[Propagator],
        max_processes: Optional[int] = 1,
        test_orbit_ephemeris: Optional[Union[TestOrbitEphemeris, ray.ObjectRef]] = None,
    ) -> RangedPointSourceDetections:
        """
        Given a set of observations, propagate this test orbit to the times of the observations and calculate the
        topocentric distance (range) assuming they lie at the same heliocentric distance as the test orbit.

        Parameters
        ----------
        observations : `~thor.observations.observations.Observations`
            Observations to range.
        propagator_class : `~adam_core.propagator.propagator.Propagator`
            Propagator class to use to propagate the orbit.
        max_processes : int, optional
            Number of processes to use to propagate the orbit. Defaults to 1.
        test_orbit_ephemeris : `~thor.orbit.TestOrbitEphemeris` or ray.ObjectRef, optional
            Pre-computed test orbit ephemeris. If provided, ephemeris generation will be
            skipped. If None, ephemeris will be generated from observations.

        Returns
        -------
        ranged_point_source_detections : `~thor.orbit.RangedPointSourceDetections`
            The ranged detections.
        """
        # Use provided ephemeris or generate one
        if test_orbit_ephemeris is not None:
            if isinstance(test_orbit_ephemeris, ray.ObjectRef):
                ephemeris = ray.get(test_orbit_ephemeris)
            else:
                ephemeris = test_orbit_ephemeris
        else:
            ephemeris = self.generate_ephemeris_from_observations(
                observations, propagator_class=propagator_class, max_processes=max_processes
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

    def split(
        self,
        dt,
        k: int = 3,
        beta: float = 0.6,
        gamma: float = 1.0,
        num: int = 2,
        current_depth: int = 0,
        max_depth: int = 1,
    ):
        """
        Split the test orbits into multiple test orbits.

        Parameters
        ----------
        dt : float
            Time step used to scale position/velocity coupling.
        k : int, optional
            Spread factor for child offsets along principal axis, by default 3 (3-sigma).
        beta : float, optional
            Shrink factor applied to leading eigenvalue for children, by default 0.6.
        gamma : float, optional
            Shrink factor applied to non-leading eigenvalues for children, by default 0.9.
        num : int, optional
            Number of children to create per split, by default 2.
        current_depth : int, optional
            Current recursion depth, by default 0.
        max_depth : int, optional
            Maximum recursion depth (inclusive), by default 0.

        Returns
        -------
        TestOrbits
            New test orbits spanning the phase space of the original test orbits.
        """
        from .phase_space.split import split_phase_space

        test_orbits = TestOrbits.empty()
        for test_orbit in self:
            states, covariances = split_phase_space(
                test_orbit.coordinates.values[0],
                test_orbit.coordinates.covariance.to_matrix()[0],
                dt,
                k,
                beta,
                gamma,
                num,
                current_depth,
                max_depth,
            )

            num_orbits = len(states)
            test_orbits_i = TestOrbits.from_kwargs(
                orbit_id=pc.binary_join_element_wise(
                    pa.repeat(test_orbit.orbit_id[0], num_orbits),
                    pa.array([f"{i:03d}" for i in range(num_orbits)], pa.large_string()),
                    pc.cast(pa.scalar("_"), pa.large_string()),
                ),
                object_id=pa.repeat(test_orbit.object_id[0], num_orbits),
                bundle_id=pa.repeat(test_orbit.bundle_id[0], num_orbits),
                coordinates=CartesianCoordinates.from_kwargs(
                    x=states[:, 0],
                    y=states[:, 1],
                    z=states[:, 2],
                    vx=states[:, 3],
                    vy=states[:, 4],
                    vz=states[:, 5],
                    covariance=CoordinateCovariances.from_matrix(covariances),
                    time=Timestamp.from_kwargs(
                        days=pa.repeat(test_orbit.coordinates.time.days[0], num_orbits),
                        nanos=pa.repeat(test_orbit.coordinates.time.nanos[0], num_orbits),
                        scale=test_orbit.coordinates.time.scale,
                    ),
                    origin=Origin.from_kwargs(
                        code=pa.repeat(test_orbit.coordinates.origin.code[0], num_orbits)
                    ),
                    frame=test_orbit.coordinates.frame,
                ),
            )

            test_orbits = qv.concatenate([test_orbits, test_orbits_i])

        return test_orbits

    def split_healpixel(
        self,
    ):
        """
        Refine sky localization by subdividing lon/lat and their variances from a parent
        HEALPix pixel into all child pixels at a larger nside (2x the parent nside).
        Keeps rho, vrho, vlon, vlat and their covariances unchanged; only lon/lat and
        sigma(lon/lat) are adjusted to the child pixel centers and half-widths.

        Returns
        -------
        TestOrbits
            New table with rows replicated per child pixel and lon/lat refined.
        """
        from .phase_space.split import split_healpixel as _split_healpixel_states

        test_orbits = TestOrbits.empty()
        test_orbit_ids = self.orbit_id
        object_ids = self.object_id
        bundle_ids = self.bundle_id
        spherical = self.coordinates.to_spherical()
        mu = spherical.values
        cov = spherical.covariance.to_matrix()
        current_nside = self.nside.to_numpy(zero_copy_only=False)
        new_nside = current_nside * 2

        for i in range(len(self)):

            states, covariances = _split_healpixel_states(
                mu[i],
                cov[i],
                current_nside[i],
                new_nside[i],
            )

            num_orbits = len(states)

            coords_sph = SphericalCoordinates.from_kwargs(
                rho=states[:, 0],
                lon=states[:, 1],
                lat=states[:, 2],
                vrho=states[:, 3],
                vlon=states[:, 4],
                vlat=states[:, 5],
                time=Timestamp.from_kwargs(
                    days=pa.repeat(spherical[i].time.days[0], num_orbits),
                    nanos=pa.repeat(spherical[i].time.nanos[0], num_orbits),
                    scale=spherical[i].time.scale,
                ),
                covariance=CoordinateCovariances.from_matrix(covariances),
                origin=Origin.from_kwargs(code=pa.repeat(spherical[i].origin.code[0], num_orbits)),
                frame=spherical[i].frame,
            )
            coords_cart = coords_sph.to_cartesian()

            test_orbits_i = TestOrbits.from_kwargs(
                orbit_id=pc.binary_join_element_wise(
                    pa.repeat(test_orbit_ids[i], num_orbits),
                    pa.array([f"{i:03d}" for i in range(num_orbits)], pa.large_string()),
                    pc.cast(pa.scalar("_"), pa.large_string()),
                ),
                object_id=pa.repeat(object_ids[i], num_orbits),
                bundle_id=pa.repeat(bundle_ids[i], num_orbits),
                nside=pa.repeat(new_nside[i], num_orbits),
                coordinates=coords_cart,
            )

            test_orbits = qv.concatenate([test_orbits, test_orbits_i])

        return test_orbits


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
    assert pc.all(pc.equal(origin_coords.origin.code, "SUN")).as_py()

    # Calculate the heliocentric distance
    r_mag = np.linalg.norm(r)

    # Extract the topocentric distance and topocentric radial velocity from the coordinates
    rho = coords.rho.to_numpy(zero_copy_only=False)
    vrho = coords.vrho.to_numpy(zero_copy_only=False)

    # Transform the coordinates to the ecliptic frame by assuming they lie on a unit sphere
    # (this assumption will only be applied to coordinates with missing rho values)
    # Convert equatorial spherical units to a unit sphere (so we can rotate them to the ecliptic)
    # and set nan covariance terms to zero so we preserve the non-zero covariance terms through
    # the transformations
    coords_eq_unit = coords.to_unit_sphere(only_missing=True)
    coords_eq_unit = zero_covariances(coords_eq_unit)

    # Transform to the equatorial coordinates (so we can rotate them to the ecliptic)
    coords_eq_cart = coords_eq_unit.to_cartesian()

    # Rotate to the ecliptic coordinates (again set nan covariance terms to zero)
    coords_ec_cart = cartesian_to_frame(coords_eq_cart, "ecliptic")
    coords_ec_cart = zero_covariances(coords_ec_cart)

    # Transform the coordinates to cartesian and calculate the unit vectors pointing
    # from the origin to the coordinates
    unit_vectors = coords_ec_cart.r_hat

    # Calculate the topocentric distance such that the heliocentric distance to the coordinate
    # is r_mag
    dotprod = np.sum(unit_vectors * origin_coords.r, axis=1)
    sqrt = np.sqrt(dotprod**2 + r_mag**2 - origin_coords.r_mag**2)
    delta_p = -dotprod + sqrt
    delta_n = -dotprod - sqrt

    # Where rho was not defined, replace it with the calculated topocentric distance
    # By default we take the positive solution which applies for all orbits exterior to the
    # observer's orbit
    coords_ec = coords_ec_cart.to_spherical()
    coords_ec = zero_covariances(coords_ec)
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
