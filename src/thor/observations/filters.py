import abc
import importlib
import logging
import multiprocessing as mp
import time
from typing import TYPE_CHECKING, List, Optional, Type, Union

import numpy as np
import pyarrow.parquet as pq
import quivr as qv
import ray
from adam_core.coordinates import SphericalCoordinates, CartesianCoordinates, OriginCodes, transform_coordinates
from adam_core.propagator import Propagator
from adam_core.ray_cluster import initialize_use_ray


from thor.config import Config
from thor.observations.observations import Observations, observations_iterator

from ..orbit import TestOrbits

if TYPE_CHECKING:
    from .observations import Observations


logger = logging.getLogger(__name__)


class ObservationFilter(abc.ABC):
    """An ObservationFilter is reduces a collection of observations to
    a subset of those observations.

    """

    @abc.abstractmethod
    def apply(
        self,
        observations: Observations,
        test_orbit: TestOrbits,
        propagator_class: Type[Propagator],
    ) -> "Observations":
        """
        Apply the filter to a collection of observations.

        Parameters
        ----------
        observations : `~thor.observations.Observations`
            The observations to filter.
        test_orbit : `~thor.orbit.TestOrbits`
            The test orbit to use for filtering.
        max_processes : int, optional
            Maximum number of processes to use for parallelization. If
            an existing ray cluster is already running, this parameter
            will be ignored if larger than 1 or not None.

        Returns
        -------
        filtered_observations : `~thor.observations.Observations`
            The filtered observations.
        """
        ...


class TestOrbitRadiusObservationFilter(ObservationFilter):
    """A TestOrbitRadiusObservationFilter is an ObservationFilter that
    gathers observations within a fixed radius of the test orbit's
    ephemeris at each exposure time within a collection of exposures.

    """

    def __init__(self, radius: float):
        """
        Parameters
        ----------
        radius : float
            The radius in degrees.
        """
        self.radius = radius

    def apply(
        self,
        observations: Union["Observations", ray.ObjectRef],
        test_orbit: TestOrbits,
        propagator_class: Type[Propagator],
    ) -> "Observations":
        """
        Apply the filter to a collection of observations.

        Parameters
        ----------
        observations : `~thor.observations.Observations`
            The observations to filter.
        test_orbit : `~thor.orbit.TestOrbits`
            The test orbit to use for filtering.
        max_processes : int, optional
            Maximum number of processes to use for parallelization. If
            an existing ray cluster is already running, this parameter
            will be ignored if larger than 1 or not None.

        Returns
        -------
        filtered_observations : `~thor.observations.Observations`
            The filtered observations. This will return a copy of the original
            observations.
        """
        time_start = time.perf_counter()
        logger.info("Applying TestOrbitRadiusObservationFilter...")
        logger.info(f"Using radius = {self.radius:.5f} deg")

        # Generate an ephemeris for every observer time/location in the dataset
        ephemeris = test_orbit.generate_ephemeris_from_observations(observations, propagator_class)

        filtered_observations = Observations.empty()
        state_ids = observations.state_id.unique()

        for state_id in state_ids:
            # Select the ephemeris and observations for this state
            ephemeris_state = ephemeris.select("id", state_id)
            observations_state = observations.select("state_id", state_id)
            coordinates_state = observations_state.coordinates

            assert len(ephemeris_state) == 1, "there should be exactly one ephemeris per exposure"

            ephem_ra = ephemeris_state.ephemeris.coordinates.lon[0].as_py()
            ephem_dec = ephemeris_state.ephemeris.coordinates.lat[0].as_py()

            # Filter the observations by radius from the predicted position of the test orbit
            filtered_observations_chunk = observations_state.apply_mask(
                _within_radius(coordinates_state, ephem_ra, ephem_dec, self.radius)
            )

            filtered_observations = qv.concatenate([filtered_observations, filtered_observations_chunk])
            if filtered_observations.fragmented():
                filtered_observations = qv.defragment(filtered_observations)

        filtered_observations = filtered_observations.sort_by(
            [
                "coordinates.time.days",
                "coordinates.time.nanos",
                "coordinates.origin.code",
            ]
        )

        time_end = time.perf_counter()
        logger.info(
            f"Filtered {len(observations)} observations to {len(filtered_observations)} observations."
        )
        logger.info(f"TestOrbitRadiusObservationFilter completed in {time_end - time_start:.3f} seconds.")
        return filtered_observations


class Spherical6DVolumeObservationFilter(ObservationFilter):
    """
    A Spherical6DVolumeObservationFilter gathers observations within a spherical
    6D phase space volume around the test orbit's state at each exposure time.
    """

    def __init__(self, rp_au: float, rv_au_per_day: float):
        """
        Parameters
        ----------
        rp_au : float
            Position radius in AU.
        rv_au_per_day : float
            Velocity radius in AU/day.
        """
        self.rp_au = rp_au
        self.rv_au_per_day = rv_au_per_day

    def apply(
        self,
        observations: Union["Observations", ray.ObjectRef],
        test_orbit: TestOrbits,
        propagator_class: Type[Propagator],
    ) -> "Observations":
        """
        Apply the filter to a collection of observations.

        Parameters
        ----------
        observations : `~thor.observations.Observations`
            The observations to filter.
        test_orbit : `~thor.orbit.TestOrbits`
            The test orbit to use for filtering.
        propagator_class : Type[Propagator]
            Propagator class for ephemeris generation.

        Returns
        -------
        filtered_observations : `~thor.observations.Observations`
            The filtered observations.
        """
        time_start = time.perf_counter()
        logger.info("Applying Spherical6DVolumeObservationFilter...")
        logger.info(f"Using position radius = {self.rp_au:.3f} AU")
        logger.info(f"Using velocity radius = {self.rv_au_per_day:.5f} AU/day")

        # Generate an ephemeris for every observer time/location in the dataset
        ephemeris = test_orbit.generate_ephemeris_from_observations(observations, propagator_class)

        # Choose a reference time to accumulate velocity uncertainty from (earliest state)
        obs_times = observations.coordinates.time
        t_ref_mjd = obs_times.min().mjd()[0].as_py() if len(observations) > 0 else 0.0

        filtered_observations = Observations.empty()
        state_ids = observations.state_id.unique()

        for state_id in state_ids:
            # Select the ephemeris and observations for this state
            ephemeris_state = ephemeris.select("id", state_id)
            observations_state = observations.select("state_id", state_id)
            coordinates_state = observations_state.coordinates

            assert len(ephemeris_state) == 1, "there should be exactly one ephemeris per exposure"

            # Pull the ephemeris center in the observation (equatorial) frame
            ra0_deg = ephemeris_state.ephemeris.coordinates.lon[0].as_py()
            dec0_deg = ephemeris_state.ephemeris.coordinates.lat[0].as_py()
            ra0 = np.deg2rad(ra0_deg)
            dec0 = np.deg2rad(dec0_deg)

            # Compute 3D unit vector to target and local tangent basis at (ra0, dec0)
            b_vec = np.array([
                np.cos(dec0) * np.cos(ra0),
                np.cos(dec0) * np.sin(ra0),
                np.sin(dec0),
            ])
            e_ra = np.array([-np.sin(ra0), np.cos(ra0), 0.0])  # unit along +RA
            e_dec = np.array([
                -np.sin(dec0) * np.cos(ra0),
                -np.sin(dec0) * np.sin(ra0),
                np.cos(dec0),
            ])  # unit along +Dec

            # Get test orbit and observer states in equatorial cartesian to derive range and relative velocity
            test_cart_eq = transform_coordinates(
                ephemeris_state.ephemeris.aberrated_coordinates,
                representation_out=CartesianCoordinates,
                frame_out="equatorial",
                origin_out=OriginCodes.SUN,
            )
            obs_cart_eq = transform_coordinates(
                ephemeris_state.observer.coordinates,
                representation_out=CartesianCoordinates,
                frame_out="equatorial",
                origin_out=OriginCodes.SUN,
            )

            X0 = test_cart_eq.values[0, 0:3]  # [AU]
            V0 = test_cart_eq.values[0, 3:6]  # [AU/day]
            O = obs_cart_eq.values[0, 0:3]    # [AU]
            O_dot = obs_cart_eq.values[0, 3:6]  # [AU/day]

            # Topocentric range and relative velocity
            r_vec = X0 - O
            rho = float(np.linalg.norm(r_vec))
            Vrel = V0 - O_dot

            # Apparent (tangent-plane) velocity direction
            Vrel_tan = Vrel - np.dot(Vrel, b_vec) * b_vec
            Vrel_tan_norm = np.linalg.norm(Vrel_tan)
            if Vrel_tan_norm > 0:
                vt_unit = Vrel_tan / Vrel_tan_norm
                # components in local tangent basis
                vt_x = float(np.dot(vt_unit, e_ra))
                vt_y = float(np.dot(vt_unit, e_dec))
            else:
                vt_x, vt_y = 1.0, 0.0  # arbitrary; footprint will be circular in this case

            # Time offset from reference in days
            t_state_mjd = ephemeris_state.ephemeris.coordinates.time.mjd()[0].as_py()
            dt_days = abs(t_state_mjd - t_ref_mjd)

            # Map 6D sphere to sky-plane elliptical footprint (radians)
            # Position contribution (all directions)
            r_pos = self.rp_au / rho
            # Velocity contribution grows with time along apparent motion direction
            r_vel_along = (self.rv_au_per_day * dt_days) / rho
            # Define ellipse axes: major along vt_unit, minor orthogonal
            a_major = r_pos + r_vel_along
            b_minor = r_pos

            # Build per-detection offsets in tangent plane (radians)
            ra_det = np.deg2rad(coordinates_state.lon.to_numpy(zero_copy_only=False))
            dec_det = np.deg2rad(coordinates_state.lat.to_numpy(zero_copy_only=False))

            # RA difference wrapped to [-pi, pi]
            dra = ra_det - ra0
            dra = (dra + np.pi) % (2 * np.pi) - np.pi
            ddec = dec_det - dec0

            # Small-angle tangent-plane coords
            x = dra * np.cos(dec0)  # along +RA
            y = ddec                # along +Dec

            # Rotate into (along, perp) wrt vt_unit on the tangent plane
            along = vt_x * x + vt_y * y
            perp = -vt_y * x + vt_x * y

            # Ellipse test: (along/a_major)^2 + (perp/b_minor)^2 <= 1
            # Guard against zero axes
            a_major_safe = a_major if a_major > 0 else r_pos
            b_minor_safe = b_minor if b_minor > 0 else r_pos
            vals = (along / a_major_safe) ** 2 + (perp / b_minor_safe) ** 2
            mask = vals <= 1.0

            filtered_observations_chunk = observations_state.apply_mask(mask)
            filtered_observations = qv.concatenate([filtered_observations, filtered_observations_chunk])
            if filtered_observations.fragmented():
                filtered_observations = qv.defragment(filtered_observations)

        filtered_observations = filtered_observations.sort_by(
            [
                "coordinates.time.days",
                "coordinates.time.nanos",
                "coordinates.origin.code",
            ]
        )

        time_end = time.perf_counter()
        logger.info(
            f"Filtered {len(observations)} observations to {len(filtered_observations)} observations."
        )
        logger.info(f"Spherical6DVolumeObservationFilter completed in {time_end - time_start:.3f} seconds.")
        return filtered_observations


def _within_spherical_volume(
    rays: np.array,
    observer_xyz: np.array,
    target_xyz: np.array, 
    target_vel: np.array,
    rp_au: float,
    rv_au_per_day: float,
) -> np.array:
    """
    Return a boolean mask identifying which observation rays intersect
    a spherical volume around the target position and velocity.

    Parameters
    ----------
    rays : np.array
        Unit vectors for each observation ray (N x 3).
    observer_xyz : np.array
        Heliocentric position of observer [AU] (3,).
    target_xyz : np.array
        Heliocentric position of target [AU] (3,).
    target_vel : np.array
        Heliocentric velocity of target [AU/day] (3,).
    rp_au : float
        Position radius in AU.
    rv_au_per_day : float
        Velocity radius in AU/day.

    Returns
    -------
    mask : np.array
        Boolean mask (N,) indicating which rays intersect the volume.
    """
    mask = np.zeros(len(rays), dtype=bool)
    
    for i, ray_u in enumerate(rays):
        # Find closest approach point along ray to target position
        r_star = np.dot(ray_u, target_xyz - observer_xyz)
        if r_star <= 0:
            continue
            
        # Compute minimum distance from ray to target position
        closest_point = observer_xyz + r_star * ray_u
        d_min = np.linalg.norm(closest_point - target_xyz)
        
        # Check if within position radius
        if d_min > rp_au:
            continue
            
        # Check velocity constraint: project target velocity onto ray direction
        # and check if it's within reasonable bounds
        vr_target = np.dot(target_vel, ray_u)  # Radial velocity component
        vt_target = np.linalg.norm(target_vel - vr_target * ray_u)  # Tangential component
        v_total = np.linalg.norm(target_vel)
        
        # For spherical constraint, check if total velocity is within bounds
        # (This is a simplified constraint - could be refined)
        if v_total <= rv_au_per_day:
            mask[i] = True
    
    return mask


def _within_radius(
    coords: SphericalCoordinates,
    ra: float,
    dec: float,
    radius: float,
) -> np.array:
    """
    Return a boolean mask that identifies which of
    the coords are within a given radius of a given ra and dec.

    Parameters
    ----------
    coords : `~adam_core.coordinates.spherical.SphericalCoordinates`
        The coords to filter.
    ra : float
        The right ascension of the center of the radius in degrees.
    dec : float
        The declination of the center of the radius in degrees.
    radius : float
        The radius in degrees.

    Returns
    -------
    mask : `~numpy.ndarray`
        A boolean mask that identifies which of the coords are within
        the radius.
    """
    det_ra = np.deg2rad(coords.lon.to_numpy())
    det_dec = np.deg2rad(coords.lat.to_numpy())

    center_ra = np.deg2rad(ra)
    center_dec = np.deg2rad(dec)

    dist_lon = det_ra - center_ra
    sin_dist_lon = np.sin(dist_lon)
    cos_dist_lon = np.cos(dist_lon)

    sin_center_lat = np.sin(center_dec)
    sin_det_lat = np.sin(det_dec)
    cos_center_lat = np.cos(center_dec)
    cos_det_lat = np.cos(det_dec)

    num1 = cos_det_lat * sin_dist_lon
    num2 = cos_center_lat * sin_det_lat - sin_center_lat * cos_det_lat * cos_dist_lon
    denominator = sin_center_lat * sin_det_lat + cos_center_lat * cos_det_lat * cos_dist_lon

    distances = np.arctan2(np.hypot(num1, num2), denominator)
    return distances <= np.deg2rad(radius)


def filter_observations_worker(
    observations: Observations,
    test_orbit: TestOrbits,
    filters: List[ObservationFilter],
    propagator_class: Type[Propagator],
) -> Observations:
    """
    Apply a list of filters to the observations.

    Parameters
    ----------
    state_id_chunk : list of int
        List of state IDs to filter.
    observations : `~thor.observations.observations.Observations`
        Observations to filter.
    test_orbit : `~thor.orbit.TestOrbits`
        Test orbit to use for filtering.
    filters : list of `~thor.observations.filters.ObservationFilter`
        List of filters to apply to the observations.

    Returns
    -------
    filtered_observations : `~thor.observations.observations.Observations`
        Filtered observations.
    """
    for filter_i in filters:
        observations = filter_i.apply(
            observations,
            test_orbit,
            propagator_class,
        )

    # Defragment the observations
    if len(observations) > 0:
        observations = qv.defragment(observations)

    return observations


filter_observations_worker_remote = ray.remote(filter_observations_worker)
filter_observations_worker_remote.options(num_cpus=1, num_returns=1)


def filter_observations(
    observations: Union[str, Observations],
    test_orbit: TestOrbits,
    config: Config,
    filters: Optional[List[ObservationFilter]] = None,
    chunk_size: int = 1_000_000,
) -> Observations:
    """
    Filter observations by applying a list of filters. The input observations
    can be either be a path to a parquet file or an Observations object already loaded
    into memory.

    Parameters
    ----------
    observations : str or `~thor.observations.observations.Observations`
        Observations to filter.
    test_orbit : `~thor.orbit.TestOrbits`
        Test orbit to use for filtering.
    config : `~thor.config.Config`
        Configuration parameters.
    filters : list of `~thor.observations.filters.ObservationFilter`, optional
        List of filters to apply to the observations. If None, the default
        TestOrbitRadiusObservationFilter will be used.
    chunk_size : int, optional
        Chunk size of state IDs to use when filtering the observations. Each worker
        will process a chunk of state IDs in parallel. If not using ray, then each
        chunk is processed serially.

    Returns
    -------
    filtered_observations : `~thor.observations.observations.Observations`
        Filtered observations.
    """
    time_start = time.perf_counter()
    logger.info("Running observation filters...")

    module_path, class_name = config.propagator_namespace.rsplit(".", 1)
    propagator_module = importlib.import_module(module_path)
    propagator_class = getattr(propagator_module, class_name)

    if len(test_orbit) != 1:
        raise ValueError(f"filter_observations received {len(test_orbit)} orbits but expected 1.")

    if isinstance(observations, str):
        num_obs = pq.read_metadata(observations).num_rows
        logger.info(f"Filtering {num_obs} observations in parquet file.")

    elif isinstance(observations, Observations):
        num_obs = len(observations)
        logger.info(f"Reading {num_obs} observations in memory.")

    else:
        raise ValueError("observations should be a parquet file or an Observations object.")

    if filters is None:
        # By default we always filter by radius from the predicted position of the test orbit
        if config.volume_type == "circular_cell":
            filters = [TestOrbitRadiusObservationFilter(radius=config.cell_radius)]
        elif config.volume_type == "spherical6d":
            filters = [Spherical6DVolumeObservationFilter(rp_au=config.ps_rp_au, rv_au_per_day=config.ps_rv_au_per_day)]
        elif config.volume_type == "conical6d":
            # Fallback to legacy circular cell until conical is implemented
            filters = [TestOrbitRadiusObservationFilter(radius=config.cell_radius)]
        else:
            raise ValueError(f"Unknown volume type: {config.volume_type}")

    if config.max_processes is None:
        max_processes = mp.cpu_count()
    else:
        max_processes = config.max_processes

    filtered_observations = Observations.empty()
    logger.info(f"{config.json()}")
    use_ray = initialize_use_ray(num_cpus=max_processes)
    if use_ray:

        futures: List[ray.ObjectRef] = []
        for observations_chunk in observations_iterator(observations, chunk_size=chunk_size):
            futures.append(
                filter_observations_worker_remote.remote(
                    observations_chunk, test_orbit, filters, propagator_class
                )
            )
            if len(futures) > max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                filtered_observations = qv.concatenate([filtered_observations, ray.get(finished[0])])
                if filtered_observations.fragmented():
                    filtered_observations = qv.defragment(filtered_observations)

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            filtered_observations = qv.concatenate([filtered_observations, ray.get(finished[0])])
            if filtered_observations.fragmented():
                filtered_observations = qv.defragment(filtered_observations)

        if isinstance(observations, ray.ObjectRef):
            ray.internal.free([observations])
            logger.info("Removed observations from the object store.")

    else:
        for observations_chunk in observations_iterator(observations, chunk_size=chunk_size):
            filtered_observations_chunk = filter_observations_worker(
                observations_chunk,
                test_orbit,
                filters,
                propagator_class,
            )
            filtered_observations = qv.concatenate([filtered_observations, filtered_observations_chunk])
            if filtered_observations.fragmented():
                filtered_observations = qv.defragment(filtered_observations)

    filtered_observations = filtered_observations.sort_by(
        [
            "coordinates.time.days",
            "coordinates.time.nanos",
            "coordinates.origin.code",
        ]
    )

    time_end = time.perf_counter()
    logger.info(f"Filtered {num_obs} observations to {len(filtered_observations)} observations.")
    logger.info(f"Observations filters completed in {time_end - time_start:.3f} seconds.")
    return filtered_observations