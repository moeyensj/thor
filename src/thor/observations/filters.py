import abc
import glob
import logging
import multiprocessing as mp
import os
import time
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import pyarrow.parquet as pq
import quivr as qv
import ray

from adam_core.coordinates import SphericalCoordinates
from adam_core.coordinates.residuals import Residuals
from adam_core.propagator import Propagator
from adam_core.ray_cluster import initialize_use_ray

from ..orbit import TestOrbitEphemeris, TestOrbits
from .observations import Observations, observations_iterator

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
    ) -> Tuple["Observations", TestOrbitEphemeris]:
        """
        Apply the filter to a collection of observations.

        Parameters
        ----------
        observations : `~thor.observations.Observations`
            The observations to filter.
        test_orbit : `~thor.orbit.TestOrbits`
            The test orbit to use for filtering.
        propagator_class : `~adam_core.propagator.Propagator`
            Propagator class to use for generating ephemeris.

        Returns
        -------
        filtered_observations : `~thor.observations.Observations`
            The filtered observations.
        test_orbit_ephemeris : `~thor.orbit.TestOrbitEphemeris`
            The test orbit ephemeris generated during filtering.
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
    ) -> Tuple["Observations", TestOrbitEphemeris]:
        """
        Apply the filter to a collection of observations.

        Parameters
        ----------
        observations : `~thor.observations.Observations`
            The observations to filter.
        test_orbit : `~thor.orbit.TestOrbits`
            The test orbit to use for filtering.
        propagator_class : `~adam_core.propagator.Propagator`
            Propagator class to use for generating ephemeris.

        Returns
        -------
        filtered_observations : `~thor.observations.Observations`
            The filtered observations. This will return a copy of the original
            observations.
        test_orbit_ephemeris : `~thor.orbit.TestOrbitEphemeris`
            The test orbit ephemeris generated during filtering (without covariance).
        """
        time_start = time.perf_counter()
        logger.info("Applying TestOrbitRadiusObservationFilter...")
        logger.info(f"Using radius = {self.radius:.5f} deg")

        # Generate an ephemeris for every observer time/location in the dataset
        ephemeris = test_orbit.generate_ephemeris_from_observations(
            observations, propagator_class, covariance=False
        )

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
        return filtered_observations, ephemeris


class TestOrbitMahalanobisObservationFilter(ObservationFilter):
    """
    Filter observations by Mahalanobis distance in spherical RA/Dec, combining
    observation covariance and (optionally) propagated predicted covariance.
    """

    def __init__(
        self,
        mahalanobis_distance: float = 5.0,
    ):
        """
        Parameters
        ----------
        mahalanobis_distance: float
            Mahalanobis distance threshold for filtering observations.
            For a 5-sigma ellipse in 2D (RA/Dec), set this to 5.0.
            For a 3-sigma ellipse, set this to 3.0.
            [Default = 5.0]
        """
        self.mahalanobis_distance = mahalanobis_distance

    def apply(
        self,
        observations: Union["Observations", ray.ObjectRef],
        test_orbit: TestOrbits,
        propagator_class: Type[Propagator],
    ) -> Tuple["Observations", TestOrbitEphemeris]:
        time_start = time.perf_counter()
        logger.info("Applying TestOrbitMahalanobisObservationFilter...")
        logger.info(f"Using mahalanobis_distance = {self.mahalanobis_distance:.1f}-sigma")

        # Only compute predicted covariance if we’re going to use it
        ephemeris = test_orbit.generate_ephemeris_from_observations(
            observations,
            propagator_class,
            covariance=True,
        )

        filtered_observations = Observations.empty()
        state_ids = observations.state_id.unique()
        chi2_threshold = self.mahalanobis_distance**2

        for state_id in state_ids:
            ephemeris_state = ephemeris.select("id", state_id)
            observations_state = observations.select("state_id", state_id)
            assert len(ephemeris_state) == 1, "there should be exactly one ephemeris per exposure/state"

            pred_sph = ephemeris_state.ephemeris.coordinates
            obs_sph = observations_state.coordinates

            # RA wrap and cos(lat) handling occur inside Residuals.calculate
            residuals = Residuals.calculate(
                obs_sph,
                pred_sph,
                use_predicted_covariance=True,
            )
            chi2 = residuals.chi2.to_numpy(zero_copy_only=False)
            mask = chi2 <= chi2_threshold

            filtered_chunk = observations_state.apply_mask(mask)
            filtered_observations = qv.concatenate([filtered_observations, filtered_chunk])
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
        logger.info(
            f"TestOrbitMahalanobisSphericalObservationFilter completed in {time_end - time_start:.3f} seconds."
        )
        return filtered_observations, ephemeris


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
) -> Tuple[Observations, TestOrbitEphemeris]:
    """
    Apply a list of filters to the observations.

    Parameters
    ----------
    observations : `~thor.observations.observations.Observations`
        Observations to filter.
    test_orbit : `~thor.orbit.TestOrbits`
        Test orbit to use for filtering.
    filters : list of `~thor.observations.filters.ObservationFilter`
        List of filters to apply to the observations.
    propagator_class : `~adam_core.propagator.Propagator`
        Propagator class to use for generating ephemeris.

    Returns
    -------
    filtered_observations : `~thor.observations.observations.Observations`
        Filtered observations.
    test_orbit_ephemeris : `~thor.orbit.TestOrbitEphemeris`
        Test orbit ephemeris generated during filtering.
    """
    test_orbit_ephemeris = TestOrbitEphemeris.empty()
    for filter_i in filters:
        observations, ephemeris = filter_i.apply(
            observations,
            test_orbit,
            propagator_class,
        )
        # Keep the most recent ephemeris (later filters may have covariance)
        test_orbit_ephemeris = ephemeris

    # Defragment the observations
    if len(observations) > 0:
        observations = qv.defragment(observations)

    return observations, test_orbit_ephemeris


filter_observations_worker_remote = ray.remote(filter_observations_worker).options(num_cpus=1)


def filter_observations(
    observations: Union[str, Observations],
    test_orbit: TestOrbits,
    filters: List[ObservationFilter],
    propagator_class: Type[Propagator],
    max_processes: Optional[int] = None,
    chunk_size: int = 1_000_000,
) -> Tuple[Observations, TestOrbitEphemeris]:
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
    filters : list of `~thor.observations.filters.ObservationFilter`,
        List of filters to apply to the observations.
    propagator_class : `~adam_core.propagator.Propagator`
        Propagator class to use for generating the ephemeris.
    max_processes : int, optional
        Maximum number of processes to use for parallelization. If
        an existing ray cluster is already running, this parameter
        will be ignored if larger than 1 or not None.
    chunk_size : int, optional
        Chunk size to use when filtering the observations. When using ray,
        chunks are distributed to multiple workers for parallel processing.
        When not using ray, chunks are processed sequentially.

    Returns
    -------
    filtered_observations : `~thor.observations.observations.Observations`
        Filtered observations.
    test_orbit_ephemeris : `~thor.orbit.TestOrbitEphemeris`
        Test orbit ephemeris generated during filtering. Will be empty if no
        filters were applied.
    """
    time_start = time.perf_counter()
    logger.info("Running observation filters...")

    if len(test_orbit) != 1:
        raise ValueError(f"filter_observations received {len(test_orbit)} orbits but expected 1.")

    if isinstance(observations, str):
        if os.path.isdir(observations):
            num_files = len(glob.glob(os.path.join(observations, "*.parquet")))
            num_obs = sum(
                pq.read_metadata(f).num_rows for f in glob.glob(os.path.join(observations, "*.parquet"))
            )
            logger.info(f"Filtering {num_obs} observations in {num_files} parquet files in directory.")
        else:
            num_obs = pq.read_metadata(observations).num_rows
            logger.info(f"Filtering {num_obs} observations in 1 parquet file.")

    elif isinstance(observations, Observations):
        num_obs = len(observations)
        logger.info(f"Reading {num_obs} observations in memory.")

    else:
        raise ValueError("observations should be a parquet file or an Observations object.")

    if len(filters) == 0:
        filtered_observations = observations
        logger.info("No filters provided, returning observations unchanged.")
        time_end = time.perf_counter()
        logger.info(f"Filtered {num_obs} observations to {len(filtered_observations)} observations.")
        logger.info(f"Observations filters completed in {time_end - time_start:.3f} seconds.")
        # Return empty ephemeris when no filters applied
        return filtered_observations, TestOrbitEphemeris.empty()

    if max_processes is None:
        max_processes = mp.cpu_count()

    filtered_observations = Observations.empty()
    test_orbit_ephemeris = TestOrbitEphemeris.empty()
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
                obs_chunk, ephemeris_chunk = ray.get(finished[0])
                filtered_observations = qv.concatenate([filtered_observations, obs_chunk])
                test_orbit_ephemeris = qv.concatenate([test_orbit_ephemeris, ephemeris_chunk])
                if filtered_observations.fragmented():
                    filtered_observations = qv.defragment(filtered_observations)
                if test_orbit_ephemeris.fragmented():
                    test_orbit_ephemeris = qv.defragment(test_orbit_ephemeris)

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            obs_chunk, ephemeris_chunk = ray.get(finished[0])
            filtered_observations = qv.concatenate([filtered_observations, obs_chunk])
            test_orbit_ephemeris = qv.concatenate([test_orbit_ephemeris, ephemeris_chunk])
            if filtered_observations.fragmented():
                filtered_observations = qv.defragment(filtered_observations)
            if test_orbit_ephemeris.fragmented():
                test_orbit_ephemeris = qv.defragment(test_orbit_ephemeris)

        if isinstance(observations, ray.ObjectRef):
            ray.internal.free([observations])
            logger.info("Removed observations from the object store.")

    else:
        for observations_chunk in observations_iterator(observations, chunk_size=chunk_size):
            obs_chunk, ephemeris_chunk = filter_observations_worker(
                observations_chunk,
                test_orbit,
                filters,
                propagator_class,
            )
            filtered_observations = qv.concatenate([filtered_observations, obs_chunk])
            test_orbit_ephemeris = qv.concatenate([test_orbit_ephemeris, ephemeris_chunk])
            if filtered_observations.fragmented():
                filtered_observations = qv.defragment(filtered_observations)
            if test_orbit_ephemeris.fragmented():
                test_orbit_ephemeris = qv.defragment(test_orbit_ephemeris)

    filtered_observations = filtered_observations.sort_by(
        [
            "coordinates.time.days",
            "coordinates.time.nanos",
            "coordinates.origin.code",
        ]
    )

    # Drop duplicate ephemeris entries (same time/observer combination)
    if len(test_orbit_ephemeris) > 0:
        test_orbit_ephemeris = test_orbit_ephemeris.drop_duplicates(subset=["id"])
        test_orbit_ephemeris = test_orbit_ephemeris.sort_by(
            [
                "ephemeris.coordinates.time.days",
                "ephemeris.coordinates.time.nanos",
                "ephemeris.coordinates.origin.code",
            ]
        )

    time_end = time.perf_counter()
    logger.info(f"Filtered {num_obs} observations to {len(filtered_observations)} observations.")
    logger.info(f"Observations filters completed in {time_end - time_start:.3f} seconds.")

    return filtered_observations, test_orbit_ephemeris
