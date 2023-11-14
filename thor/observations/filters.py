import abc
import logging
import time
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import quivr as qv
import ray
from adam_core.coordinates import SphericalCoordinates

from thor.config import Config
from thor.observations.observations import Observations

from ..orbit import TestOrbit, TestOrbitEphemeris

if TYPE_CHECKING:
    from .observations import Observations


logger = logging.getLogger(__name__)


def TestOrbitRadiusObservationFilter_worker(
    observations: "Observations",
    ephemeris: TestOrbitEphemeris,
    state_id: int,
    radius: float,
) -> "Observations":
    """
    Apply the filter to a collection of observations for a particular state.

    Parameters
    ----------
    observations : `~thor.observations.Observations`
        The observations to filter.
    ephemeris : `~thor.orbit.TestOrbitEphemeris`
        The ephemeris to use for filtering.
    state_id : int
        The state ID.
    radius : float
        The radius in degrees.

    Returns
    -------
    filtered_observations : `~thor.observations.Observations`
        The filtered observations.
    """
    # Select the ephemeris and observations for this state
    ephemeris_state = ephemeris.select("id", state_id)
    observations_state = observations.select("state_id", state_id)
    coordinates_state = observations_state.coordinates

    assert (
        len(ephemeris_state) == 1
    ), "there should be exactly one ephemeris per exposure"

    ephem_ra = ephemeris_state.ephemeris.coordinates.lon[0].as_py()
    ephem_dec = ephemeris_state.ephemeris.coordinates.lat[0].as_py()

    # Return the observations within the radius for this particular state
    return observations_state.apply_mask(
        _within_radius(coordinates_state, ephem_ra, ephem_dec, radius)
    )


TestOrbitRadiusObservationFilter_remote = ray.remote(
    TestOrbitRadiusObservationFilter_worker
)


class ObservationFilter(abc.ABC):
    """An ObservationFilter is reduces a collection of observations to
    a subset of those observations.

    """

    @abc.abstractmethod
    def apply(
        self,
        observations: "Observations",
        test_orbit: TestOrbit,
        max_processes: Optional[int] = 1,
    ) -> "Observations":
        """
        Apply the filter to a collection of observations.

        Parameters
        ----------
        observations : `~thor.observations.Observations`
            The observations to filter.
        test_orbit : `~thor.orbit.TestOrbit`
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
        test_orbit: TestOrbit,
        max_processes: Optional[int] = 1,
    ) -> "Observations":
        """
        Apply the filter to a collection of observations.

        Parameters
        ----------
        observations : `~thor.observations.Observations`
            The observations to filter.
        test_orbit : `~thor.orbit.TestOrbit`
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
        ephemeris = test_orbit.generate_ephemeris_from_observations(observations)

        filtered_observations_list = []
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

            state_ids = observations.state_id.unique().sort()
            futures = []
            for state_id in state_ids:
                futures.append(
                    TestOrbitRadiusObservationFilter_remote.remote(
                        observations_ref,
                        ephemeris_ref,
                        state_id,
                        self.radius,
                    )
                )

            while futures:
                finished, futures = ray.wait(futures, num_returns=1)
                filtered_observations_list.append(ray.get(finished[0]))

        else:
            state_ids = observations.state_id.unique().sort()
            for state_id in state_ids:
                filtered_observations = TestOrbitRadiusObservationFilter_worker(
                    observations,
                    ephemeris,
                    state_id,
                    self.radius,
                )
                filtered_observations_list.append(filtered_observations)

        observations_filtered = qv.concatenate(filtered_observations_list)
        observations_filtered = observations_filtered.sort_by(
            [
                "coordinates.time.days",
                "coordinates.time.nanos",
                "coordinates.origin.code",
            ]
        )

        time_end = time.perf_counter()
        logger.info(
            f"Filtered {len(observations)} observations to {len(observations_filtered)} observations."
        )
        logger.info(
            f"TestOrbitRadiusObservationFilter completed in {time_end - time_start:.3f} seconds."
        )
        return observations_filtered


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
    denominator = (
        sin_center_lat * sin_det_lat + cos_center_lat * cos_det_lat * cos_dist_lon
    )

    distances = np.arctan2(np.hypot(num1, num2), denominator)
    return distances <= np.deg2rad(radius)


def filter_observations(
    observations: Observations,
    test_orbit: TestOrbit,
    config: Config,
    filters: Optional[List[ObservationFilter]] = None,
) -> Observations:
    """
    Apply a list of filters to the observations.

    Parameters
    ----------
    observations : `~thor.observations.observations.Observations`
        Observations to filter.
    filters : list of `~thor.observations.filters.ObservationFilter`
        List of filters to apply to the observations.

    Returns
    -------
    filtered_observations : `~thor.observations.observations.Observations`
        Filtered observations.
    """
    if filters is None:
        # By default we always filter by radius from the predicted position of the test orbit
        filters = [TestOrbitRadiusObservationFilter(radius=config.cell_radius)]

    filtered_observations = observations
    for filter_i in filters:
        filtered_observations = filter_i.apply(
            filtered_observations, test_orbit, config.max_processes
        )

    # Defragment the observations
    if len(filtered_observations) > 0:
        filtered_observations = qv.defragment(filtered_observations)
        filtered_observations = filtered_observations.sort_by(
            ["coordinates.time.days", "coordinates.time.nanos", "coordinates.origin.code"]
        )

    return filtered_observations
