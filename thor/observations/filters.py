import abc
from typing import TYPE_CHECKING

import numpy as np
import quivr as qv
from adam_core.observations import PointSourceDetections

from ..orbit import TestOrbit

if TYPE_CHECKING:
    from .observations import Observations


class ObservationFilter(abc.ABC):
    """An ObservationFilter is reduces a collection of observations to
    a subset of those observations.

    """

    @abc.abstractmethod
    def apply(
        self, observations: "Observations", test_orbit: TestOrbit
    ) -> "Observations":
        """
        Apply the filter to a collection of observations.

        Parameters
        ----------
        observations : `~thor.observations.Observations`
            The observations to filter.
        test_orbit : `~thor.orbit.TestOrbit`
            The test orbit to use for filtering.

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
        self, observations: "Observations", test_orbit: TestOrbit
    ) -> "Observations":
        """
        Apply the filter to a collection of observations.

        Parameters
        ----------
        observations : `~thor.observations.Observations`
            The observations to filter.
        test_orbit : `~thor.orbit.TestOrbit`
            The test orbit to use for filtering.

        Returns
        -------
        filtered_observations : `~thor.observations.Observations`
            The filtered observations.
        """
        # Generate an ephemeris for every observer time/location in the dataset
        test_orbit_ephemeris = test_orbit.generate_ephemeris_from_observations(
            observations
        )

        # Link the ephemeris to the observations
        link = qv.Linkage(
            test_orbit_ephemeris,
            observations,
            left_keys=test_orbit_ephemeris.id,
            right_keys=observations.state_id,
        )

        # Loop over states and build a mask of detections within the radius
        state_ids = observations.state_id.to_numpy(zero_copy_only=False)
        mask = np.zeros(len(observations), dtype=bool)
        for state_id in np.unique(state_ids):
            # Compute the indices for observations belonging to this state
            idx_state = np.where(state_ids == state_id)[0]

            # Select the ephemeris and observations for this state
            ephemeris_i = link.select_left(state_id)
            observations_i = link.select_right(state_id)
            detections_i = observations_i.detections

            assert (
                len(ephemeris_i) == 1
            ), "there should be exactly one ephemeris per exposure"

            ephem_ra = ephemeris_i.ephemeris.coordinates.lon[0].as_py()
            ephem_dec = ephemeris_i.ephemeris.coordinates.lat[0].as_py()

            # Compute indices for the detections within the radius
            idx_within = idx_state[
                _within_radius(detections_i, ephem_ra, ephem_dec, self.radius)
            ]

            # Update the mask
            mask[idx_within] = True

        return observations.apply_mask(mask)


def _within_radius(
    detections: PointSourceDetections,
    ra: float,
    dec: float,
    radius: float,
) -> np.array:
    """
    Return a boolean mask that identifies which of
    the detections are within a given radius of a given ra and dec.

    Parameters
    ----------
    detections : `~adam_core.observations.detections.PointSourceDetections`
        The detections to filter.
    ra : float
        The right ascension of the center of the radius in degrees.
    dec : float
        The declination of the center of the radius in degrees.
    radius : float
        The radius in degrees.

    Returns
    -------
    mask : `~numpy.ndarray`
        A boolean mask that identifies which of the detections are within
        the radius.
    """
    det_ra = np.deg2rad(detections.ra.to_numpy())
    det_dec = np.deg2rad(detections.dec.to_numpy())

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
