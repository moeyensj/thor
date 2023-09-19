import abc
from typing import TYPE_CHECKING

import numpy as np
import quivr as qv
from adam_core.observations import Exposures, PointSourceDetections

from ..orbit import TestOrbit

if TYPE_CHECKING:
    from .observations import Observations


class ObservationFilter(abc.ABC):
    """An ObservationFilter is reduces a collection of observations to
    a subset of those observations.

    """

    @abc.abstractmethod
    def apply(self, observations: "Observations") -> "Observations":
        ...


class TestOrbitRadiusObservationFilter(ObservationFilter):
    """A TestOrbitRadiusObservationFilter is an ObservationFilter that
    gathers observations within a fixed radius of the test orbit's
    ephemeris at each exposure time within a collection of exposures.

    """

    def __init__(self, radius: float, test_orbit: TestOrbit):
        """
        Parameters
        ----------
        radius: float
            The radius of the cell in degrees.
        test_orbit: `~thor.orbit.TestOrbit`
            The test orbit to use for filtering.
        """
        self.radius = radius
        self.test_orbit = test_orbit

    def apply(self, observations: "Observations") -> "Observations":
        from .observations import Observations

        # Generate an ephemeris for every observer time/location in the dataset
        observers = observations.get_observers()
        ephems_linkage = self.test_orbit.generate_ephemeris(
            observers=observers,
        )

        matching_detections = PointSourceDetections.empty()
        matching_exposures = Exposures.empty()

        # Build a mapping of exposure_id to ephemeris ra and dec
        for exposure in observations.exposures:
            key = ephems_linkage.key(
                code=exposure.observatory_code[0].as_py(),
                mjd=exposure.midpoint().mjd()[0].as_py(),
            )
            ephem = ephems_linkage.select_left(key)
            assert len(ephem) == 1, "there should be exactly one ephemeris per exposure"

            ephem_ra = ephem.coordinates.lon[0].as_py()
            ephem_dec = ephem.coordinates.lat[0].as_py()

            exp_dets = observations.linkage.select_left(exposure.id[0])

            nearby_dets = _within_radius(exp_dets, ephem_ra, ephem_dec, self.radius)
            if len(nearby_dets) > 0:
                matching_exposures = qv.concatenate([matching_exposures, exposure])
                matching_detections = qv.concatenate([matching_detections, nearby_dets])

        return Observations(matching_detections, matching_exposures)


def _within_radius(
    detections: PointSourceDetections,
    ra: float,
    dec: float,
    radius: float,
) -> PointSourceDetections:
    """
    Return the detections within a given radius of a given ra and dec.

    ra, dec, and radius should be in degrees.
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

    mask = distances <= np.deg2rad(radius)
    return detections.apply_mask(mask)
