import abc
from typing import TypeAlias

import numpy as np
import quivr as qv
from adam_core.observations import detections, exposures

from . import orbit


class Observations:
    """Observations represents a collection of exposures and the
    detections they contain.

    The detections may be a filtered subset of the detections in the
    exposures.
    """

    detections: detections.PointSourceDetections
    exposures: exposures.Exposures
    linkage: qv.Linkage[detections.PointSourceDetections, exposures.Exposures]

    def __init__(
        self,
        detections: detections.PointSourceDetections,
        exposures: exposures.Exposures,
    ):
        self.detections = detections
        self.exposures = exposures
        self.linkage = qv.Linkage(
            detections,
            exposures,
            left_keys=detections.exposure_id,
            right_keys=exposures.id,
        )


class ObservationSource(abc.ABC):
    """An ObservationSource is a source of observations for a given test orbit.

    It has one method, gather_observations, which takes a test orbit
    and returns a collection of Observations.

    """

    @abc.abstractmethod
    def gather_observations(self, test_orbit: orbit.TestOrbit) -> Observations:
        pass


class FixedRadiusObservationSource(ObservationSource):
    """A FixedRadiusObservationSource is an ObservationSource that
    gathers observations within a fixed radius of the test orbit's
    ephemeris at each exposure time within a collection of exposures.

    """

    def __init__(self, radius: float, all_observations: Observations):
        """
        radius: The radius of the cell in degrees
        """
        self.radius = radius
        self.all_observations = all_observations

    def gather_observations(self, test_orbit: orbit.TestOrbit) -> Observations:
        # Generate an ephemeris for every observer time/location in the dataset
        observers = self.all_observations.exposures.observers()
        ephems_linkage = test_orbit.generate_ephemeris(
            observers=observers,
        )

        matching_detections = detections.PointSourceDetections.empty()
        matching_exposures = exposures.Exposures.empty()

        # Build a mapping of exposure_id to ephemeris ra and dec
        for exposure in self.all_observations.exposures:
            key = ephems_linkage.key(
                code=exposure.observatory_code[0].as_py(),
                mjd=exposure.midpoint().mjd()[0].as_py(),
            )
            ephem = ephems_linkage.select_left(key)
            assert len(ephem) == 1, "there should be exactly one ephemeris per exposure"

            ephem_ra = ephem.coordinates.lon[0].as_py()
            ephem_dec = ephem.coordinates.lat[0].as_py()

            exp_dets = self.all_observations.linkage.select_left(exposure.id[0])

            nearby_dets = _within_radius(exp_dets, ephem_ra, ephem_dec, self.radius)
            if len(nearby_dets) > 0:
                matching_exposures = qv.concatenate([matching_exposures, exposure])
                matching_detections = qv.concatenate([matching_detections, nearby_dets])

        return Observations(matching_detections, matching_exposures)


def _within_radius(
    detections: detections.PointSourceDetections,
    ra: float,
    dec: float,
    radius: float,
) -> detections.PointSourceDetections:
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


class StaticObservationSource(ObservationSource):
    """A StaticObservationSource is an ObservationSource that
    returns a fixed collection of observations for any test orbit.
    """

    def __init__(self, observations: Observations):
        self.observations = observations

    def gather_observations(self, test_orbit: orbit.TestOrbit) -> Observations:
        return self.observations
