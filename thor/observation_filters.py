import abc

import numpy as np
import pandas as pd
import pyarrow as pa
import quivr as qv
from adam_core.observations import Exposures, PointSourceDetections
from adam_core.observers import Observers
from astropy.time import Time

from .orbit import TestOrbit


class Observations:
    """Observations represents a collection of exposures and the
    detections they contain.

    The detections may be a filtered subset of the detections in the
    exposures.
    """

    detections: PointSourceDetections
    exposures: Exposures
    linkage: qv.Linkage[PointSourceDetections, Exposures]

    def __init__(
        self,
        detections: PointSourceDetections,
        exposures: Exposures,
    ):
        self.detections = detections
        self.exposures = exposures
        self.linkage = qv.Linkage(
            detections,
            exposures,
            left_keys=detections.exposure_id,
            right_keys=exposures.id,
        )

    def observers(self, exposures_only: bool = False) -> Observers:
        """
        Get the Observers associated with these observations. By default, this will return
        observers for each unique detection time and observatory code. However, if exposures_only
        is set to True, then only the observers for each exposure midpoint will be returned.

        Parameters
        ----------
        exposures_only : bool, optional
            If True, only return the observers for each exposure midpoint, by default False.

        Returns
        -------
        observers : `~adam_core.observers.observers.Observers`
            Observer for each unique detection time and observatory code, or each exposure midpoint.
        """
        if exposures_only:
            return self.exposures.observers()

        else:
            # This could be changed / improved with additional functionality in quivr:
            # Features needed:
            # 1. A way to join two tables (or we simply add observatory code to each detection)
            # 2. A method to drop row duplicates considering a user-defined set of column names
            # 3. By extension of 2, having a .unique() attached to a table would be nice

            # Get the time scale
            scale = self.detections.time.scale

            # Create a table of exposure IDs and times for each detection
            detection_times = pa.table(
                [self.detections.exposure_id, self.detections.time.mjd()],
                names=["exposure_id", "time_mjd"],
            )

            # Create a table of exposure IDs and observatory codes for each exposure
            exposure_obscodes = pa.table(
                [self.exposures.id, self.exposures.observatory_code],
                names=["exposure_id", "observatory_code"],
            )

            # Merge the two tables on exposure ID (we need to get each individual detection's observatory code)
            detection_obscodes_times = pd.merge(
                exposure_obscodes.to_pandas(),
                detection_times.to_pandas(),
                on="exposure_id",
            )

            # Now, lets only keep the unique occurrences of observatory code and time
            unique_obscodes_times = detection_obscodes_times.drop_duplicates(
                subset=["observatory_code", "time_mjd"]
            )

            # Get unique codes and times
            unique_obscodes = unique_obscodes_times["observatory_code"].unique()
            observers_list = []
            for code in unique_obscodes:
                observers_list.append(
                    Observers.from_code(
                        code,
                        Time(
                            unique_obscodes_times[
                                unique_obscodes_times["observatory_code"] == code
                            ]["time_mjd"].values,
                            format="mjd",
                            scale=scale,
                        ),
                    )
                )

            return qv.concatenate(observers_list)


class ObservationFilter(abc.ABC):
    """An ObservationFilter is reduces a collection of observations to
    a subset of those observations.

    """

    @abc.abstractmethod
    def apply(self, observations: Observations) -> Observations:
        ...


class TestOrbitRadiusObservationFilter(ObservationFilter):
    """A TestOrbitRadiusObservationFilter is an ObservationFilter that
    gathers observations within a fixed radius of the test orbit's
    ephemeris at each exposure time within a collection of exposures.

    """

    def __init__(self, radius: float, test_orbit: TestOrbit):
        """
        radius: The radius of the cell in degrees
        """
        self.radius = radius
        self.test_orbit = test_orbit

    def apply(self, observations: Observations) -> Observations:
        # Generate an ephemeris for every observer time/location in the dataset
        observers = observations.observers(exposures_only=True)
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
