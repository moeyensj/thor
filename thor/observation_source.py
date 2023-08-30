import abc
from typing import TypeAlias

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
        raise NotImplementedError


class StaticObservationsSource(ObservationSource):
    """A StaticObservationsSource is an ObservationSource that
    returns a fixed collection of observations for any test orbit.
    """

    def __init__(self, observations: Observations):
        self.observations = observations

    def gather_observations(self, test_orbit: orbit.TestOrbit) -> Observations:
        return self.observations
