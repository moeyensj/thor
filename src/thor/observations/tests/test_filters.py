import pyarrow.compute as pc
from adam_assist import ASSISTPropagator

from ..filters import TestOrbitRadiusObservationFilter, filter_observations


def test_observation_fixtures(fixed_test_orbit, fixed_observations):
    assert len(fixed_test_orbit) == 1
    assert len(pc.unique(fixed_observations.exposure_id)) == 5
    assert len(fixed_observations.coordinates) == 100 * 100 * 5


def test_orbit_radius_observation_filter(fixed_test_orbit, fixed_observations):
    fos = TestOrbitRadiusObservationFilter(
        radius=0.5,
    )
    have, ephemeris = fos.apply(fixed_observations, fixed_test_orbit, ASSISTPropagator)
    assert len(pc.unique(have.exposure_id)) == 5
    assert pc.all(
        pc.equal(
            pc.unique(have.exposure_id),
            pc.unique(fixed_observations.exposure_id),
        )
    )
    # Should be about pi/4 fraction of the detections (0.785
    assert len(have.coordinates) < 0.80 * len(fixed_observations.coordinates)
    assert len(have.coordinates) > 0.76 * len(fixed_observations.coordinates)
    assert len(ephemeris) > 0


def test_filter_observations(fixed_observations, fixed_test_orbit):
    # Test filter_observations with a radius filter
    filters = [TestOrbitRadiusObservationFilter(radius=0.5)]

    have, ephemeris = filter_observations(
        fixed_observations, fixed_test_orbit, filters, propagator_class=ASSISTPropagator, max_processes=1
    )
    assert len(pc.unique(have.exposure_id)) == 5
    assert len(have.coordinates) < 0.80 * len(fixed_observations.coordinates)
    assert len(have.coordinates) > 0.76 * len(fixed_observations.coordinates)
    assert len(ephemeris) > 0

    # Test with no filters: observations should be returned unchanged
    have_no_filter, ephemeris_no_filter = filter_observations(
        fixed_observations, fixed_test_orbit, [], propagator_class=ASSISTPropagator, max_processes=1
    )
    assert len(have_no_filter.coordinates) == len(fixed_observations.coordinates)
    assert len(ephemeris_no_filter) == 0
