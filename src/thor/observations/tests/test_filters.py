from unittest import mock

import pyarrow.compute as pc

from ...config import Config
from ..filters import TestOrbitRadiusObservationFilter, filter_observations


def test_observation_fixtures(fixed_test_orbit, fixed_observations):
    assert len(fixed_test_orbit) == 1
    assert len(pc.unique(fixed_observations.exposure_id)) == 5
    assert len(fixed_observations.coordinates) == 100 * 100 * 5


def test_orbit_radius_observation_filter(fixed_test_orbit, fixed_observations):
    fos = TestOrbitRadiusObservationFilter(
        radius=0.5,
    )
    have = fos.apply(fixed_observations, fixed_test_orbit)
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


def test_filter_observations(fixed_observations, fixed_test_orbit):
    # Test that if not filters are passed, we use
    # TestOrbitRadiusObservationFilter by defualt
    config = Config(cell_radius=0.5, max_processes=1)

    have = filter_observations(fixed_observations, fixed_test_orbit, config)
    assert len(pc.unique(have.exposure_id)) == 5
    assert len(have.coordinates) < 0.80 * len(fixed_observations.coordinates)
    assert len(have.coordinates) > 0.76 * len(fixed_observations.coordinates)

    # Make sure if we pass a custom list of filters, they are used
    # instead
    noop_filter = mock.Mock()
    noop_filter.apply.return_value = fixed_observations

    filters = [noop_filter]
    have = filter_observations(fixed_observations, fixed_test_orbit, config, filters=filters)
    assert len(have.coordinates) == len(fixed_observations.coordinates)

    # Ensure NOOPFilter.apply was run
    filters[0].apply.assert_called_once()
