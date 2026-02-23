import pytest
from adam_core.coordinates import Origin, SphericalCoordinates
from adam_core.time import Timestamp

from ...observations.observations import Observations
from ...observations.photometry import Photometry
from ..attribution import Attributions


@pytest.fixture
def observations():
    # Observations 01 and 02 are at the same time
    # Observations 03, 04, and 05 are at the same time
    observations = Observations.from_kwargs(
        id=["01", "02", "03", "04", "05"],
        exposure_id=["e01", "e01", "e02", "e02", "e02"],
        coordinates=SphericalCoordinates.from_kwargs(
            time=Timestamp.from_mjd([59001.1, 59001.1, 59002.1, 59002.1, 59002.1], scale="utc"),
            lon=[1, 2, 3, 4, 5],
            lat=[5, 6, 7, 8, 9],
            origin=Origin.from_kwargs(code=["500", "500", "500", "500", "500"]),
        ),
        photometry=Photometry.from_kwargs(
            filter=["g", "g", "g", "g", "g"],
            mag=[10, 11, 12, 13, 14],
        ),
        state_id=["a", "a", "b", "b", "b"],
    )
    return observations


@pytest.fixture
def attributions():
    attributions = Attributions.from_kwargs(
        orbit_id=["o01", "o01", "o02", "o03", "o04", "o04", "o05"],
        obs_id=["01", "02", "03", "03", "04", "05", "01"],
        distance=[
            1 / 3600,
            0.5 / 3600,
            2 / 3600,
            1 / 3600,
            2 / 3600,
            1 / 3600,
            0.5 / 3600,
        ],
    )
    return attributions


def test_Attributions_drop_coincident_attributions(observations, attributions):
    # Test that we can drop coincident attributions (attributions of multiple observations
    # at the same time to the same orbit)

    filtered = attributions.drop_coincident_attributions(observations)
    # Orbit 1 gets linked to two observations at the same time
    # We should expect to only keep the one with the smallest distance
    # Orbit 2 and 3 get linked to the same observation but we should keep both
    # Orbit 5 gets linked to same observation as orbit 1 but we should keep both
    assert len(filtered) == 5
    assert filtered.orbit_id.to_pylist() == ["o01", "o02", "o03", "o04", "o05"]
    assert filtered.obs_id.to_pylist() == ["02", "03", "03", "05", "01"]
    assert filtered.distance.to_pylist() == [
        0.5 / 3600,
        2 / 3600,
        1 / 3600,
        1 / 3600,
        0.5 / 3600,
    ]


def test_Attributions_drop_multiple_attributions(attributions):
    # Test that we can drop multiple attributions (attributions of the same observation
    # to multiple orbits)

    # We should drop the attribution of Orbit 2 to Observation 03, since Orbit 3
    # is closer to the observation
    # We should drop the attribution of Orbit 1 to Observation 01, since Orbit 5
    # is closer to the observation
    filtered = attributions.drop_multiple_attributions()
    assert len(filtered) == 5
    assert filtered.orbit_id.to_pylist() == ["o05", "o01", "o03", "o04", "o04"]
    assert filtered.obs_id.to_pylist() == ["01", "02", "03", "04", "05"]
    assert filtered.distance.to_pylist() == [
        0.5 / 3600,
        0.5 / 3600,
        1 / 3600,
        2 / 3600,
        1 / 3600,
    ]
