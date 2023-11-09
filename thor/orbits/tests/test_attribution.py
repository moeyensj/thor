import pytest
from adam_core.observations import PointSourceDetections
from adam_core.time import Timestamp

from ...observations.observations import Observations
from ..attribution import Attributions


def test_Attributions_drop_coincident_attributions():
    observations = Observations.from_kwargs(
        detections=PointSourceDetections.from_kwargs(
            id=["01", "02", "03", "04"],
            exposure_id=["e01", "e01", "e02", "e02"],
            time=Timestamp.from_mjd([59001.1, 59001.1, 59002.1, 59002.1], scale="utc"),
            ra=[1, 2, 3, 4],
            dec=[5, 6, 7, 8],
            mag=[10, 11, 12, 13],
        ),
        state_id=[0, 0, 1, 1],
        observatory_code=["500", "500", "500", "500"],
    )

    attributions = Attributions.from_kwargs(
        orbit_id=["o01", "o01", "o02", "o03"],
        obs_id=["01", "02", "03", "03"],
        distance=[0.5 / 3600, 1 / 3600, 2 / 3600, 1 / 3600],
    )

    filtered = attributions.drop_coincident_attributions(observations)
    # Orbit 1 gets linked to two observations at the same time
    # We should expect to only keep the one with the smallest distance
    # Orbit 2 and 3 get linked to the same observation but we should keep both
    assert len(filtered) == 3
    assert filtered.orbit_id.to_pylist() == ["o01", "o02", "o03"]
    assert filtered.obs_id.to_pylist() == ["01", "03", "03"]
    assert filtered.distance.to_pylist() == [0.5 / 3600, 2 / 3600, 1 / 3600]
