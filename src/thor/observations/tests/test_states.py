import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from adam_core.coordinates import Origin, SphericalCoordinates
from adam_core.time import Timestamp

from ..states import calculate_state_ids


def test_calculate_state_ids():
    # Create some coordinates with different times and origins and make sure
    # that the state IDs are assigned correctly
    origins = Origin.from_kwargs(code=["500", "500", "I41", "500", "500", "I41"])
    times = Timestamp.from_mjd(
        [59002.1, 59001.1, 59001.1, 59001.1, 59002.1, 59002.1],
        scale="utc",
    )
    coords = SphericalCoordinates.from_kwargs(
        rho=np.random.rand(6),
        lon=np.random.rand(6),
        lat=np.random.rand(6),
        time=times,
        origin=origins,
        frame="equatorial",
    )

    state_ids = calculate_state_ids(coords)
    expected_state_ids = pa.array([2, 0, 1, 0, 2, 3])

    assert pc.all(pc.equal(state_ids, expected_state_ids)).as_py()
