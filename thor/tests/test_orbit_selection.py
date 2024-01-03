import numpy as np
from adam_core.coordinates import KeplerianCoordinates, Origin
from adam_core.time import Timestamp

from ..orbit_selection import select_average_within_region


def test_select_average_within_region():
    # Given a set of Keperian coordinates test to make
    # sure that select_average_within_region selects
    # the orbit closest to the median in semi-major axis,
    # eccentricity, and inclination.

    # Create a set of Keplerian coordinates
    coords = KeplerianCoordinates.from_kwargs(
        a=[1.0, 2.0, 3.0, 4.0, 5.0],
        e=[0.10, 0.15, 0.20, 0.25, 0.30],
        i=[0, 5.0, 10.0, 15.0, 20.0],
        raan=np.random.rand(5) * 360.0,
        ap=np.random.rand(5) * 360.0,
        M=np.random.rand(5) * 360.0,
        time=Timestamp.from_mjd([59000.0 for _ in range(5)], scale="tdb"),
        origin=Origin.from_kwargs(code=np.full(5, "SUN", dtype="object")),
        frame="ecliptic",
    )

    # Select the average orbit
    index = select_average_within_region(coords)
    assert index == 2
