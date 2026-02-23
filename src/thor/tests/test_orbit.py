import numpy as np
from adam_core.coordinates import CartesianCoordinates, Origin, SphericalCoordinates
from adam_core.time import Timestamp

from ..orbit import assume_heliocentric_distance


def test_assume_heliocentric_distance_missing_rho():
    # Test that we can correctly calculate the heliocentric distance only
    # for those observations that have missing topocentric distances.

    # Lets define an origin located at the Sun
    origin_coords = CartesianCoordinates.from_kwargs(
        x=[0.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.0],
        vz=[0.0],
        time=Timestamp.from_mjd([59000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
    )

    lon = np.zeros(2)
    lat = np.zeros(2)
    rho = np.array([np.nan, 2.5])
    num_detections = len(lon)
    coords = SphericalCoordinates.from_kwargs(
        rho=rho,
        lon=lon,
        lat=lat,
        time=Timestamp.from_mjd([59000.0] * num_detections, scale="tdb"),
        origin=Origin.from_kwargs(code=["Observatory"] * num_detections),
        frame="equatorial",
    )

    # Lets assume the heliocentric distance is 3 au, the first detection
    # should have a heliocentric distance of 3 au (as rho), the second
    # detection should have a heliocentric distance of 2.5 au (as rho)
    # Since the origin is the Sun the heliocentric distance is also the
    # topocentric distance
    r = np.array([3.0, 0.0, 0.0])
    coords_assumed = assume_heliocentric_distance(r, coords, origin_coords)
    np.testing.assert_equal(coords_assumed.rho, np.array([r[0], rho[1]]))


def test_assume_heliocentric_distance_zero_origin():
    # Test that we can correctly calculate the heliocentric distance
    # Lets define an origin located at the Sun
    origin_coords = CartesianCoordinates.from_kwargs(
        x=[0.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.0],
        vz=[0.0],
        time=Timestamp.from_mjd([59000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
    )

    # Lets define a few detections located spherically about
    # the origin
    lon = np.arange(0, 360, 90)
    lat = np.arange(-90, 90, 90)
    lon, lat = np.meshgrid(lon, lat)
    lon = lon.flatten()
    lat = lat.flatten()
    num_detections = len(lon)
    coords = SphericalCoordinates.from_kwargs(
        lon=lon,
        lat=lat,
        time=Timestamp.from_mjd([59000.0] * num_detections, scale="tdb"),
        origin=Origin.from_kwargs(code=["Observatory"] * num_detections),
        frame="equatorial",
    )

    # Lets assume the heliocentric distance is 3 au, each detection
    # should have a heliocentric distance of 3 au (as rho)
    # Since the origin is the Sun the heliocentric distance is also the
    # topocentric distance
    r = np.array([3.0, 0.0, 0.0])
    r_mag = np.linalg.norm(r)
    coords_assumed = assume_heliocentric_distance(r, coords, origin_coords)
    np.testing.assert_equal(coords_assumed.rho, r_mag * np.ones(num_detections))


def test_assume_heliocentric_distance():
    # Test that we can correctly calculate the heliocentric distance with
    # a non-zero origin
    # Lets define an origin located at 1 au from the Sun
    origin_coords = CartesianCoordinates.from_kwargs(
        x=[1.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.0],
        vz=[0.0],
        time=Timestamp.from_mjd([59000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
    )

    # Lets define a few detections located at the six cardinal points
    # (two poles, four along the compass rose)
    # South pole, North pole, North, East, South, West
    lon = np.array([0.0, 0.0, 0.0, 90, 180, 270])
    lat = np.array([-90.0, 90.0, 0.0, 0.0, 0.0, 0.0])
    num_detections = len(lon)
    coords = SphericalCoordinates.from_kwargs(
        lon=lon,
        lat=lat,
        time=Timestamp.from_mjd([59000.0] * num_detections, scale="tdb"),
        origin=Origin.from_kwargs(code=["Observatory"] * num_detections),
        frame="equatorial",
    )

    # If we now assume a distance of sqrt(2) au, then two poles and the East and
    # West points should have a topocentric distance of 1 au (as rho), the North
    # point should have topocentric distance sqrt(2) - 1 au (as rho), and the South
    # point should have a topocentric distance of sqrt(2) + 1 au (on the opposite side of the
    # Sun)
    r = np.array([1.0, 1.0, 0.0])
    r_mag = np.linalg.norm(r)
    coords_assumed = assume_heliocentric_distance(r, coords, origin_coords)
    rho_assumed = coords_assumed.rho.to_numpy()
    rho_expected = np.array([1.0, 1.0, r_mag - 1, 1.0, r_mag + 1, 1.0])
    np.testing.assert_almost_equal(rho_assumed, rho_expected)
