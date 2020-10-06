import pytest
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from ..coordinate_transforms import _convertSphericalToCartesian
from ..coordinate_transforms import _convertCartesianToSpherical
from ..coordinate_transforms import transformCoordinates

# Define cardinal axes in cartesian coordinates
X_AXIS = np.array([1., 0., 0.])
NEG_X_AXIS = -X_AXIS
Y_AXIS = np.array([0., 1., 0.])
NEG_Y_AXIS = -Y_AXIS
Z_AXIS = np.array([0., 0., 1.])
NEG_Z_AXIS = -Z_AXIS

CARTESIAN_AXES = [
    X_AXIS,
    NEG_X_AXIS,
    Y_AXIS,
    NEG_Y_AXIS,
    Z_AXIS,
    NEG_Z_AXIS,
]

# Define cardinal axes in spherical coordinates
X_AXIS_SPHERICAL = np.array([1., 0., 0.])
NEG_X_AXIS_SPHERICAL = np.array([1., np.pi, 0.])
Y_AXIS_SPHERICAL = np.array([1., np.pi/2, 0])
NEG_Y_AXIS_SPHERICAL = np.array([1, 3*np.pi/2, 0.])
Z_AXIS_SPHERICAL = np.array([1., 0., np.pi/2])
NEG_Z_AXIS_SPHERICAL = np.array([1., np.pi, -np.pi/2])

SPHERICAL_AXES = [
    X_AXIS_SPHERICAL,
    NEG_X_AXIS_SPHERICAL,
    Y_AXIS_SPHERICAL,
    NEG_Y_AXIS_SPHERICAL,
    Z_AXIS_SPHERICAL,
    NEG_Z_AXIS_SPHERICAL,
]
    
ZERO_V = np.array([0.])

def __cardinalityCartesian(func):
    # This function runs all cardinal spherical axes through func
    # and tests that the returned cartesian axes correspond
    # to the cardinal cartesian axes.
    for s_axis, c_axis in zip(SPHERICAL_AXES, CARTESIAN_AXES):
        coords = func(
            s_axis[0:1],
            s_axis[1:2],
            s_axis[2:3],
            *[ZERO_V for i in range(3)]
        )
        np.testing.assert_array_almost_equal(
            np.concatenate([coords[0], coords[1], coords[2]]),
            c_axis
        )
    return

def __cardinalitySpherical(func):
    # This function runs all cardinal cartesian axes through func
    # and tests that the returned spherical axes correspond
    # to the cardinal spherical axes.
    for s_axis, c_axis in zip(SPHERICAL_AXES, CARTESIAN_AXES):
        coords = func(
            c_axis[0:1],
            c_axis[1:2],
            c_axis[2:3],
            *[ZERO_V for i in range(3)]
        )
        np.testing.assert_array_almost_equal(
            np.concatenate([coords[0], coords[1], coords[2]]),
            s_axis
        )
    return

def test__convertSphericalToCartesian():
    __cardinalityCartesian(_convertSphericalToCartesian)
    return

def test__convertCartesianToSpherical():
    __cardinalitySpherical(_convertCartesianToSpherical)
    return

def test_transformCoordinates_errors():
    # Test that no errors are raised
    for frame in ["equatorial", "ecliptic"]:
        for representation in ["cartesian", "spherical"]:
            transformCoordinates(
                np.ones((100, 6)),
                frame,
                frame,
                representation_in=representation,
                representation_out=representation
            )

    # Test that ValueErrors are raised for incorrect frames
    # and representations
    for frame in ["a", "a"]:
        for representation in ["a", "a"]:
            with pytest.raises(ValueError):
                transformCoordinates(
                    np.ones((100, 6)),
                    frame,
                    frame,
                    representation_in=representation,
                    representation_out=representation
                )

    # Test that ValueErrors are not raised for correct coordinate shapes
    for frame in ["equatorial", "ecliptic"]:
        for representation in ["cartesian", "spherical"]:
            for shape in [3, 6]:    
                transformCoordinates(
                    np.ones((100, shape)),
                    frame,
                    frame,
                    representation_in=representation,
                    representation_out=representation
                )
    # Test that ValueErrors are raised for incorrect coordinate shapes
    for frame in ["equatorial", "ecliptic"]:
        for representation in ["cartesian", "spherical"]:
            for shape in [1,2,4,5,7,8]:    
                with pytest.raises(ValueError):
                    transformCoordinates(
                        np.ones((100, shape)),
                        frame,
                        frame,
                        representation_in=representation,
                        representation_out=representation
                    )
    return

def test_transformCoordinates_representations():
    
    for frame in ["equatorial", "ecliptic"]:

        lon = np.degrees(np.random.uniform(0, 2*np.pi, 100))
        lat = np.degrees(np.random.uniform(-np.pi/2, np.pi/2, 100))
        rho = np.ones(100)

        coords_spherical = np.vstack([rho, lon, lat]).T
        coords_cartesian = transformCoordinates(
            coords_spherical, 
            frame,
            frame, 
            representation_in="spherical",
            representation_out="cartesian"
        )

        # Check shape is correct
        assert coords_cartesian.shape == (100, 3)

        # Check that the radius is 1
        np.testing.assert_allclose(np.linalg.norm(coords_cartesian, axis=1), np.ones(100))

        # Test that we get our original inputs if we go back to spherical
        coords_spherical_2 = transformCoordinates(
            coords_cartesian, 
            frame,
            frame, 
            representation_in="cartesian",
            representation_out="spherical"
        )
        np.testing.assert_allclose(coords_spherical, coords_spherical_2)
        
        vrho = np.zeros(100)
        vlat = np.ones(100) * np.degrees(0.005)
        vlon = np.ones(100) * np.degrees(0.005)
        coords_spherical = np.vstack([rho, lon, lat, vrho, vlon, vlat]).T
        coords_cartesian = transformCoordinates(
            coords_spherical, 
            frame,
            frame, 
            representation_in="spherical",
            representation_out="cartesian"
        )
        # Check shape is correct
        assert coords_cartesian.shape == (100, 6)

        # Check that the radius is 1
        np.testing.assert_allclose(np.linalg.norm(coords_cartesian[:, :3], axis=1), np.ones(100))

        # Test that we get our original inputs if we go back to spherical
        coords_spherical_2 = transformCoordinates(
            coords_cartesian, 
            frame,
            frame, 
            representation_in="cartesian",
            representation_out="spherical"
        )
        np.testing.assert_allclose(coords_spherical, coords_spherical_2, atol=1e-16)

def test_transformCoordinates_frames():
    lon = np.degrees(np.random.uniform(0, 2*np.pi, 100))
    lat = np.degrees(np.random.uniform(-np.pi/2, np.pi/2, 100))
    rho = np.ones(100)
    vrho = np.zeros(100)
    vlat = np.zeros(100)
    vlon = np.zeros(100) 
    coords_spherical_ec = np.array([rho, lon, lat, vrho, vlon, vlat]).T

    # Set up astropy coordinates in heliocentric ecliptic frame
    coords_eclipiau76 = SkyCoord(
        lon=lon*u.degree, 
        lat=lat*u.degree, 
        distance=rho*u.AU, 
        frame="heliocentriceclipticiau76")

    # Transform to heliocentric equatorial (in spherical and cartesian)
    coords_hcrs = coords_eclipiau76.transform_to("hcrs")
    coords_hcrs_spherical = np.array([coords_hcrs.distance.value, coords_hcrs.ra.value, coords_hcrs.dec.value]).T
    coords_hcrs.representation_type = "cartesian"
    coords_hcrs_cartesian = np.array([coords_hcrs.x.value, coords_hcrs.y.value, coords_hcrs.z.value]).T

    coords_spherical_eq = transformCoordinates(
        coords_spherical_ec, 
        "ecliptic", 
        "equatorial", 
        representation_in="spherical", 
        representation_out="spherical")

    np.testing.assert_allclose(coords_spherical_eq[:, :3], coords_hcrs_spherical)

    coords_cartesian_eq = transformCoordinates(
        coords_spherical_ec, 
        "ecliptic", 
        "equatorial", 
        representation_in="spherical", 
        representation_out="cartesian")

    np.testing.assert_allclose(coords_cartesian_eq[:, :3], coords_hcrs_cartesian)

    lon = np.degrees(np.random.uniform(0, 2*np.pi, 100))
    lat = np.degrees(np.random.uniform(-np.pi/2, np.pi/2, 100))
    rho = np.ones(100)
    vrho = np.zeros(100)
    vlat = np.zeros(100)
    vlon = np.zeros(100) 
    coords_spherical_eq = np.array([rho, lon, lat, vrho, vlon, vlat]).T

    # Set up astropy coordinates in heliocentric equatorial frame
    coords_hcrs = SkyCoord(
        ra=lon*u.degree, 
        dec=lat*u.degree, 
        distance=rho*u.AU, 
        frame="hcrs")

    # Transform to heliocentric ecliptic (in spherical and cartesian)
    coords_eclipiau76 = coords_hcrs.transform_to("heliocentriceclipticiau76")
    coords_eclipiau76_spherical = np.array([coords_eclipiau76.distance.value, coords_eclipiau76.lon.value, coords_eclipiau76.lat.value]).T
    coords_eclipiau76.representation_type = "cartesian"
    coords_eclipiau76_cartesian = np.array([coords_eclipiau76.x.value, coords_eclipiau76.y.value, coords_eclipiau76.z.value]).T

    coords_spherical_ec = transformCoordinates(
        coords_spherical_eq, 
        "equatorial", 
        "ecliptic", 
        representation_in="spherical", 
        representation_out="spherical")

    np.testing.assert_allclose(coords_spherical_ec[:, :3], coords_eclipiau76_spherical)

    coords_cartesian_ec = transformCoordinates(
        coords_spherical_eq, 
        "equatorial",
        "ecliptic", 
        representation_in="spherical", 
        representation_out="cartesian")

    np.testing.assert_allclose(coords_cartesian_ec[:, :3], coords_eclipiau76_cartesian)

def test_transformCoordinates_noTransform():
    lon = np.degrees(np.random.uniform(0, 2*np.pi, 100))
    lat = np.degrees(np.random.uniform(-np.pi/2, np.pi/2, 100))
    rho = np.ones(100)
    vrho = np.zeros(100)
    vlat = np.zeros(100)
    vlon = np.zeros(100) 
    coords_spherical_ec = np.array([rho, lon, lat, vrho, vlon, vlat]).T
    coords_spherical_eq = np.array([rho, lon, lat, vrho, vlon, vlat]).T

    coords_spherical_ec_2 = transformCoordinates(
        coords_spherical_ec, 
        "ecliptic", 
        "ecliptic", 
        representation_in="spherical", 
        representation_out="spherical")

    np.testing.assert_allclose(coords_spherical_ec, coords_spherical_ec_2)

    coords_spherical_eq_2 = transformCoordinates(
        coords_spherical_eq, 
        "equatorial", 
        "equatorial", 
        representation_in="spherical", 
        representation_out="spherical")

    np.testing.assert_allclose(coords_spherical_eq, coords_spherical_eq_2)

    x = np.random.uniform(-10, 10, 100)
    y = np.random.uniform(-10, 10, 100)
    z = np.random.uniform(-1, 1, 100)
    vx = np.ones(100)
    vy = np.ones(100)
    vz = np.ones(100) 
    coords_cartesian_ec = np.array([x, y, z, vx, vy, vz]).T
    coords_cartesian_eq = np.array([x, y, z, vx, vy, vz]).T

    coords_cartesian_ec_2 = transformCoordinates(
        coords_cartesian_ec, 
        "ecliptic", 
        "ecliptic", 
        representation_in="cartesian", 
        representation_out="cartesian")

    np.testing.assert_allclose(coords_cartesian_ec, coords_cartesian_ec_2)

    coords_cartesian_eq_2 = transformCoordinates(
        coords_cartesian_eq, 
        "equatorial", 
        "equatorial", 
        representation_in="cartesian", 
        representation_out="cartesian")

    np.testing.assert_allclose(coords_cartesian_eq, coords_cartesian_eq_2)

