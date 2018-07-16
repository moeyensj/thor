import numpy as np

OBLIQUITY = np.radians(23.43928)
TRANSFORM_EQ2EC = np.matrix([[1, 0, 0],
                             [0, np.cos(OBLIQUITY), np.sin(OBLIQUITY)],
                             [0, -np.sin(OBLIQUITY), np.cos(OBLIQUITY)]])
TRANSFORM_EC2EQ = np.matrix([[1, 0, 0],
                             [0, np.cos(OBLIQUITY), -np.sin(OBLIQUITY)],
                             [0, np.sin(OBLIQUITY), np.cos(OBLIQUITY)]])


__all__ = ["equatorialToEclipticCartesian",
           "eclipticToEquatorialCartesian",
           "_angularToCartesian",
           "eclipticAngularToCartesian",
           "equatorialAngularToCartesian",
           "_cartesianToAngular",
           "eclipticCartesianToAngular",
           "equatorialCartesianToAngular",
           "eclipticToEquatorialAngular",
           "equatorialToEclipticAngular"]

# Cartesian to Cartesian


def equatorialToEclipticCartesian(coords_eq_cart):
    """
    Transform equatorial cartesian coordinates to ecliptic cartesian
    coordinates.

    Parameters
    ----------
    coords_eq_cart : `~numpy.ndarray` (N, 3)
        Cartesian equatorial x, y, z coordinates.

    Returns
    -------
    coords_ec_cart : `~numpy.ndarray` (N, 3)
        Cartesian ecliptic x, y, z coordinates.

    See Also
    --------
    eclipticToEquatorialCartesian : Transform ecliptic cartesian coordinates
        to equatorial cartesian coordinates.
    """
    return np.array(TRANSFORM_EQ2EC @ coords_eq_cart.T).T


def eclipticToEquatorialCartesian(coords_ec_cart):
    """
    Transform ecliptic cartesian coordinates to equatorial cartesian
    coordinates.


    Parameters
    ----------
    coords_ec_cart : `~numpy.ndarray` (N, 3)
        Cartesian ecliptic x, y, z coordinates.

    Returns
    -------
    coords_eq_cart : `~numpy.ndarray` (N, 3)
        Cartesian equatorial x, y, z coordinates.

    See Also
    --------
    equatorialToEclipticCartesian : Transform ecliptic cartesian coordinates
        to equatorial cartesian coordinates.
    """
    return np.array(TRANSFORM_EC2EQ @ coords_ec_cart.T).T

# Angular to Cartesian


def _angularToCartesian(phi, theta, r):
    """
    Converts angular coordinates to cartesian given
    an angle or angles of longitude (phi), an angle or angles of latitude (theta),
    and distance (r).

    Parameters
    ----------
    phi : `~numpy.ndarray` (N)
        Longitude in radians.
    theta : `~numpy.ndarray` (N)
        Latitude in radians.
    r : float or `~numpy.ndarray` (N)
        Radial distance in arbitrary units.

    Returns
    -------
    x, y, z: `~numpy.ndarray` (N, 3)
        Cartesian x, y, z coordinates.
    """
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    return np.array([x, y, z]).T


def eclipticAngularToCartesian(coords_ec_ang, delta=1):
    """
    Convert ecliptic coordinates to
    cartesian coordinates.

    Input coordinate array should have shape (N, 2):
    np.array([[lon1, lat1],
              [lon2, lat2],
              [lon3, lat3],
                  .....
              [lonN, latN]])

    Parameters
    ----------
    coords_ec_ang : `~numpy.ndarray` (N, 2)
        Ecliptic longitude and latitude in radians.
    dist : float or `~numpy.ndarray` (N), optional
        Distance in arbitrary units.
        [Default = 1]

    Returns
    -------
    coords_ec_cart : `~numpy.ndarray` (N, 3)
        Cartesian x, y, z coordinates.

    See Also
    --------
    equatorialAngularToCartesian : Convert angular equatorial
    coordinates to equatorial cartesian.
    """
    return _angularToCartesian(coords_ec_ang[:, 0], coords_ec_ang[:, 1], delta)


def equatorialAngularToCartesian(coords_eq_ang, r=1):
    """
    Convert angular equatorial coordinates to
    cartesian coordinates.

    Input coordinate array should have shape (N, 2):
    np.array([[ra1, dec1],
              [ra2, dec2],
              [ra3, dec3],
                  .....
              [raN, decN]])

    Parameters
    ----------
    coords_eq_ang : `~numpy.ndarray` (N, 2)
        Right Ascenion and declination in radians.
    r : float or `~numpy.ndarray` (N), optional
        Distance in arbitrary units.
        [Default = 1]

    Returns
    -------
    coords_eq_cart : `~numpy.ndarray` (N, 3)
        Cartesian x, y, z coordinates.

    See Also
    --------
    eclipticAngularToCartesian : Convert angular ecliptic
    coordinates to ecliptic cartesian.
    """
    return _angularToCartesian(coords_eq_ang[:, 0], coords_eq_ang[:, 1], r)

# Cartesian to Angular


def _cartesianToAngular(x, y, z):
    """
    Convert cartesian coordinates to angular coordinates.

    Parameters
    ----------
    x : `~numpy.ndarray` (N)
        Cartesian x coordinate.
    y : `~numpy.ndarray` (N)
        Cartesian y coordinate.
    z : `~numpy.ndarray` (N)
        Cartesian z coordinate.

    Returns
    -------
    phi, theta, r : `~numpy.ndarray` (N, 3)
        Longitude and latitude in radians and
        radial distance in arbitrary units.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arcsin(z / r)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return np.array([phi, theta, r]).T


def eclipticCartesianToAngular(coords_ec_cart):
    """
    Convert ecliptic cartesian coordinates to
    angular coordinates.

    Parameters
    ----------
    coords_ec_cart : `~numpy.ndarray` (N, 3)
        Cartesian ecliptic x, y, z coordinates.

    Returns
    -------
    coords_eq_ang : `~numpy.ndarray` (N, 3)
        Ecliptic longitude and latitude in radians
        and radial distance in arbitrary units.
    """
    return _cartesianToAngular(*coords_ec_cart.T)


def equatorialCartesianToAngular(coords_eq_cart):
    """
    Convert equatorial cartesian coordinates to
    angular coordinates.


    Parameters
    ----------
    coords_eq_cart : `~numpy.ndarray` (N, 3)
        Cartesian equatorial x, y, z coordinates.

    Returns
    -------
    coords_eq_ang : `~numpy.ndarray` (N, 3)
        Equatorial longitude and latitude in radians
        and radial distance in arbitrary units.
    """
    return _cartesianToAngular(*coords_eq_cart.T)

# Angular to Angular


def equatorialToEclipticAngular(coords_eq_ang, r=1):
    """
    Transform angular equatorial coordinates to
    angular ecliptic coordinates.

    Input coordinate array should have shape (N, 2):
    np.array([[lon1, lat1],
              [lon2, lat2],
              [lon3, lat3],
                  .....
              [lonN, latN]])

    Parameters
    ----------
    coords_eq_ang : `~numpy.ndarray` (N, 2)
        Right Ascension and Declination in radians.
    r : float or `~numpy.ndarray` (N), optional
        Distance from origin in arbitrary units.
        [Default = 1]

    Returns
    -------
    coords_ec_ang : `~numpy.ndarray` (N, 3)
        Ecliptic longitude and latitude in radians
        and radial distance in arbitrary units.
    """
    coords_eq_cart = equatorialAngularToCartesian(coords_eq_ang, r=r)
    coords_ec_cart = equatorialToEclipticCartesian(coords_eq_cart)
    return eclipticCartesianToAngular(coords_ec_cart)


def eclipticToEquatorialAngular(coords_ec_ang, delta=1):
    """
    Transform angular ecliptic coordinates to
    angular equatorial coordinates.

    Input coordinate array should have shape (N, 2):
    np.array([[ra1, dec1],
              [ra2, dec2],
              [ra3, dec3],
                  .....
              [raN, decN]])

    Parameters
    ----------
    coords_ec_ang : `~numpy.ndarray` (N, 2)
        Ecliptic longitude and latitude in radians.
    delta : float or `~numpy.ndarray` (N)
        Distance from origin in arbitrary units.
        [Default = 1]

    Returns
    -------
    coords_eq_ang : `~numpy.ndarray` (N, 3)
        Right Ascension and Declination in radians
        and radial distance in arbitrary units.
    """
    coords_ec_cart = eclipticAngularToCartesian(coords_ec_ang, delta=delta)
    coords_eq_cart = eclipticToEquatorialCartesian(coords_ec_cart)
    return equatorialCartesianToAngular(coords_eq_cart)
