import numpy as np
from numba import jit

from ..constants import Constants as c

TRANSFORM_EQ2EC = c.TRANSFORM_EQ2EC
TRANSFORM_EC2EQ = c.TRANSFORM_EC2EQ

__all__ = [
    "_convertCartesianToSpherical",
    "_convertSphericalToCartesian",
    "transformCoordinates",
]


@jit(["UniTuple(f8[:], 6)(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])"], nopython=True, cache=True)
def _convertCartesianToSpherical(x, y, z, vx, vy, vz):
    """
    Convert spherical coordinates to cartesian coordinates.

    Parameters
    ----------
    x : `~numpy.ndarray` (N)
        x-position in units of distance.
    y : `~numpy.ndarray` (N)
        y-position in units of distance.
    z : `~numpy.ndarray` (N)
        z-position in units of distance.
    vx : `~numpy.ndarray` (N)
        x-velocity in the same units of x per arbitrary unit
        of time.
    vy : `~numpy.ndarray` (N)
        y-velocity in the same units of y per arbitrary unit
        of time.
    vz : `~numpy.ndarray` (N)
        z-velocity in the same units of z per arbitrary unit
        of time.

    Returns
    -------
    rho : `~numpy.ndarray` (N)
        Radial distance in the same units of x, y, and z.
    lon : `~numpy.ndarray` (N)
        Longitude ranging from 0 to 2 pi radians.
    lat : `~numpy.ndarray` (N)
        Latitude ranging from -pi/2 to pi/2 radians with 0 at the
        equator.
    vrho : `~numpy.ndarray` (N)
        Radial velocity in radians per arbitrary unit of time (same
        unit of time as the x, y, and z velocities).
    vlon : `~numpy.ndarray` (N)
        Longitudinal velocity in radians per arbitrary unit of time
        (same unit of time as the x, y, and z velocities).
    vlat : `~numpy.ndarray` (N)
        Latitudinal velocity in radians per arbitrary unit of time.
        (same unit of time as the x, y, and z velocities).
    """
    rho = np.sqrt(x**2 + y**2 + z**2)
    lon = np.arctan2(y, x)
    lon = np.where(lon < 0.0, 2 * np.pi + lon, lon)
    lat = np.arcsin(z / rho)
    lat = np.where((lat >= 3*np.pi/2) & (lat <= 2*np.pi), lat - 2*np.pi, lat)

    if np.all(vx == 0) & (np.all(vy == 0)) & (np.all(vz == 0)):
        vrho = np.zeros(len(rho))
        vlon = np.zeros(len(lon))
        vlat = np.zeros(len(lat))
    else:
        vrho = (x * vx + y * vy + z * vz) / rho
        vlon = (vy * x - vx * y) / (x**2 + y**2)
        vlat = (vz - vrho * z / rho) / np.sqrt(x**2 + y**2)

    return rho, lon, lat, vrho, vlon, vlat


@jit(["UniTuple(f8[:], 6)(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])"], nopython=True, cache=True)
def _convertSphericalToCartesian(rho, lon, lat, vrho, vlon, vlat):
    """
    Convert spherical coordinates to cartesian coordinates.

    Parameters
    ----------
    rho : `~numpy.ndarray` (N)
        Radial distance in units of distance.
    lon : `~numpy.ndarray` (N)
        Longitude ranging from 0 to 2 pi radians.
    lat : `~numpy.ndarray` (N)
        Latitude ranging from -pi/2 to pi/2 radians with 0 at the
        equator.
    vrho : `~numpy.ndarray` (N)
        Radial velocity in radians per arbitrary unit of time.
    vlon : `~numpy.ndarray` (N)
        Longitudinal velocity in radians per arbitrary unit of time.
    vlat : `~numpy.ndarray` (N)
        Latitudinal velocity in radians per arbitrary unit of time.

    Returns
    -------
    x : `~numpy.ndarray` (N)
        x-position in the same units of rho.
    y : `~numpy.ndarray` (N)
        y-position in the same units of rho.
    z : `~numpy.ndarray` (N)
        z-position in the same units of rho.
    vx : `~numpy.ndarray` (N)
        x-velocity in the same units of rho per unit of time
        (same unit of time as the rho, lon, and lat velocities,
        for example, radians per day > AU per day).
    vy : `~numpy.ndarray` (N)
        y-velocity in the same units of rho per unit of time
        (same unit of time as the rho, lon, and lat velocities,
        for example, radians per day > AU per day).
    vz : `~numpy.ndarray` (N)
        z-velocity in the same units of rho per unit of time
        (same unit of time as the rho, lon, and lat velocities,
        for example, radians per day > AU per day).
    """
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    x = rho * cos_lat * cos_lon
    y = rho * cos_lat * sin_lon
    z = rho * sin_lat

    if np.all(vlon == 0) & (np.all(vlat == 0)) & (np.all(vrho == 0)):
        vx = np.zeros(len(x))
        vy = np.zeros(len(y))
        vz = np.zeros(len(z))
    else:
        vx = cos_lat * cos_lon * vrho - rho * cos_lat * sin_lon * vlon - rho * sin_lat * cos_lon * vlat
        vy = cos_lat * sin_lon * vrho + rho * cos_lat * cos_lon * vlon - rho * sin_lat * sin_lon * vlat
        vz = sin_lat * vrho + rho * cos_lat * vlat

    return x, y, z, vx, vy, vz

def transformCoordinates(coords, frame_in, frame_out, representation_in="cartesian", representation_out="cartesian"):
    """
    Transform coordinates between frames ('ecliptic', 'equatorial') and/or representations ('cartesian', 'spherical').
    Coordinates may include only positions or they may also include velocities.

    Parameters
    ----------
    coords : `~numpy.ndarray` (N, 3) or (N, 6)
        Coordinates to transform. These may include velocities in which case array shape should
        be (N, 6).
    frame_in : {'equatorial', 'ecliptic'}
        Reference frame of the input coordinates.
    frame_out : {'equatorial', 'ecliptic'}
        Reference frame of the output coordinates.
    representation_in : {'cartesian', 'spherical'}, optional
        The coordinate system of the input coordinates.
        If spherical, the expected columns are:
            rho, lon, lat, vrho, vlon, vlat
        Spherical velocites should be in units of degrees per unit
        time.
        If cartesian, the expected columns are:
            x, y, z, vx, vy, vz

    representation_out : {'cartesian', 'spherical'}, optional
        The coordinate system of the transformed coordinates.
        If spherical, the columns returned will be:
            rho, lon, lat, vrho, vlon, vlat
        Spherical velocites wil be in units of degrees per unit
        time.
        If cartesian, the columns returned will be:
            x, y, z, vx, vy, vz

    Returns
    -------
    coords_out : `~numpy.ndarray` (N, 3) or (N, 6)
        Tranformed coordinates with velocities if the input coordinates
        also had velocities.

    Raises
    ------
    ValueError
        If frame_in, frame_out are not one of 'equatorial', 'ecliptic'.
        If representation_in, representation_out are not one of 'cartesian', 'spherical'.
        If coords does not have shape (N, 3) or (N, 6).
    """
    # Check that frame_in and frame_out are one of equatorial
    # or ecliptic, raise errors otherwise
    frame_err = [
        "{} should be one of:\n",
        "'equatorial' or 'ecliptic'"
    ]
    if frame_in != "equatorial" and frame_in != "ecliptic":
        raise ValueError("".join(frame_err).format("frame_in"))

    if frame_out != "equatorial" and frame_out != "ecliptic":
        raise ValueError("".join(frame_err).format("frame_out"))

    # Check that representation_in and representation_out are one of cartesian
    # or spherical, raise errors otherwise
    representation_err = [
        "{} should be one of:\n",
        "'cartesian' or 'spherical'"
    ]
    if representation_in != "cartesian" and representation_in != "spherical":
        raise ValueError("".join(representation_err).format("representation_in"))

    if representation_out != "cartesian" and representation_out != "spherical":
        raise ValueError("".join(representation_err).format("representation_out"))

    # Check that the coords has either shape (N, 3), or (N, 6), if
    # neither raise an error!
    coords_ = np.zeros((len(coords), 6))
    if coords.shape[1] == 3:
        coords_[:, :3] = coords[:, :3]
    elif coords.shape[1] == 6:
        coords_ = coords.copy()
    else:
        err = (
            "coords should have shape (N, 3) or (N, 6).\n"
        )
        raise ValueError(err)

     # Convert angles and angular velocites from degrees to radians
    if representation_in == "spherical":
        coords_[:, 1:3] = np.radians(coords_[:, 1:3])
        coords_[:, 4:6] = np.radians(coords_[:, 4:6])

    # If the input and output frames don't match, regardless of the representation types,
    # we need to convert to cartesian coordinates so we can rotate to the correct frame.
    coords_rotated = np.zeros_like(coords_)
    if frame_in != frame_out:

        if representation_in == "spherical":
            coords_ = np.array(_convertSphericalToCartesian(*coords_.T)).T

        if frame_in == "ecliptic":
            rotation_matrix = TRANSFORM_EC2EQ
        else: # frame_in == "equatorial"
            rotation_matrix = TRANSFORM_EQ2EC

        coords_rotated[:, 0:3] = (rotation_matrix @ coords_[:, 0:3].T).T
        coords_rotated[:, 3:6] = (rotation_matrix @ coords_[:, 3:6].T).T

        if representation_in == "spherical":
            coords_rotated = np.array(_convertCartesianToSpherical(*coords_rotated.T)).T

    else:
        coords_rotated = coords_.copy()

    # At this point, it is assumed that the rotated coordinates are in the correct
    # frame.
    if representation_in == "cartesian" and representation_out == "spherical":
        coords_out = np.array(_convertCartesianToSpherical(*coords_rotated.T)).T

    elif representation_in == "spherical" and representation_out == "cartesian":
        coords_out = np.array(_convertSphericalToCartesian(*coords_rotated.T)).T

    else:
        coords_out = coords_rotated

    # Convert angles and angular velocities back to units of degrees
    if representation_out == "spherical":
        coords_out[:, 1:3] = np.degrees(coords_out[:, 1:3])
        coords_out[:, 4:6] = np.degrees(coords_out[:, 4:6])

    if coords.shape[1] == 3:
        return coords_out[:, :3]

    return coords_out

