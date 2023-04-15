import numpy as np

__all__ = ["angularToGnomonic", "cartesianToGnomonic"]


def angularToGnomonic(coords_ang, coords_ang_center=np.array([0, 0])):
    """
    Project angular spherical coordinates onto a gnomonic tangent plane.

    Parameters
    ----------
    coords_ang : `~numpy.ndarray` (N, 2)
        Longitude (between 0 and 2 pi) and latitude (between -pi/2 and pi/2)
        in radians.
    coords_ang_center : `~numpy.ndarray` (2), optional
        Longitude (between 0 and 2 pi) and latitude (between -pi/2 and pi/2)
        in radians about which to center projection.
        [Default = np.array([0, 0])]

    Returns
    -------
    coords_gnomonic : `~numpy.ndarray` (N, 2)
        Gnomonic longitude and latitude in radians.
    """
    lon = coords_ang[:, 0]
    lon = np.where(lon > np.pi, lon - 2 * np.pi, lon)
    lat = coords_ang[:, 1]
    lon_0, lat_0 = coords_ang_center

    c = np.sin(lat_0) * np.sin(lat) + np.cos(lat_0) * np.cos(lat) * np.cos(lon - lon_0)
    u = np.cos(lat) * np.sin(lon - lon_0) / c
    v = (
        np.cos(lat_0) * np.sin(lat) - np.sin(lat_0) * np.cos(lat) * np.cos(lon - lon_0)
    ) / c
    return np.array([u, v]).T


def cartesianToGnomonic(coords_cart):
    """
    Project cartesian coordinates onto a gnomonic tangent plane centered about
    the x-axis.

    Parameters
    ----------
    coords_cart : `~numpy.ndarray` (N, 3) or (N, 6)
        Cartesian x, y, z coordinates (can optionally include cartesian
        velocities)

    Returns
    -------
    coords_gnomonic : `~numpy.ndarray` (N, 2) or (N, 4)
        Gnomonic longitude and latitude in radians.
    """
    x = coords_cart[:, 0]
    y = coords_cart[:, 1]
    z = coords_cart[:, 2]

    u = y / x
    v = z / x

    if coords_cart.shape[1] == 6:
        vx = coords_cart[:, 3]
        vy = coords_cart[:, 4]
        vz = coords_cart[:, 5]

        vu = (x * vy - vx * y) / x**2
        vv = (x * vz - vx * z) / x**2

        gnomonic_coords = np.array([u, v, vu, vv]).T

    else:
        gnomonic_coords = np.array([u, v]).T

    return gnomonic_coords
