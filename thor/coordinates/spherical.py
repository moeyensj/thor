import warnings
import numpy as np
import jax.numpy as jnp
from jax import config, jit
from astropy.time import Time
from astropy import units as u
from typing import Optional

config.update("jax_enable_x64", True)

from .coordinates import Coordinates
from .cartesian import CartesianCoordinates

__all__ = [
    "_cartesian_to_spherical",
    "_spherical_to_cartesian",
    "SphericalCoordinates"
]

SPHERICAL_COLS = ["rho", "lon", "lat", "vrho", "vlon", "vlat"]
SPHERICAL_UNITS = [u.au, u.degree, u.degree, u.au / u.d, u.degree / u.d, u.degree / u.d]

@jit
def _cartesian_to_spherical(coords_cartesian):
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
    coords_spherical = jnp.zeros_like(coords_cartesian, dtype=jnp.float64)
    x = coords_cartesian[:, 0]
    y = coords_cartesian[:, 1]
    z = coords_cartesian[:, 2]
    vx = coords_cartesian[:, 3]
    vy = coords_cartesian[:, 4]
    vz = coords_cartesian[:, 5]

    rho = jnp.sqrt(x**2 + y**2 + z**2)
    lon = jnp.arctan2(y, x)
    lon = jnp.where(lon < 0.0, 2 * jnp.pi + lon, lon)
    lat = jnp.arcsin(z / rho)
    lat = jnp.where((lat >= 3*jnp.pi/2) & (lat <= 2*jnp.pi), lat - 2*jnp.pi, lat)

    vrho = (x * vx + y * vy + z * vz) / rho
    vlon = (vy * x - vx * y) / (x**2 + y**2)
    vlat = (vz - vrho * z / rho) / jnp.sqrt(x**2 + y**2)

    coords_spherical = coords_spherical.at[:, 0].set(rho)
    coords_spherical = coords_spherical.at[:, 1].set(jnp.degrees(lon))
    coords_spherical = coords_spherical.at[:, 2].set(jnp.degrees(lat))
    coords_spherical = coords_spherical.at[:, 3].set(vrho)
    coords_spherical = coords_spherical.at[:, 4].set(jnp.degrees(vlon))
    coords_spherical = coords_spherical.at[:, 5].set(jnp.degrees(vlat))

    return coords_spherical

@jit
def _spherical_to_cartesian(coords_spherical):
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
    coords_cartesian = jnp.zeros_like(coords_spherical, dtype=jnp.float64)
    rho = coords_spherical[:, 0]
    lon = jnp.radians(coords_spherical[:, 1])
    lat = jnp.radians(coords_spherical[:, 2])
    vrho = coords_spherical[:, 3]
    vlon = jnp.radians(coords_spherical[:, 4])
    vlat = jnp.radians(coords_spherical[:, 5])

    cos_lat = jnp.cos(lat)
    sin_lat = jnp.sin(lat)
    cos_lon = jnp.cos(lon)
    sin_lon = jnp.sin(lon)

    x = rho * cos_lat * cos_lon
    y = rho * cos_lat * sin_lon
    z = rho * sin_lat

    vx = cos_lat * cos_lon * vrho - rho * cos_lat * sin_lon * vlon - rho * sin_lat * cos_lon * vlat
    vy = cos_lat * sin_lon * vrho + rho * cos_lat * cos_lon * vlon - rho * sin_lat * sin_lon * vlat
    vz = sin_lat * vrho + rho * cos_lat * vlat

    coords_cartesian = coords_cartesian.at[:, 0].set(x)
    coords_cartesian = coords_cartesian.at[:, 1].set(y)
    coords_cartesian = coords_cartesian.at[:, 2].set(z)
    coords_cartesian = coords_cartesian.at[:, 3].set(vx)
    coords_cartesian = coords_cartesian.at[:, 4].set(vy)
    coords_cartesian = coords_cartesian.at[:, 5].set(vz)

    return coords_cartesian

class SphericalCoordinates(Coordinates):

    def __init__(
            self,
            rho: Optional[np.ndarray] = None,
            lon: Optional[np.ndarray] = None,
            lat: Optional[np.ndarray] = None,
            vrho: Optional[np.ndarray] = None,
            vlon: Optional[np.ndarray] = None,
            vlat: Optional[np.ndarray] = None,
            times: Optional[Time] = None,
            covariances: Optional[np.ndarray] = None,
            origin: str = "heliocentric",
            frame: str = "ecliptic"
        ):
        """

        Parameters
        ----------
        rho : `~numpy.ndarray` (N)
            Radial distance in units of au.
        lon : `~numpy.ndarray` (N)
            Longitudinal angle in units of degrees.
        lat : `~numpy.ndarray` (N)
            Latitudinal angle in units of degrees (geographic coordinate
            style with 0 degrees at the equator and ranging from -90 to 90).
        vrho : `~numpy.ndarray` (N)
            Radial velocity in units of au per day.
        vlon : `~numpy.ndarray` (N)
            Longitudinal velocity in units of degrees per day.
        vlat : `~numpy.ndarray` (N)
            Latitudinal velocity in units of degrees per day.
        """

        Coordinates.__init__(self,
            rho,
            lon,
            lat,
            vrho,
            vlon,
            vlat,
            covariances=covariances,
            times=times,
            origin=origin,
            frame=frame,
            names=SPHERICAL_COLS
        )

        self._rho = self._coords[:, 0]
        self._lon = self._coords[:, 1]
        self._lat = self._coords[:, 2]
        self._vrho = self._coords[:, 3]
        self._vlon = self._coords[:, 4]
        self._vlat = self._coords[:, 5]
        return

    @property
    def rho(self):
        return self._rho

    @property
    def lon(self):
        return self._lon

    @property
    def lat(self):
        return self._lat

    @property
    def vrho(self):
        return self._vrho

    @property
    def vlon(self):
        return self._vlon

    @property
    def vlat(self):
        return self._vlat

    def to_cartesian(self) -> CartesianCoordinates:

        coords_cartesian = _spherical_to_cartesian(self.coords.filled())
        coords_cartesian = np.array(coords_cartesian)

        if self.covariances is not None:
            warnings.warn("Covariance transformations have not been implemented yet.")

        coords = CartesianCoordinates(
            x=coords_cartesian[:, 0],
            y=coords_cartesian[:, 1],
            z=coords_cartesian[:, 2],
            vx=coords_cartesian[:, 3],
            vy=coords_cartesian[:, 4],
            vz=coords_cartesian[:, 5],
            times=self.times,
            covariances=None,
            origin=self.origin,
            frame=self.frame
        )
        return coords

    @classmethod
    def from_cartesian(cls, cartesian: CartesianCoordinates):

        coords_spherical = _cartesian_to_spherical(cartesian.coords.filled())
        coords_spherical = np.array(coords_spherical)

        if cartesian.covariances is not None:
            warnings.warn("Covariance transformations have not been implemented yet.")

        coords = cls(
            rho=coords_spherical[:, 0],
            lon=coords_spherical[:, 1],
            lat=coords_spherical[:, 2],
            vrho=coords_spherical[:, 3],
            vlon=coords_spherical[:, 4],
            vlat=coords_spherical[:, 5],
            times=cartesian.times,
            covariances=None,
            origin=cartesian.origin,
            frame=cartesian.frame
        )

        return coords
