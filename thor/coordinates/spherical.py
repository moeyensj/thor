import numpy as np
import jax.numpy as jnp
from jax.experimental import loops
from jax import config, jit
from astropy.time import Time
from astropy import units as u
from typing import Optional, Union

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

from .coordinates import Coordinates
from .cartesian import CartesianCoordinates
from .covariances import transform_covariances_jacobian

__all__ = [
    "_cartesian_to_spherical",
    "_spherical_to_cartesian",
    "SphericalCoordinates"
]

SPHERICAL_COLS = ["rho", "lon", "lat", "vrho", "vlon", "vlat"]
SPHERICAL_UNITS = [u.au, u.degree, u.degree, u.au / u.d, u.degree / u.d, u.degree / u.d]

@jit
def _cartesian_to_spherical(coords_cartesian: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """
    Convert a single Cartesian coordinate to a spherical coordinate.

    Parameters
    ----------
    coords_cartesian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (6)
        3D Cartesian coordinate including time derivatives.
        x : x-position in units of distance.
        y : y-position in units of distance.
        z : z-position in units of distance.
        vx : x-velocity in the same units of x per arbitrary unit of time.
        vy : y-velocity in the same units of y per arbitrary unit of time.
        vz : z-velocity in the same units of z per arbitrary unit of time.

    Returns
    -------
    coords_spherical : `~jax.numpy.ndarray` (6)
        3D Spherical coordinate including time derivatives.
        rho : Radial distance in the same units of x, y, and z.
        lon : Longitude ranging from 0.0 to 360.0 degrees.
        lat : Latitude ranging from -90.0 to 90.0 degrees with 0 at the equator.
        vrho : Radial velocity in the same units as rho per arbitrary unit of time
            (same unit of time as the x, y, and z velocities).
        vlon : Longitudinal velocity in degrees per arbitrary unit of time
            (same unit of time as the x, y, and z velocities).
        vlat :Latitudinal velocity in degrees per arbitrary unit of time.
            (same unit of time as the x, y, and z velocities).
    """
    coords_spherical = jnp.zeros(6, dtype=jnp.float64)
    x = coords_cartesian[0]
    y = coords_cartesian[1]
    z = coords_cartesian[2]
    vx = coords_cartesian[3]
    vy = coords_cartesian[4]
    vz = coords_cartesian[5]

    rho = jnp.sqrt(x**2 + y**2 + z**2)
    lon = jnp.arctan2(y, x)
    lon = jnp.where(lon < 0.0, 2 * jnp.pi + lon, lon)
    lat = jnp.arcsin(z / rho)
    lat = jnp.where((lat >= 3*jnp.pi/2) & (lat <= 2*jnp.pi), lat - 2*jnp.pi, lat)

    vrho = (x * vx + y * vy + z * vz) / rho
    vlon = (vy * x - vx * y) / (x**2 + y**2)
    vlat = (vz - vrho * z / rho) / jnp.sqrt(x**2 + y**2)

    coords_spherical = coords_spherical.at[0].set(rho)
    coords_spherical = coords_spherical.at[1].set(jnp.degrees(lon))
    coords_spherical = coords_spherical.at[2].set(jnp.degrees(lat))
    coords_spherical = coords_spherical.at[3].set(vrho)
    coords_spherical = coords_spherical.at[4].set(jnp.degrees(vlon))
    coords_spherical = coords_spherical.at[5].set(jnp.degrees(vlat))

    return coords_spherical

@jit
def cartesian_to_spherical(coords_cartesian: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """
    Convert Cartesian coordinates to a spherical coordinates.

    Parameters
    ----------
    coords_cartesian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N, 6)
        3D Cartesian coordinates including time derivatives.
        x : x-position in units of distance.
        y : y-position in units of distance.
        z : z-position in units of distance.
        vx : x-velocity in the same units of x per arbitrary unit of time.
        vy : y-velocity in the same units of y per arbitrary unit of time.
        vz : z-velocity in the same units of z per arbitrary unit of time.

    Returns
    -------
    coords_spherical : ~jax.numpy.ndarray` (N, 6)
        3D Spherical coordinates including time derivatives.
        rho : Radial distance in the same units of x, y, and z.
        lon : Longitude ranging from 0.0 to 360.0 degrees.
        lat : Latitude ranging from -90.0 to 90.0 degrees with 0 at the equator.
        vrho : Radial velocity in the same units as rho per arbitrary unit of time
            (same unit of time as the x, y, and z velocities).
        vlon : Longitudinal velocity in degrees per arbitrary unit of time
            (same unit of time as the x, y, and z velocities).
        vlat : Latitudinal velocity in degrees per arbitrary unit of time.
            (same unit of time as the x, y, and z velocities).
    """
    with loops.Scope() as s:
        N = len(coords_cartesian)
        s.arr = jnp.zeros((N, 6), dtype=jnp.float64)

        for i in s.range(s.arr.shape[0]):
            s.arr = s.arr.at[i].set(
                _cartesian_to_spherical(
                    coords_cartesian[i],
                )
            )

        coords_spherical = s.arr

    return coords_spherical

@jit
def _spherical_to_cartesian(coords_spherical: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """
    Convert a single spherical coordinate to a Cartesian coordinate.

    Parameters
    ----------
    coords_spherical : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (6)
        3D Spherical coordinate including time derivatives.
        rho : Radial distance in the same units of x, y, and z.
        lon : Longitude ranging from 0.0 to 360.0 degrees.
        lat : Latitude ranging from -90.0 to 90.0 degrees with 0 at the equator.
        vrho : Radial velocity in the same units as rho per arbitrary unit of time
            (same unit of time as the x, y, and z velocities).
        vlon : Longitudinal velocity in degrees per arbitrary unit of time
            (same unit of time as the x, y, and z velocities).
        vlat : Latitudinal velocity in degrees per arbitrary unit of time.
            (same unit of time as the x, y, and z velocities).

    Returns
    -------
    coords_cartesian : `~jax.numpy.ndarray` (6)
        3D Cartesian coordinate including time derivatives.
        x : x-position in units of distance.
        y : y-position in units of distance.
        z : z-position in units of distance.
        vx : x-velocity in the same units of x per arbitrary unit of time.
        vy : y-velocity in the same units of y per arbitrary unit of time.
        vz : z-velocity in the same units of z per arbitrary unit of time.
    """
    coords_cartesian = jnp.zeros(6, dtype=jnp.float64)
    rho = coords_spherical[0]
    lon = jnp.radians(coords_spherical[1])
    lat = jnp.radians(coords_spherical[2])
    vrho = coords_spherical[3]
    vlon = jnp.radians(coords_spherical[4])
    vlat = jnp.radians(coords_spherical[5])

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

    coords_cartesian = coords_cartesian.at[0].set(x)
    coords_cartesian = coords_cartesian.at[1].set(y)
    coords_cartesian = coords_cartesian.at[2].set(z)
    coords_cartesian = coords_cartesian.at[3].set(vx)
    coords_cartesian = coords_cartesian.at[4].set(vy)
    coords_cartesian = coords_cartesian.at[5].set(vz)

    return coords_cartesian

@jit
def spherical_to_cartesian(coords_spherical: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    coords_spherical : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N, 6)
        3D Spherical coordinates including time derivatives.
        rho : Radial distance in the same units of x, y, and z.
        lon : Longitude ranging from 0.0 to 360.0 degrees.
        lat : Latitude ranging from -90.0 to 90.0 degrees with 0 at the equator.
        vrho : Radial velocity in the same units as rho per arbitrary unit of time
            (same unit of time as the x, y, and z velocities).
        vlon : Longitudinal velocity in degrees per arbitrary unit of time
            (same unit of time as the x, y, and z velocities).
        vlat :Latitudinal velocity in degrees per arbitrary unit of time.
            (same unit of time as the x, y, and z velocities).

    Returns
    -------
    coords_cartesian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N, 6)
        3D Cartesian coordinates including time derivatives.
        x : x-position in units of distance.
        y : y-position in units of distance.
        z : z-position in units of distance.
        vx : x-velocity in the same units of x per arbitrary unit of time.
        vy : y-velocity in the same units of y per arbitrary unit of time.
        vz : z-velocity in the same units of z per arbitrary unit of time.
    """
    with loops.Scope() as s:
        N = len(coords_spherical)
        s.arr = jnp.zeros((N, 6), dtype=jnp.float64)

        for i in s.range(s.arr.shape[0]):
            s.arr = s.arr.at[i].set(
                _spherical_to_cartesian(
                    coords_spherical[i],
                )
            )

        coords_cartesian = s.arr

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

        coords_cartesian = spherical_to_cartesian(self.coords.filled())
        coords_cartesian = np.array(coords_cartesian)

        if self.covariances is not None:
            covariances_cartesian = transform_covariances_jacobian(
                self.coords.filled(),
                self.covariances.filled(),
                _spherical_to_cartesian
            )
        else:
            covariances_cartesian = None

        coords = CartesianCoordinates(
            x=coords_cartesian[:, 0],
            y=coords_cartesian[:, 1],
            z=coords_cartesian[:, 2],
            vx=coords_cartesian[:, 3],
            vy=coords_cartesian[:, 4],
            vz=coords_cartesian[:, 5],
            times=self.times,
            covariances=covariances_cartesian,
            origin=self.origin,
            frame=self.frame
        )
        return coords

    @classmethod
    def from_cartesian(cls, cartesian: CartesianCoordinates):

        coords_spherical = cartesian_to_spherical(cartesian.coords.filled())
        coords_spherical = np.array(coords_spherical)

        if cartesian.covariances is not None:
            covariances_spherical = transform_covariances_jacobian(
                cartesian.coords.filled(),
                cartesian.covariances.filled(),
                _cartesian_to_spherical
            )
        else:
            covariances_spherical = None

        coords = cls(
            rho=coords_spherical[:, 0],
            lon=coords_spherical[:, 1],
            lat=coords_spherical[:, 2],
            vrho=coords_spherical[:, 3],
            vlon=coords_spherical[:, 4],
            vlat=coords_spherical[:, 5],
            times=cartesian.times,
            covariances=covariances_spherical,
            origin=cartesian.origin,
            frame=cartesian.frame
        )

        return coords