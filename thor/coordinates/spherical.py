import warnings
import numpy as np
from numba import jit
from astropy.time import Time
from astropy import units as u
from typing import Optional

from .coordinates import Coordinates
from .cartesian import CartesianCoordinates

__all__ = [
    "_cartesian_to_spherical",
    "_spherical_to_cartesian",
    "SphericalCoordinates"
]

SPHERICAL_COLS = ["rho", "lon", "lat", "vrho", "vlon", "vlat"]
SPHERICAL_UNITS = [u.au, u.degree, u.degree, u.au / u.d, u.degree / u.d, u.degree / u.d]


@jit(["UniTuple(f8[:], 6)(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])"], nopython=True, cache=True)
def _cartesian_to_spherical(x, y, z, vx, vy, vz):
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
        vrho = np.zeros((len(rho)), dtype=np.float64)
        vlon = np.zeros((len(lon)), dtype=np.float64)
        vlat = np.zeros((len(lat)), dtype=np.float64)
    else:
        vrho = (x * vx + y * vy + z * vz) / rho
        vlon = (vy * x - vx * y) / (x**2 + y**2)
        vlat = (vz - vrho * z / rho) / np.sqrt(x**2 + y**2)

    return rho, lon, lat, vrho, vlon, vlat


@jit(["UniTuple(f8[:], 6)(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])"], nopython=True, cache=True)
def _spherical_to_cartesian(rho, lon, lat, vrho, vlon, vlat):
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
        vx = np.zeros((len(x)), dtype=np.float64)
        vy = np.zeros((len(y)), dtype=np.float64)
        vz = np.zeros((len(z)), dtype=np.float64)
    else:
        vx = cos_lat * cos_lon * vrho - rho * cos_lat * sin_lon * vlon - rho * sin_lat * cos_lon * vlat
        vy = cos_lat * sin_lon * vrho + rho * cos_lat * cos_lon * vlon - rho * sin_lat * sin_lon * vlat
        vz = sin_lat * vrho + rho * cos_lat * vlat

    return x, y, z, vx, vy, vz

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

        x, y, z, vx, vy, vz = _spherical_to_cartesian(
            self._rho.filled(),
            np.radians(self._lon).filled(),
            np.radians(self._lat).filled(),
            self._vrho.filled(),
            np.radians(self._vlon).filled(),
            np.radians(self._vlat).filled(),
        )

        if self.covariances is not None:
            warnings.warn("Covariance transformations have not been implemented yet.")

        coords = CartesianCoordinates(
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            times=self.times,
            covariances=None,
            origin=self.origin,
            frame=self.frame
        )
        return coords

    @classmethod
    def from_cartesian(cls, cartesian: CartesianCoordinates):

        rho, lon, lat, vrho, vlon, vlat = _cartesian_to_spherical(
            cartesian._x.filled(),
            cartesian._y.filled(),
            cartesian._z.filled(),
            cartesian._vx.filled(),
            cartesian._vy.filled(),
            cartesian._vz.filled(),
        )
        lon = np.radians(lon)
        lat = np.radians(lat)
        vlon = np.radians(vlon)
        vlat = np.radians(vlat)

        if cartesian.covariances is not None:
            warnings.warn("Covariance transformations have not been implemented yet.")

        coords = cls(
            rho=rho,
            lon=lon,
            lat=lat,
            vrho=vrho,
            vlon=vlon,
            vlat=vlat,
            times=cartesian.times,
            covariances=None,
            origin=cartesian.origin,
            frame=cartesian.frame
        )

        return coords
