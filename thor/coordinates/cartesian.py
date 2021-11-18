import numpy as np
from astropy.time import Time
from astropy import units as u
from typing import Optional

from .coordinates import Coordinates

__all__ = [
    "CartesianCoordinates"
]

CARTESIAN_COLS = ["x", "y", "z", "vx", "vy", "vz"]
CARTESIAN_UNITS = [u.au, u.au, u.au, u.au / u.d, u.au / u.d, u.au / u.d]

class CartesianCoordinates(Coordinates):

    def __init__(
            self,
            x: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None,
            z: Optional[np.ndarray] = None,
            vx: Optional[np.ndarray] = None,
            vy: Optional[np.ndarray] = None,
            vz: Optional[np.ndarray] = None,
            times: Optional[Time] = None,
            covariances: Optional[np.ndarray] = None,
            origin: str = "heliocentric",
            frame: str = "ecliptic"
        ):
        """

        Parameters
        ----------
        x : `~numpy.ndarray` (N)
            X-coordinate in units of au.
        y : `~numpy.ndarray` (N)
            Y-coordinate in units of au.
        z : `~numpy.ndarray` (N)
            Z-coordinate in units of au.
        vx : `~numpy.ndarray` (N)
            X-coordinate velocity in in units of au per day.
        vy : `~numpy.ndarray` (N)
            Y-coordinate velocity in in units of au per day.
        vz : `~numpy.ndarray` (N)
            Z-coordinate velocity in in units of au per day.
        """
        Coordinates.__init__(self,
            x,
            y,
            z,
            vx,
            vy,
            vz,
            covariances=covariances,
            times=times,
            origin=origin,
            frame=frame,
            names=CARTESIAN_COLS
        )

        self._x = self._coords[:, 0]
        self._y = self._coords[:, 1]
        self._z = self._coords[:, 2]
        self._vx = self._coords[:, 3]
        self._vy = self._coords[:, 4]
        self._vz = self._coords[:, 5]
        return

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def vx(self):
        return self._vx

    @property
    def vy(self):
        return self._vy

    @property
    def vz(self):
        return self._vz

    def to_cartesian(self):
        return self

    @classmethod
    def from_cartesian(cls, cartesian):
        return cartesian
