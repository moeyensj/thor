import numpy as np
from astropy.time import Time
from astropy import units as u
from typing import (
    List,
    Optional
)
from collections import OrderedDict

from ..constants import Constants as c
from .coordinates import Coordinates

__all__ = [
    "CartesianCoordinates"
]
TRANSFORM_EQ2EC = np.zeros((6, 6))
TRANSFORM_EQ2EC[0:3, 0:3] = c.TRANSFORM_EQ2EC
TRANSFORM_EQ2EC[3:6, 3:6] = c.TRANSFORM_EQ2EC
TRANSFORM_EC2EQ = TRANSFORM_EQ2EC.T

CARTESIAN_COLS = OrderedDict()
for i in ["x", "y", "z", "vx", "vy", "vz"]:
    CARTESIAN_COLS[i] = i
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
            frame: str = "ecliptic",
            names: List[str] = CARTESIAN_COLS,
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
            names=names
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

    @property
    def r(self):
        return self._coords[:, 0:3]

    @property
    def v(self):
        return self._coords[:, 3:6]

    def to_cartesian(self):
        return self

    @classmethod
    def from_cartesian(cls, cartesian):
        return cartesian

    def _rotate(self, matrix):

        coords = self._coords.dot(matrix.T)
        coords.mask = self._coords.mask
        coords.fill_value = np.NaN

        if self._covariances is not None:
            covariances = np.ma.zeros(self._covariances.shape)
            for i, cov in enumerate(self._covariances):
                covariances[i] = matrix @ cov @ matrix.T
        else:
            covariances = None

        return coords, covariances

    def to_equatorial(self):
        if self.frame == "equatorial":
            return self
        elif self.frame == "ecliptic":
            coords, covariances = self._rotate(TRANSFORM_EC2EQ)
            data = {}
            data["x"] = coords[:, 0].filled()
            data["y"] = coords[:, 1].filled()
            data["z"] = coords[:, 2].filled()
            data["vx"] = coords[:, 3].filled()
            data["vy"] = coords[:, 4].filled()
            data["vz"] = coords[:, 5].filled()
            data["times"] = self.times
            data["covariances"] = covariances
            data["origin"] = self.origin
            data["frame"] = "ecliptic"
            return CartesianCoordinates(**data)
        else:
            raise ValueError

    def to_ecliptic(self):
        if self.frame == "equatorial":
            coords, covariances = self._rotate(TRANSFORM_EQ2EC)
            data = {}
            data["x"] = coords[:, 0].filled()
            data["y"] = coords[:, 1].filled()
            data["z"] = coords[:, 2].filled()
            data["vx"] = coords[:, 3].filled()
            data["vy"] = coords[:, 4].filled()
            data["vz"] = coords[:, 5].filled()
            data["times"] = self.times
            data["covariances"] = covariances
            data["origin"] = self.origin
            data["frame"] = "equatorial"
            return CartesianCoordinates(**data)
        elif self.frame == "ecliptic":
            return self
        else:
            raise ValueError

    @classmethod
    def from_df(cls,
            df,
            coord_cols=CARTESIAN_COLS,
            covariance_col="cartesian_covariances",
            origin_col="origin"
        ):
        """
        Create a CartesianCoordinates class from a dataframe.

        Parameters
        ----------
        df : `~pandas.DataFrame`
            Pandas DataFrame containing Cartesian coordinates and optionally their
            times and covariances.
        coord_cols : OrderedDict
            Ordered dictionary containing as keys the coordinate dimensions and their equivalent columns
            as values. For example,
                coord_cols = OrderedDict()
                coord_cols["x"] = Column name of x distance values
                coord_cols["y"] = Column name of y distance values
                coord_cols["z"] = Column name of z distance values
                coord_cols["vx"] = Column name of x velocity values
                coord_cols["vy"] = Column name of y velocity values
                coord_cols["vz"] = Column name of z velocity values
        covariance_col : str
            Name of the column containing covariance matrices.
        origin_col : str
            Name of the column containing the origin of each coordinate.
        """
        data = Coordinates._dict_from_df(
            df,
            coord_cols=coord_cols,
            covariance_col=covariance_col,
            origin_col=origin_col
        )
        return cls(**data)
