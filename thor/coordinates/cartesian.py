import numpy as np
from astropy.time import Time
from astropy import units as u
from typing import (
    Optional
)
from collections import OrderedDict

from ..constants import Constants as c
from .coordinates import Coordinates

__all__ = [
    "CartesianCoordinates",
    "CARTESIAN_COLS",
    "CARTESIAN_UNITS"
]

TRANSFORM_EQ2EC = np.zeros((6, 6))
TRANSFORM_EQ2EC[0:3, 0:3] = c.TRANSFORM_EQ2EC
TRANSFORM_EQ2EC[3:6, 3:6] = c.TRANSFORM_EQ2EC
TRANSFORM_EC2EQ = TRANSFORM_EQ2EC.T

CARTESIAN_COLS = OrderedDict()
CARTESIAN_UNITS = OrderedDict()
for i in ["x", "y", "z"]:
    CARTESIAN_COLS[i] = i
    CARTESIAN_UNITS[i] = u.au
for i in ["vx", "vy", "vz"]:
    CARTESIAN_COLS[i] = i
    CARTESIAN_UNITS[i] = u.au / u.d

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
            sigma_x: Optional[np.ndarray] = None,
            sigma_y: Optional[np.ndarray] = None,
            sigma_z: Optional[np.ndarray] = None,
            sigma_vx: Optional[np.ndarray] = None,
            sigma_vy: Optional[np.ndarray] = None,
            sigma_vz: Optional[np.ndarray] = None,
            origin: str = "heliocentric",
            frame: str = "ecliptic",
            names: OrderedDict = CARTESIAN_COLS,
            units: OrderedDict = CARTESIAN_UNITS,
        ):
        """

        Parameters
        ----------
        x : `~numpy.ndarray` (N)
            X-coordinate.
        y : `~numpy.ndarray` (N)
            Y-coordinate.
        z : `~numpy.ndarray` (N)
            Z-coordinate.
        vx : `~numpy.ndarray` (N)
            X-coordinate velocity.
        vy : `~numpy.ndarray` (N)
            Y-coordinate velocity.
        vz : `~numpy.ndarray` (N)
            Z-coordinate velocity.
        """
        sigmas = (
            sigma_x, sigma_y, sigma_z,
            sigma_vx, sigma_vy, sigma_vz
        )
        Coordinates.__init__(self,
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            covariances=covariances,
            sigmas=sigmas,
            times=times,
            origin=origin,
            frame=frame,
            names=names,
            units=units
        )
        return

    @property
    def x(self):
        return self._values[:, 0]

    @property
    def y(self):
        return self._values[:, 1]

    @property
    def z(self):
        return self._values[:, 2]

    @property
    def vx(self):
        return self._values[:, 3]

    @property
    def vy(self):
        return self._values[:, 4]

    @property
    def vz(self):
        return self._values[:, 5]

    @property
    def sigma_x(self):
        return self.sigmas[:, 0]

    @property
    def sigma_y(self):
        return self.sigmas[:, 1]

    @property
    def sigma_z(self):
        return self.sigmas[:, 2]

    @property
    def sigma_vx(self):
        return self.sigmas[:, 3]

    @property
    def sigma_vy(self):
        return self.sigmas[:, 4]

    @property
    def sigma_vz(self):
        return self.sigmas[:, 5]

    @property
    def r(self):
        return self._values[:, 0:3]

    @property
    def v(self):
        return self._values[:, 3:6]

    @property
    def r_mag(self):
        return np.linalg.norm(self.r.filled(), axis=1)

    @property
    def v_mag(self):
        return np.linalg.norm(self.v.filled(), axis=1)

    @property
    def r_hat(self):
        return self.r.filled() / self.r_mag.reshape(-1, 1)

    @property
    def v_hat(self):
        return self.v.filled() / self.v_mag.reshape(-1, 1)

    def to_cartesian(self):
        return self

    @classmethod
    def from_cartesian(cls, cartesian):
        return cartesian

    def _rotate(self, matrix):

        coords = self._values.dot(matrix.T)
        coords.mask = self._values.mask
        coords.fill_value = np.NaN

        if self._covariances is not None:
            covariances = matrix @ self._covariances @ matrix.T
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
            data["frame"] = "equatorial"
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
            data["frame"] = "ecliptic"
            return CartesianCoordinates(**data)
        elif self.frame == "ecliptic":
            return self
        else:
            raise ValueError

    @classmethod
    def from_df(cls,
            df,
            coord_cols=CARTESIAN_COLS,
            origin_col="origin",
            frame_col="frame",
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
        origin_col : str
            Name of the column containing the origin of each coordinate.
        """
        data = Coordinates._dict_from_df(
            df,
            coord_cols=coord_cols,
            origin_col=origin_col,
            frame_col=frame_col
        )
        return cls(**data)
