import logging
import numpy as np
from astropy.time import Time
from astropy import units as u
from typing import (
    Optional,
    Union
)
from copy import deepcopy
from collections import OrderedDict

from ..constants import Constants as c
from ..utils.spice import get_perturber_state
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

COVARIANCE_ROTATION_TOLERANCE = 1e-25

logger = logging.getLogger(__name__)

class CartesianCoordinates(Coordinates):

    def __init__(
            self,
            x: Optional[Union[int, float, np.ndarray]] = None,
            y: Optional[Union[int, float, np.ndarray]] = None,
            z: Optional[Union[int, float, np.ndarray]] = None,
            vx: Optional[Union[int, float, np.ndarray]] = None,
            vy: Optional[Union[int, float, np.ndarray]] = None,
            vz: Optional[Union[int, float, np.ndarray]] = None,
            times: Optional[Time] = None,
            covariances: Optional[np.ndarray] = None,
            sigma_x: Optional[np.ndarray] = None,
            sigma_y: Optional[np.ndarray] = None,
            sigma_z: Optional[np.ndarray] = None,
            sigma_vx: Optional[np.ndarray] = None,
            sigma_vy: Optional[np.ndarray] = None,
            sigma_vz: Optional[np.ndarray] = None,
            origin: str = "heliocenter",
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
    def sigma_r(self):
        return np.sqrt(np.sum(self.sigmas.filled()[:, 0:3]**2, axis=1))

    @property
    def sigma_v(self):
        return np.sqrt(np.sum(self.sigmas.filled()[:, 3:6]**2, axis=1))

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

    def rotate(self,
            matrix: np.ndarray,
            frame_out: str
        ) -> "CartesianCoordinates":
        """
        Rotate Cartesian coordinates and their covariances by the
        given rotation matrix. A copy is made of the coordinates and a new
        instance of the CartesianCoordinates class is returned.

        Covariance matrices are also rotated. Rotations will sometimes result
        in covariance matrix elements very near zero but not exactly zero. Any
        elements that are smaller than +-1e-25 are rounded down to 0.

        Parameters
        ----------
        matrix : `~numpy.ndarray` (6, 6)
            Rotation matrix.
        frame_out : str
            Name of the frame to which coordinates are being rotated.

        Returns
        -------
        CartesianCoordinates : `~thor.coordinates.cartesian.CartesianCoordinates`
            Rotated Cartesian coordinates and their covariances.
        """
        coords_rotated = deepcopy(np.ma.dot(self._values, matrix.T))
        coords_rotated[self._values.mask] = np.NaN

        if self._covariances is not None:
            covariances_rotated = deepcopy(matrix @ self._covariances @ matrix.T)
            near_zero = len(covariances_rotated[np.abs(covariances_rotated) < COVARIANCE_ROTATION_TOLERANCE])
            if near_zero > 0:
                logger.debug(f"{near_zero} covariance elements are within {COVARIANCE_ROTATION_TOLERANCE:.0e} of zero after rotation, setting these elements to 0.")
                covariances_rotated = np.where(np.abs(covariances_rotated) < COVARIANCE_ROTATION_TOLERANCE, 0, covariances_rotated)

        else:
            covariances_rotated = None

        data = {}
        data["x"] = coords_rotated[:, 0]
        data["y"] = coords_rotated[:, 1]
        data["z"] = coords_rotated[:, 2]
        data["vx"] = coords_rotated[:, 3]
        data["vy"] = coords_rotated[:, 4]
        data["vz"] = coords_rotated[:, 5]
        data["times"] = deepcopy(self.times)
        data["covariances"] = covariances_rotated
        data["origin"] = deepcopy(self.origin)
        data["frame"] = deepcopy(frame_out)
        data["units"] = deepcopy(self.units)
        data["names"] = deepcopy(self.names)
        return CartesianCoordinates(**data)

    def to_frame(self, frame: str):
        """
        Rotate Cartesian coordinates and their covariances to the given frame.

        Parameters
        ----------
        frame : {'ecliptic', 'equatorial'}
            Desired reference frame of the output coordinates.

        Returns
        -------
        CartesianCoordinates : `~thor.coordinates.cartesian.CartesianCoordinates`
            Rotated Cartesian coordinates and their covariances.
        """
        if frame == "ecliptic" and self.frame != "ecliptic":
            return self.rotate(TRANSFORM_EC2EQ, "ecliptic")
        elif frame == "equatorial" and self.frame != "equatorial":
            return self.rotate(TRANSFORM_EQ2EC, "equatorial")
        elif frame == self.frame:
            return self
        else:
            err = (
                "frame should be one of {'ecliptic', 'equatorial'}"
            )
            raise ValueError(err)

        return

    def translate(self,
            vectors: Union[np.ndarray, np.ma.masked_array],
            origin_out: str
        ) -> "CartesianCoordinates":
        """
        Translate CartesianCoordinates by the given coordinate vector(s).
        A copy is made of the coordinates and a new instance of the
        CartesianCoordinates class is returned.

        Translation will only be applied to those coordinates that do not already
        have the desired origin (self.origin != origin_out).

        Parameters
        ----------
        vectors : {`~numpy.ndarray`, `~numpy.ma.masked_array`} (N, 6), (1, 6) or (6)
            Translation vector(s) for each coordinate or a single vector with which
            to translate all coordinates.
        origin_out : str
            Name of the origin to which coordinates are being translated.

        Returns
        -------
        CartesianCoordinates : `~thor.coordinates.cartesian.CartesianCoordinates`
            Translated Cartesian coordinates and their covariances.

        Raises
        ------
        ValueError: If vectors does not have shape (N, 6), (1, 6), or (6)
        TypeError: If vectors is not a `~numpy.ndarray` or a `~numpy.ma.masked_array`
        """
        if not isinstance(vectors, (np.ndarray, np.ma.masked_array)):
            err = (
                "coords should be one of {`~numpy.ndarray`, `~numpy.ma.masked_array`}"
            )
            raise TypeError(err)

        if len(vectors.shape) == 2:
            N, D = vectors.shape
        elif len(vectors.shape) == 1:
            N, D = vectors.shape[0], None
        else:
            err = (
                f"vectors should be 2D or 1D, instead vectors is {len(vectors.shape)}D."
            )
            raise ValueError(err)

        N_self, D_self = self.values.shape
        if (N != len(self) and N != 1) and (D is None and N != D_self):
            err = (
                f"Translation vector(s) should have shape ({N_self}, {D_self}), (1, {D_self}) or ({D_self},).\n"
                f"Given translation vector(s) has shape {vectors.shape}."
            )
            raise ValueError(err)

        coords_translated = deepcopy(self.values)

        # Only apply translation to coordinates that do not already have the desired origin
        origin_different_mask = np.where(self.origin != origin_out)[0]
        origin_same_mask = np.where(self.origin == origin_out)[0]
        if len(coords_translated[origin_same_mask]) > 0:
            info = (
                f"Translation will not be applied to the {len(coords_translated[origin_same_mask])} coordinates that already have the desired origin."
            )
            logger.info(info)

        if len(vectors.shape) == 2:
            coords_translated[origin_different_mask] = coords_translated[origin_different_mask] + vectors[origin_different_mask]
        else:
            coords_translated[origin_different_mask] = coords_translated[origin_different_mask] + vectors

        covariances_translated = deepcopy(self.covariances)

        data = {}
        data["x"] = coords_translated[:, 0]
        data["y"] = coords_translated[:, 1]
        data["z"] = coords_translated[:, 2]
        data["vx"] = coords_translated[:, 3]
        data["vy"] = coords_translated[:, 4]
        data["vz"] = coords_translated[:, 5]
        data["times"] = deepcopy(self.times)
        data["covariances"] = covariances_translated
        data["origin"] = deepcopy(origin_out)
        data["frame"] = deepcopy(self.frame)
        data["units"] = deepcopy(self.units)
        data["names"] = deepcopy(self.names)
        return CartesianCoordinates(**data)

    def to_origin(self, origin: str):
        """
        Translate coordinates to a different origin.

        Parameters
        ----------
        origin : {'heliocenter', 'barycenter'}
            Name of the desired origin.

        Returns
        -------
        CartesianCoordinates : `~thor.coordinates.cartesian.CartesianCoordinates`
            Translated Cartesian coordinates and their covariances.
        """
        unique_origins = np.unique(self.origin)
        vectors = np.zeros((len(self), 6), dtype=np.float64)

        for origin_in in unique_origins:

            mask = np.where(self.origin == origin_in)[0]

            vectors[mask] = get_perturber_state(
                origin_in,
                self.times[mask],
                frame=self.frame,
                origin=origin
            )

        return self.translate(vectors, origin)

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
