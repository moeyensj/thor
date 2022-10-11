import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import (
    config,
    jit,
    vmap
)
from astropy.time import Time
from astropy import units as u
from typing import (
    Optional,
    Union
)
from collections import OrderedDict

config.update("jax_enable_x64", True)

from .coordinates import Coordinates
from .cartesian import CartesianCoordinates
from .covariances import transform_covariances_jacobian

__all__ = [
    "_cartesian_to_bernstein_khushalani",
    "_bernstein_khushalani_to_cartesian",
    "BernsteinKhushalaniCoordinates",
    "BERNSTEIN_KHUSHALANI_COLS",
    "BERNSTEIN_KHUSHALANI_UNITS"
]

BERNSTEIN_KHUSHALANI_COLS = OrderedDict()
BERNSTEIN_KHUSHALANI_UNITS = OrderedDict()
for i in ["gamma", "alpha", "beta", "vgamma", "valpha", "vbeta",]:
    BERNSTEIN_KHUSHALANI_COLS[i] = i
BERNSTEIN_KHUSHALANI_UNITS["gamma"] = 1 / u.au
BERNSTEIN_KHUSHALANI_UNITS["alpha"] = u.dimensionless_unscaled
BERNSTEIN_KHUSHALANI_UNITS["beta"] = u.dimensionless_unscaled
BERNSTEIN_KHUSHALANI_UNITS["vgamma"] = 1 / u.d
BERNSTEIN_KHUSHALANI_UNITS["valpha"] = 1 / u.d
BERNSTEIN_KHUSHALANI_UNITS["vbeta"] = 1 / u.d

@jit
def _cartesian_to_bernstein_khushalani(coords_cartesian: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """
    Convert a single Cartesian coordinate to a Bernstein-Khushalani coordinate.

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
    coords_bernstein_khushalani : `~jax.numpy.ndarray` (6)
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
    coords_bernstein_khushalani = jnp.zeros(6, dtype=jnp.float64)
    x = coords_cartesian[0]
    y = coords_cartesian[1]
    z = coords_cartesian[2]
    vx = coords_cartesian[3]
    vy = coords_cartesian[4]
    vz = coords_cartesian[5]

    r = jnp.sqrt(x**2 + y**2 + z**2)
    vr = (x*vx + y*vy + z*vz) / r

    alpha = x / r
    valpha = vx / r
    beta = y / r
    vbeta = vy / r
    gamma = 1. / r
    vgamma = vr / r

    coords_bernstein_khushalani = coords_bernstein_khushalani.at[0].set(gamma)
    coords_bernstein_khushalani = coords_bernstein_khushalani.at[1].set(alpha)
    coords_bernstein_khushalani = coords_bernstein_khushalani.at[2].set(beta)
    coords_bernstein_khushalani = coords_bernstein_khushalani.at[3].set(vgamma)
    coords_bernstein_khushalani = coords_bernstein_khushalani.at[4].set(valpha)
    coords_bernstein_khushalani = coords_bernstein_khushalani.at[5].set(vbeta)

    return coords_bernstein_khushalani

# Vectorization Map: _cartesian_to_bernstein_khushalani
_cartesian_to_bernstein_khushalani_vmap = vmap(
    _cartesian_to_bernstein_khushalani,
    in_axes=(0,),
)

def cartesian_to_bernstein_khushalani(coords_cartesian: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """
    Convert Cartesian coordinates to a Bernstein-Khushalani coordinates.

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
    coords_bernstein_khushalani : ~jax.numpy.ndarray` (N, 6)
        3D Spherical coordinates including time derivatives.
        gamma : Radial distance in the same units of x, y, and z.
        alpha : Longitude ranging from 0.0 to 360.0 degrees.
        beta : Latitude ranging from -90.0 to 90.0 degrees with 0 at the equator.
        vgamma : Radial velocity in the same units as rho per arbitrary unit of time
            (same unit of time as the x, y, and z velocities).
        valpha : Longitudinal velocity in degrees per arbitrary unit of time
            (same unit of time as the x, y, and z velocities).
        vbeta : Latitudinal velocity in degrees per arbitrary unit of time.
            (same unit of time as the x, y, and z velocities).
    """
    coords_bernstein_khushalani = _cartesian_to_bernstein_khushalani_vmap(coords_cartesian)
    return coords_bernstein_khushalani

@jit
def _bernstein_khushalani_to_cartesian(coords_bernstein_khushalani: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """
    Convert a single Bernstein-Khushalani coordinate to a Cartesian coordinate.

    Parameters
    ----------
    coords_bernstein_khushalani : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (6)
        3D Spherical coordinate including time derivatives.
        gamma : Inverse radial distance in the same units of x, y, and z inversed.
        alpha : Radially normalized longitude ranging from 0.0 to 360.0 degrees.
        beta : Radially normalized latitude ranging from -90.0 to 90.0 degrees with 0 at the equator.
        vgamma : Radially normalized radial velocity in the same units as gamma per arbitrary unit of time
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
    gamma = coords_bernstein_khushalani[0]
    alpha = coords_bernstein_khushalani[1]
    beta = coords_bernstein_khushalani[2]
    vgamma = coords_bernstein_khushalani[3]
    valpha = coords_bernstein_khushalani[4]
    vbeta = coords_bernstein_khushalani[5]

    r = 1. / gamma
    x = alpha * r
    y = beta * r
    vx = valpha * r
    vy = vbeta * r
    vr = vgamma * r

    z = jnp.sqrt(r**2 - x**2 - y**2)
    vz = (vr * r - vx * x - vy * y) / z

    coords_cartesian = coords_cartesian.at[0].set(x)
    coords_cartesian = coords_cartesian.at[1].set(y)
    coords_cartesian = coords_cartesian.at[2].set(z)
    coords_cartesian = coords_cartesian.at[3].set(vx)
    coords_cartesian = coords_cartesian.at[4].set(vy)
    coords_cartesian = coords_cartesian.at[5].set(vz)

    return coords_cartesian

# Vectorization Map: _bernstein_khushalani_to_cartesian
_bernstein_khushalani_to_cartesian_vmap = vmap(
    _bernstein_khushalani_to_cartesian,
    in_axes=(0,),
)

def bernstein_khushalani_to_cartesian(coords_bernstein_khushalani: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """
    Convert Bernstein-Khushalani coordinates to Cartesian coordinates.

    Parameters
    ----------
    coords_bernstein_khushalani : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N, 6)
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
    coords_cartesian = _bernstein_khushalani_to_cartesian_vmap(coords_bernstein_khushalani)
    return coords_cartesian

class BernsteinKhushalaniCoordinates(Coordinates):

    def __init__(
            self,
            gamma: Optional[Union[int, float, np.ndarray]] = None,
            alpha: Optional[Union[int, float, np.ndarray]] = None,
            beta: Optional[Union[int, float, np.ndarray]] = None,
            vgamma: Optional[Union[int, float, np.ndarray]] = None,
            valpha: Optional[Union[int, float, np.ndarray]] = None,
            vbeta: Optional[Union[int, float, np.ndarray]] = None,
            times: Optional[Time] = None,
            covariances: Optional[np.ndarray] = None,
            sigma_gamma: Optional[np.ndarray] = None,
            sigma_alpha: Optional[np.ndarray] = None,
            sigma_beta: Optional[np.ndarray] = None,
            sigma_vgamma: Optional[np.ndarray] = None,
            sigma_valpha: Optional[np.ndarray] = None,
            sigma_vbeta: Optional[np.ndarray] = None,
            origin: str = "heliocenter",
            frame: str = "ecliptic",
            names: OrderedDict = BERNSTEIN_KHUSHALANI_COLS,
            units: OrderedDict = BERNSTEIN_KHUSHALANI_UNITS
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
        sigmas = (
            sigma_gamma, sigma_alpha, sigma_beta,
            sigma_vgamma, sigma_valpha, sigma_vbeta
        )
        Coordinates.__init__(self,
            gamma=gamma,
            alpha=alpha,
            beta=beta,
            vgamma=vgamma,
            valpha=valpha,
            vbeta=vbeta,
            covariances=covariances,
            sigmas=sigmas,
            times=times,
            origin=origin,
            frame=frame,
            names=names,
            units=units,
        )
        return

    @property
    def gamma(self):
        """
        Inverse radial distance
        """
        return self._values[:, 0]

    @property
    def alpha(self):
        """
        Radially normalized longitude
        """
        return self._values[:, 1]

    @property
    def beta(self):
        """
        Radially normalized latitude
        """
        return self._values[:, 2]

    @property
    def vgamma(self):
        """
        Radially normalized radial velocity
        """
        return self._values[:, 3]

    @property
    def valpha(self):
        """
        Radially normalized longitudinal velocity
        """
        return self._values[:, 4]

    @property
    def vbeta(self):
        """
        Radially normalized latitudinal velocity
        """
        return self._values[:, 5]

    @property
    def sigma_gamma(self):
        """
        1-sigma uncertainty in inverse radial distance
        """
        return self.sigmas[:, 0]

    @property
    def sigma_alpha(self):
        """
        1-sigma uncertainty in radially normalized longitude
        """
        return self.sigmas[:, 1]

    @property
    def sigma_beta(self):
        """
        1-sigma uncertainty in radially normalized latitude
        """
        return self.sigmas[:, 2]

    @property
    def sigma_vgamma(self):
        """
        1-sigma uncertainty in radially normalized radial velocity
        """
        return self.sigmas[:, 3]

    @property
    def sigma_valpha(self):
        """
        1-sigma uncertainty in radially normalized longitudinal velocity
        """
        return self.sigmas[:, 4]

    @property
    def sigma_vbeta(self):
        """
        1-sigma uncertainty in radially normalized latitudinal velocity
        """
        return self.sigmas[:, 5]

    def to_cartesian(self) -> CartesianCoordinates:

        coords_cartesian = bernstein_khushalani_to_cartesian(self.values.filled())
        coords_cartesian = np.array(coords_cartesian)

        if self.covariances is not None:
            covariances_cartesian = transform_covariances_jacobian(
                self.values,
                self.covariances,
                _bernstein_khushalani_to_cartesian
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
            frame=self.frame,
        )
        return coords

    @classmethod
    def from_cartesian(cls,
            cartesian: CartesianCoordinates
        ) -> "BernsteinKhushalaniCoordinates":

        coords_bernstein_khushalani = cartesian_to_bernstein_khushalani(cartesian.values.filled())
        coords_bernstein_khushalani = np.array(coords_bernstein_khushalani)

        if cartesian.covariances is not None and (~np.all(cartesian.covariances.mask)):
            covariances_bernstein_khushalani = transform_covariances_jacobian(
                cartesian.values,
                cartesian.covariances,
                _cartesian_to_bernstein_khushalani
            )
        else:
            covariances_bernstein_khushalani = None

        coords = cls(
            gamma=coords_bernstein_khushalani[:, 0],
            alpha=coords_bernstein_khushalani[:, 1],
            beta=coords_bernstein_khushalani[:, 2],
            vgamma=coords_bernstein_khushalani[:, 3],
            valpha=coords_bernstein_khushalani[:, 4],
            vbeta=coords_bernstein_khushalani[:, 5],
            times=cartesian.times,
            covariances=covariances_bernstein_khushalani,
            origin=cartesian.origin,
            frame=cartesian.frame
        )

        return coords

    @classmethod
    def from_df(cls,
            df: pd.DataFrame,
            coord_cols: OrderedDict = BERNSTEIN_KHUSHALANI_COLS,
            origin_col: str = "origin",
            frame_col: str = "frame"
        ) -> "BernsteinKhushalaniCoordinates":
        """
        Create a BernsteinKhushalaniCoordinates class from a dataframe.

        Parameters
        ----------
        df : `~pandas.DataFrame`
            Pandas DataFrame containing Bernstein-Khushalani coordinates and optionally their
            times and covariances.
        coord_cols : OrderedDict
            Ordered dictionary containing as keys the coordinate dimensions and their equivalent columns
            as values. For example,
                coord_cols = OrderedDict()
                coord_cols["gamma"] = Column name of inverse radial distance values
                coord_cols["alpha"] = Column name of radially normalized longitudinal values
                coord_cols["beta"] = Column name of radially normalized latitudinal values
                coord_cols["vgamma"] = Column name of the radially normalized radial velocity values
                coord_cols["valpha"] = Column name of radially normalized longitudinal velocity values
                coord_cols["vbeta"] = Column name of radially normalized latitudinal velocity values
        origin_col : str
            Name of the column containing the origin of each coordinate.
        frame_col : str
            Name of the column containing the coordinate frame.
        """
        data = Coordinates._dict_from_df(
            df,
            coord_cols=coord_cols,
            origin_col=origin_col,
            frame_col=frame_col
        )
        return cls(**data)
