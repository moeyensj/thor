import numpy as np
import jax.numpy as jnp
from jax import (
    config,
    jit
)
from jax.experimental import loops
from astropy.time import Time
from astropy import units as u
from typing import (
    List,
    Optional,
    Union
)
from collections import OrderedDict

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

from ..constants import Constants as c
from ..dynamics.kepler import solve_kepler
from .coordinates import Coordinates
from .cartesian import CartesianCoordinates
from .covariances import transform_covariances_jacobian

__all__ = [
    "_cartesian_to_keplerian",
    "_cartesian_to_keplerian6",
    "cartesian_to_keplerian",
    "_keplerian_to_cartesian",
    "keplerian_to_cartesian",
    "KeplerianCoordinates"
]

KEPLERIAN_COLS = OrderedDict()
KEPLERIAN_UNITS = OrderedDict()
for i in ["a", "e", "i", "raan", "ap", "M"]:
    KEPLERIAN_COLS[i] = i
KEPLERIAN_UNITS["a"] = u.au
KEPLERIAN_UNITS["e"] = u.dimensionless_unscaled
KEPLERIAN_UNITS["i"] = u.deg
KEPLERIAN_UNITS["raan"] = u.deg
KEPLERIAN_UNITS["ap"] = u.deg
KEPLERIAN_UNITS["M"] = u.deg

MU = c.MU
Z_AXIS = jnp.array([0., 0., 1.])

@jit
def _cartesian_to_keplerian(
        coords_cartesian: Union[np.ndarray, jnp.ndarray],
        t0: float,
        mu: float = MU,
    ) -> jnp.ndarray:
    """
    Convert a single Cartesian coordinate to a Keplerian coordinate.

    Parameters
    ----------
    coords_cartesian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (6)
        3D Cartesian coordinate including time derivatives.
        x : x-position in units of au.
        y : y-position in units of au.
        z : z-position in units of au.
        vx : x-velocity in units of au per day.
        vy : y-velocity in units of au per day.
        vz : z-velocity in units of au per day.
    t0 : float (1)
        Epoch at which cometary elements are defined in MJD TDB.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.

    Returns
    -------
    coords_keplerian : `~jax.numpy.ndarray` (11)
        11D Keplerian coordinate.
        a : semi-major axis in au.
        q : periapsis distance in au.
        e : eccentricity.
        i : inclination in degrees.
        raan : Right ascension (longitude) of the ascending node in degrees.
        ap : argument of periapsis in degrees.
        M : mean anomaly in degrees.
        nu : true anomaly in degrees.
        n : mean motion in degrees per day.
        P : period in days.
        tp : time of pericenter passage in days.
    """
    with loops.Scope() as s:

        s.arr = np.zeros(11, dtype=jnp.float64)
        r = coords_cartesian[0:3]
        v = coords_cartesian[3:6]

        r_mag = jnp.linalg.norm(r)
        v_mag = jnp.linalg.norm(v)

        sme = v_mag**2 / 2 - mu / r_mag

        h = jnp.cross(r, v)
        h_mag = jnp.linalg.norm(h)

        n = jnp.cross(Z_AXIS, h)
        n_mag = jnp.linalg.norm(n)

        e_vec = ((v_mag**2 - mu / r_mag) * r - (jnp.dot(r, v)) * v) / mu
        e = jnp.linalg.norm(e_vec)

        for _ in s.cond_range(e != 1.0):
            a1 = mu / (-2 * sme)
            p1 = a1 * (1 - e**2)
            q1 = a1 * (1 - e)

        for _ in s.cond_range(e == 1.0):
            a2 = jnp.inf
            p2 = -h_mag**2 / mu
            q2 = a2

        a = jnp.where(e != 1.0, a1, a2)
        p = jnp.where(e != 1.0, p1, p2)
        q = jnp.where(e != 1.0, q1, q2)

        i = jnp.arccos(h[2] / h_mag)

        raan = jnp.arccos(n[0] / n_mag)
        raan = jnp.where(n[1] < 0, 2*jnp.pi - raan, raan)

        ap = jnp.arccos(jnp.dot(n, e_vec) / (n_mag * e))
        ap = jnp.where(e_vec[2] < 0, 2*jnp.pi - ap, ap)

        nu = jnp.arccos(jnp.dot(e_vec, r) / (e * r_mag))
        nu = jnp.where(jnp.dot(r, v) < 0, 2*jnp.pi - nu, nu)

        n = jnp.sqrt(mu / jnp.abs(a)**3)

        for _ in s.cond_range(e < 1.0):
            E = jnp.arctan2(jnp.sqrt(1 - e**2) * jnp.sin(nu), e + jnp.cos(nu))
            M_E = E - e * jnp.sin(E)
            M_E = jnp.where(M_E < 0.0, M_E + 2*jnp.pi, M_E)

        for _ in s.cond_range(e > 1.0):
            H = jnp.arcsinh(jnp.sin(nu) * jnp.sqrt(e**2 - 1) / (1 + e * jnp.cos(nu)))
            M_H = e * jnp.sinh(H) - H

        M = jnp.where(e < 1.0, M_E, M_H)
        P = 2*jnp.pi / n

        # If the mean anomaly is greater than 180 degrees
        # then the orbit is approaching pericenter passage
        # in which case the pericenter will occur in the future
        # in less than half a period. If the mean anomaly is less
        # than 180 degrees, then the orbit is ascending from pericenter
        # passage and the most recent pericenter was in the past.
        dtp = jnp.where(M > jnp.pi, P - M / n, - M / n)
        tp = t0 + dtp

        s.arr = s.arr.at[0].set(a)
        s.arr = s.arr.at[1].set(q)
        s.arr = s.arr.at[2].set(e)
        s.arr = s.arr.at[3].set(jnp.degrees(i))
        s.arr = s.arr.at[4].set(jnp.degrees(raan))
        s.arr = s.arr.at[5].set(jnp.degrees(ap))
        s.arr = s.arr.at[6].set(jnp.degrees(M))
        s.arr = s.arr.at[7].set(jnp.degrees(nu))
        s.arr = s.arr.at[8].set(jnp.degrees(n))
        s.arr = s.arr.at[9].set(P)
        s.arr = s.arr.at[10].set(tp)

        coords_keplerian = s.arr

    return coords_keplerian

@jit
def _cartesian_to_keplerian6(
        coords_cartesian: Union[np.ndarray, jnp.ndarray],
        t0: float,
        mu: float = MU,
    ) -> jnp.ndarray:
    """
    Limit conversion of Cartesian coordinates to Keplerian 6 fundamental coordinates.

    Parameters
    ----------
    coords_cartesian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (6)
        3D Cartesian coordinate including time derivatives.
        x : x-position in units of au.
        y : y-position in units of au.
        z : z-position in units of au.
        vx : x-velocity in units of au per day.
        vy : y-velocity in units of au per day.
        vz : z-velocity in units of au per day.
    t0 : float (1)
        Epoch at which cometary elements are defined in MJD TDB.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.

    Returns
    -------
    coords_keplerian : `~jax.numpy.ndarray` (6)
        6D Keplerian coordinate.
        a : semi-major axis in au.
        e : eccentricity.
        i : inclination in degrees.
        raan : Right ascension (longitude) of the ascending node in degrees.
        ap : argument of periapsis in degrees.
        M : mean anomaly in degrees.
    """
    coords_keplerian = _cartesian_to_keplerian(coords_cartesian, t0=t0, mu=mu)
    return coords_keplerian[jnp.array([0, 2, 3, 4, 5, 6])]

@jit
def cartesian_to_keplerian(
        coords_cartesian: Union[np.ndarray, jnp.ndarray],
        t0: Union[np.ndarray, jnp.ndarray],
        mu: float = MU,
    ) -> jnp.ndarray:
    """
    Convert Cartesian coordinates to Keplerian coordinates.

    Parameters
    ----------
    coords_cartesian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N, 6)
        3D Cartesian coordinates including time derivatives.
        x : x-position in units of au.
        y : y-position in units of au.
        z : z-position in units of au.
        vx : x-velocity in units of au per day.
        vy : y-velocity in units of au per day.
        vz : z-velocity in units of au per day.
    t0 : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N)
        Epoch at which cometary elements are defined in MJD TDB.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.

    Returns
    -------
    coords_keplerian : `~jax.numpy.ndarray` (N, 11)
        11D Keplerian coordinates.
        a : semi-major axis in au.
        q : periapsis distance in au.
        e : eccentricity.
        i : inclination in degrees.
        raan : Right ascension (longitude) of the ascending node in degrees.
        ap : argument of periapsis in degrees.
        M : mean anomaly in degrees.
        nu : true anomaly in degrees.
        n : mean motion in degrees per day.
        P : period in days.
        tp : time of pericenter passage in days.
    """
    with loops.Scope() as s:
        N = len(coords_cartesian)
        s.arr = jnp.zeros((N, 11), dtype=jnp.float64)

        for i in s.range(s.arr.shape[0]):
            s.arr = s.arr.at[i].set(
                _cartesian_to_keplerian(
                    coords_cartesian[i],
                    t0[i],
                    mu=mu
                )
            )

        coords_keplerian = s.arr

    return coords_keplerian

@jit
def _keplerian_to_cartesian(
        coords_keplerian: Union[np.ndarray, jnp.ndarray],
        mu: float = MU,
        max_iter: int = 100,
        tol: float = 1e-15
    ) -> jnp.ndarray:
    """
    Convert a single Keplerian coordinate to a Cartesian coordinate.

    Parameters
    ----------
    coords_keplerian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (6)
        6D Keplerian coordinate.
        a : semi-major axis in au.
        e : eccentricity.
        i : inclination in degrees.
        raan : Right ascension (longitude) of the ascending node in degrees.
        ap : argument of periapsis in degrees.
        M : mean anomaly in degrees.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.
    max_iter : int, optional
        Maximum number of iterations over which to converge. If number of iterations is
        exceeded, will use the value of the relevant anomaly at the last iteration.
    tol : float, optional
        Numerical tolerance to which to compute anomalies using the Newtown-Raphson
        method.

    Returns
    -------
    coords_cartesian : `~jax.numpy.ndarray` (6)
        3D Cartesian coordinate including time derivatives.
        x : x-position in units of au.
        y : y-position in units of au.
        z : z-position in units of au.
        vx : x-velocity in units of au per day.
        vy : y-velocity in units of au per day.
        vz : z-velocity in units of au per day.
    """
    with loops.Scope() as s:
        s.arr = jnp.zeros(6, dtype=jnp.float64)

        a = coords_keplerian[0]
        e = coords_keplerian[1]
        i = jnp.radians(coords_keplerian[2])
        raan = jnp.radians(coords_keplerian[3])
        ap = jnp.radians(coords_keplerian[4])
        M = jnp.radians(coords_keplerian[5])
        p = a * (1 - e**2)

        nu = solve_kepler(e, M, max_iter=max_iter, tol=tol)

        r_PQW = jnp.array([
            p * jnp.cos(nu) / (1 + e * jnp.cos(nu)),
            p * jnp.sin(nu) / (1 + e * jnp.cos(nu)),
            0
        ])

        v_PQW = jnp.array([
            -jnp.sqrt(mu/p) * jnp.sin(nu),
            jnp.sqrt(mu/p) * (e + jnp.cos(nu)),
            0
        ])

        cos_raan = jnp.cos(raan)
        sin_raan = jnp.sin(raan)
        cos_ap = jnp.cos(ap)
        sin_ap = jnp.sin(ap)
        cos_i = jnp.cos(i)
        sin_i = jnp.sin(i)

        P1 = jnp.array([
            [cos_ap, -sin_ap, 0.],
            [sin_ap, cos_ap, 0.],
            [0., 0., 1.],
        ],  dtype=jnp.float64
        )

        P2 = jnp.array([
            [1., 0., 0.],
            [0., cos_i, -sin_i],
            [0., sin_i, cos_i],
        ],  dtype=jnp.float64
        )

        P3 = jnp.array([
            [cos_raan, -sin_raan, 0.],
            [sin_raan, cos_raan, 0.],
            [0., 0., 1.],
        ],  dtype=jnp.float64
        )

        rotation_matrix = P3 @ P2 @ P1
        r = rotation_matrix @ r_PQW
        v = rotation_matrix @ v_PQW

        s.arr = s.arr.at[0].set(r[0])
        s.arr = s.arr.at[1].set(r[1])
        s.arr = s.arr.at[2].set(r[2])
        s.arr = s.arr.at[3].set(v[0])
        s.arr = s.arr.at[4].set(v[1])
        s.arr = s.arr.at[5].set(v[2])

        coords_cartesian = s.arr

    return coords_cartesian

@jit
def keplerian_to_cartesian(
        coords_keplerian: Union[np.ndarray, jnp.ndarray],
        mu: float = MU,
        max_iter: int = 100,
        tol: float = 1e-15
    ) -> jnp.ndarray:
    """
    Convert Keplerian coordinates to Cartesian coordinates.

    Parameters
    ----------
    coords_keplerian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N, 6)
        6D Keplerian coordinate.
        a : semi-major axis in au.
        e : eccentricity.
        i : inclination in degrees.
        raan : Right ascension (longitude) of the ascending node in degrees.
        ap : argument of periapsis in degrees.
        M : mean anomaly in degrees.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.
    max_iter : int, optional
        Maximum number of iterations over which to converge. If number of iterations is
        exceeded, will use the value of the relevant anomaly at the last iteration.
    tol : float, optional
        Numerical tolerance to which to compute anomalies using the Newtown-Raphson
        method.

    Returns
    -------
    coords_cartesian : `~jax.numpy.ndarray` (N, 6)
        3D Cartesian coordinates including time derivatives.
        x : x-position in units of au.
        y : y-position in units of au.
        z : z-position in units of au.
        vx : x-velocity in units of au per day.
        vy : y-velocity in units of au per day.
        vz : z-velocity in units of au per day.
    """
    with loops.Scope() as s:
        N = len(coords_keplerian)
        s.arr = jnp.zeros((N, 6), dtype=jnp.float64)

        for i in s.range(s.arr.shape[0]):
            s.arr = s.arr.at[i].set(
                _keplerian_to_cartesian(
                    coords_keplerian[i],
                    mu=mu,
                    max_iter=max_iter,
                    tol=tol
                )
            )

        coords_cartesian = s.arr

    return coords_cartesian

class KeplerianCoordinates(Coordinates):

    def __init__(
            self,
            a: Optional[np.ndarray] = None,
            e: Optional[np.ndarray] = None,
            i: Optional[np.ndarray] = None,
            raan: Optional[np.ndarray] = None,
            ap: Optional[np.ndarray] = None,
            M: Optional[np.ndarray] = None,
            times: Optional[Time] = None,
            covariances: Optional[np.ndarray] = None,
            origin: str = "heliocentric",
            frame: str = "ecliptic",
            names: OrderedDict = KEPLERIAN_COLS,
            units: OrderedDict = KEPLERIAN_UNITS,
            mu: float = MU,
        ):
        Coordinates.__init__(self,
            a=a,
            e=e,
            i=i,
            raan=raan,
            ap=ap,
            M=M,
            covariances=covariances,
            times=times,
            origin=origin,
            frame=frame,
            names=names,
            units=units
        )
        self._mu = mu

        return

    @property
    def a(self):
        return self._values[:, 0]

    @property
    def e(self):
        return self._values[:, 1]

    @property
    def i(self):
        return self._values[:, 2]

    @property
    def raan(self):
        return self._values[:, 3]

    @property
    def ap(self):
        return self._values[:, 4]

    @property
    def M(self):
        return self._values[:, 5]

    @property
    def q(self):
        # pericenter distance
        return self.a * (1 - self.e)

    @property
    def p(self):
        # apocenter distance
        return self.a * (1 + self.e)

    @property
    def mu(self):
        return self._mu

    def to_cartesian(self) -> CartesianCoordinates:

        coords_cartesian = keplerian_to_cartesian(
            self.values.filled(),
            mu=MU,
            max_iter=100,
            tol=1e-15,
        )
        coords_cartesian = np.array(coords_cartesian)

        if self.covariances is not None:
            covariances_cartesian = transform_covariances_jacobian(
                self.values.filled(),
                self.covariances.filled(),
                _keplerian_to_cartesian
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
            covariances=covariances_cartesian,
            origin=self.origin,
            frame=self.frame
        )

        return coords

    @classmethod
    def from_cartesian(cls, cartesian: CartesianCoordinates, mu=MU):

        coords_keplerian = cartesian_to_keplerian(
            cartesian.values.filled(),
            cartesian.times.tdb.mjd,
            mu=mu,
        )
        coords_keplerian = np.array(coords_keplerian)

        if cartesian.covariances is not None:
            covariances_keplerian = transform_covariances_jacobian(
                cartesian.values.filled(),
                cartesian.covariances.filled(),
                _cartesian_to_keplerian6,
                t0=cartesian.times.tdb.mjd,
                mu=mu,
            )
        else:
            covariances_keplerian = None

        coords = cls(
            a=coords_keplerian[:, 0],
            e=coords_keplerian[:, 2],
            i=coords_keplerian[:, 3],
            raan=coords_keplerian[:, 4],
            ap=coords_keplerian[:, 5],
            M=coords_keplerian[:, 6],
            times=cartesian.times,
            covariances=covariances_keplerian,
            origin=cartesian.origin,
            frame=cartesian.frame,
            mu=mu
        )

        return coords

    @classmethod
    def from_df(cls,
            df,
            coord_cols={
                "a" : "a",
                "e" : "e",
                "i" : "i",
                "raan" : "raan",
                "ap" : "ap",
                "M" : "M"
            },
            origin_col="origin"
        ):
        """
        Create a KeplerianCoordinates class from a dataframe.

        Parameters
        ----------
        df : `~pandas.DataFrame`
            Pandas DataFrame containing Keplerian coordinates and optionally their
            times and covariances.
        coord_cols : OrderedDict
            Ordered dictionary containing as keys the coordinate dimensions and their equivalent columns
            as values. For example,
                coord_cols = OrderedDict()
                coord_cols["a"] = Column name of semi-major axis values
                coord_cols["e"] = Column name of eccentricity values
                coord_cols["i"] = Column name of inclination values
                coord_cols["raan"] = Column name of longitude of ascending node values
                coord_cols["ap"] = Column name of argument of pericenter values
                coord_cols["M"] = Column name of mean anomaly values
        origin_col : str
            Name of the column containing the origin of each coordinate.
        """
        data = Coordinates._dict_from_df(
            df,
            coord_cols=coord_cols,
            origin_col=origin_col
        )
        return cls(**data)