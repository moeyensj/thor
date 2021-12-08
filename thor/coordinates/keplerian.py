import warnings
import numpy as np
import jax.numpy as jnp
from jax import config, jit
from jax.experimental import loops
from astropy.time import Time
from astropy import units as u
from typing import Optional

config.update("jax_enable_x64", True)

from ..constants import Constants as c
from .coordinates import Coordinates
from .cartesian import CartesianCoordinates

__all__ = [
    "_cartesian_to_keplerian",
    "cartesian_to_keplerian",
    "_keplerian_to_cartesian",
    "keplerian_to_cartesian",
    "KeplerianCoordinates"
]

KEPLERIAN_COLS = ["a", "e", "i", "raan", "argperi", "M"]

MU = c.MU
Z_AXIS = np.array([0., 0., 1.])

@jit
def _cartesian_to_keplerian(coords_cartesian, mu=MU):
    """
    Convert a single Cartesian coordinate to a Keplerian coordinate.

    Keplerian coordinates are returned in an array with the following elements:
        a : semi-major axis [au]
        e : eccentricity
        i : inclination [degrees]
        Omega : longitude of the ascending node [degrees]
        omega : argument of periapsis [degrees]
        M0 : mean anomaly [degrees]

    Parameters
    ----------
    coords_cartesian : `~numpy.ndarray` (6)
        Cartesian coordinate in units of au and au per day.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.

    Returns
    -------
    coords_keplerian : `~numpy.ndarray (8)
        Keplerian coordinate with angles in degrees and semi-major axis and pericenter distance
        in au.
    """
    with loops.Scope() as s:

        s.arr = np.zeros(8, dtype=jnp.float64)
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

        for _ in s.cond_range(e != 0.0):
            a1 = mu / (-2 * sme)
            p1 = a1 * (1 - e**2)
            q1 = a1 * (1 - e)

        for _ in s.cond_range(e == 0.0):
            a2 = jnp.inf
            p2 = h_mag**2 / mu
            q2 = a2

        a = jnp.where(e != 0.0, a1, a2)
        q = jnp.where(e != 0.0, q1, q2)

        i = jnp.arccos(h[2] / h_mag)

        raan = jnp.arccos(n[0] / n_mag)
        raan = jnp.where(n[1] < 0, 2*jnp.pi - raan, raan)

        ap = jnp.arccos(jnp.dot(n, e_vec) / (n_mag * e))
        ap = jnp.where(e_vec[2] < 0, 2*jnp.pi - ap, ap)

        nu = jnp.arccos(jnp.dot(e_vec, r) / (e * r_mag))
        nu = jnp.where(jnp.dot(r, v) < 0, 2*jnp.pi - nu, nu)

        for _ in s.cond_range(e < 1.0):
            E = jnp.arctan2(jnp.sqrt(1 - e**2) * jnp.sin(nu), e + jnp.cos(nu))
            M_E = E - e * jnp.sin(E)
            M_E = jnp.where(M_E < 0.0, M_E + 2*jnp.pi, M_E)

        for _ in s.cond_range(e > 1.0):
            H = jnp.arcsinh(jnp.sin(nu) * jnp.sqrt(e**2 - 1) / (1 + e * jnp.cos(nu)))
            M_H = e * jnp.sinh(H) - H

        M = jnp.where(e < 1.0, M_E, M_H)

        s.arr = s.arr.at[0].set(a)
        s.arr = s.arr.at[1].set(q)
        s.arr = s.arr.at[2].set(e)
        s.arr = s.arr.at[3].set(jnp.degrees(i))
        s.arr = s.arr.at[4].set(jnp.degrees(raan))
        s.arr = s.arr.at[5].set(jnp.degrees(ap))
        s.arr = s.arr.at[6].set(jnp.degrees(M))
        s.arr = s.arr.at[7].set(jnp.degrees(nu))

        coords_keplerian = s.arr

    return coords_keplerian

@jit
def cartesian_to_keplerian(coords_cartesian, mu=MU):
    """
    Convert Cartesian coordinates to Keplerian coordinates.

    Keplerian coordinates are returned in an array with the following elements:
        a : semi-major axis [au]
        e : eccentricity
        i : inclination [degrees]
        Omega : longitude of the ascending node [degrees]
        omega : argument of periapsis [degrees]
        M0 : mean anomaly [degrees]

    Parameters
    ----------
    coords_cartesian : `~numpy.ndarray` (N, 6)
        Cartesian coordinates in units of au and au per day.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.

    Returns
    -------
    coords_keplerian : `~numpy.ndarray (N, 8)
        Keplerian coordinates with angles in degrees and semi-major axis and pericenter distance
        in au.
    """
    with loops.Scope() as s:
        N = len(coords_cartesian)
        s.arr = jnp.zeros((N, 8), dtype=jnp.float64)

        for i in s.range(s.arr.shape[0]):
            s.arr = s.arr.at[i].set(
                _cartesian_to_keplerian(
                    coords_cartesian[i],
                     mu=mu
                )
            )

        coords_keplerian = s.arr

    return coords_keplerian

@jit
def _keplerian_to_cartesian(coords_keplerian, mu=MU, max_iter=100, tol=1e-15):
    """
    Convert a single Keplerian coordinate to a Cartesian coordinate.

    Keplerian coordinates should have following elements:
        a : semi-major axis [au]
        e : eccentricity [degrees]
        i : inclination [degrees]
        Omega : longitude of the ascending node [degrees]
        omega : argument of periapsis [degrees]
        M0 : mean anomaly [degrees]

    Parameters
    ----------
    coords_keplerian : `~numpy.ndarray` (6)
        Keplerian coordinate with angles in degrees and semi-major axis in au.
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
    coords_cartesian : `~numpy.ndarray (6)
        Cartesian coordinate in units of au and au per day.
    """
    with loops.Scope() as s:

        a = coords_keplerian[0]
        e = coords_keplerian[1]
        i = jnp.radians(coords_keplerian[2])
        raan = jnp.radians(coords_keplerian[3])
        ap = jnp.radians(coords_keplerian[4])
        M = jnp.radians(coords_keplerian[5])
        p = a * (1 - e**2)

        s.arr = jnp.zeros(6, dtype=jnp.float64)
        for _ in s.cond_range(e < 1.0):

            with loops.Scope() as ss:
                ratio = 1e10
                enit = M
                ss.arr = jnp.array([enit, ratio,], dtype=jnp.float64)
                ss.idx = 0
                for _ in ss.while_range(lambda : (ss.idx < max_iter) & (ss.arr[1] > tol)):
                    f = ss.arr[0] - e * jnp.sin(ss.arr[0]) - M
                    fp = 1 - e * jnp.cos(ss.arr[0])
                    ratio = f / fp
                    ss.arr = ss.arr.at[0].set(ss.arr[0]-ratio)
                    ss.arr = ss.arr.at[1].set(jnp.abs(ratio))
                    ss.idx += 1

                E = ss.arr[0]
                nu_E = 2 * jnp.arctan2(jnp.sqrt(1 + e) * jnp.sin(E/2), jnp.sqrt(1 - e) * jnp.cos(E/2))

        for _ in s.cond_range(e > 1.0):

            with loops.Scope() as ss:
                ratio = 1e10
                H_init = M / (e - 1)
                ss.arr = jnp.array([H_init, ratio], dtype=jnp.float64)
                ss.idx = 0
                for _ in ss.while_range(lambda : (ss.idx < max_iter) & (ss.arr[1] > tol)):
                    f = M - e * jnp.sinh(ss.arr[0]) + ss.arr[0]
                    fp =  e * jnp.cosh(ss.arr[0]) - 1
                    ratio = f / fp
                    ss.arr = ss.arr.at[0].set(ss.arr[0]+ratio)
                    ss.arr = ss.arr.at[1].set(jnp.abs(ratio))
                    ss.idx += 1

                H = ss.arr[0]
                nu_H = 2 * jnp.arctan(jnp.sqrt(e + 1) * jnp.sinh(H / 2) / (jnp.sqrt(e - 1) * jnp.cosh(H / 2)))

        nu = jnp.where(
            e < 1.0, nu_E,
            jnp.where(e > 1.0, nu_H, jnp.nan)
        )

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
def keplerian_to_cartesian(coords_keplerian, mu=MU, max_iter=100, tol=1e-15):
    """
    Convert Keplerian coordinates to Cartesian coordinates.

    Keplerian coordinates should have following elements:
        a : semi-major axis [au]
        e : eccentricity [degrees]
        i : inclination [degrees]
        Omega : longitude of the ascending node [degrees]
        omega : argument of periapsis [degrees]
        M0 : mean anomaly [degrees]

    Parameters
    ----------
    coords_keplerian : `~numpy.ndarray` (N, 6)
        Keplerian coordinates with angles in degrees and semi-major axis in au.
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
    coords_cartesian : `~numpy.ndarray (N, 6)
        Cartesian coordinates in units of au and au per day.
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
            mu: float = MU,
        ):
        Coordinates.__init__(self,
            a,
            e,
            i,
            raan,
            ap,
            M,
            covariances=covariances,
            times=times,
            origin=origin,
            frame=frame,
            names=KEPLERIAN_COLS
        )

        self._a = self._coords[:, 0]
        self._e = self._coords[:, 1]
        self._i = self._coords[:, 2]
        self._raan = self._coords[:, 3]
        self._ap = self._coords[:, 4]
        self._M = self._coords[:, 5]
        self._mu = mu

        return

    @property
    def a(self):
        return self._a

    @property
    def e(self):
        return self._e

    @property
    def i(self):
        return self._i

    @property
    def raan(self):
        return self._raan

    @property
    def ap(self):
        return self._ap

    @property
    def M(self):
        return self._M

    @property
    def mu(self):
        return self._mu

    def to_cartesian(self) -> CartesianCoordinates:

        coords_cartesian = keplerian_to_cartesian(
            self.coords.filled(),
            mu=MU,
            max_iter=100,
            tol=1e-15,
        )
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
            covariances=None,
            origin=self.origin,
            frame=self.frame
        )

        return coords

    @classmethod
    def from_cartesian(cls, cartesian: CartesianCoordinates, mu=MU):

        coords_keplerian = cartesian_to_keplerian(
            cartesian.coords.filled(),
            mu=mu,
        )
        coords_keplerian = np.array(coords_keplerian)

        if cartesian.covariances is not None:
            warnings.warn("Covariance transformations have not been implemented yet.")

        coords = cls(
            a=coords_keplerian[:, 0],
            e=coords_keplerian[:, 2],
            i=coords_keplerian[:, 3],
            raan=coords_keplerian[:, 4],
            ap=coords_keplerian[:, 5],
            M=coords_keplerian[:, 6],
            times=cartesian.times,
            covariances=None,
            origin=cartesian.origin,
            frame=cartesian.frame,
            mu=mu
        )

        return coords

