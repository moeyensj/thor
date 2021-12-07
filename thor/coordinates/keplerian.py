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
    "_keplerian_to_cartesian",
    "KeplerianCoordinates"
]

KEPLERIAN_COLS = ["a", "e", "i", "raan", "argperi", "M"]

MU = c.MU
Z_AXIS = np.array([0., 0., 1.])

@jit
def _cartesian_to_keplerian(coords_cartesian, mu=MU):
    """
    Convert cartesian orbital elements to Keplerian orbital elements.

    Keplerian orbital elements are returned in an array with the following elements:
        a : semi-major axis [au]
        e : eccentricity
        i : inclination [degrees]
        Omega : longitude of the ascending node [degrees]
        omega : argument of periapsis [degrees]
        M0 : mean anomaly [degrees]

    Parameters
    ----------
    coords_cartesian : `~numpy.ndarray` (N, 6)
        Cartesian elements in units of au and au per day.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.

    Returns
    -------
    coords_keplerian : `~numpy.ndarray (N, 8)
        Keplerian elements with angles in degrees and semi-major axis and pericenter distance
        in au.

    """
    with loops.Scope() as s:
        N = len(coords_cartesian)
        r = coords_cartesian[:, 0:3]
        v = coords_cartesian[:, 3:6]

        s.arr = jnp.zeros_like((N, 8), dtype=jnp.float64)

        for i in s.range(s.arr.shape[0]):

            r_i = r[i]
            v_i = v[i]

            r_mag = jnp.linalg.norm(r_i)
            v_mag = jnp.linalg.norm(v_i)

            sme = v_mag**2 / 2 - mu / r_mag

            h = jnp.cross(r_i, v_i)
            h_mag = jnp.linalg.norm(h)

            n = jnp.cross(Z_AXIS, h)
            n_mag = jnp.linalg.norm(n)

            e_vec = ((v_mag**2 - mu / r_mag) * r_i - (jnp.dot(r_i, v_i)) * v_i) / mu
            e_i = jnp.linalg.norm(e_vec)

            for _ in s.cond_range(e_i != 0.0):
                a_i = mu / (-2 * sme)
                p_i = a_i * (1 - e_i**2)
                q_i = a_i * (1 - e_i)

            for _ in s.cond_range(e_i == 0.0):
                a_i = jnp.inf
                p_i = h_mag**2 / mu
                q_i = a_i

            i_i = jnp.arccos(h[2] / h_mag)

            raan_i = jnp.arccos(n[0] / n_mag)
            raan_i = jnp.where(n[1] < 0, 2*jnp.pi - raan_i, raan_i)

            ap_i = jnp.arccos(jnp.dot(n, e_vec) / (n_mag * e_i))
            ap_i = jnp.where(e_vec[2] < 0, 2*jnp.pi - ap_i, ap_i)

            nu_i = jnp.arccos(jnp.dot(e_vec, r_i) / (e_i * r_mag))
            nu_i = jnp.where(jnp.dot(r_i, v_i) < 0, 2*jnp.pi - nu_i, nu_i)

            for _ in s.cond_range(e_i < 1.0):
                E = jnp.arctan2(jnp.sqrt(1 - e_i**2) * jnp.sin(nu_i), e_i + jnp.cos(nu_i))
                M_i = jnp.degrees(E - e_i * jnp.sin(E))
                if M_i < 0:
                    M_i += 2*jnp.pi

            for _ in s.cond_range(e_i > 1.0):
                H = jnp.arcsinh(jnp.sin(nu_i) * jnp.sqrt(e_i**2 - 1) / (1 + e_i * jnp.cos(nu_i)))
                M_i = e_i * jnp.sinh(H) - H

            s.arr = s.arr.at[i, 0].set(a_i)
            s.arr = s.arr.at[i, 1].set(q_i)
            s.arr = s.arr.at[i, 2].set(e_i)
            s.arr = s.arr.at[i, 3].set(i_i)
            s.arr = s.arr.at[i, 4].set(raan_i)
            s.arr = s.arr.at[i, 5].set(ap_i)
            s.arr = s.arr.at[i, 6].set(M_i)
            s.arr = s.arr.at[i, 7].set(nu_i)

        coords_keplerian = s.arr

    return coords_keplerian

@jit
def _cartesian_to_keplerian6(coords_cartesian, mu=MU):
    coords_keplerian = _cartesian_to_keplerian(coords_cartesian, mu=mu)
    return coords_keplerian[:, [0, 2, 3, 4, 5, 6]]

@jit
def _keplerian_to_cartesian(coords_keplerian, mu=MU, max_iter=100, tol=1e-15):
    """
    Convert Keplerian orbital elements to cartesian orbital elements.

    Keplerian orbital elements should have following elements:
        a : semi-major axis [au]
        e : eccentricity [degrees]
        i : inclination [degrees]
        Omega : longitude of the ascending node [degrees]
        omega : argument of periapsis [degrees]
        M0 : mean anomaly [degrees]

    Parameters
    ----------
    elements_kepler : `~numpy.ndarray` (N, 6)
        Keplerian elements with angles in degrees and semi-major
        axis in au.
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
    elements_cart : `~numpy.ndarray (N, 6)
        Cartesian elements in units of au and au per day.
    """
    with loops.Scope() as s:

        a = coords_keplerian[:, 0]
        e = coords_keplerian[:, 1]
        i = coords_keplerian[:, 2]
        raan = coords_keplerian[:, 3]
        ap = coords_keplerian[:, 4]
        M = coords_keplerian[:, 5]

        i_rad = jnp.radians(i)
        raan_rad = jnp.radians(raan)
        ap_rad = jnp.radians(ap)
        M_rad = jnp.radians(M)

        s.arr = jnp.zeros_like(coords_keplerian, dtype=jnp.float64)

        for i in s.range(s.arr.shape[0]):
            a_i = a[i]
            e_i = e[i]
            i_i = i_rad[i]
            raan_i = raan_rad[i]
            ap_i = ap_rad[i]
            M_i = M_rad[i]

            p_i = a_i * (1 - e_i**2)

            for _ in s.cond_range(e_i < 1.0):

                with loops.Scope() as ss:
                    ratio = 1e10
                    E_init = M_i
                    ss.arr = jnp.array([E_init, ratio,], dtype=jnp.float64)
                    ss.idx = 0
                    for j in ss.while_range(lambda : (ss.idx < max_iter) & (ss.arr[1] > tol)):
                        f = ss.arr[0] - e_i * jnp.sin(ss.arr[0]) - M_i
                        fp = 1 - e_i * jnp.cos(ss.arr[0])
                        ratio = f / fp
                        ss.arr = ss.arr.at[0].set(ss.arr[0]-ratio)
                        ss.arr = ss.arr.at[1].set(jnp.abs(ratio))
                        ss.idx += 1

                    E = ss.arr[0]
                    nu_E = 2 * jnp.arctan2(jnp.sqrt(1 + e_i) * jnp.sin(E/2), jnp.sqrt(1 - e_i) * jnp.cos(E/2))

            for _ in s.cond_range(e_i > 1.0):

                with loops.Scope() as ss:
                    ratio = 1e10
                    H_init = M_i / (e_i - 1)
                    ss.arr = jnp.array([H_init, ratio], dtype=jnp.float64)
                    ss.idx = 0
                    for j in ss.while_range(lambda : (ss.idx < max_iter) & (ss.arr[1] > tol)):
                        f = M_i - e_i * jnp.sinh(ss.arr[0]) + ss.arr[0]
                        fp =  e_i * jnp.cosh(ss.arr[0]) - 1
                        ratio = f / fp
                        ss.arr = ss.arr.at[0].set(ss.arr[0]+ratio)
                        ss.arr = ss.arr.at[1].set(jnp.abs(ratio))
                        ss.idx += 1

                    H = ss.arr[0]
                    nu_H = 2 * jnp.arctan(jnp.sqrt(e_i + 1) * jnp.sinh(H / 2) / (jnp.sqrt(e_i - 1) * jnp.cosh(H / 2)))

            nu = jnp.where(
                e_i < 1.0, nu_E,
                jnp.where(e_i > 1.0, nu_H, jnp.nan)
            )

            r_PQW = jnp.array([
                p_i * jnp.cos(nu) / (1 + e_i * jnp.cos(nu)),
                p_i * jnp.sin(nu) / (1 + e_i * jnp.cos(nu)),
                0
            ])

            v_PQW = jnp.array([
                -jnp.sqrt(mu/p_i) * jnp.sin(nu),
                jnp.sqrt(mu/p_i) * (e_i + jnp.cos(nu)),
                0
            ])

            cos_raan = jnp.cos(raan_i)
            sin_raan = jnp.sin(raan_i)
            cos_ap = jnp.cos(ap_i)
            sin_ap = jnp.sin(ap_i)
            cos_i = jnp.cos(i_i)
            sin_i = jnp.sin(i_i)

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

            s.arr = s.arr.at[i, 0].set(r[0])
            s.arr = s.arr.at[i, 1].set(r[1])
            s.arr = s.arr.at[i, 2].set(r[2])
            s.arr = s.arr.at[i, 3].set(v[0])
            s.arr = s.arr.at[i, 4].set(v[1])
            s.arr = s.arr.at[i, 5].set(v[2])

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

        coords_cartesian = _keplerian_to_cartesian(
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

        coords_keplerian = _cartesian_to_keplerian(
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

