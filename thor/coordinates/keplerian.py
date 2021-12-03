import warnings
import numpy as np
from numba import jit
from astropy.time import Time
from astropy import units as u
from typing import Optional

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

@jit(["UniTuple(f8[:], 8)(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8)"], nopython=True, cache=False)
def _cartesian_to_keplerian(x, y, z, vx, vy, vz, mu=MU):
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
    elements_cart : `~numpy.ndarray` (N, 6)
        Cartesian elements in units of au and au per day.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.

    Returns
    -------
    elements_kepler : `~numpy.ndarray (N, 8)
        Keplerian elements with angles in degrees and semi-major axis and pericenter distance
        in au.

    """
    N = len(x)
    a = np.zeros((N), dtype=np.float64)
    q = np.zeros((N), dtype=np.float64)
    e = np.zeros((N), dtype=np.float64)
    i = np.zeros((N), dtype=np.float64)
    raan = np.zeros((N), dtype=np.float64)
    ap = np.zeros((N), dtype=np.float64)
    M = np.zeros((N), dtype=np.float64)
    nu = np.zeros((N), dtype=np.float64)

    r = np.zeros((N, 3), dtype=np.float64)
    r[:, 0] = x
    r[:, 1] = y
    r[:, 2] = z
    v = np.zeros((N, 3), dtype=np.float64)
    v[:, 0] = vx
    v[:, 1] = vy
    v[:, 2] = vz

    for j in range(N):

        r_i = r[j]
        v_i = v[j]

        r_mag = np.linalg.norm(r_i)
        v_mag = np.linalg.norm(v_i)

        sme = v_mag**2 / 2 - mu / r_mag

        h = np.cross(r_i, v_i)
        h_mag = np.linalg.norm(h)

        n = np.cross(Z_AXIS, h)
        n_mag = np.linalg.norm(n)

        e_vec = ((v_mag**2 - mu / r_mag) * r_i - (np.dot(r_i, v_i)) * v_i) / mu
        e_i = np.linalg.norm(e_vec)

        if e_i != 0.0:
            a_i = mu / (-2 * sme)
            p_i = a_i * (1 - e_i**2)
            q_i = a_i * (1 - e_i)
        else:
            a_i = np.inf
            p_i = h_mag**2 / mu
            q_i = a_i

        i_i = np.arccos(h[2] / h_mag)

        raan_i = np.arccos(n[0] / n_mag)
        if n[1] < 0:
            raan_i = 2*np.pi - raan_i

        ap_i = np.arccos(np.dot(n, e_vec) / (n_mag * e_i))
        if e_vec[2] < 0:
            ap_i = 2*np.pi - ap_i

        nu_i = np.arccos(np.dot(e_vec, r_i) / (e_i * r_mag))
        if np.dot(r_i, v_i) < 0:
            nu_i = 2*np.pi - nu_i

        if e_i < 1.0:
            E = np.arctan2(np.sqrt(1 - e_i**2) * np.sin(nu_i), e_i + np.cos(nu_i))
            M_i = np.degrees(E - e_i * np.sin(E))
            if M_i < 0:
                M_i += 2*np.pi
        elif e_i == 1.0:
            raise ValueError("Parabolic orbits not yet implemented!")
        else:
            H = np.arcsinh(np.sin(nu_i) * np.sqrt(e_i**2 - 1) / (1 + e_i * np.cos(nu_i)))
            M_i = e_i * np.sinh(H) - H


        a[j] = a_i
        q[j] = q_i
        e[j] = e_i
        i[j] = i_i
        raan[j] = raan_i
        ap[j] = ap_i
        M[j] = M_i
        nu[j] = nu_i

    return a, q, e, i, raan, ap, M, nu

@jit(["UniTuple(f8[:], 6)(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8, i8, f8)"], nopython=True, cache=False)
def _keplerian_to_cartesian(a, e, i, raan, ap, M, mu=MU, max_iter=100, tol=1e-15):
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
    N = len(a)
    x = np.zeros((N), dtype=np.float64)
    y = np.zeros((N), dtype=np.float64)
    z = np.zeros((N), dtype=np.float64)
    vx = np.zeros((N), dtype=np.float64)
    vy = np.zeros((N), dtype=np.float64)
    vz = np.zeros((N), dtype=np.float64)

    i_rad = np.radians(i)
    raan_rad = np.radians(raan)
    ap_rad = np.radians(ap)
    M_rad = np.radians(M)

    for i in range(N):
        a_i = a[i]
        e_i = e[i]
        i_i = i_rad[i]
        raan_i = raan_rad[i]
        ap_i = ap_rad[i]
        M_i = M_rad[i]

        p_i = a_i * (1 - e_i**2)

        if e_i < 1.0:
            iterations = 0
            ratio = 1e10
            E = M_i

            while np.abs(ratio) > tol:
                f = E - e_i * np.sin(E) - M_i
                fp = 1 - e_i * np.cos(E)
                ratio = f / fp
                E -= ratio
                iterations += 1
                if iterations >= max_iter:
                    break

            nu = 2 * np.arctan2(np.sqrt(1 + e_i) * np.sin(E/2), np.sqrt(1 - e_i) * np.cos(E/2))

        elif e_i == 1.0:
            raise ValueError("Parabolic orbits not yet implemented!")

        else:
            iterations = 0
            ratio = 1e10
            H = M_i / (e_i - 1)

            while np.abs(ratio) > tol:
                f = M_i - e_i * np.sinh(H) + H
                fp =  e_i * np.cosh(H) - 1
                ratio = f / fp
                H += ratio
                iterations += 1
                if iterations >= max_iter:
                    break

            nu = 2 * np.arctan(np.sqrt(e_i + 1) * np.sinh(H / 2) / (np.sqrt(e_i - 1) * np.cosh(H / 2)))

        r_PQW = np.array([
            p_i * np.cos(nu) / (1 + e_i * np.cos(nu)),
            p_i * np.sin(nu) / (1 + e_i * np.cos(nu)),
            0
        ])

        v_PQW = np.array([
            -np.sqrt(mu/p_i) * np.sin(nu),
            np.sqrt(mu/p_i) * (e_i + np.cos(nu)),
            0
        ])

        cos_raan = np.cos(raan_i)
        sin_raan = np.sin(raan_i)
        cos_ap = np.cos(ap_i)
        sin_ap = np.sin(ap_i)
        cos_i = np.cos(i_i)
        sin_i = np.sin(i_i)

        P1 = np.array([
            [cos_ap, -sin_ap, 0.],
            [sin_ap, cos_ap, 0.],
            [0., 0., 1.],
        ])

        P2 = np.array([
            [1., 0., 0.],
            [0., cos_i, -sin_i],
            [0., sin_i, cos_i],
        ])

        P3 = np.array([
            [cos_raan, -sin_raan, 0.],
            [sin_raan, cos_raan, 0.],
            [0., 0., 1.],
        ])

        rotation_matrix = P3 @ P2 @ P1
        r = rotation_matrix @ r_PQW
        v = rotation_matrix @ v_PQW

        x[i] = r[0]
        y[i] = r[1]
        z[i] = r[2]
        vx[i] = v[0]
        vy[i] = v[1]
        vz[i] = v[2]

    return x, y, z, vx, vy, vz

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

        x, y, z, vx, vy, vz = _keplerian_to_cartesian(
            self._a.filled(),
            self._e.filled(),
            self._i.filled(),
            self._raan.filled(),
            self._ap.filled(),
            self._M.filled(),
            mu=MU,
            max_iter=100,
            tol=1e-15,
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
            covariances=None,
            origin=self.origin,
            frame=self.frame
        )

        return coords

    @classmethod
    def from_cartesian(cls, cartesian: CartesianCoordinates, mu=MU):

        a, q, e, i, raan, ap, M, nu = _cartesian_to_keplerian(
            cartesian._x.filled(),
            cartesian._y.filled(),
            cartesian._z.filled(),
            cartesian._vx.filled(),
            cartesian._vy.filled(),
            cartesian._vz.filled(),
            mu=mu,
        )

        if cartesian.covariances is not None:
            warnings.warn("Covariance transformations have not been implemented yet.")

        coords = cls(
            a=a,
            e=e,
            i=i,
            raan=raan,
            ap=ap,
            M=M,
            times=cartesian.times,
            covariances=None,
            origin=cartesian.origin,
            frame=cartesian.frame,
            mu=mu
        )

        return coords

