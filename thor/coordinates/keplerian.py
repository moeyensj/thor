import logging
import numpy as np
import jax.numpy as jnp
from jax import (
    config,
    jit,
    lax,
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

from ..constants import Constants as c
from ..dynamics.kepler import (
    calc_mean_anomaly,
    solve_kepler,
)
from .coordinates import Coordinates
from .cartesian import CartesianCoordinates
from .covariances import transform_covariances_jacobian

__all__ = [
    "_cartesian_to_keplerian",
    "_cartesian_to_keplerian6",
    "cartesian_to_keplerian",
    "_keplerian_to_cartesian",
    "keplerian_to_cartesian",
    "KeplerianCoordinates",
    "KEPLERIAN_COLS",
    "KEPLERIAN_UNITS"
]

logger = logging.getLogger(__name__)

FLOAT_TOLERANCE = 1e-15

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

    If the orbit is found to be circular (e = 0 +- 1e-15) then
    the argument of periapsis is set to 0. The anomalies are then accordingly
    defined with this assumption.

    If the orbit's inclination is zero or 180 degrees (i = 0 +- 1e-15 or i = 180 +- 1e-15),
    then the longitude of the ascending node is set to 0 (located in the direction of
    the reference axis).

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
    coords_keplerian : `~jax.numpy.ndarray` (13)
        13D Keplerian coordinate.
        a : semi-major axis in au.
        p : semi-latus rectum in au.
        q : periapsis distance in au.
        Q : apoapsis distance in au.
        e : eccentricity.
        i : inclination in degrees.
        raan : Right ascension (longitude) of the ascending node in degrees.
        ap : argument of periapsis in degrees.
        M : mean anomaly in degrees.
        nu : true anomaly in degrees.
        n : mean motion in degrees per day.
        P : period in days.
        tp : time of pericenter passage in days.

    References
    ----------
    [1] Bate, R. R; Mueller, D. D; White, J. E. (1971). Fundamentals of Astrodynamics. 1st ed.,
        Dover Publications, Inc. ISBN-13: 978-0486600611
    """
    coords_keplerian = jnp.zeros(13, dtype=jnp.float64)
    r = coords_cartesian[0:3]
    v = coords_cartesian[3:6]

    r_mag = jnp.linalg.norm(r)
    v_mag = jnp.linalg.norm(v)

    sme = v_mag**2 / 2 - mu / r_mag

    # Calculate the angular momentum vector
    # Equation 2.4-1 in Bate, Mueller, & White [1]
    h = jnp.cross(r, v)
    h_mag = jnp.linalg.norm(h)

    # Calculate the vector which is perpendicular to the
    # momentum vector and the Z-axis and points towards
    # the direction of the ascending node.
    # Equation 2.4-3 in Bate, Mueller, & White [1]
    n = jnp.cross(Z_AXIS, h)
    n_mag = jnp.linalg.norm(n)

    # Calculate the eccentricity vector which lies in the orbital plane
    # and points toward periapse.
    # Equation 2.4-5 in Bate, Mueller, & White [1]
    e_vec = ((v_mag**2 - mu / r_mag) * r - (jnp.dot(r, v)) * v) / mu
    e = jnp.linalg.norm(e_vec)

    # Calculate the semi-latus rectum
    p = h_mag**2 / mu

    # Calculate the inclination
    # Equation 2.4-7 inin Bate, Mueller, & White [1]
    i = jnp.arccos(h[2] / h_mag)

    # Calculate the longitude of the ascending node
    # Equation 2.4-8 in Bate, Mueller, & White [1]
    raan = jnp.arccos(n[0] / n_mag)
    raan = jnp.where(n[1] < 0, 2*jnp.pi - raan, raan)
    # In certain conventions when the orbit is zero inclined or 180 inclined
    # the ascending node is set to 0 as opposed to being undefined. This is what
    # SPICE does so we will do the same.
    raan = jnp.where(
         (i < FLOAT_TOLERANCE) | (jnp.abs(i - 2*jnp.pi) < FLOAT_TOLERANCE),
         0.,
         raan
    )

    # Calculate the argument of pericenter
    # Equation 2.4-9 in Bate, Mueller, & White [1]
    ap = jnp.arccos(jnp.dot(n, e_vec) / (n_mag * e))
    # Adopt convention that if the orbit is circular the argument of
    # periapsis is set to 0
    ap = jnp.where(e_vec[2] < 0, 2*jnp.pi - ap, ap)
    ap = jnp.where(jnp.abs(e) < FLOAT_TOLERANCE, 0., ap)

    # Calculate true anomaly (undefined for
    # circular orbits)
    # Equation 2.4-10 in Bate, Mueller, & White [1]
    nu = jnp.arccos(jnp.dot(e_vec, r) / (e * r_mag))
    nu = jnp.where(jnp.dot(r, v) < 0, 2*jnp.pi - nu, nu)
    #nu = jnp.where(jnp.abs(e) < FLOAT_TOLERANCE, jnp.nan, nu)

    # Calculate the semi-major axis (undefined for parabolic
    # orbits)
    a = jnp.where(
        (e > (1.0 - FLOAT_TOLERANCE)) & (e < (1.0 + FLOAT_TOLERANCE)),
        jnp.nan,
        mu / (-2 * sme)
    )

    # Calculate the periapsis distance
    q = jnp.where(
        (e > (1.0 - FLOAT_TOLERANCE)) & (e < (1.0 + FLOAT_TOLERANCE)),
        p / 2,
        a * (1 - e)
    )

    # Calculate the apoapsis distance (infinite for
    # parabolic and hyperbolic orbits)
    Q = jnp.where(
        e < 1.0,
        a * (1 + e),
        jnp.inf
    )

    # Calculate the mean anomaly
    M = calc_mean_anomaly(nu, e)

    # Calculate the mean motion
    n = lax.cond(
        (e > (1.0 - FLOAT_TOLERANCE)) & (e < (1.0 + FLOAT_TOLERANCE)),
        lambda a, q: jnp.sqrt(mu / (2 * q**3)),
        lambda a, q: jnp.sqrt(mu / jnp.abs(a)**3),
        a,
        q,
    )

    # Calculate the orbital period which for parabolic and hyperbolic
    # orbits is infinite while for all closed orbits
    # is well defined.
    P = lax.cond(
        e < 1.0,
        lambda n: 2*jnp.pi / n,
        lambda n: jnp.inf,
        n
    )

    # In the case of closed orbits, if the mean anomaly is
    # greater than 180 degrees then the orbit is
    # approaching pericenter passage in which case
    # the pericenter will occur in the future
    # in less than half a period. If the mean anomaly is less
    # than 180 degrees, then the orbit is ascending from pericenter
    # passage and the most recent pericenter was in the past.
    dtp = M / n
    dtp = jnp.where((M > jnp.pi) & (e < 1.0), P - M / n, - M / n)
    tp = t0 + dtp

    coords_keplerian = coords_keplerian.at[0].set(a)
    coords_keplerian = coords_keplerian.at[1].set(p)
    coords_keplerian = coords_keplerian.at[2].set(q)
    coords_keplerian = coords_keplerian.at[3].set(Q)
    coords_keplerian = coords_keplerian.at[4].set(e)
    coords_keplerian = coords_keplerian.at[5].set(jnp.degrees(i))
    coords_keplerian = coords_keplerian.at[6].set(jnp.degrees(raan))
    coords_keplerian = coords_keplerian.at[7].set(jnp.degrees(ap))
    coords_keplerian = coords_keplerian.at[8].set(jnp.degrees(M))
    coords_keplerian = coords_keplerian.at[9].set(jnp.degrees(nu))
    coords_keplerian = coords_keplerian.at[10].set(jnp.degrees(n))
    coords_keplerian = coords_keplerian.at[11].set(P)
    coords_keplerian = coords_keplerian.at[12].set(tp)

    return coords_keplerian

# Vectorization Map: _cartesian_to_keplerian
_cartesian_to_keplerian_vmap = vmap(
    _cartesian_to_keplerian,
    in_axes=(0, 0, None),
)

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
    coords_keplerian = _cartesian_to_keplerian_vmap(
        coords_cartesian,
        t0,
        mu
    )
    return coords_keplerian[jnp.array([0, 4, 5, 6, 7, 8])]

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
    coords_keplerian : `~jax.numpy.ndarray` (N, 13)
        13D Keplerian coordinates.
        a : semi-major axis in au.
        p : semi-latus rectum in au.
        q : periapsis distance in au.
        Q : apoapsis distance in au.
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
    coords_keplerian = _cartesian_to_keplerian_vmap(
        coords_cartesian,
        t0,
        mu
    )
    return coords_keplerian

@jit
def _keplerian_to_cartesian(
        coords_keplerian: Union[np.ndarray, jnp.ndarray],
        mu: float = MU,
        max_iter: int = 1000,
        tol: float = 1e-15
    ) -> jnp.ndarray:
    """
    Convert a single Keplerian coordinate to a Cartesian coordinate.

    Parabolic orbits (e = 1.0 +- 1e-15) with elements (a, e, i, raan, ap, M) cannot be converted
    to Cartesian orbits since their semi-major axes are by definition undefined.
    Please consider representing the orbits with Cometary elements
    and using those to convert to Cartesian. See `~thor.coordinates.cometary._cometary_to_cartesian`.

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
    coords_cartesian = jnp.zeros(6, dtype=jnp.float64)

    a = coords_keplerian[0]
    e = coords_keplerian[1]
    i = jnp.radians(coords_keplerian[2])
    raan = jnp.radians(coords_keplerian[3])
    ap = jnp.radians(coords_keplerian[4])
    M = jnp.radians(coords_keplerian[5])

    # Calculate semi-major axis (undefined for
    # parabolic orbits)
    # a = lax.cond(
    #     (e > (1.0 - FLOAT_TOLERANCE)) & (e < (1.0 + FLOAT_TOLERANCE)),
    #     lambda e, q: jnp.nan,
    #     lambda e, q: q / (1 - e),
    #     e, q
    # )
    # Calculate the periapsis distance (for parabolic orbits this is the defining
    # parameter and it cannot be calculated from any combination of the default
    # Keplerian elements)
    q = lax.cond(
        (e > (1.0 - FLOAT_TOLERANCE)) & (e < (1.0 + FLOAT_TOLERANCE)),
        lambda a, e: jnp.nan,
        lambda a, e: a * (1 - e),
        a, e
    )

    # Calculate the semi-latus rectum
    p = lax.cond(
        (e > (1.0 - FLOAT_TOLERANCE)) & (e < (1.0 + FLOAT_TOLERANCE)),
        lambda a, e, q: 2*q,
        lambda a, e, q: a * (1 - e**2),
        a, e, q
    )

    # Calculate the true anomaly
    nu = lax.cond(
        (e > (1.0 - FLOAT_TOLERANCE)) & (e < (1.0 + FLOAT_TOLERANCE)),
        lambda e_i, M_i: jnp.nan,
        lambda e_i, M_i: solve_kepler(e_i, M_i, max_iter=max_iter, tol=tol),
        e, M
    )

    # Calculate the perifocal rotation matrices
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

    coords_cartesian = coords_cartesian.at[0].set(r[0])
    coords_cartesian = coords_cartesian.at[1].set(r[1])
    coords_cartesian = coords_cartesian.at[2].set(r[2])
    coords_cartesian = coords_cartesian.at[3].set(v[0])
    coords_cartesian = coords_cartesian.at[4].set(v[1])
    coords_cartesian = coords_cartesian.at[5].set(v[2])

    return coords_cartesian

# Vectorization Map: _keplerian_to_cartesian
_keplerian_to_cartesian_vmap = vmap(
    _keplerian_to_cartesian,
    in_axes=(0, None, None, None),
)

def keplerian_to_cartesian(
        coords_keplerian: Union[np.ndarray, jnp.ndarray],
        mu: float = MU,
        max_iter: int = 100,
        tol: float = 1e-15
    ) -> jnp.ndarray:
    """
    Convert Keplerian coordinates to Cartesian coordinates.

    Parabolic orbits (e = 1.0 +- 1e-15) with elements (a, e, i, raan, ap, M) cannot be converted
    to Cartesian orbits since their semi-major axes are by definition undefined.
    Please consider representing these orbits with Cometary elements
    and using those to convert to Cartesian. See `~thor.coordinates.cometary.cometary_to_cartesian`.

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

    Raises
    ------
    ValueError: When semi-major axis is less than 0 for elliptical orbits or when
        semi-major axis is greater than 0 for hyperbolic orbits.
    """
    a = coords_keplerian[:, 0]
    e = coords_keplerian[:, 1]

    parabolic = np.where((e < (1.0 + FLOAT_TOLERANCE)) & (e > (1.0 - FLOAT_TOLERANCE)))[0]
    if len(parabolic) > 0:
        msg = (
            "Parabolic orbits (e = 1.0 +- 1e-15) are best represented using Cometary coordinates.\n"
            "Conversion to Cartesian coordinates will not yield correct results as semi-major axis\n"
            "for parabolic orbits is undefined."
        )
        logger.critical(msg)

    hyperbolic_invalid = np.where((e > (1.0 + FLOAT_TOLERANCE)) & (a > 0))[0]
    if len(hyperbolic_invalid) > 0:
        err = (
            "Semi-major axis (a) for hyperbolic orbits (e > 1 + 1e-15) should be negative. "
            f"Instead found a = {a[hyperbolic_invalid][0]} with e = {e[hyperbolic_invalid][0]}."
        )
        raise ValueError(err)

    elliptical_invalid = np.where((e < (1.0 - FLOAT_TOLERANCE)) & (a < 0))[0]
    if len(elliptical_invalid) > 0:
        err = (
            "Semi-major axis (a) for elliptical orbits (e < 1 - 1e-15) should be positive. "
            f"Instead found a = {a[elliptical_invalid][0]} with e = {e[elliptical_invalid][0]}."
        )
        raise ValueError(err)

    coords_cartesian = _keplerian_to_cartesian_vmap(
        coords_keplerian,
        mu,
        max_iter,
        tol
    )
    return coords_cartesian

class KeplerianCoordinates(Coordinates):

    def __init__(
            self,
            a: Optional[Union[int, float, np.ndarray]] = None,
            e: Optional[Union[int, float, np.ndarray]] = None,
            i: Optional[Union[int, float, np.ndarray]] = None,
            raan: Optional[Union[int, float, np.ndarray]] = None,
            ap: Optional[Union[int, float, np.ndarray]] = None,
            M: Optional[Union[int, float, np.ndarray]] = None,
            times: Optional[Time] = None,
            covariances: Optional[np.ndarray] = None,
            sigma_a: Optional[np.ndarray] = None,
            sigma_e: Optional[np.ndarray] = None,
            sigma_i: Optional[np.ndarray] = None,
            sigma_raan: Optional[np.ndarray] = None,
            sigma_ap: Optional[np.ndarray] = None,
            sigma_M: Optional[np.ndarray] = None,
            origin: str = "heliocentric",
            frame: str = "ecliptic",
            names: OrderedDict = KEPLERIAN_COLS,
            units: OrderedDict = KEPLERIAN_UNITS,
            mu: float = MU,
        ):
        sigmas = (
            sigma_a, sigma_e, sigma_i,
            sigma_raan, sigma_ap, sigma_M
        )
        Coordinates.__init__(self,
            a=a,
            e=e,
            i=i,
            raan=raan,
            ap=ap,
            M=M,
            covariances=covariances,
            sigmas=sigmas,
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
    def sigma_a(self):
        return self.sigmas[:, 0]

    @property
    def sigma_e(self):
        return self.sigmas[:, 1]

    @property
    def sigma_i(self):
        return self.sigmas[:, 2]

    @property
    def sigma_raan(self):
        return self.sigmas[:, 3]

    @property
    def sigma_ap(self):
        return self.sigmas[:, 4]

    @property
    def sigma_M(self):
        return self.sigmas[:, 5]

    @property
    def q(self):
        # periapsis distance
        return self.a * (1 - self.e)

    @property
    def Q(self):
        # apoapsis distance
        return self.a * (1 + self.e)

    @property
    def P(self):
        return np.sqrt(4 * np.pi**2 * self.a**3 / self.mu)

    @property
    def mu(self):
        return self._mu

    def to_cartesian(self) -> CartesianCoordinates:

        coords_cartesian = keplerian_to_cartesian(
            self.values.filled(),
            mu=MU,
            max_iter=1000,
            tol=1e-15,
        )
        coords_cartesian = np.array(coords_cartesian)

        if self.covariances is not None:
            covariances_cartesian = transform_covariances_jacobian(
                self.values,
                self.covariances,
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
            times=self.times,
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

        if cartesian.covariances is not None and (~np.all(cartesian.covariances.mask)):
            covariances_keplerian = transform_covariances_jacobian(
                cartesian.values,
                cartesian.covariances,
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
            coord_cols=KEPLERIAN_COLS,
            origin_col="origin",
            frame_col="frame"
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