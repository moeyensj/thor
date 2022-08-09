import numpy as np
import jax.numpy as jnp
from jax import (
    config,
    jit,
)
from typing import Tuple

config.update("jax_enable_x64", True)

from ..constants import Constants as C
from .stumpff import STUMPFF_TYPES
from .chi import calc_chi

__all__ = [
    "calc_lagrange_coefficients",
    "apply_lagrange_coefficients",
]

MU = C.MU
LAGRANGE_TYPES = Tuple[jnp.float64, jnp.float64, jnp.float64, jnp.float64]

@jit
def calc_lagrange_coefficients(
        r: np.ndarray,
        v: np.ndarray,
        dt: float,
        mu: float = MU,
        max_iter: int = 100,
        tol: float = 1e-16
    ) -> Tuple[LAGRANGE_TYPES, STUMPFF_TYPES, np.float64]:
    """
    Calculate the exact Lagrange coefficients given an initial state defined at t0,
    and the change in time from t0 to t1 (dt = t1 - t0).

    Parameters
    ----------
    r : `~numpy.ndarray` (3)
        Position vector in au.
    v : `~numpy.ndarray` (3)
        Velocity vector in au per day.
    dt : float
        Time from epoch to which calculate chi in units of decimal days.
    mu : float
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.
    max_iter : int
        Maximum number of iterations over which to converge. If number of iterations is
        exceeded, will return the value of the universal anomaly at the last iteration.
    tol : float
        Numerical tolerance to which to compute chi using the Newtown-Raphson
        method.

    Returns
    -------
    lagrange_coeffs : (float x 4)
        f : float
            Langrange f coefficient.
        g : float
            Langrange g coefficient.
        f_dot : float
            Time deriviative of the Langrange f coefficient.
        g_dot : float
            Time deriviative of the Langrange g coefficient.
    stumpff_coeffs : (float x 6)
        First six Stumpff functions (c0, c1, c2, c3, c4, c5)
    chi : float
        Universal anomaly.

    References
    ----------
    [1] Curtis, H. D. (2014). Orbital Mechanics for Engineering Students. 3rd ed.,
        Elsevier Ltd. ISBN-13: 978-0080977478
    """
    sqrt_mu = jnp.sqrt(mu)
    chi, c0, c1, c2, c3, c4, c5 = calc_chi(
        r,
        v,
        dt,
        mu=mu,
        max_iter=max_iter,
        tol=tol
    )
    stumpff_coeffs = (c0, c1, c2, c3, c4, c5)
    chi2 = chi**2

    r_mag = jnp.linalg.norm(r)
    v_mag = jnp.linalg.norm(v)

    # Equations 3.48 and 3.50 in Curtis (2014) [1]
    alpha = -v_mag**2 / mu + 2 / r_mag

    # Equations 3.69a and 3.69b in Curtis (2014) [1]
    f = 1 - chi**2 / r_mag * c2
    g = dt - 1 / sqrt_mu * chi**3 * c3

    r_new = f * r + g * v
    r_new_mag = jnp.linalg.norm(r_new)

    # Equations 3.69c and 3.69d in Curtis (2014) [1]
    f_dot = sqrt_mu / (r_mag * r_new_mag) * (alpha * chi**3 * c3 - chi)
    g_dot = 1 - chi2 / r_new_mag * c2

    lagrange_coeffs = (f, g, f_dot, g_dot)

    return lagrange_coeffs, stumpff_coeffs, chi

@jit
def apply_lagrange_coefficients(
        r: np.ndarray,
        v: np.ndarray,
        f: float,
        g: float,
        f_dot: float,
        g_dot: float
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the Lagrange coefficients to r and v.

    Parameters
    ----------
    r : `~numpy.ndarray` (3)
        Position vector in au.
    v : `~numpy.ndarray` (3)
        Velocity vector in au per day.
    f : float
        Langrange f coefficient.
    g : float
        Langrange g coefficient.
    f_dot : float
        Time deriviative of the Langrange f coefficient.
    g_dot : float
        Time deriviative of the Langrange g coefficient.

    Returns
    -------
    r_new : `~numpy.ndarray` (3)
        New position vector in au propagated with the Lagrange coefficients.
    v_new : `~numpy.ndarray` (3)
        New velocity vector in au per day propagated with the Lagrange coefficients.
    """
    r_new = f * r + g * v
    v_new = f_dot * r + g_dot * v

    return r_new, v_new