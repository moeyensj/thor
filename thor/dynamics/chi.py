import numpy as np
import jax.numpy as jnp
from jax import (
    config,
    jit,
    lax
)
from typing import Tuple

config.update("jax_enable_x64", True)

from ..constants import Constants as c
from .stumpff import calc_stumpff

__all__ = [
    "calc_chi",
]

MU = c.MU

@jit
def calc_chi(
        r: np.ndarray,
        v: np.ndarray,
        dt: float,
        mu: float = MU,
        max_iter: int = 100,
        tol: float = 1e-16
    ) -> Tuple[np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64]:
    """
    Calculate universal anomaly chi using Newton-Raphson.

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
    chi : float
        Universal anomaly.
    c0, c1, c2, c3, c4, c5 : 6 x float
        First six Stumpff functions.

    References
    ----------
    [1] Curtis, H. D. (2014). Orbital Mechanics for Engineering Students. 3rd ed.,
        Elsevier Ltd. ISBN-13: 978-0080977478
    """
    v_mag = jnp.linalg.norm(v)
    r_mag = jnp.linalg.norm(r)
    rv_mag = jnp.dot(r, v) / r_mag
    sqrt_mu = jnp.sqrt(mu)

    # Equations 3.48 and 3.50 in Curtis (2014) [1]
    alpha = -v_mag**2 / mu + 2 / r_mag

    # Equation 3.66 in Curtis (2014) [1]
    chi = sqrt_mu * jnp.abs(alpha) * dt

    ratio = 1e15
    iterations = 0
    # Define parameters array (arguments that will be changing as
    # the while loop iterates):
    # chi, c0, c1, c2, c3, c4, c5, ratio, iterations
    p = [chi, 0., 0., 0., 0., 0., 0., ratio, iterations]

    # Define while loop body function
    @jit
    def _chi_newton_raphson(p):

        chi = p[0]
        ratio = p[-2]
        iterations = p[-1]

        chi2 = chi**2
        psi = alpha * chi2
        c0, c1, c2, c3, c4, c5 = calc_stumpff(psi)

        # Newton-Raphson
        # Equation 3.65 in Curtis (2014) [1]
        f = (r_mag * rv_mag / sqrt_mu * chi2 * c2
             + (1 - alpha*r_mag) * chi**3 * c3
             + r_mag * chi
             - sqrt_mu * dt)
        fp = (r_mag * rv_mag / sqrt_mu * chi * (1 - alpha * chi2 * c3)
              + (1 - alpha * r_mag) * chi2 * c2
              + r_mag)

        ratio = f / fp
        chi -= ratio
        iterations += 1

        p[0] = chi
        p[1] = c0
        p[2] = c1
        p[3] = c2
        p[4] = c3
        p[5] = c4
        p[6] = c5
        p[7] = ratio
        p[8] = iterations
        return p

    # Define while loop condition function
    @jit
    def _while_condition(p):
        ratio = p[-2]
        iterations = p[-1]
        return (jnp.abs(ratio) > tol) & (iterations <= max_iter)

    p = lax.while_loop(
        _while_condition,
        _chi_newton_raphson,
        p
    )
    chi = p[0]
    c0 = p[1]
    c1 = p[2]
    c2 = p[3]
    c3 = p[4]
    c4 = p[5]
    c5 = p[6]

    return chi, c0, c1, c2, c3, c4, c5