import numpy as np
from numba import jit

from ..constants import Constants as c
from .stumpff import calc_stumpff

__all__ = [
    "calc_chi",
]

MU = c.MU

@jit(["UniTuple(f8, 7)(f8[:], f8[:], f8, f8, i8, f8)"], nopython=True, cache=True)
def calc_chi(r, v, dt, mu=MU, max_iter=100, tol=1e-16):
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
    r = np.ascontiguousarray(r)
    v = np.ascontiguousarray(v)

    v_mag = np.linalg.norm(v)
    r_mag = np.linalg.norm(r)
    rv_mag = np.dot(r, v) / r_mag
    sqrt_mu = np.sqrt(mu)

    # Equations 3.48 and 3.50 in Curtis (2014) [1]
    alpha = -v_mag**2 / mu + 2 / r_mag

    # Equation 3.66 in Curtis (2014) [1]
    chi = sqrt_mu * np.abs(alpha) * dt

    ratio = 1e10
    iterations = 0
    while np.abs(ratio) > tol:
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
        if iterations >= max_iter:
            break

    return chi, c0, c1, c2, c3, c4, c5