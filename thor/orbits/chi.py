import numpy as np
from numba import jit

from ..constants import Constants as c
from .stumpff import calcStumpff

__all__ = [
    "calcChi",
]

MU = c.MU

@jit(["UniTuple(f8, 7)(f8[:], f8[:], f8, f8, i8, f8)"], nopython=True, cache=True)
def calcChi(r, v, dt, mu=MU, max_iter=100, tol=1e-16):
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
    [1] Danby, J. M. A. (1992). Fundamentals of Celestial Mechanics. 2nd ed.,
        William-Bell, Inc. ISBN-13: 978-0943396200
        Notes: of particular interest is Danby's fantastic chapter on universal
            variables (6.9)
    """
    r = np.ascontiguousarray(r)
    v = np.ascontiguousarray(v)

    v_mag = np.linalg.norm(v)
    r_mag = np.linalg.norm(r)

    # Equivalent to dot{r_0} in Danby's textbook, derived
    # from the text below Equation 6.9.10 in Danby 1992 [1]
    rv_mag = np.dot(r, v) / r_mag

    # Equation 6.9.9 in Danby 1992 [1]
    alpha = 2 * mu / r_mag - v_mag**2
    chi =  np.abs(alpha) * dt
    ratio = 1e10

    iterations = 0
    while np.abs(ratio) > tol:
        chi2 = chi**2
        psi = alpha * chi2
        c0, c1, c2, c3, c4, c5 = calcStumpff(psi)

        # Newton-Raphson
        # Equation 6.9.29 in Danby 1992 [1]
        f = (r_mag * chi * c1) + (r_mag * rv_mag * chi**2 * c2) + (mu * chi**3 * c3) - dt
        fp = (r_mag * c0) + (r_mag * rv_mag * chi * c1) + (mu * chi**2 * c2)

        ratio = f / fp
        chi -= ratio
        iterations += 1
        if iterations >= max_iter:
            break

    return chi, c0, c1, c2, c3, c4, c5