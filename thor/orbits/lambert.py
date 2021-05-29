import numpy as np
from numba import jit

from ..constants import Constants as c
from .stumpff import calcStumpff

MU = c.MU

__all__ = [
    "calcLambert"
]


@jit(["UniTuple(f8[:], 2)(f8[:], f8, f8[:], f8, f8, f8, f8)"], nopython=True, cache=True)
def calcLambert(r0, t0, r1, t1, mu=MU, max_iter=1000, dt_tol=1e-12):
    """
    Solve the Lambert problem using the universal variable formulation and
    Newton-Raphson. Given two position vectors and their corresponding
    times calculate the velocity vectors at both times.

    Parameters
    ----------
    r0 : `~numpy.ndarray` (3)
        Cartesian position vector of the target at t0 in units of AU.
    t0 : float
        Time at which r0 is true in units of decimal days.
    r1 : `~numpy.ndarray` (3)
        Cartesian position vector of the target at t1.
    t1 : float
        Time at which r1 is true in units of decimal days.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        AU**3 / d**2.
    max_iter : int
        Maximum number of iterations to reach convergence.
    dt_tol : float
        Tolerance up to which to iterate the time of flight.

    Returns
    -------
    v0 : `~numpy.ndarray` (3)
        Velocity of the target at t0 in units of AU per day.
    v1 : `~numpy.ndarray` (3)
        Velocity of the target at t1 in units of AU per day.
    """
    dt = t1 - t0
    r0_mag = np.linalg.norm(r0)
    r1_mag = np.linalg.norm(r1)

    delta_nu = np.arccos(np.dot(r0, r1) / (r0_mag * r1_mag))
    A = np.sqrt(r0_mag * r1_mag * (1 + np.cos(delta_nu)))
    sqrt_mu = np.sqrt(mu)

    psi_iter = 0
    iterations = 0
    converged = False
    while not converged:
        c0, c1, c2, c3, c4, c5 = calcStumpff(psi_iter)

        y = r0_mag + r1_mag - A * (1 - psi_iter * c3) / np.sqrt(c2)

        while y < 0 and A > 0:
            psi_iter += 1e-8
            y = r0_mag + r1_mag - A * (1 - psi_iter * c3) / np.sqrt(c2)

        chi = np.sqrt(y / c2)

        dt_iter = (chi**3 * c3 + A * np.sqrt(y)) / sqrt_mu

        if np.abs(dt_iter - dt) < dt_tol:
            converged = True

        if np.abs(psi_iter) > 1e-8:
            c2p = (1 - psi_iter * c3 - 2 * c2) / (2 * psi_iter)
            c3p = (c2 - 3 * c3) / (2 * psi_iter)
            dtp = (chi**3 * (c3p - 3/2 * (c3 * c2p / c2)) + 1/8 * A * ((3 * c3 * np.sqrt(y)) / c2 + A / chi)) / sqrt_mu

        else:
            c2 = 1/2
            y0 = r0_mag + r1_mag - A / np.sqrt(c2)
            dtp = np.sqrt(2)/40 * y0**(3/2) + A / 8 * (np.sqrt(y0) + A * np.sqrt(1/2/y0))

        ratio = (dt_iter - dt) / dtp
        psi_iter -= ratio

        iterations += 1

        if iterations >= max_iter:
            break

    f = 1 - y / r0_mag
    g_dot = 1 - y / r1_mag
    g = A * np.sqrt(y / mu)

    v0 = (r1 - f * r0 ) / g
    v1 = (g_dot * r1 - r0 ) / g

    return v0, v1