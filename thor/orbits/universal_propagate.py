import numpy as np
from numba import jit

from ..constants import Constants as c
from .lagrange import calcLagrangeCoeffs
from .lagrange import applyLagrangeCoeffs

__all__ = [
    "propagateUniversal",
]

MU = c.MU

@jit(["f8[:,:](f8[:,:], f8[:], f8[:], f8, i8, f8)"], nopython=True, cache=True)
def propagateUniversal(orbits, t0, t1, mu=MU, max_iter=100, tol=1e-14):
    """
    Propagate orbits using the universal anomaly formalism.

    Parameters
    ----------
    orbits : `~numpy.ndarray` (N, 6)
        Orbital state vectors (X_0) with position in units of AU and velocity in units of AU per day.
    t0 : `~numpy.ndarray` (N)
        Epoch in MJD at which orbits are defined.
    t1 : `~numpy.ndarray` (M)
        Epochs to which to propagate each orbit. If a single epoch is given, all orbits are propagated to this
        epoch. If multiple epochs are given, then will propagate each orbit to that epoch.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        AU**3 / d**2.
    max_iter : int, optional
        Maximum number of iterations over which to converge. If number of iterations is
        exceeded, will return the value of the universal anomaly at the last iteration.
    tol : float, optional
        Numerical tolerance to which to compute universal anomaly using the Newtown-Raphson
        method.

    Returns
    -------
    orbits : `~numpy.ndarray` (N*M, 8)
        Orbits propagated to each MJD with position in units of AU and velocity in units of AU per day.
        The first two columns are the orbit ID (a zero-based integer value assigned to each unique input orbit)
        and the MJD of each propagated state.
    """
    new_orbits = []
    num_orbits = orbits.shape[0]
    sqrt_mu = np.sqrt(mu)

    for i in range(num_orbits):
        for j, t in enumerate(t1):

            r = np.ascontiguousarray(orbits[i, 0:3])
            v = np.ascontiguousarray(orbits[i, 3:6])
            dt = t - t0[i]

            f, g, f_dot, g_dot = calcLagrangeCoeffs(
                r,
                v,
                dt,
                mu=mu,
                max_iter=max_iter,
                tol=tol
            )
            r_new, v_new = applyLagrangeCoeffs(r, v, f, g, f_dot, g_dot)

            new_orbits.append([i, t, r_new[0], r_new[1], r_new[2], v_new[0], v_new[1], v_new[2]])

    return np.array(new_orbits)