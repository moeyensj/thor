import numpy as np
import jax.numpy as jnp
from jax import (
    jit,
    vmap
)

from ..constants import Constants as c
from .lagrange import (
    calc_lagrange_coefficients,
    apply_lagrange_coefficients
)

__all__ = [
    "propagate_2body",
]

MU = c.MU

@jit
def _propagate_2body(
        orbit: jnp.ndarray,
        t0: float,
        t1: jnp.ndarray,
        mu: float = MU,
        max_iter: int = 1000,
        tol: float = 1e-14
    ) -> jnp.ndarray:
    """
    Propagate an orbit from t0 to t1.

    Parameters
    ----------
    orbit : `~jax.numpy.ndarray` (6)
        Cartesian orbit with position in units of au and velocity in units of au per day.
    t0 : `float` (1)
        Epoch in MJD at which the orbit are defined.
    t1 : `~jax.numpy.ndarray` (N)
        Epochs to which to propagate the given orbit. If a single epoch is given, all orbits are propagated to this
        epoch. If multiple epochs are given, then will propagate each orbit to that epoch.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.
    max_iter : int, optional
        Maximum number of iterations over which to converge. If number of iterations is
        exceeded, will return the value of the universal anomaly at the last iteration.
    tol : float, optional
        Numerical tolerance to which to compute universal anomaly using the Newtown-Raphson
        method.

    Returns
    -------
    orbits : `~jax.numpy.ndarray` (N, 8)
        Orbit propagated to each MJD with position in units of au and velocity in units of au per day.
        The first two columns are the orbit ID (a zero-based integer value assigned to each unique input orbit)
        and the MJD of each propagated state.
    """
    r = orbit[0:3]
    v = orbit[3:6]
    dt = t1 - t0

    lagrange_coeffs, stumpff_coeffs, chi = calc_lagrange_coefficients(
        r,
        v,
        dt,
        mu=mu,
        max_iter=max_iter,
        tol=tol
    )
    r_new, v_new = apply_lagrange_coefficients(r, v, *lagrange_coeffs)

    return jnp.array([t1, r_new[0], r_new[1], r_new[2], v_new[0], v_new[1], v_new[2]])

# Vectorization Map: _propagate_2body
_propagate_2body_vmap = vmap(
    _propagate_2body,
    in_axes=(0, 0, 0, None, None, None),
    out_axes=(0)
)

def propagate_2body(
        orbits: np.ndarray,
        t0: np.ndarray,
        t1: np.ndarray,
        mu: float = MU,
        max_iter: int = 1000,
        tol: float = 1e-14
    ) -> np.ndarray:
    """
    Propagate orbits using the universal anomaly formalism.

    Parameters
    ----------
    orbits : `~jax.numpy.ndarray` (N, 6)
        Cartesian orbits with position in units of au and velocity in units of au per day.
    t0 : `~jax.numpy.ndarray` (N)
        Epoch in MJD at which orbits are defined.
    t1 : `~jax.numpy.ndarray` (M)
        Epochs to which to propagate each orbit. If a single epoch is given, all orbits are propagated to this
        epoch. If multiple epochs are given, then will propagate each orbit to that epoch.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.
    max_iter : int, optional
        Maximum number of iterations over which to converge. If number of iterations is
        exceeded, will return the value of the universal anomaly at the last iteration.
    tol : float, optional
        Numerical tolerance to which to compute universal anomaly using the Newtown-Raphson
        method.

    Returns
    -------
    orbits : `~numpy.ndarray` (N*M, 8)
        Orbits propagated to each MJD with position in units of au and velocity in units of au per day.
        The first two columns are the orbit ID (a zero-based integer value assigned to each unique input orbit)
        and the MJD of each propagated state.
    """
    n_orbits = orbits.shape[0]
    n_times = t1.shape[0]
    orbits_ = jnp.vstack([orbits[i] for i in range(n_orbits) for j in range(n_times)])
    t0_ = jnp.hstack([t0[i] for i in range(n_orbits) for j in range(n_times)])
    t1_ = jnp.hstack([t1 for i in range(n_orbits)])

    orbits_propagated = _propagate_2body_vmap(
        orbits_,
        t0_,
        t1_,
        mu,
        max_iter,
        tol
    )
    return np.array(orbits_propagated)