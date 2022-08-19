import jax.numpy as jnp
from jax import (
    config,
    jit,
    lax,
    vmap
)
from typing import Tuple

config.update("jax_enable_x64", True)

from ..constants import Constants as c
from .propagate_2body import _propagate_2body

__all__ = [
    "add_light_time",
    "add_stellar_aberration"
]

MU = c.MU
C = c.C

@jit
def _add_light_time(
        orbit: jnp.ndarray,
        t0: float,
        observer_position: jnp.ndarray,
        lt_tol: float = 1e-10,
        mu: float = MU,
        max_iter: int = 1000,
        tol: float = 1e-15
    ) -> Tuple[jnp.ndarray, jnp.float64]:
    """
    When generating ephemeris, orbits need to be backwards propagated to the time
    at which the light emitted or relflected from the object towards the observer.

    Light time correction must be added to orbits in expressed in an inertial frame (ie, orbits
    must be barycentric).

    Parameters
    ----------
    orbit : `~jax.numpy.ndarray` (6)
        Barycentric orbit in cartesian elements to correct for light time delay.
    t0 : float
        Epoch at which orbits are defined.
    observer_positions : `~jax.numpy.ndarray` (3)
        Location of the observer in barycentric cartesian elements at the time of observation.
    lt_tol : float, optional
        Calculate aberration to within this value in time (units of days.)
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        AU**3 / d**2.
    max_iter : int, optional
        Maximum number of iterations over which to converge for propagation.
    tol : float, optional
        Numerical tolerance to which to compute universal anomaly during propagation using the Newtown-Raphson
        method.

    Returns
    -------
    corrected_orbit : `~jax.numpy.ndarray` (6)
        Orbit adjusted for light travel time.
    lt : float
        Light time correction (t0 - corrected_t0).
    """
    dlt = 1e30
    lt = 1e30

    @jit
    def _iterate_light_time(p):

        orbit = p[0]
        t0 = p[1]
        lt0 = p[2]
        dlt = p[3]

        # Calculate topocentric distance
        rho = jnp.linalg.norm(orbit[:3] - observer_position)

        # Calculate initial guess of light time
        lt = rho / C

        # Calculate difference between previous light time correction
        # and current guess
        dlt = jnp.abs(lt - lt0)

        # Propagate backwards to new epoch
        t1 = t0 - lt
        orbit_propagated = _propagate_2body(orbit, t0, t1, mu=mu, max_iter=max_iter, tol=tol)

        p[0] = orbit_propagated
        p[1] = t1
        p[2] = lt
        p[3] = dlt
        return p

    @jit
    def _while_condition(p):
        dlt = p[-1]
        return dlt > lt_tol

    p = [orbit, t0, lt, dlt]
    p = lax.while_loop(
        _while_condition,
        _iterate_light_time,
        p
    )

    orbit_aberrated = p[0]
    t0_aberrated = p[1]
    lt = p[2]
    return orbit_aberrated, lt

# Vectorization Map: _add_light_time
_add_light_time_vmap = vmap(
    _add_light_time,
    in_axes=(0, 0, 0, None, None, None, None)
)

@jit
def add_light_time(
        orbits: jnp.ndarray,
        t0: jnp.ndarray,
        observer_positions: jnp.ndarray,
        lt_tol: float = 1e-10,
        mu: float = MU,
        max_iter: int = 1000,
        tol: float = 1e-15
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    When generating ephemeris, orbits need to be backwards propagated to the time
    at which the light emitted or relflected from the object towards the observer.

    Light time correction must be added to orbits in expressed in an inertial frame (ie, orbits
    must be barycentric).

    Parameters
    ----------
    orbits : `~jax.numpy.ndarray` (N, 6)
        Barycentric orbits in cartesian elements to correct for light time delay.
    t0 : `~jax.numpy.ndarray` (N)
        Epoch at which orbits are defined.
    observer_positions : `~jax.numpy.ndarray` (N, 3)
        Location of the observer in barycentric cartesian elements at the time of observation.
    lt_tol : float, optional
        Calculate aberration to within this value in time (units of days.)
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        AU**3 / d**2.
    max_iter : int, optional
        Maximum number of iterations over which to converge for propagation.
    tol : float, optional
        Numerical tolerance to which to compute universal anomaly during propagation using the Newtown-Raphson
        method.

    Returns
    -------
    corrected_orbits : `~jax.numpy.ndarray` (N, 6)
        Orbits adjusted for light travel time.
    lt : `~jax.numpy.ndarray` (N)
        Light time correction (t0 - corrected_t0).
    """
    orbits_aberrated, lts = _add_light_time_vmap(
        orbits,
        t0,
        observer_positions,
        lt_tol,
        mu,
        max_iter,
        tol
    )
    return orbits_aberrated, lts

@jit
def add_stellar_aberration(
        orbits: jnp.ndarray,
        observer_states: jnp.ndarray
    ) -> jnp.ndarray:
    """
    The motion of the observer in an inertial frame will cause an object
    to appear in a different location than its true geometric location. This
    aberration is typically applied after light time corrections have been added.

    The velocity of the input orbits are unmodified only the position
    vector is modified with stellar aberration.

    Parameters
    ----------
    orbits : `~jax.numpy.ndarray` (N, 6)
        Orbits in barycentric cartesian elements.
    observer_states : `~jax.numpy.ndarray` (N, 6)
        Observer states in barycentric cartesian elements.

    Returns
    -------
    rho_aberrated : `~jax.numpy.ndarray` (N, 3)
        The topocentric position vector for each orbit with
        added stellar aberration.

    References
    ----------
    [1] Urban, S. E; Seidelmann, P. K. (2013) Explanatory Supplement to the Astronomical Almanac. 3rd ed.,
        University Science Books. ISBN-13: 978-1891389856
    """
    topo_states = orbits - observer_states
    rho_aberrated = jnp.zeros((len(topo_states), 3), dtype=jnp.float64)
    rho_aberrated = rho_aberrated.at[:].set(topo_states[:, :3])

    v_obs = observer_states[:, 3:]
    beta = v_obs / C
    gamma_inv = jnp.sqrt(1 - jnp.linalg.norm(beta, axis=1, keepdims=True)**2)
    delta = jnp.linalg.norm(topo_states[:, :3], axis=1, keepdims=True)

    # Equation 7.40 in Urban & Seidelmann (2013) [1]
    rho = topo_states[:, :3] / delta
    rho_aberrated = rho_aberrated.at[:].set((gamma_inv * rho + beta + rho * beta * beta / (1 + gamma_inv)) / (1 + rho*beta))
    rho_aberrated = rho_aberrated.at[:].multiply(delta)

    return rho_aberrated