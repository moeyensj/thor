import numpy as np
import jax.numpy as jnp
from jax import (
    config,
    jit,
    vmap
)
from copy import deepcopy
from astropy.time import Time

config.update("jax_enable_x64", True)

from ..constants import Constants as c
from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.covariances import transform_covariances_jacobian
from ..orbits.orbits import Orbits
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
        t1: float,
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
    t0 : float (1)
        Epoch in MJD at which the orbit are defined.
    t1 : float (N)
        Epochs to which to propagate the given orbit.
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
    orbits : `~jax.numpy.ndarray` (N, 6)
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

    return jnp.array([r_new[0], r_new[1], r_new[2], v_new[0], v_new[1], v_new[2]])

# Vectorization Map: _propagate_2body
_propagate_2body_vmap = vmap(
    _propagate_2body,
    in_axes=(0, 0, 0, None, None, None),
    out_axes=(0)
)

def propagate_2body(
        orbits: Orbits,
        times: Time,
        mu: float = MU,
        max_iter: int = 1000,
        tol: float = 1e-14
    ) -> Orbits:
    """
    Propagate orbits using the 2-body universal anomaly formalism.

    Parameters
    ----------
    orbits : `~jax.numpy.ndarray` (N, 6)
        Cartesian orbits with position in units of au and velocity in units of au per day.
    times : `~astropy.time.core.Time` (M)
        Epochs to which to propagate each orbit. If a single epoch is given, all orbits are propagated to this
        epoch. If multiple epochs are given, then each orbit to will be propagated to each epoch.
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
    orbits : `~thor.orbits.orbits.Orbits` (N*M)
        Orbits propagated to each MJD.
    """
    orbits_ = deepcopy(orbits)
    orbits_.to_frame("ecliptic")
    orbits_.to_origin("heliocenter")

    cartesian_orbits = orbits_.cartesian.values.filled()
    t0 = orbits_.cartesian.times.tdb.mjd
    t1 = times.tdb.mjd

    n_orbits = cartesian_orbits.shape[0]
    n_times = len(times)
    orbit_ids = np.hstack([orbits_.orbit_ids[i] for i in range(n_orbits) for j in range(n_times)])
    object_ids = np.hstack([orbits_.object_ids[i] for i in range(n_orbits) for j in range(n_times)])
    orbits_array = np.vstack([cartesian_orbits[i] for i in range(n_orbits) for j in range(n_times)])
    t0_ = np.hstack([t0[i] for i in range(n_orbits) for j in range(n_times)])
    t1_ = np.hstack([t1 for i in range(n_orbits)])

    orbits_propagated = _propagate_2body_vmap(
        orbits_array,
        t0_,
        t1_,
        mu,
        max_iter,
        tol
    )
    orbits_propagated = np.array(orbits_propagated)

    if not np.all(orbits_.cartesian.covariances.mask):
        covariances_array = np.stack([orbits_.cartesian.covariances[i] for i in range(n_orbits) for j in range(n_times)])
        covariances_cartesian = transform_covariances_jacobian(
            orbits_array,
            covariances_array,
            _propagate_2body,
            in_axes=(0, 0, 0, None, None, None),
            out_axes=0,
            t0=t0_,
            t1=t1_,
            mu=mu,
            max_iter=max_iter,
            tol=tol,
        )
    else:
        covariances_cartesian = None

    orbits_propagated = Orbits(
        CartesianCoordinates(
            x=orbits_propagated[:, 0],
            y=orbits_propagated[:, 1],
            z=orbits_propagated[:, 2],
            vx=orbits_propagated[:, 3],
            vy=orbits_propagated[:, 4],
            vz=orbits_propagated[:, 5],
            covariances=covariances_cartesian,
            times=Time(
                t1_,
                scale="tdb",
                format="mjd"
            ),
            origin="heliocenter",
            frame="ecliptic",
        ),
        orbit_ids=orbit_ids,
        object_ids=object_ids,
    )

    return orbits_propagated