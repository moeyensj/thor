import numpy as np
import jax.numpy as jnp
from jax import (
    jit,
    vmap
)
from typing import (
    Tuple,
    Union
)

from ..constants import Constants as c
from ..coordinates.spherical import (
    _cartesian_to_spherical,
    SphericalCoordinates
)
from ..coordinates.covariances import transform_covariances_jacobian
from ..orbits.orbits import Orbits
from ..orbits.ephemeris import Ephemeris
from ..observers.observers import Observers
from .aberrations import (
    _add_light_time,
    add_stellar_aberration
)

__all__ = [
    "generate_ephemeris_2body"
]

MU = c.MU

@jit
def _generate_ephemeris_2body(
        propagated_orbit: Union[np.ndarray, jnp.ndarray],
        observation_time: float,
        observer_coordinates: Union[np.ndarray, jnp.ndarray],
        lt_tol: float = 1e-10,
        mu: float = MU,
        max_iter: int = 1000,
        tol: float = 1e-15
    ) -> Tuple[jnp.ndarray, jnp.float64]:
    """
    Given a propagated orbit, generate its on-sky ephemeris as viewed from the observer.
    This function calculates the light time delay between the propagated orbit and the observer,
    and then propagates the orbit backward by that amount to when the light from object was actually
    emitted towards the observer.

    The motion of the observer in an inertial frame will cause an object
    to appear in a different location than its true geometric location, this is known as
    stellar aberration. Stellar aberration is will also be applied after
    light time correction has been added.

    The velocity of the input orbits are unmodified only the position
    vector is modified with stellar aberration.

    Parameters
    ----------
    propagated_orbit : `~jax.numpy.ndarray` (6)
        Barycentric Cartesian orbit propagated to the given time.
    observation_time : float
        Epoch at which orbit and observer coordinates are defined.
    observer_coordinates : `~jax.numpy.ndarray` (3)
        Barycentric Cartesian observer coordinates.
    lt_tol : float, optional
        Calculate aberration to within this value in time (units of days).
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
    ephemeris_spherical : `~jax.numpy.ndarray` (6)
        Topocentric Spherical ephemeris.
    lt : float
        Light time correction (t0 - corrected_t0).
    """
    propagated_orbits_aberrated, light_time = _add_light_time(
        propagated_orbit,
        observation_time,
        observer_coordinates[0:3],
        lt_tol=lt_tol,
        mu=mu,
        max_iter=max_iter,
        tol=tol,
    )

    propagated_orbits_aberrated = propagated_orbits_aberrated.at[0:3].set(
        add_stellar_aberration(
            propagated_orbits_aberrated.reshape(1, -1),
            observer_coordinates.reshape(1, -1),
        )[0]
    )

    topocentric_coordinates = propagated_orbits_aberrated[0] - observer_coordinates
    ephemeris_spherical = _cartesian_to_spherical(topocentric_coordinates)

    return ephemeris_spherical, light_time

# Vectorization Map: _generate_ephemeris_2body
_generate_ephemeris_2body_vmap = vmap(
    _generate_ephemeris_2body,
    in_axes=(0, 0, 0, None, None, None, None),
    out_axes=(0, 0)
)

def generate_ephemeris_2body(
        propagated_orbits: Orbits,
        observers: Observers,
        lt_tol: float = 1e-10,
        mu: float = MU,
        max_iter: int = 1000,
        tol: float = 1e-15
    ):
    """
    Generate on-sky ephemerides for each propagated orbit as viewed by the observers.
    This function calculates the light time delay between the propagated orbits and the observers,
    and then propagates the orbits backward by that amount to when the light from each object was actually
    emitted towards the observer.

    The motion of the observer in an inertial frame will cause an object
    to appear in a different location than its true geometric location, this is known as
    stellar aberration. Stellar aberration is will also be applied after
    light time correction has been added.

    The velocity of the input orbits are unmodified only the position
    vector is modified with stellar aberration.

    Parameters
    ----------
    propagated_orbits : `~thor.orbits.orbits.Orbits` (N)
        Propagated orbits.
    observers : `~thor.observers.observers.Observers` (N)
        Observers for which to generate ephemerides. Orbits should already have been
        propagated to the same times as the observers.
    lt_tol : float, optional
        Calculate aberration to within this value in time (units of days).
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
    ephemeris : `~thor.orbits.ephemeris.Ephemeris` (N)
        Topocentric ephemerides for each propagated orbit as observed by the given observers.
    """
    ephemeris_spherical, light_time = _generate_ephemeris_2body_vmap(
        propagated_orbits.cartesian.values,
        propagated_orbits.cartesian.times.utc.mjd,
        observers.cartesian.values,
        lt_tol,
        mu,
        max_iter,
        tol
    )
    ephemeris_spherical = np.array(ephemeris_spherical)
    light_time = np.array(light_time)

    if not np.all(propagated_orbits.cartesian.covariances.mask):
        covariances_spherical = transform_covariances_jacobian(
            propagated_orbits.cartesian.values,
            propagated_orbits.cartesian.covariances,
            _generate_ephemeris_2body,
            in_axes=(0, 0, 0, None, None, None, None),
            out_axes=(0, 0),
            observation_times=propagated_orbits.cartesian.times.utc.mjd,
            observer_coordinates=observers.cartesian.values,
            lt_tol=lt_tol,
            mu=mu,
            max_iter=max_iter,
            tol=tol,
        )
        covariances_spherical = np.array(covariances_spherical)
    else:
        covariances_spherical = None

    spherical_coordinates = SphericalCoordinates(
        rho=ephemeris_spherical[:, 0],
        lon=ephemeris_spherical[:, 1],
        lat=ephemeris_spherical[:, 2],
        vrho=ephemeris_spherical[:, 3],
        vlon=ephemeris_spherical[:, 4],
        vlat=ephemeris_spherical[:, 5],
        covariances=covariances_spherical,
        origin=observers.codes,
        frame="ecliptic"
    )

    ephemeris = Ephemeris(
        spherical_coordinates,
        orbit_ids=propagated_orbits.orbit_ids,
        object_ids=propagated_orbits.object_ids,
        light_time=light_time,
    )
    return ephemeris