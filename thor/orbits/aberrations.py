import numpy as np
from numba import jit

from ..constants import Constants as c
from .universal_propagate import propagateUniversal

__all__ = [
    "addLightTime",
    "addStellarAberration"
]

MU = c.MU
C = c.C

@jit(["Tuple((f8[:,:], f8[:]))(f8[:,:], f8[:], f8[:,:], f8, f8, i8, f8)"], nopython=True, cache=True)
def addLightTime(orbits, t0, observer_positions, lt_tol=1e-10, mu=MU, max_iter=1000, tol=1e-15):
    """
    When generating ephemeris, orbits need to be backwards propagated to the time
    at which the light emitted or relflected from the object towards the observer.

    Light time correction must be added to orbits in expressed in an inertial frame (ie, orbits
    must be barycentric)

    Parameters
    ----------
    orbits : `~numpy.ndarray` (N, 6)
        Barycentric orbits in cartesian elements to correct for light time delay.
    t0 : `~numpy.ndarray` (N)
        Epoch at which orbits are defined.
    observer_positions : `numpy.ndarray` (N, 3)
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
    corrected_orbits : `~numpy.ndarray` (N, 6)
        Orbits adjusted for light travel time.
    lt : `~numpy.ndarray` (N)
        Light time correction (t0 - corrected_t0).
    """
    corrected_orbits = np.zeros((len(orbits), 6))
    lts = np.zeros(len(orbits))
    num_orbits = len(orbits)
    for i in range(num_orbits):

        # Set up running variables
        orbit_i = orbits[i:i+1, :]
        observer_position_i = observer_positions[i:i+1, :]
        t0_i = t0[i:i+1]
        dlt = 1e30
        lt_i = 1e30

        while dlt > lt_tol:
            # Calculate topocentric distance
            rho = np.linalg.norm(orbit_i[:, :3] - observer_position_i)

            # Calculate initial guess of light time
            lt = rho / C

            # Calculate difference between previous light time correction
            # and current guess
            dlt = np.abs(lt - lt_i)

            # Propagate backwards to new epoch
            orbit = propagateUniversal(orbits[i:i+1, :], t0[i:i+1], t0[i:i+1] - lt, mu=mu, max_iter=max_iter, tol=tol)

            # Update running variables
            t0_i = orbit[:, 1]
            orbit_i = orbit[:, 2:]
            lt_i = lt

        corrected_orbits[i, :] = orbit[0, 2:]
        lts[i] = lt

    return corrected_orbits, lts

@jit(["f8[:,:](f8[:,:], f8[:,:])"], nopython=True, cache=True)
def addStellarAberration(orbits, observer_states):
    """
    The motion of the observer in an inertial frame will cause an object
    to appear in a different location than its true geometric location. This
    aberration is typically applied after light time corrections have been added.

    The velocity of the input orbits are unmodified only the position
    vector is modified with stellar aberration.

    Parameters
    ----------
    orbits : `~numpy.ndarray` (N, 6)
        Orbits in barycentric cartesian elements.
    observer_states : `~numpy.ndarray` (N, 6)
        Observer states in barycentric cartesian elements.

    Returns
    -------
    rho_aberrated : `~numpy.ndarray` (N, 3)
        The topocentric position vector for each orbit with
        added stellar aberration.

    References
    ----------
    [1] Urban, S. E; Seidelmann, P. K. (2013) Explanatory Supplement to the Astronomical Almanac. 3rd ed.,
        University Science Books. ISBN-13: 978-1891389856
    """
    topo_states = orbits - observer_states
    rho_aberrated = topo_states[:, :3].copy()
    for i in range(len(orbits)):
        v_obs = observer_states[i, 3:]
        beta = v_obs / C
        gamma_inv = np.sqrt(1 - np.linalg.norm(beta)**2)
        delta = np.linalg.norm(topo_states[i, :3])

        # Equation 7.40 in Urban & Seidelmann (2013) [1]
        rho = topo_states[i, :3] / delta
        rho_aberrated[i, :] = (gamma_inv * rho + beta + np.dot(rho, beta) * beta / (1 + gamma_inv)) / (1 + np.dot(rho, beta))
        rho_aberrated[i, :] *= delta

    return rho_aberrated