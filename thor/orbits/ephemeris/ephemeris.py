import numpy as np
from numba import jit

from ...constants import Constants as c
from ..propagate import propagateUniversal

__all__ = [
    "addPlanetaryAberration",
]

MU = c.G * c.M_SUN
C = c.C

@jit(["Tuple((f8[:,:], f8[:], f8[:]))(f8[:,:], f8[:], f8[:,:], f8, f8, i8, f8)"], nopython=True)
def addPlanetaryAberration(orbits, t0, observer_states, lt_tol=1e-10, mu=MU, max_iter=1000, tol=1e-15):
    """
    When generating ephemeris, orbits need to be backwards propagated to the time
    at which the light emitted or relflected from the object towards the observer.
    
    Parameters
    ----------
    orbits : `~numpy.ndarray` (N, 6)
        Orbits to correct for planetary aberration.
    t0 : `~numpy.ndarray` (N)
        Epoch at which orbits are defined.
    observer_states : `numpy.ndarray` (N, 3)
        Location of the observer at the time of observeration.
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
        Orbits adjusted for planetary aberration.
    corrected_t0 : `~numpy.ndarray` (N)
        Aberration adjusted epochs.
    lt : `~numpy.ndarray` (N)
        Light time correction (t0 - corrected_t0).
    """
    corrected_orbits = np.zeros((len(orbits), 6))
    corrected_t0 = np.zeros(len(orbits))
    lts = np.zeros(len(orbits))
    num_orbits = len(orbits)
    for i in range(num_orbits):
        
        # Set up running variables
        orbit_i = orbits[i:i+1, :]
        observer_state_i = observer_states[i:i+1, :]
        t0_i = t0[i:i+1]
        dlt = 1e30
        lt_i = 1e30
     
        j = 0 
        while dlt > lt_tol:
            # Calculate topocentric distance
            rho = np.linalg.norm(orbit_i[:, :3] - observer_state_i)
        
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
            
        corrected_orbits[i, :] = orbit[:, 2:]
        corrected_t0[i] = orbit[0, 1]
        lts[i] = lt
        
    return corrected_orbits, corrected_t0, lts
    
    
    