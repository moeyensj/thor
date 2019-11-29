import numpy as np
from numba import jit

from ...constants import Constants as c
from .stumpff import calcC2C3

__all__ = [
    "calcChi",
    "propagateUniversal"
]

MU = c.G * c.M_SUN

@jit(["f8(f8[::1], f8, f8, i8, f8)"], nopython=True)
def calcChi(orbit, dt, mu=MU, max_iter=10000, tol=1e-14):
    """
    Calculate universal anomaly chi using Newton-Raphson. 
    
    Parameters
    ----------
    orbit : `~numpy.ndarray` (6)
        Orbital state vector (X_0) with position in units of AU and velocity in units of AU per day. [J2000 ECLIPTIC]
    dt : float
        Time from epoch to which calculate chi in units of decimal days.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of 
        AU**3 / d**2. 
    max_iter : int, optional
        Maximum number of iterations over which to converge. If number of iterations is 
        exceeded, will return the value of the universal anomaly at the last iteration. 
    tol : float, optional
        Numerical tolerance to which to compute chi using the Newtown-Raphson 
        method. 
    
    Returns
    -------
    chi : float
        Universal anomaly. 
    """
    r = orbit[:3]
    v = orbit[3:]
    v_mag = np.linalg.norm(v)
    r_mag = np.linalg.norm(r)
    rv_mag = np.dot(r, v) / r_mag
    sqrt_mu = np.sqrt(mu)
    
    alpha = -v_mag**2 / mu + 2 / r_mag
    chi = np.sqrt(mu) * np.abs(alpha) * dt
    ratio = 1e10
    
    iterations = 0
    while np.abs(ratio) > tol:
        chi2 = chi**2
        psi = alpha * chi2
        c2, c3 = calcC2C3(psi)
        
        # Newton-Raphson
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
        
    return chi

@jit(["f8[::1](f8[::1], f8, f8, i8, f8)"], nopython=True)
def propagateUniversal(orbit, dt, mu=MU, max_iter=10000, tol=1e-14):
    """
    Propagate an orbit using the universal anomaly formalism. 
    
    Parameters
    ----------
    orbit : `~numpy.ndarray` (6)
        Orbital state vector (X_0) with position in units of AU and velocity in units of AU per day. [J2000 ECLIPTIC]
    dt : float
        Time from current epoch in units of decimal days to which to propagate
        the orbit.
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
    orbit : `~numpy.ndarray` (6)
        Orbital state vector at time dt from original state vector 
        with position in units of AU and velocity in units of AU per day. [J2000 ECLIPTIC]
    """
    chi = calcChi(orbit, dt, mu=mu, max_iter=max_iter, tol=tol)
    
    r = orbit[:3]
    v = orbit[3:]
    v_mag = np.linalg.norm(v)
    r_mag = np.linalg.norm(r)
    sqrt_mu = np.sqrt(mu)
    chi2 = chi**2

    alpha = -v_mag**2 / mu + 2 / r_mag
    psi = alpha * chi2
    c2, c3 = calcC2C3(psi)
    
    f = 1 - chi**2 / r_mag * c2
    g = dt - 1 / sqrt_mu * chi**3 * c3
    
    r_new = f * r + g * v
    r_new_mag = np.linalg.norm(r_new)
    
    f_dot = sqrt_mu / (r_mag * r_new_mag) * (alpha * chi**3 * c3 - chi)
    g_dot = 1 - chi2 / r_new_mag * c2
    
    v_new = f_dot * r + g_dot * v
    
    return np.concatenate((r_new, v_new))   