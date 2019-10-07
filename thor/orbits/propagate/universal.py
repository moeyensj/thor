import numpy as np
from numba import jit
from astropy import constants as c
from astropy import units as u

from .stumpff import calcC2C3

__all__ = [
    "calcChi"
]

MU = (c.G * c.M_sun).to(u.AU**3 / u.day**2).value

@jit("f8(f8[:], f8[:], f8, f8, i8, f8)", nopython=True)
def calcChi(r, v, dt, mu=MU, maxIterations=10000, tol=1e-14):
    """
    Calculate universal anomaly chi using Newton-Raphson. 
    
    Parameters
    ----------
    r : `~numpy.ndarray` (3)
        Heliocentric position vector in units of AU. [J2000 ECLIPTIC]
    v : `~numpy.ndarray` (3)
        Heliocentric velocity vector in units of AU per day. [J2000 ECLIPTIC]
    dt : float
        Time from epoch to which calculate chi in units of decimal days.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of 
        AU**3 / d**2. 
    iterations : int, optional
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
        if iterations >= maxIterations:
            break
        
    return chi