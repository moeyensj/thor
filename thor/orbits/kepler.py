import numpy as np
from numba import jit

from ..constants import Constants as c

__all__ = ["convertCartesianToKeplerian"]

MU = c.G * c.M_SUN

@jit(["f8[::1](f8[::1], f8)"], nopython=True)
def convertCartesianToKeplerian(elements_cart, mu=MU):
    """
    Convert cartesian orbital elements to Keplerian orbital elements.
    
    Keplerian orbital elements are returned in an array with the following elements:
        a: semi-major axis in AU
        q: pericenter distance in AU
        e: eccentricity 
        i: inclination in degrees
        ascNode: right ascension of the ascending node in degrees
        argPeri : argument of perihelion/perigee/pericenter in degrees
        anomaly : mean anomaly for elliptical orbits, parabolic anomaly for 
            parabolic orbits, and hyperbolic anomaly for hyperbolic orbits
            in degrees
        trueAnom_deg : true anomaly in degrees
    
    Parameters
    ----------
    elements_cart : `~numpy.ndarray` (6)
        Cartesian elements in units of AU and AU per day. 
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of 
        AU**3 / d**2. 
    
    Returns
    -------
    elements_kepler : `~numpy.ndarray (8)
        Keplerian elements with angles in degrees and semi-major
        axis in AU. 
        
    """
    r = elements_cart[:3]
    v = elements_cart[3:]
    v_mag = np.linalg.norm(v)
    r_mag = np.linalg.norm(r)
    
    sme = v_mag**2 / 2 - mu / r_mag

    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    n = np.cross(np.array([0, 0, 1]), h)
    n_mag = np.linalg.norm(n)

    e_vec = ((v_mag**2 - mu / r_mag) * r - (np.dot(r, v)) * v) / mu
    e = np.linalg.norm(e_vec)

    if e != 0.0:
        a = mu / (-2 * sme)
        p = a * (1 - e**2)
        q = a * (1 - e)
    else:
        a = np.inf
        p = h_mag**2 / mu
        q = a

    i_deg = np.degrees(np.arccos(h[2] / h_mag))

    ascNode_deg = np.degrees(np.arccos(n[0] / n_mag))
    if n[1] < 0:
        ascNode_deg = 360.0 - ascNode_deg

    argPeri_deg = np.degrees(np.arccos(np.dot(n, e_vec) / (n_mag * e)))
    if e_vec[2] < 0:
        argPeri_deg = 360.0 - argPeri_deg

    trueAnom_deg = np.degrees(np.arccos(np.dot(e_vec, r) / (e * r_mag)))
    if np.dot(r, v) < 0:
        trueAnom_deg = 360.0 - trueAnom_deg
    trueAnom_rad = np.radians(trueAnom_deg)
    
    if e < 1.0:
        eccentricAnom_rad = np.arctan2(np.sqrt(1 - e**2) * np.sin(trueAnom_rad), e + np.cos(trueAnom_rad))
        meanAnom_deg = np.degrees(eccentricAnom_rad - e * np.sin(eccentricAnom_rad))
        if meanAnom_deg < 0:
            meanAnom_deg += 360.0
    elif e == 1.0:
        parabolicAnom_rad = np.arctan(trueAnom_rad / 2)
        meanAnom_deg = np.inf
    else:
        hyperbolicAnom_rad = np.arcsinh(np.sin(trueAnom_rad) * np.sqrt(e**2 - 1) / (1 + e * np.cos(trueAnom_rad)))
        meanAnom_deg = np.degrees(e * np.sinh(hyperbolicAnom_rad) - hyperbolicAnom_rad)
        
    return np.array([a, q, e, i_deg, ascNode_deg, argPeri_deg, meanAnom_deg, trueAnom_deg])