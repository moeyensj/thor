import numpy as np
from astropy import constants as c
from astropy import units as u

__all__ = ["convertCartesianToKeplerian"]

MU = (c.G * c.M_sun).to(u.AU**3 / u.day**2).value

def convertCartesianToKeplerian(elements_cart):
    """
    Convert cartesian orbital elements to Keplerian orbital elements.
    
    Parameters
    ----------
    elements_cart : `~numpy.ndarray` (6)
        Cartesian elements in units of AU and AU per day. 
    
    Returns
    -------
    elements_kepler : `~numpy.ndarray (6)
        Keplerian elements with angles in degrees and semi-major
        axis in AU. 
    """
    r = elements_cart[:3]
    v = elements_cart[3:]
    v_mag = np.linalg.norm(v)
    r_mag = np.linalg.norm(r)
    
    # Calculate specific mechanical energy
    sme = v_mag**2 / 2 - MU / r_mag

    # Calculate angular momentum
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    n = np.cross(np.array([0, 0, 1]), h)
    n_mag = np.linalg.norm(n)

    e = ((v_mag**2 - MU / r_mag) * r - (np.dot(r, v)) * v) / MU
    e_mag = np.linalg.norm(e)

    if e_mag != 0.0:
        a = MU / (-2 * sme)
        p = a * (1 - e_mag**2)
    else:
        a = np.inf
        p = h_mag**2 / MU

    i = np.degrees(np.arccos(h[2] / h_mag))

    ascNode = np.degrees(np.arccos(n[0] / n_mag))
    if n[1] < 0:
        ascNode = 360.0 - ascNode

    argPeri = np.degrees(np.arccos(np.dot(n, e) / (n_mag * e_mag)))
    if e[2] < 0:
        argPeri = 360.0 - argPeri

    trueAnom = np.degrees(np.arccos(np.dot(e, r) / (e_mag * r_mag)))
    if np.dot(r, v) < 0:
        trueAnom = 360.0 - trueAnom
    
    if e_mag < 1.0:
        eccentricAnom = np.arccos((e_mag + np.cos(np.radians(trueAnom))) / (1 + e_mag * np.cos(np.radians(trueAnom))))
        
    meanAnom = np.degrees(eccentricAnom - e_mag * np.sin(eccentricAnom))
        
    return np.array([a, e_mag, i, ascNode, argPeri, meanAnom])