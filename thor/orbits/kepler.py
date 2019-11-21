import numpy as np
from numba import jit

from ..constants import Constants as c

__all__ = ["convertCartesianToKeplerian",
           "convertKeplerianToCartesian"]

MU = c.G * c.M_SUN

@jit(["f8[:,:](f8[:,:], f8)"], nopython=True)
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
        meanAnom : mean anomaly in degrees
        trueAnom_deg : true anomaly in degrees
    
    Parameters
    ----------
    elements_cart : `~numpy.ndarray` (N, 6)
        Cartesian elements in units of AU and AU per day. 
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of 
        AU**3 / d**2. 
    
    Returns
    -------
    elements_kepler : `~numpy.ndarray (N, 8)
        Keplerian elements with angles in degrees and semi-major
        axis in AU. 
        
    """
    elements_kepler = []
    for i in range(len(elements_cart)):
        r = elements_cart[i, :3]
        v = elements_cart[i, 3:]
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
            raise ValueError("Parabolic orbits not yet implemented!")
            parabolicAnom_rad = np.arctan(trueAnom_rad / 2)
            meanAnom_deg = np.degrees(parabolicAnom_rad + parabolicAnom_rad**3 / 3)
        else:
            hyperbolicAnom_rad = np.arcsinh(np.sin(trueAnom_rad) * np.sqrt(e**2 - 1) / (1 + e * np.cos(trueAnom_rad)))
            meanAnom_deg = np.degrees(e * np.sinh(hyperbolicAnom_rad) - hyperbolicAnom_rad)
        
        elements_kepler.append([a, q, e, i_deg, ascNode_deg, argPeri_deg, meanAnom_deg, trueAnom_deg])
        
    return np.array(elements_kepler)

@jit(["f8[:,:](f8[:,:], f8, f8, i8)"], nopython=True)
def convertKeplerianToCartesian(elements_kepler, mu=MU, tol=1e-15, maxIterations=100):
    """
    Convert Keplerian orbital elements to cartesian orbital elements.
    
    Keplerian orbital elements should have following elements:
        a: semi-major axis in AU
        e: eccentricity 
        i: inclination in degrees
        ascNode: right ascension of the ascending node in degrees
        argPeri : argument of perihelion/perigee/pericenter in degrees
        meanAnom : mean anomaly in degrees
    
    Parameters
    ----------
    elements_kepler : `~numpy.ndarray` (N, 6)
        Keplerian elements with angles in degrees and semi-major
        axis in AU.   
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of 
        AU**3 / d**2. 
    
    Returns
    -------
    elements_cart : `~numpy.ndarray (N, 6)
        Cartesian elements in units of AU and AU per day.
    """
    elements_cart = []
    for i in range(len(elements_kepler)):
        a = elements_kepler[i, 0]
        e = elements_kepler[i, 1]
        p = a * (1 - e**2)

        i_deg = elements_kepler[i, 2]
        i_rad = np.radians(i_deg)

        ascNode_deg = elements_kepler[i, 3]
        ascNode_rad = np.radians(ascNode_deg)

        argPeri_deg = elements_kepler[i, 4]
        argPeri_rad = np.radians(argPeri_deg)

        meanAnom_deg = elements_kepler[i, 5]
        meanAnom_rad = np.radians(meanAnom_deg)

        if e < 1.0:
            iterations = 0
            ratio = 1e10
            eccentricAnom_rad = meanAnom_rad 

            while np.abs(ratio) > tol:
                f = eccentricAnom_rad - e * np.sin(eccentricAnom_rad) - meanAnom_rad 
                fp = 1 - e * np.cos(eccentricAnom_rad)
                ratio = f / fp
                eccentricAnom_rad -= ratio
                iterations += 1
                if iterations >= maxIterations:
                    break

            trueAnom_rad = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(eccentricAnom_rad/2), np.sqrt(1 - e) * np.cos(eccentricAnom_rad/2))

        elif e == 1.0:
            raise ValueError("Parabolic orbits not yet implemented!")
            
        else:
            iterations = 0
            ratio = 1e10
            hyperbolicAnom_rad = meanAnom_rad / (e - 1)

            while np.abs(ratio) > tol:
                f = meanAnom_rad - e * np.sinh(hyperbolicAnom_rad) + hyperbolicAnom_rad 
                fp =  e * np.cosh(hyperbolicAnom_rad) - 1
                ratio = f / fp
                hyperbolicAnom_rad += ratio
                iterations += 1
                if iterations >= maxIterations:
                    break

            trueAnom_rad = 2 * np.arctan(np.sqrt(e + 1) * np.sinh(hyperbolicAnom_rad / 2) / (np.sqrt(e - 1) * np.cosh(hyperbolicAnom_rad / 2)))

        r_PQW = np.array([
            p * np.cos(trueAnom_rad) / (1 + e * np.cos(trueAnom_rad)),
            p * np.sin(trueAnom_rad) / (1 + e * np.cos(trueAnom_rad)),
            0
        ])

        v_PQW = np.array([
            -np.sqrt(mu/p) * np.sin(trueAnom_rad),
            np.sqrt(mu/p) * (e + np.cos(trueAnom_rad)),
            0
        ])

        cos_ascNode = np.cos(ascNode_rad)
        sin_ascNode = np.sin(ascNode_rad)
        cos_argPeri = np.cos(argPeri_rad)
        sin_argPeri = np.sin(argPeri_rad)
        cos_i = np.cos(i_rad)
        sin_i = np.sin(i_rad)

        P1 = np.array([
            [cos_argPeri, -sin_argPeri, 0.],
            [sin_argPeri, cos_argPeri, 0.],
            [0., 0., 1.],
        ])

        P2 = np.array([
            [1., 0., 0.],
            [0., cos_i, -sin_i],
            [0., sin_i, cos_i],
        ])

        P3 = np.array([
            [cos_ascNode, -sin_ascNode, 0.],
            [sin_ascNode, cos_ascNode, 0.],
            [0., 0., 1.],
        ])

        rotation_matrix = P3 @ P2 @ P1
        r = rotation_matrix @ r_PQW
        v = rotation_matrix @ v_PQW

        elements_cart.append([r[0], r[1], r[2], v[0], v[1], v[2]])
    
    return np.array(elements_cart)