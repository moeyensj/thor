import numpy as np
from scipy import roots

from ...constants import Constants as c
from ...coordinates import equatorialToEclipticCartesian
from ...coordinates import equatorialAngularToCartesian
from .gibbs import calcGibbs
from .herrick_gibbs import calcHerrickGibbs

__all__ = [
    "_calcV",
    "_calcA",
    "_calcB",
    "_calcLambdas",
    "_calcRhos",
    "_calcFG",
    "calcGauss",
    "gaussIOD"
]

MU = c.G * c.M_SUN

def _calcV(rho1hat, rho2hat, rho3hat):
    # Vector triple product that gives the area of 
    # the "volume of the parallelepiped" or according to 
    # to Milani et al. 2008: 3x volume of the pyramid with vertices q, r1, r2, r3.
    # Note that vector triple product rules apply here.
    return np.dot(np.cross(rho1hat, rho2hat), rho3hat)

def _calcA(q1, q2, q3, rho1hat, rho3hat, t31, t32, t21):
    # Equation 21 from Milani et al. 2008
    return np.dot(np.cross(np.linalg.norm(q2)**3 * rho1hat, rho3hat), (t32 * q1 - t31 * q2 + t21 * q3))

def _calcB(q1, q3, rho1hat, rho3hat, t31, t32, t21):
    # Equation 19 from Milani et al. 2008
    return np.dot(np.cross(MU / 6 * t32 * t21 * rho1hat, rho3hat), ((t31 + t32) * q1 + (t31 + t21) * q3))

def _calcLambdas(r2_mag, t31, t32, t21):
    # Equations 16 and 17 from Milani et al. 2008
    lambda1 = t32 / t31 * (1 + MU / (6 * r2_mag**3) * (t31**2 - t32**2))
    lambda3 = t21 / t31 * (1 + MU / (6 * r2_mag**3) * (t31**2 - t21**2))
    return lambda1, lambda3

def _calcRhos(lambda1, lambda3, q1, q2, q3, rho1hat, rho2hat, rho3hat, V):
    # This can be derived by taking a series of scalar products of the coplanarity condition equation
    # with cross products of unit vectors in the direction of the observer, in particular, see Chapter 9 in
    # Milani's book on the theory of orbit determination 
    numerator = -lambda1 * q1 + q2 - lambda3 * q3
    rho1_mag = np.dot(numerator, np.cross(rho2hat, rho3hat)) / (lambda1 * V)
    rho2_mag = np.dot(numerator, np.cross(rho1hat, rho3hat)) / V
    rho3_mag = np.dot(numerator, np.cross(rho1hat, rho2hat)) / (lambda3 * V)
    return np.dot(rho1_mag, rho1hat), np.dot(rho2_mag, rho2hat), np.dot(rho3_mag, rho3hat)

def _calcFG(r2_mag, t32, t21):
    # Calculate the Lagrange coefficients (Gauss f and g series)
    f1 = 1 - (1 / 2) * MU / r2_mag**3 * (-t21)**2
    f3 = 1 - (1 / 2) * MU / r2_mag**3 * t32**2
    g1 = -t21 - (1 / 6) * MU / r2_mag**3 * (-t21)**2
    g3 = t32 - (1 / 6) * MU / r2_mag**3 * t32**2
    return f1, g1, f3, g3

def calcGauss(r1, r2, r3, t1, t2, t3):
    """
    Calculates the velocity vector at the location of the second position vector (r2) with Gauss'
    original method.
    
    .. math::        
        f_1 \approx 1 - \frac{1}{2}\frac{\mu}{r_2^3} (t_1 - t_2)^2
        
        f_3 \approx 1 - \frac{1}{2}\frac{\mu}{r_2^3} (t_3 - t_2)^2
        
        g_1 \approx (t_1 - t_2) - \frac{1}{6}\frac{\mu}{r_2^3} (t_1 - t_2)^2
        
        g_3 \approx (t_3 - t_2) - \frac{1}{6}\frac{\mu}{r_2^3} (t_3 - t_2)^2
        
        \vec{v}_2 = \frac{1}{f_1 g_3 - f_3 g_1} (-f_3 \vec{r}_1 + f_1 \vec{r}_3)

        
    For more details on theory see Chapter 5 in Howard D. Curtis' "Orbital Mechanics for
    Engineering Students". 
    
    Parameters
    ----------
    r1 : `~numpy.ndarray` (3)
        Heliocentric position vector at time 1 in cartesian coordinates in units
        of AU. 
    r2 : `~numpy.ndarray` (3)
        Heliocentric position vector at time 2 in cartesian coordinates in units
        of AU. 
    r3 : `~numpy.ndarray` (3)
        Heliocentric position vector at time 3 in cartesian coordinates in units
        of AU.
    t1 : float 
        Time at r1. Units of MJD or JD work or any decimal time format (one day is 1.00) as 
        long as all times are given in the same format. 
    t2 : float
        Time at r2. Units of MJD or JD work or any decimal time format (one day is 1.00) as 
        long as all times are given in the same format. 
    t3 : float
        Time at r3. Units of MJD or JD work or any decimal time format (one day is 1.00) as 
        long as all times are given in the same format. 
    
    Returns
    -------
    v2 : `~numpy.ndarray` (3)
        Velocity of object at position r2 at time t2 in units of AU per day. 
    """
    t21 = t2 - t1
    t32 = t3 - t2
    r2_mag = np.linalg.norm(r2)
    f1, g1, f3, g3 = _calcFG(r2_mag, t32, t21)
    return (1 / (f1 * g3 - f3 * g1)) * (-f3 * r1 + f1 * r3)

def gaussIOD(coords_eq_ang, t, coords_obs, velocityMethod="gauss"):
    """
    Compute up to three intial orbits using three observations in angular equatorial
    coordinates. 
    
    Parameters
    ----------
    coords_eq_ang : `~numpy.ndarray` (3, 2)
        RA and Dec of three observations in units of degrees.
    t : `~numpy.ndarray` (3)
        Times of the three observations in units of decimal days (MJD or JD for example).
    coords_obs : `~numpy.ndarray` (3, 2)
        Heliocentric position vector of the observer at times t in units of AU.
    velocityMethod : {'gauss', gibbs', 'herrick+gibbs'}, optional
        Which method to use for calculating the velocity at the second observation.
        [Default = 'gauss']
        
    Returns
    -------
    orbits : `~numpy.ndarray` ((<=3, 6) or (0))
        Up to three preliminary orbits (as cartesian state vectors).
    """
    rho = equatorialToEclipticCartesian(equatorialAngularToCartesian(coords_eq_ang))
    rho1 = rho[0]
    rho2 = rho[1]
    rho3 = rho[2]
    q1 = coords_obs[0]
    q2 = coords_obs[1]
    q3 = coords_obs[2]
    q2_mag = np.linalg.norm(q2)

    rho1hat = rho1 / np.linalg.norm(rho1)
    rho2hat = rho2 / np.linalg.norm(rho2)
    rho3hat = rho3 / np.linalg.norm(rho3)
    
    t1 = t[0]
    t2 = t[1]
    t3 = t[2]
    t31 = t3 - t1
    t21 = t2 - t1
    t32 = t3 - t2
    
    A = _calcA(q1, q2, q3, rho1hat, rho3hat, t31, t32, t21)
    B = _calcB(q1, q3, rho1hat, rho3hat, t31, t32, t21)
    V = _calcV(rho1hat, rho2hat, rho3hat)
    coseps2 = np.dot(q2 / np.linalg.norm(q2), rho2hat) 
    C0 = V * t31 * q2_mag**4 / B
    h0 = - A / B
    
    if np.isnan(C0) or np.isnan(h0):
        return np.array([])
    
    # Find roots to eighth order polynomial
    all_roots = roots([
        C0**2,
        0,
        -q2_mag**2 * (h0**2 + 2 * C0 * h0 * coseps2 + C0**2),
        0,
        0,
        2 * q2_mag**5 * (h0 + C0 * coseps2),
        0,
        0,
        -q2_mag**8
    ])
    
    # Keep only positive real roots (which should at most be 3)
    r2_mags = np.real(all_roots[np.isreal(all_roots) & (np.real(all_roots) >= 0)])
    num_solutions = len(r2_mags)
    if num_solutions == 0:
        return np.array([])
    
    orbits = []
    for r2_mag in r2_mags:
        lambda1, lambda3 = _calcLambdas(r2_mag, t31, t32, t21)
        rho1, rho2, rho3 = _calcRhos(lambda1, lambda3, q1, q2, q3, rho1hat, rho2hat, rho3hat, V)

        # Test if we get the same rho2 as using equation 22 in Milani et al. 2008
        rho2_mag = (h0 - q2_mag**3 / r2_mag**3) * q2_mag / C0
        np.testing.assert_almost_equal(np.dot(rho2_mag, rho2hat), rho2)

        r1 = q1 + rho1
        r2 = q2 + rho2
        r3 = q3 + rho3
        
        if velocityMethod == "gauss":
            v2 = calcGauss(r1, r2, r3, t1, t2, t3)
        elif velocityMethod == "gibbs":
            v2 = calcGibbs(r1, r2, r3)
        elif velocityMethod == "herrick+gibbs":
            v2 = calcHerrickGibbs(r1, r2, r3, t1, t2, t3)
        else:
            raise ValueError("velocityMethod should be one of {'gauss', 'gibbs', 'herrick+gibbs'}")
        orbit = np.concatenate([r2, v2])
        
        if np.linalg.norm(v2) >= C:
            print("Velocity is greater than speed of light!")
        orbits.append(orbit)
    
    return np.array(orbits)
