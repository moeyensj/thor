import numpy as np

from scipy import roots
from astropy import constants as c
from astropy import units as u

MU = (c.G * c.M_sun).to(u.AU**3 / u.day**2).value

__all__ = [
    "_calcV",
    "_calcA",
    "_calcB",
    "_calcLambdas",
    "_calcRhos",
    "_calcFG",
    "calcGauss",
]


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
    Calculates the velocity vector at the second position of an object Gauss'
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
