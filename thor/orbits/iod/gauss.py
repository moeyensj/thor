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
