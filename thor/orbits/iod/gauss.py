import numpy as np
from scipy import roots

from ...constants import Constants as c
from ...coordinates import equatorialToEclipticCartesian
from ...coordinates import equatorialAngularToCartesian
from ..propagate import calcC2C3
from ..propagate import calcChi
from .gibbs import calcGibbs
from .herrick_gibbs import calcHerrickGibbs

__all__ = [
    "_calcV",
    "_calcA",
    "_calcB",
    "_calcLambdas",
    "_calcRhos",
    "_calcFG",
    "_iterateGaussIOD",
    "calcGauss",
    "gaussIOD"
]

MU = c.G * c.M_SUN
C = c.C

#@jit(["f8(f8[::1], f8[::1], f8[::1])"], nopython=True)
def _calcV(rho1hat, rho2hat, rho3hat):
    # Vector triple product that gives the area of 
    # the "volume of the parallelepiped" or according to 
    # to Milani et al. 2008: 3x volume of the pyramid with vertices q, r1, r2, r3.
    # Note that vector triple product rules apply here.
    return np.dot(np.cross(rho1hat, rho2hat), rho3hat)

#@jit(["f8(f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8, f8, f8)"], nopython=True)
def _calcA(q1, q2, q3, rho1hat, rho3hat, t31, t32, t21):
    # Equation 21 from Milani et al. 2008
    return np.linalg.norm(q2)**3 * np.dot(np.cross(rho1hat, rho3hat), (t32 * q1 - t31 * q2 + t21 * q3))

#@jit(["f8(f8[::1], f8[::1], f8[::1], f8[::1], f8, f8, f8)"], nopython=True)
def _calcB(q1, q3, rho1hat, rho3hat, t31, t32, t21):
    # Equation 19 from Milani et al. 2008
    return MU / 6 * t32 * t21 * np.dot(np.cross(rho1hat, rho3hat), ((t31 + t32) * q1 + (t31 + t21) * q3))

#@jit(["UniTuple(f8, 2)(f8, f8, f8, f8)"], nopython=True)
def _calcLambdas(r2_mag, t31, t32, t21):
    # Equations 16 and 17 from Milani et al. 2008
    lambda1 = t32 / t31 * (1 + MU / (6 * r2_mag**3) * (t31**2 - t32**2))
    lambda3 = t21 / t31 * (1 + MU / (6 * r2_mag**3) * (t31**2 - t21**2))
    return lambda1, lambda3

#@jit(["UniTuple(f8[::1], 3)(f8, f8, f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8[::1], f8)"], nopython=True)
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

def _iterateGaussIOD(orbit, t21, t32, q1, q2, q3, rho1hat, rho2hat, rho3hat, mu=MU, max_iter=10, tol=1e-15):
    # Iterate over the polynomial solution from Gauss using the universal anomaly 
    # formalism until the solution converges or the maximum number of iterations is reached
    
    # Calculate variables that won't change per iteration
    sqrt_mu = np.sqrt(mu)
    V = _calcV(rho1hat, rho2hat, rho3hat)
    
    # Set up temporary variables that change iteration to iteration
    orbit_iter = orbit
    rho1_ratio = 1e10
    rho2_ratio = 1e10
    rho3_ratio = 1e10
    rho1_mag = -1e10
    rho2_mag = -1e10
    rho3_mag = -1e10
    
    i = 0
    while ((np.abs(rho1_ratio) > tol) 
            & (np.abs(rho2_ratio) > tol) 
            & (np.abs(rho3_ratio) > tol)):
        
        r = orbit_iter[:3]
        v = orbit_iter[3:]
        v_mag = np.linalg.norm(v)
        r_mag = np.linalg.norm(r)
        
        # Calculate the universal anomaly for both the first and third observations
        # then calculate the Lagrange coefficients
        for j, dt in enumerate([-t21, t32]):
            alpha = -v_mag**2 / mu + 2 / r_mag
            
            chi = calcChi(orbit_iter, dt, mu=mu, max_iter=max_iter, tol=tol)
            chi2 = chi**2

            psi = alpha * chi2
            c2, c3 = calcC2C3(psi)

            f = 1 - chi**2 / r_mag * c2
            g = dt - 1 / sqrt_mu * chi**3 * c3

            if j == 0:
                g1 = g
                f1 = f
            else:
                g3 = g
                f3 = f
    
        # Calculate the coplanarity coefficients
        lambda1 = g3 / (f1 * g3 - f3 * g1) 
        lambda3 = -g1 / (f1 * g3 - f3 * g1)   
        
        # Calculate new topocentric observer to target vectors
        rho1_temp, rho2_temp, rho3_temp = _calcRhos(lambda1, lambda3, 
                                                    q1, q2, 
                                                    q3, rho1hat, 
                                                    rho2hat, rho3hat, 
                                                    V)
        
        # Calculate heliocentric position vector
        r1 = q1 + rho1_temp
        r2 = q2 + rho2_temp
        r3 = q3 + rho3_temp
        
        # Update topocentric observer to target magnitudes ratios
        rho1_ratio = rho1_mag / np.linalg.norm(rho1_temp)
        rho2_ratio = rho2_mag / np.linalg.norm(rho2_temp)
        rho3_ratio = rho3_mag / np.linalg.norm(rho3_temp)
        
        # Update running topocentric observer to target magnitudes variables
        rho1_mag = np.linalg.norm(rho1_temp)
        rho2_mag = np.linalg.norm(rho2_temp)
        rho3_mag = np.linalg.norm(rho3_temp)

        # Calculate the velocity at the second observation
        v2 = 1 / (f1 * g3 - f3 * g1) * (-f3 * r1 + f1 * r3)

        # Update the guess of the orbit
        orbit_iter = np.concatenate((r2, v2))
        
        i += 1
        if i >= max_iter:
            break
    
    return orbit_iter

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

def gaussIOD(coords_eq_ang, t, coords_obs, velocity_method="gibbs", iterate=True, mu=MU, max_iter=10, tol=1e-15):
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
    velocity_method : {'gauss', gibbs', 'herrick+gibbs'}, optional
        Which method to use for calculating the velocity at the second observation.
        [Default = 'gibbs']
    iterate : bool, optional
        Iterate initial orbit using universal anomaly to better approximate the 
        Lagrange coefficients. 
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of 
        AU**3 / d**2. 
    max_iter : int, optional
        Maximum number of iterations over which to converge to solution.
    tol : float, optional
        Numerical tolerance to which to compute chi using the Newtown-Raphson 
        method. 
        
    Returns
    -------
    orbits : `~numpy.ndarray` ((<3, 6) or (0))
        Up to three preliminary orbits (as cartesian state vectors).
    """
    rho = equatorialToEclipticCartesian(equatorialAngularToCartesian(np.radians(coords_eq_ang)))
    rho1hat = rho[0, :]
    rho2hat = rho[1, :]
    rho3hat = rho[2, :]
    q1 = coords_obs[0,:]
    q2 = coords_obs[1,:]
    q3 = coords_obs[2,:]
    q2_mag = np.linalg.norm(q2)
    
    # Make sure rhohats are unit vectors
    rho1hat = rho1hat / np.linalg.norm(rho1hat)
    rho2hat = rho2hat / np.linalg.norm(rho2hat)
    rho3hat = rho3hat / np.linalg.norm(rho3hat)
    
    t1 = t[0]
    t2 = t[1]
    t3 = t[2]
    t31 = t3 - t1
    t21 = t2 - t1
    t32 = t3 - t2
    
    A = _calcA(q1, q2, q3, rho1hat, rho3hat, t31, t32, t21)
    B = _calcB(q1, q3, rho1hat, rho3hat, t31, t32, t21)
    V = _calcV(rho1hat, rho2hat, rho3hat)
    coseps2 = np.dot(q2, rho2hat) / q2_mag
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
        
        if velocity_method == "gauss":
            v2 = calcGauss(r1, r2, r3, t1, t2, t3)
        elif velocity_method == "gibbs":
            v2 = calcGibbs(r1, r2, r3)
        elif velocity_method == "herrick+gibbs":
            v2 = calcHerrickGibbs(r1, r2, r3, t1, t2, t3)
        else:
            raise ValueError("velocity_method should be one of {'gauss', 'gibbs', 'herrick+gibbs'}")
        orbit = np.concatenate([r2, v2])
        
        if iterate == True:
            orbit = _iterateGaussIOD(orbit, t21, t32, 
                                     q1, q2, q3, 
                                     rho1hat, rho2hat, rho3hat, 
                                     mu=MU, max_iter=max_iter, tol=tol)
        
        if np.linalg.norm(v2) >= C:
            print("Velocity is greater than speed of light!")
        orbits.append(orbit)
    
    return np.array(orbits)