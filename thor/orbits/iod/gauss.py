import numpy as np
from numpy import roots
from numba import jit

from ...constants import Constants as c
from ...coordinates import equatorialToEclipticCartesian
from ...coordinates import equatorialAngularToCartesian
from ..propagate import calcStumpff
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
    "_calcM",
    "_calcStateTransitionMatrix",
    "_iterateGaussIOD",
    "calcGauss",
    "gaussIOD"
]

MU = c.G * c.M_SUN
C = c.C

def _calcV(rho1_hat, rho2_hat, rho3_hat):
    # Vector triple product that gives the area of 
    # the "volume of the parallelepiped" or according to 
    # to Milani et al. 2008: 3x volume of the pyramid with vertices q, r1, r2, r3.
    # Note that vector triple product rules apply here.
    return np.dot(np.cross(rho1_hat, rho2_hat), rho3_hat)

def _calcA(q1, q2, q3, rho1_hat, rho3_hat, t31, t32, t21):
    # Equation 21 from Milani et al. 2008
    return np.linalg.norm(q2)**3 * np.dot(np.cross(rho1_hat, rho3_hat), (t32 * q1 - t31 * q2 + t21 * q3))

def _calcB(q1, q3, rho1_hat, rho3_hat, t31, t32, t21, mu=MU):
    # Equation 19 from Milani et al. 2008
    return mu / 6 * t32 * t21 * np.dot(np.cross(rho1_hat, rho3_hat), ((t31 + t32) * q1 + (t31 + t21) * q3))

def _calcLambdas(r2_mag, t31, t32, t21, mu=MU):
    # Equations 16 and 17 from Milani et al. 2008
    lambda1 = t32 / t31 * (1 + mu / (6 * r2_mag**3) * (t31**2 - t32**2))
    lambda3 = t21 / t31 * (1 + mu / (6 * r2_mag**3) * (t31**2 - t21**2))
    return lambda1, lambda3

def _calcRhos(lambda1, lambda3, q1, q2, q3, rho1_hat, rho2_hat, rho3_hat, V):
    # This can be derived by taking a series of scalar products of the coplanarity condition equation
    # with cross products of unit vectors in the direction of the observer, in particular, see Chapter 9 in
    # Milani's book on the theory of orbit determination 
    numerator = -lambda1 * q1 + q2 - lambda3 * q3
    rho1_mag = np.dot(numerator, np.cross(rho2_hat, rho3_hat)) / (lambda1 * V)
    rho2_mag = np.dot(numerator, np.cross(rho1_hat, rho3_hat)) / V
    rho3_mag = np.dot(numerator, np.cross(rho1_hat, rho2_hat)) / (lambda3 * V)
    rho1 = np.dot(rho1_mag, rho1_hat)
    rho2 = np.dot(rho2_mag, rho2_hat)
    rho3 = np.dot(rho3_mag, rho3_hat)
    return rho1, rho2, rho3

def _calcFG(r2_mag, t32, t21, mu=MU):
    # Calculate the Lagrange coefficients (Gauss f and g series)
    f1 = 1 - (1 / 2) * mu / r2_mag**3 * (-t21)**2 
    f3 = 1 - (1 / 2) * mu / r2_mag**3 * t32**2
    g1 = -t21 - (1 / 6) * mu / r2_mag**3 * (-t21)**2
    g3 = t32 - (1 / 6) * mu / r2_mag**3 * t32**2
    return f1, g1, f3, g3

def _calcM(r0_mag, r_mag, f, g, f_dot, g_dot, c0, c1, c2, c3, c4, c5, alpha, chi, mu=MU):
    # Universal variables will differ between different texts and works in the literature.
    # c0, c1, c2, c3, c4, c5 are expected to be 
    w = chi / np.sqrt(mu)
    alpha_alt = - mu * alpha
    U0 = (1 - alpha_alt * chi**2) * c0
    U1 = (chi - alpha_alt * chi**3) * c1 / np.sqrt(mu)
    U2 = chi**2 * c2 / mu
    U3 = chi**3 * c3 / mu**(3/2)
    U4 = chi**4 * c4 / mu**(2)
    U5 = chi**5 * c5 / mu**(5/2)
    
    F = f_dot
    G = g_dot
    
    # Equations 18 and 19 in Sheppard 1985
    U = (U2 * U3 + w * U4 - 3 * U5) / 3
    W = g * U2 + 3 * mu * U
    
    # Calculate elements of the M matrix
    m11 = (U0 / (r_mag * r0_mag) + 1 / r0_mag**2 + 1 / r_mag**2) * F - (mu**2 * W) / (r_mag * r0_mag)**3
    m12 = F * U1 / r_mag + (G - 1) / r_mag**2
    m13 = (G - 1) * U1 / r_mag - (mu * W) / r_mag**3
    m21 = -F * U1 / r0_mag - (f - 1) / r0_mag**2
    m22 = -F * U2
    m23 = -(G - 1) * U2
    m31 = (f - 1) * U1 / r0_mag - (mu * W) / r0_mag**3
    m32 = (f - 1) * U2
    m33 = g * U2 - W
    
    # Combine elements into matrix
    M = np.array([
        [m11, m12, m13],
        [m21, m22, m23],
        [m31, m32, m33],
    ])
    
    return M

def _calcStateTransitionMatrix(M, r0, v0, f, g, f_dot, g_dot, r, v):
    I = np.identity(3)
    state_0 = np.vstack((r0, v0))
    state_1 = np.vstack((r, v))
    
    phi11 = f * I + state_1.T @ M[1:, :2] @ state_0 
    phi12 = g * I + state_1.T @ M[1:, 1:] @ state_0
    phi21 = f_dot * I + state_1.T @ M[:2, :2] @ state_0
    phi22 = g_dot * I + state_1.T @ M[:2, 1:] @ state_0
    
    phi = np.block([
        [phi11, phi12],
        [phi21, phi22]
    ])
    return phi

def _iterateGaussIOD(orbit, t21, t32, q1, q2, q3, rho1, rho2, rho3, mu=MU, max_iter=10, tol=1e-15):
    # Iterate over the polynomial solution from Gauss using the universal anomaly 
    # formalism until the solution converges or the maximum number of iterations is reached
    
    # Calculate variables that won't change per iteration
    sqrt_mu = np.sqrt(mu)
    
    # Calculate magntiude and unit rho vectors
    rho1_mag = np.linalg.norm(rho1)
    rho2_mag = np.linalg.norm(rho2)
    rho3_mag = np.linalg.norm(rho3)
    rho1_hat = rho1 / rho1_mag
    rho2_hat = rho2 / rho2_mag
    rho3_hat = rho3 / rho3_mag
    
    orbit_iter = orbit
    i = 0
    for i in range(max_iter):
        # Grab orbit position and velocity vectors 
        # These should belong to the state of the object at the time of the second
        # observation after applying Gauss's method the first time
        r = orbit_iter[:3]
        v = orbit_iter[3:]
        v_mag = np.linalg.norm(v)
        r_mag = np.linalg.norm(r)

        # Calculate the inverse semi-major axis
        # Note: the definition of alpha will change between different works in the literature.
        #   Here alpha is defined as 1 / a where a is the semi-major axis of the orbit
        alpha = -v_mag**2 / mu + 2 / r_mag

        # Calculate the universal anomaly for both the first and third observations
        # then calculate the Lagrange coefficients and the state for each observation.
        # Use those to calculate the state transition matrix
        for j, dt in enumerate([-t21, t32]):
            # Calculate the universal anomaly 
            # Universal anomaly here is defined in such a way that it satisfies the following
            # differential equation:
            #   d\chi / dt = \sqrt{mu} / r
            chi = calcChi(orbit_iter, dt, mu=mu, max_iter=10, tol=tol)
            chi2 = chi**2

            # Calculate the values of the Stumpff functions
            psi = alpha * chi2 
            c0, c1, c2, c3, c4, c5 = calcStumpff(psi)

            # Calculate the Lagrange coefficients 
            # and the corresponding state vector
            f = 1 - chi2 / r_mag * c2
            g = dt - 1 / sqrt_mu * chi**3 * c3

            r_new = f * r + g * v
            r_new_mag = np.linalg.norm(r_new)

            f_dot = sqrt_mu / (r_mag * r_new_mag) * (alpha * chi**3 * c3 - chi)
            g_dot = 1 - chi2 / r_new_mag * c2

            v_new = f_dot * r + g_dot * v
            
            # Calculate M matrix and use it to calculate the state transition matrix
            M = _calcM(r_mag, r_new_mag, f, g, f_dot, g_dot, c0, c1, c2, c3, c4, c5, alpha, chi, mu=mu)
            STM = _calcStateTransitionMatrix(M, r, v, f, g, f_dot, g_dot, r_new, v_new)
            
            if j == 0:
                STM1 = STM
                v1 = v_new
                r1 = r_new
            else:
                STM3 = STM
                v3 = v_new
                r3 = r_new

        # Create phi error vector: as the estimate of the orbit 
        # improves the elements in this vector should approach 0.
        phi = np.hstack((
            r1 - q1 - rho1_mag * rho1_hat, 
            r - q2 - rho2_mag * rho2_hat, 
            r3 - q3 - rho3_mag * rho3_hat))
        if np.linalg.norm(phi) == 0:
            break

        dphi = np.zeros((9, 9), dtype=float)
        dphi[0:3, 0:3] = STM1[0:3, 0:3]   # dr1/dr2
        dphi[3:6, 0:3] = np.identity(3)   # dr2/dr2
        dphi[6:9, 0:3] = STM3[0:3, 0:3]   # dr3/dr2

        dphi[0:3, 3:6] = STM1[0:3, 3:6]   # dr1/dv2
        dphi[3:6, 3:6] = np.zeros((3, 3)) # dr2/dv2
        dphi[6:9, 3:6] = STM3[0:3, 3:6]   # dr3/dv2

        dphi[0:3,6] = -v1 / C - rho1_hat

        dphi[0:3,7] = v1 / C
        dphi[3:6,7] = -rho2_hat
        dphi[6:9,7] = v3 / C

        dphi[6:9,8] = -v3 / C - rho3_hat

        delta = np.linalg.solve(dphi, phi)
        orbit_iter -= delta[0:6]
        rho1_mag -= delta[6]
        rho2_mag -= delta[7]
        rho3_mag -= delta[8]
                
        i += 1
        if i >= max_iter:
            break
    
    return orbit_iter

def calcGauss(r1, r2, r3, t1, t2, t3):
    """
    Calculates the velocity vector at the location of the second position vector (r2) with Gauss's
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
    rho1_hat = rho[0, :]
    rho2_hat = rho[1, :]
    rho3_hat = rho[2, :]
    q1 = coords_obs[0,:]
    q2 = coords_obs[1,:]
    q3 = coords_obs[2,:]
    q2_mag = np.linalg.norm(q2)
    
    # Make sure rhohats are unit vectors
    rho1_hat = rho1_hat / np.linalg.norm(rho1_hat)
    rho2_hat = rho2_hat / np.linalg.norm(rho2_hat)
    rho3_hat = rho3_hat / np.linalg.norm(rho3_hat)
    
    t1 = t[0]
    t2 = t[1]
    t3 = t[2]
    t31 = t3 - t1
    t21 = t2 - t1
    t32 = t3 - t2
    
    A = _calcA(q1, q2, q3, rho1_hat, rho3_hat, t31, t32, t21)
    B = _calcB(q1, q3, rho1_hat, rho3_hat, t31, t32, t21)
    V = _calcV(rho1_hat, rho2_hat, rho3_hat)
    coseps2 = np.dot(q2, rho2_hat) / q2_mag
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
        rho1, rho2, rho3 = _calcRhos(lambda1, lambda3, q1, q2, q3, rho1_hat, rho2_hat, rho3_hat, V)

        # Test if we get the same rho2 as using equation 22 in Milani et al. 2008
        rho2_mag = (h0 - q2_mag**3 / r2_mag**3) * q2_mag / C0
        np.testing.assert_almost_equal(np.dot(rho2_mag, rho2_hat), rho2)

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
                                     rho1, rho2, rho3,
                                     mu=mu, max_iter=max_iter, tol=tol)
        
        if np.linalg.norm(orbit[3:]) >= C:
            print("Velocity is greater than speed of light!")
        orbits.append(orbit)
    
    return np.array(orbits)