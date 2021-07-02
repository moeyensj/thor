import numpy as np

from ..constants import Constants as c
from .stumpff import calcStumpff
from .chi import calcChi

__all__ = [
    "_calcM",
    "_calcStateTransitionMatrix",
    "iterateStateTransition"
]

MU = c.MU
C = c.C

def _calcM(r0_mag, r_mag, f, g, f_dot, g_dot, c0, c1, c2, c3, c4, c5, alpha, chi, mu=MU):
    # Universal variables will differ between different texts and works in the literature.
    # c0, c1, c2, c3, c4, c5 are expected to follow the Battin formalism (adopted by both
    # Vallado and Curtis in their books). The M matrix is proposed by Shepperd 1985 and follows
    # the Goodyear formalism. Conversions between the two formalisms can be derived from Table 1 in
    # Everhart & Pitkin 1982.
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

    # Equations 18 and 19 in Shepperd 1985
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

def iterateStateTransition(orbit, t21, t32, q1, q2, q3, rho1, rho2, rho3, light_time=True, mu=MU, max_iter=10, tol=1e-15):
    """
    Improve an initial orbit by iteratively solving for improved Langrange coefficients and minimizing the phi error vector
    by calculating the state transition matrix required to achieve this minimization.

    Parameters
    ----------
    orbit : `~numpy.ndarray` (6)
        Preliminary orbit from IOD to improve by iteration.
    t21 : float
        Time between the second and first observation (units of decimal days).
    t32 : float
        Time between the third and second observation (units of decimal days).
    q1 : `~numpy.ndarray` (3)
        Observer position vector at first observation.
    q2 : `~numpy.ndarray` (3)
        Observer position vector at second observation.
    q3 : `~numpy.ndarray` (3)
        Observer position vector at third observation.
    rho1 : `~numpy.ndarray` (3)
        Observer to target position vector at the first observation.
    rho2 : `~numpy.ndarray` (3)
        Observer to target position vector at the second observation.
    rho3 : `~numpy.ndarray` (3)
        Observer to target position vector at the third observation.
    light_time : bool, optional
        Correct for light travel time.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        AU**3 / d**2.
    max_iter : int, optional
        Maximum number of iterations over which to converge.
    tol : float, optional
        Numerical tolerance to which to compute chi using the Newtown-Raphson
        method.


    Returns
    -------
    orbit_iter : `~numpy.ndarray` (7)
        Improved orbit after iterating using the state transition matrix.
    """
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
    orbit_iter_prev = orbit
    i = 0
    phi_mag_prev = 1e10
    for i in range(max_iter):
        # Grab orbit position and velocity vectors
        # These should belong to the state of the object at the time of the second
        # observation after applying Gauss's method the first time
        r = np.ascontiguousarray(orbit_iter[0:3])
        v = np.ascontiguousarray(orbit_iter[3:6])
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
            if light_time is True:
                if j == 1:
                    dt += (rho2_mag - rho1_mag) / C
                else:
                    dt -= (rho3_mag - rho2_mag) / C

            # Calculate the universal anomaly
            # Universal anomaly here is defined in such a way that it satisfies the following
            # differential equation:
            #   d\chi / dt = \sqrt{mu} / r
            chi, c0, c1, c2, c3, c4, c5 = calcChi(r, v, dt, mu=mu, max_iter=100, tol=tol)
            chi2 = chi**2

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
        phi_mag_iter = np.linalg.norm(phi)
        if phi_mag_iter < 1e-15:
            break

        dphi = np.zeros((9, 9), dtype=float)
        dphi[0:3, 0:3] = STM1[0:3, 0:3]   # dr1/dr2
        dphi[3:6, 0:3] = np.identity(3)   # dr2/dr2
        dphi[6:9, 0:3] = STM3[0:3, 0:3]   # dr3/dr2

        dphi[0:3, 3:6] = STM1[0:3, 3:6]   # dr1/dv2
        dphi[3:6, 3:6] = np.zeros((3, 3)) # dr2/dv2
        dphi[6:9, 3:6] = STM3[0:3, 3:6]   # dr3/dv2

        if light_time is True:
            dphi[0:3,6] = -v1 / C - rho1_hat
            dphi[0:3,7] = v1 / C
            dphi[3:6,7] = -rho2_hat
            dphi[6:9,7] = v3 / C
            dphi[6:9,8] = -v3 / C - rho3_hat
        else:
            dphi[0:3,6] = -rho1_hat
            dphi[3:6,7] = -rho2_hat
            dphi[6:9,8] = -rho3_hat

        delta = np.linalg.solve(dphi, phi)
        orbit_iter -= delta[0:6]
        rho1_mag -= delta[6]
        rho2_mag -= delta[7]
        rho3_mag -= delta[8]

        if np.any(np.isnan(orbit_iter)):
            orbit_iter = orbit_iter_prev
            break

        i += 1
        orbit_iter_prev = orbit_iter
        phi_mag_prev = phi_mag_iter
        if i >= max_iter:
            break

    return orbit_iter
