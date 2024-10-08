import numpy as np
from adam_core.constants import Constants as c
from adam_core.dynamics.lagrange import (
    apply_lagrange_coefficients,
    calc_lagrange_coefficients,
)

from .state_transition import calcStateTransitionMatrix

__all__ = ["iterateStateTransition"]

MU = c.MU
C = c.C


def iterateStateTransition(
    orbit,
    t21,
    t32,
    q1,
    q2,
    q3,
    rho1,
    rho2,
    rho3,
    light_time=True,
    mu=MU,
    max_iter=10,
    tol=1e-15,
):
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
    for i in range(max_iter):
        # Grab orbit position and velocity vectors
        # These should belong to the state of the object at the time of the second
        # observation after applying Gauss's method the first time
        r = np.ascontiguousarray(orbit_iter[0:3])
        v = np.ascontiguousarray(orbit_iter[3:6])

        # Calculate the universal anomaly for both the first and third observations
        # then calculate the Lagrange coefficients and the state for each observation.
        # Use those to calculate the state transition matrix
        for j, dt in enumerate([-t21, t32]):
            if light_time is True:
                if j == 1:
                    dt += (rho2_mag - rho1_mag) / C
                else:
                    dt -= (rho3_mag - rho2_mag) / C

            # Calculate the universal anomaly, stumpff functions and the Lagrange coefficients
            # Universal anomaly here is defined in such a way that it satisfies the following
            # differential equation:
            #   d\chi / dt = \sqrt{mu} / r
            # and the corresponding state vector
            lagrange_coeffs, stumpff_coeffs, chi = calc_lagrange_coefficients(
                r, v, dt, mu=mu, max_iter=max_iter, tol=tol
            )
            r_new, v_new = apply_lagrange_coefficients(r, v, *lagrange_coeffs)

            # Calculate the state transition matrix
            STM = calcStateTransitionMatrix(orbit_iter, dt, mu=mu, max_iter=100, tol=1e-15)

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
        phi = np.hstack(
            (
                r1 - q1 - rho1_mag * rho1_hat,
                r - q2 - rho2_mag * rho2_hat,
                r3 - q3 - rho3_mag * rho3_hat,
            )
        )
        phi_mag_iter = np.linalg.norm(phi)
        if phi_mag_iter < 1e-15:
            break

        dphi = np.zeros((9, 9), dtype=float)
        dphi[0:3, 0:3] = STM1[0:3, 0:3]  # dr1/dr2
        dphi[3:6, 0:3] = np.identity(3)  # dr2/dr2
        dphi[6:9, 0:3] = STM3[0:3, 0:3]  # dr3/dr2

        dphi[0:3, 3:6] = STM1[0:3, 3:6]  # dr1/dv2
        dphi[3:6, 3:6] = np.zeros((3, 3))  # dr2/dv2
        dphi[6:9, 3:6] = STM3[0:3, 3:6]  # dr3/dv2

        if light_time is True:
            dphi[0:3, 6] = -v1 / C - rho1_hat
            dphi[0:3, 7] = v1 / C
            dphi[3:6, 7] = -rho2_hat
            dphi[6:9, 7] = v3 / C
            dphi[6:9, 8] = -v3 / C - rho3_hat
        else:
            dphi[0:3, 6] = -rho1_hat
            dphi[3:6, 7] = -rho2_hat
            dphi[6:9, 8] = -rho3_hat

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
        if i >= max_iter:
            break

    return orbit_iter
