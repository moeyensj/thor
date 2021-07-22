import numpy as np
from numba import jit

from ..constants import Constants as c
from .lagrange import calcLagrangeCoeffs
from .lagrange import applyLagrangeCoeffs

__all__ = [
    "calcMMatrix",
    "calcStateTransitionMatrix"
]

MU = c.MU

@jit("f8[:,:](f8[:], f8[:], UniTuple(f8, 4), UniTuple(f8, 6), f8, f8, f8)", nopython=True, cache=True)
def calcMMatrix(r0, r1, lagrange_coeffs, stumpff_coeffs, chi, alpha, mu=MU):

    # Extract relevant quanitities and calculate vector magnitudes
    c0, c1, c2, c3, c4, c5 = stumpff_coeffs
    f, g, f_dot, g_dot = lagrange_coeffs
    r0_mag = np.linalg.norm(r0)
    r1_mag = np.linalg.norm(r1)
    sqrt_mu = np.sqrt(mu)

    # Universal variables will differ between different texts and works in the literature.
    # c0, c1, c2, c3, c4, c5 are expected to follow the Battin formalism (adopted by both
    # Vallado and Curtis in their books). The M matrix is proposed by Shepperd 1985 and follows
    # the Goodyear formalism. Conversions between the two formalisms can be derived from Table 1 in
    # Everhart & Pitkin 1982.
    w = chi / sqrt_mu
    alpha_alt = - mu * alpha
    U0 = (1 - alpha * chi**2) * c2
    U1 = (chi - alpha * chi**3) * c3 / sqrt_mu
    U2 = chi**2 * c2 / mu
    U3 = chi**3 * c3 / mu**(3/2)
    U4 = chi**4 * c2 / mu**(2)
    U5 = chi**5 * c3 / mu**(5/2)

    F = f_dot
    G = g_dot

    # Equations 18 and 19 in Shepperd 1985
    U = (U2 * U3 + w * U4 - 3 * U5) / 3
    W = g * U2 + 3 * mu * U

    # Calculate elements of the M matrix
    m11 = (U0 / (r1_mag * r0_mag) + 1 / r0_mag**2 + 1 / r1_mag**2) * F - (mu**2 * W) / (r1_mag * r0_mag)**3
    m12 = F * U1 / r1_mag + (G - 1) / r1_mag**2
    m13 = (G - 1) * U1 / r1_mag - (mu * W) / r1_mag**3
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

@jit("f8[:,:](f8[:], f8, f8, i8, f8)", nopython=True, cache=True)
def calcStateTransitionMatrix(orbit, dt, mu=0.0002959122082855911, max_iter=100, tol=1e-15):

    r0 = orbit[0:3]
    v0 = orbit[3:6]

    r0_mag = np.linalg.norm(r0)
    v0_mag = np.linalg.norm(v0)

    # Calculate the inverse semi-major axis
    # Note: the definition of alpha will change between different works in the literature.
    #   Here alpha is defined as 1 / a where a is the semi-major axis of the orbit
    alpha = -v0_mag**2 / mu + 2 / r0_mag

    lagrange_coeffs, stumpff_coeffs, chi = calcLagrangeCoeffs(
        r0,
        v0,
        dt,
        mu=mu,
        max_iter=max_iter,
        tol=tol
    )
    f, g, f_dot, g_dot = lagrange_coeffs
    r1, v1 = applyLagrangeCoeffs(r0, v0, *lagrange_coeffs)
    M = calcMMatrix(r0, r1, lagrange_coeffs, stumpff_coeffs, chi, alpha, mu=mu)

    I = np.identity(3)
    state_0 = np.zeros((3, 2))
    state_0[:, 0] = r0
    state_0[:, 1] = v0
    state_1 = np.zeros((3, 2))
    state_1[:, 0] = r1
    state_1[:, 1] = v1
    #state_0 = np.vstack((r0, v0))
    #state_1 = np.vstack((r1, v1))

    phi11 = f * I + state_1 @ (M[1:3, 0:2] @ state_0.T)
    phi12 = g * I + state_1 @ (M[1:3, 1:3] @ state_0.T)
    phi21 = f_dot * I - state_1 @ (M[0:2, 0:2] @ state_0.T)
    phi22 = g_dot * I - state_1 @ (M[0:2, 1:3] @ state_0.T)

    #phi = np.block([
    #    [phi11, phi12],
    #    [phi21, phi22]
    #])
    phi = np.zeros((6, 6))
    phi[0:3, 0:3] = phi11
    phi[0:3, 3:6] = phi12
    phi[3:6, 0:3] = phi21
    phi[3:6, 3:6] = phi22
    return phi