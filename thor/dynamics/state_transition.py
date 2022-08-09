import jax.numpy as jnp
from jax import (
    config,
    jit
)

config.update("jax_enable_x64", True)

from ..constants import Constants as c
from .stumpff import STUMPFF_TYPES
from .lagrange import (
    LAGRANGE_TYPES,
    calc_lagrange_coefficients,
    apply_lagrange_coefficients
)

__all__ = [
    "_calc_M_matrix",
    "calc_state_transition_matrix"
]

MU = c.MU

@jit
def _calc_M_matrix(
        r0: jnp.ndarray,
        r1: jnp.ndarray,
        lagrange_coeffs: LAGRANGE_TYPES,
        stumpff_coeffs: STUMPFF_TYPES,
        chi: float,
        alpha: float,
        mu: float = MU
    ) -> jnp.ndarray:
    """
    Calculate the M matrix proposed by S. W. Shepperd in 1985.

    Parameters
    ----------
    r0 : `~jax.numpy.ndarray` (3)
        Cartesian position vector at t0 in au.
    r1 : `~jax.numpy.ndarray` (3)
        Cartesian position vector at t1 in au.
    langrange_coeffs : (float x 4)
        The langrage coefficients f, g, f_dot, and g_dot from t0
        to t1.
    stumpff_coeffs : (float x 6)
        The Stumpff functions / coefficients evaluated with at alpha * chi**2.
    chi : float
        Universal anomaly in units of au^(1/2).
    alpha : float
        Inverse of the semi-major axis defined as 1/a.
    mu : float
        Gravitational parameter (GM) of the attracting body in units of
        AU**3 / d**2.

    Returns
    -------
    M : `~jax.numpy.ndarray` (3, 3)
        The M matrix proposed by S. W. Shepperd in 1985 as a convenient way to
        package the quantities required to calculate the 2-body state transition
        matrix.

    References
    ----------
    [1] Goodyear, W. H. (1965). Completely general closed-form solution
        for coordinates and partial derivative of the two-body problem.
        The Astronomical Journal, 70, 189. https://doi.org/10.1086/109713
    [2] Goodyear, W. H. (1965). Errata: Completely general closed-form solution
        for coordinates and partial derivative of the two-body problem.
        The Astronomical Journal, 70, 446. https://doi.org/10.1086/109760
    [3] Everhart, E., & Pitkin, E. T. (1983). Universal variables in the two‐body problem.
        American Journal of Physics, 51(8), 712–717. https://doi.org/10.1119/1.13152
    [4] Shepperd, S. W. (1985). Universal Keplerian state transition matrix.
        Celestial Mechanics, 35(2), 129–144. https://doi.org/10.1007/BF01227666

    TODO
    ----
        - Check formalism conversions and constant definitions.
    """
    # Extract relevant quanitities and calculate vector magnitudes
    c0, c1, c2, c3, c4, c5 = stumpff_coeffs
    f, g, f_dot, g_dot = lagrange_coeffs
    r0_mag = jnp.linalg.norm(r0)
    r1_mag = jnp.linalg.norm(r1)
    sqrt_mu = jnp.sqrt(mu)

    # Universal variables will differ between different texts and works in the literature.
    # c0, c1, c2, c3, c4, c5 are expected to follow the Battin formalism (adopted by both
    # Vallado and Curtis in their books). The M matrix is proposed by Shepperd 1985 [4] and follows
    # the Goodyear formalism [1][2]. Conversions between the two formalisms can be derived from Table 1 in
    # Everhart & Pitkin 1983 [3].
    # TODO : These conversions were not robustly tested and needed further investigation
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

    # Equations 18 and 19 in Shepperd 1985 [4]
    U = (U2 * U3 + w * U4 - 3 * U5) / 3
    W = g * U2 + 3 * mu * U

    # Calculate elements of the M matrix
    # See equation A.41 in Shepperd 1985 [4]
    m11 = (U0 / (r1_mag * r0_mag) + 1 / r0_mag**2 + 1 / r1_mag**2) * F - (mu**2 * W) / (r1_mag * r0_mag)**3
    m12 = F * U1 / r1_mag + (G - 1) / r1_mag**2
    m13 = (G - 1) * U1 / r1_mag - (mu * W) / r1_mag**3
    m21 = -F * U1 / r0_mag - (f - 1) / r0_mag**2
    m22 = -F * U2
    m23 = -(G - 1) * U2
    m31 = (f - 1) * U1 / r0_mag - (mu * W) / r0_mag**3
    m32 = (f - 1) * U2
    m33 = g * U2 - W

    # Combine elements into a 3 x 3 matrix
    M = jnp.array([
        [m11, m12, m13],
        [m21, m22, m23],
        [m31, m32, m33],
    ])
    return M

@jit
def calc_state_transition_matrix(
        orbit: jnp.ndarray,
        dt: float,
        mu: float = MU,
        max_iter: int = 1000,
        tol: float = 1e-15
    ) -> jnp.ndarray:
    """
    Calculate the state transition matrix for a given change in epoch. The state transition matrix
    maps deviations from a state at an epoch t0 to a different epoch t1 (dt = t1 - t0).

    Parameters
    ----------
    orbit : `~jax.numpy.ndarray` (6)
        Cartesian state vector in units of au and au per day.
    dt : float
        Time difference (dt = t1 - t0) at which to calculate the state transition matrix.
    mu : float
        Gravitational parameter (GM) of the attracting body in units of
        AU**3 / d**2.
    max_iter : int
        Maximum number of iterations over which to converge. If number of iterations is
        exceeded, will return the value of the universal anomaly at the last iteration.
    tol : float
        Numerical tolerance to which to compute chi using the Newtown-Raphson
        method.

    Returns
    -------
    Phi : `~jax.numpy.ndarray` (6, 6)
        The state transition matrix.

    References
    ----------
    [1] Shepperd, S. W. (1985). Universal Keplerian state transition matrix.
        Celestial Mechanics, 35(2), 129–144. https://doi.org/10.1007/BF01227666
    """
    r0 = orbit[0:3]
    v0 = orbit[3:6]

    r0_mag = jnp.linalg.norm(r0)
    v0_mag = jnp.linalg.norm(v0)

    # Calculate the inverse semi-major axis
    # Note: the definition of alpha will change between different works in the literature.
    #   Here alpha is defined as 1 / a where a is the semi-major axis of the orbit
    alpha = -v0_mag**2 / mu + 2 / r0_mag

    lagrange_coeffs, stumpff_coeffs, chi = calc_lagrange_coefficients(
        r0,
        v0,
        dt,
        mu=mu,
        max_iter=max_iter,
        tol=tol
    )
    f, g, f_dot, g_dot = lagrange_coeffs
    r1, v1 = apply_lagrange_coefficients(r0, v0, *lagrange_coeffs)
    M = _calc_M_matrix(r0, r1, lagrange_coeffs, stumpff_coeffs, chi, alpha, mu=mu)

    # Construct the 3 x 2 state matrices with the position vector
    # in the first column and the velocity vector in the second column
    # See equations A.42 - A.45 in Shepperd 1985 [1]
    state_0 = jnp.zeros((3, 2))
    state_0 = state_0.at[:, 0].set(r0)
    state_0 = state_0.at[:, 1].set(v0)
    state_1 = jnp.zeros((3, 2))
    state_1 = state_1.at[:, 0].set(r1)
    state_1 = state_1.at[:, 1].set(v1)

    # Construct the 4 3 x 3 submatrices that can form the
    # the 6 x 6 state transition matrix.
    # See equations A.42 - A.46 in Shepperd 1985 [1]
    I = jnp.identity(3)
    phi = jnp.zeros((6, 6))
    phi11 = f * I + state_1 @ (M[1:3, 0:2] @ state_0.T)
    phi12 = g * I + state_1 @ (M[1:3, 1:3] @ state_0.T)
    phi21 = f_dot * I - state_1 @ (M[0:2, 0:2] @ state_0.T)
    phi22 = g_dot * I - state_1 @ (M[0:2, 1:3] @ state_0.T)

    phi = phi.at[0:3, 0:3].set(phi11)
    phi = phi.at[0:3, 3:6].set(phi12)
    phi = phi.at[3:6, 0:3].set(phi21)
    phi = phi.at[3:6, 3:6].set(phi22)

    return phi