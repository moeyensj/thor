import jax.numpy as jnp
from jax import (
    config,
    jit,
    lax
)

__all__ = [
    "calc_mean_anomaly",
    "solve_kepler"
]

config.update("jax_enable_x64", True)

@jit
def calc_mean_anomaly(
        nu: float,
        e: float
    ) -> float:
    """
    Calculate the mean anomaly given true anomaly in radians
    and eccentricity.

    Parameters
    ----------
    nu : float
        True anomaly in radians.
    e : float
        Eccentricity.

    Returns
    -------
    M : float
        Mean anomaly in radians.
    """
    E, M = lax.cond(
        e < 1.0,
        _calc_elliptical_anomalies,
        lambda nu, e: lax.cond(
            e > 1.0,
            _calc_hyperbolic_anomalies,
            _calc_parabolic_anomalies,
            nu,
            e
        ),
        nu,
        e
    )

    return M

@jit
def _calc_elliptical_anomalies(
        nu: float,
        e: float
    ) -> float:
    E = jnp.arctan2(jnp.sqrt(1 - e**2) * jnp.sin(nu), e + jnp.cos(nu))
    M = E - e * jnp.sin(E)
    M = jnp.where(M < 0.0, M + 2*jnp.pi, M)
    return E, M

@jit
def _calc_hyperbolic_anomalies(
        nu: float,
        e: float
    ) -> float:
    H = jnp.arcsinh(jnp.sin(nu) * jnp.sqrt(e**2 - 1) / (1 + e * jnp.cos(nu)))
    M = e * jnp.sinh(H) - H
    M = jnp.where(M < 0.0, M + 2*jnp.pi, M)
    return H, M

@jit
def _calc_parabolic_anomalies(
        nu: float,
        e: float
    ) -> float:
    D = jnp.arctan(nu / 2)
    M = D + D**3 / 3
    M = jnp.where(M < 0.0, M + 2*jnp.pi, M)
    return D, M

@jit
def solve_kepler(
        e: float,
        M: float,
        max_iter: int = 100,
        tol: float = 1e-15
    ) -> float:
    """
    Solve Kepler's equation for true anomaly (nu) given eccentricity
    and mean anomaly using Newton-Raphson. This is only valid for orbits
    with eccentricity < 1.0 and eccentricity > 1.0. For parabolic orbits see
    `~thor.dynamics.barker.solve_barker`.

    Parameters
    ----------
    e : float
        Eccentricity
    M : float
        Mean anomaly in radians.
    max_iter : int, optional
        Maximum number of iterations over which to converge. If number of iterations is
        exceeded, will use the value of the relevant anomaly at the last iteration.
    tol : float, optional
        Numerical tolerance to which to compute anomalies using the Newtown-Raphson
        method.

    Returns
    -------
    nu : float
        True anomaly in radians.

    References
    ----------
    [1] Curtis, H. D. (2014). Orbital Mechanics for Engineering Students. 3rd ed.,
        Elsevier Ltd. ISBN-13: 978-0080977478
    """
    ratio = 1e15
    iterations = 0
    E_init = jnp.where(e < 1.0, M, M)

    p = [E_init, e, M, ratio, iterations]

    @jit
    def _elliptical_newton_raphson(p):

        E = p[0]
        e = p[1]
        M = p[2]
        iterations = p[4]

        # Newton-Raphson
        # Equation 3.17 in Curtis (2014) [1]
        f = E - e * jnp.sin(E) - M
        fp = 1 - e * jnp.cos(E)

        ratio = f / fp
        E -= ratio
        iterations += 1

        p[0] = E
        p[1] = e
        p[2] = M
        p[3] = ratio
        p[4] = iterations
        return p

    @jit
    def _hyperbolic_newton_raphson(p):

        F = p[0]
        e = p[1]
        M = p[2]
        iterations = p[4]

        # Newton-Raphson
        # Equation 3.45 in Curtis (2014) [1]
        f = e * jnp.sinh(F) - F - M
        fp = e * jnp.cosh(F) * F - 1

        ratio = f / fp
        F -= ratio
        iterations += 1

        p[0] = F
        p[1] = e
        p[2] = M
        p[3] = ratio
        p[4] = iterations
        return p

    # Define while loop condition function
    @jit
    def _while_condition(p):
        ratio = p[-2]
        iterations = p[-1]
        return (jnp.abs(ratio) > tol) & (iterations <= max_iter)

    # Calculate parameters, if e < 1.0 then the orbit is elliptical
    # if e > 1.0 then the orbit is hyperbolic
    p = lax.cond(
        e < 1.0,
        lambda p: lax.while_loop(
            _while_condition,
            _elliptical_newton_raphson,
            p
        ),
        lambda p: lax.cond(
            e > 1.0,
            lambda p: lax.while_loop(
                _while_condition,
                _hyperbolic_newton_raphson,
                p,
            ),
            lambda p: p,
            p
        ),
        p
    )

    nu = lax.cond(
        e < 1.0,
        lambda E, e: 2 * jnp.arctan2(jnp.sqrt(1 + e) * jnp.sin(E/2), jnp.sqrt(1 - e) * jnp.cos(E/2)),
        lambda E, e: lax.cond(
            e > 1.0,
            lambda H, e: 2 * jnp.arctan(jnp.sqrt(e + 1) * jnp.sinh(H / 2) / (jnp.sqrt(e - 1) * jnp.cosh(H / 2))),
            lambda H, e: jnp.nan, # TODO: replace with Barker's equation
            p[0],
            p[1]
        ),
        p[0],
        p[1]
    )

    return nu