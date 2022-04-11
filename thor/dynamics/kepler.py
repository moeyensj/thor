import jax.numpy as jnp
from jax import (
    config,
    jit
)
from jax.experimental import loops

__all__ = [
    "solve_kepler"
]

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

@jit
def solve_kepler(e, M, max_iter=100, tol=1e-15):
    """
    Solve Kepler's equation for true anomaly (nu) given eccentricity
    and mean anomaly using Newton-Raphson.

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
    """
    with loops.Scope() as s:

        s.arr = jnp.zeros(1, dtype=jnp.float64)
        for _ in s.cond_range(e < 1.0):

            with loops.Scope() as ss:
                ratio = 1e10
                e_init = M
                ss.arr = jnp.array([e_init, ratio], dtype=jnp.float64)
                ss.idx = 0
                for _ in ss.while_range(lambda : (ss.idx < max_iter) & (ss.arr[1] > tol)):
                    f = ss.arr[0] - e * jnp.sin(ss.arr[0]) - M
                    fp = 1 - e * jnp.cos(ss.arr[0])
                    ratio = f / fp
                    ss.arr = ss.arr.at[0].set(ss.arr[0]-ratio)
                    ss.arr = ss.arr.at[1].set(jnp.abs(ratio))
                    ss.idx += 1

                E = ss.arr[0]
                nu_E = 2 * jnp.arctan2(jnp.sqrt(1 + e) * jnp.sin(E/2), jnp.sqrt(1 - e) * jnp.cos(E/2))

        for _ in s.cond_range(e > 1.0):

            with loops.Scope() as ss:
                ratio = 1e10
                H_init = M / (e - 1)
                ss.arr = jnp.array([H_init, ratio], dtype=jnp.float64)
                ss.idx = 0
                for _ in ss.while_range(lambda : (ss.idx < max_iter) & (ss.arr[1] > tol)):
                    f = M - e * jnp.sinh(ss.arr[0]) + ss.arr[0]
                    fp =  e * jnp.cosh(ss.arr[0]) - 1
                    ratio = f / fp
                    ss.arr = ss.arr.at[0].set(ss.arr[0]+ratio)
                    ss.arr = ss.arr.at[1].set(jnp.abs(ratio))
                    ss.idx += 1

                H = ss.arr[0]
                nu_H = 2 * jnp.arctan(jnp.sqrt(e + 1) * jnp.sinh(H / 2) / (jnp.sqrt(e - 1) * jnp.cosh(H / 2)))

        nu = jnp.where(
            e < 1.0,
            nu_E,
            jnp.where(
                e > 1.0,
                nu_H,
                jnp.nan
            )
        )
        anomaly = jnp.where(
            e < 1.0,
            E,
            jnp.where(
                e > 1.0,
                H,
                jnp.nan
            )
        )

        s.arr = s.arr.at[0].set(nu)

    return s.arr[0]