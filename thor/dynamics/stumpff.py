import numpy as np
import jax.numpy as jnp
from jax import (
    config,
    jit,
    lax
)
from typing import Tuple

config.update("jax_enable_x64", True)

__all__ = [
    "calc_stumpff"
]

STUMPFF_TYPES = Tuple[jnp.float64, jnp.float64, jnp.float64, jnp.float64, jnp.float64, jnp.float64]

@jit
def _positive_psi(psi: np.float64) -> STUMPFF_TYPES:
    # Equation 6.9.15 in Danby (1992) [1]
    sqrt_psi = jnp.sqrt(psi)
    c0 = jnp.cos(sqrt_psi)
    c1 = jnp.sin(sqrt_psi) / sqrt_psi

    # Equation 6.9.16 in Danby (1992) [1]
    # states the recursion relation for higher
    # order Stumpff functions
    c2 = (1. - c0) / psi
    c3 = (1. - c1) / psi
    c4 = (1/2. - c2) / psi
    c5 = (1/6. - c3) / psi

    return c0, c1, c2, c3, c4, c5

@jit
def _negative_psi(psi: np.float64) -> STUMPFF_TYPES:
    # Equation 6.9.15 in Danby (1992) [1]
    sqrt_npsi = jnp.sqrt(-psi)
    c0 = jnp.cosh(sqrt_npsi)
    c1 = jnp.sinh(sqrt_npsi) / sqrt_npsi

    # Equation 6.9.16 in Danby (1992) [1]
    # states the recursion relation for higher
    # order Stumpff functions
    c2 = (1. - c0) / psi
    c3 = (1. - c1) / psi
    c4 = (1/2. - c2) / psi
    c5 = (1/6. - c3) / psi

    return c0, c1, c2, c3, c4, c5

@jit
def _null_psi(psi: np.float64) -> STUMPFF_TYPES:
    # Equation 6.9.14 in Danby (1992) [1]
    c0 = 1.
    c1 = 1.
    c2 = 1/2.
    c3 = 1/6.
    c4 = 1/24.
    c5 = 1/120.

    return c0, c1, c2, c3, c4, c5


@jit
def calc_stumpff(psi: np.float64) -> STUMPFF_TYPES:
    """
    Calculate the first 6 Stumpff functions for variable psi.

    .. math::

        \Psi = \alpha \chi^2

        \frac{d\chi}{dt} = \frac{1}{r}

        \alpha = \frac{\mu}{a}

        c_0(\Psi) = \begin{cases}
            \cos{\sqrt{\Psi}} & \text{ if } \Psi > 0 \\
            \cosh{\sqrt{-\Psi}} & \text{ if } \Psi < 0 \\
            1 & \text{ if } \Psi= 0
        \end{cases}

        c_1(\Psi) = \begin{cases}
            \frac{\sin{\sqrt{\Psi}}{\sqrt{\Psi}} & \text{ if } \Psi > 0 \\
            \frac{\sinh{\sqrt{-\Psi}}{\sqrt{-\Psi}} & \text{ if } \Psi < 0 \\
            1 & \text{ if } \Psi= 0
        \end{cases}

        \Psi c_{n+2} = \frac{1}{k!} - c_n(\Psi)

    Parameters
    ----------
    psi : float
        Dimensionless parameter at which to evaluate the Stumpff functions (equivalent to alpha * chi**2).

    Returns
    -------
    c0, c1, c2, c3, c4, c5 : 6 x float
        First six Stumpff functions.

    References
    ----------
    [1] Danby, J. M. A. (1992). Fundamentals of Celestial Mechanics. 2nd ed.,
        William-Bell, Inc. ISBN-13: 978-0943396200
        Notes: of particular interest is Danby's fantastic chapter on universal variables (6.9)
    """
    c0, c1, c2, c3, c4, c5 = lax.cond(
        psi > 0.0,
        _positive_psi,
        lambda psi: lax.cond(
            psi < 0.0,
            _negative_psi,
            _null_psi,
            psi
        ),
        psi
    )
    return c0, c1, c2, c3, c4, c5