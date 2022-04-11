import numpy as np
from numba import jit

__all__ = [
    "calc_stumpff"
]

@jit("UniTuple(f8, 6)(f8)", nopython=True, cache=True)
def calc_stumpff(psi):
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
    if psi > 0.0:
        # Equation 6.9.15 in Danby (1992) [1]
        sqrt_psi = np.sqrt(psi)
        c0 = np.cos(sqrt_psi)
        c1 = np.sin(sqrt_psi) / sqrt_psi

        # Equation 6.9.16 in Danby (1992) [1]
        # states the recursion relation for higher
        # order Stumpff functions
        c2 = (1. - c0) / psi
        c3 = (1. - c1) / psi
        c4 = (1/2. - c2) / psi
        c5 = (1/6. - c3) / psi
    elif psi < 0.0:
        # Equation 6.9.15 in Danby (1992) [1]
        sqrt_npsi = np.sqrt(-psi)
        c0 = np.cosh(sqrt_npsi)
        c1 = np.sinh(sqrt_npsi) / sqrt_npsi

        # Equation 6.9.16 in Danby (1992) [1]
        # states the recursion relation for higher
        # order Stumpff functions
        c2 = (1. - c0) / psi
        c3 = (1. - c1) / psi
        c4 = (1/2. - c2) / psi
        c5 = (1/6. - c3) / psi
    else:
        # Equation 6.9.14 in Danby (1992) [1]
        c0 = 1.
        c1 = 1.
        c2 = 1/2.
        c3 = 1/6.
        c4 = 1/24.
        c5 = 1/120.

    return c0, c1, c2, c3, c4, c5