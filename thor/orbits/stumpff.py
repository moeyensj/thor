import numpy as np
from numba import jit

__all__ = [
    "calcStumpff"
]

@jit("UniTuple(f8, 6)(f8)", nopython=True)
def calcStumpff(psi):
    """
    Calculate the first 6 Stumpff functions for universal variable psi.

    .. math::

        \Psi = \alpha \chi^2

        \frac{d\chi}{dt} = \frac{\sqrt{\mu}}{r}

        \alpha = \frac{1}{a}

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

    For more details on the universal variable formalism see Chapter 2 in David A. Vallado's "Fundamentals of Astrodynamics
    and Applications" or Chapter 3 in Howard Curtis' "Orbital Mechanics for Engineering Students".

    Parameters
    ----------
    psi : float
        Dimensionless parameter at which to evaluate the Stumpff functions (equivalent to alpha * chi**2).

    Returns
    -------
    c0, c1, c2, c3, c4, c5 : 6 x float
        First six Stumpff functions.
    """
    if psi > 0.0:
        c0 = np.cos(np.sqrt(psi))
        c1 = np.sin(np.sqrt(psi)) / np.sqrt(psi)
        c2 = (1. - c0) / psi
        c3 = (1. - c1) / psi
        c4 = (1/2. - c2) / psi
        c5 = (1/6. - c3) / psi
    elif psi < 0.0:
        c0 = np.cosh(np.sqrt(-psi))
        c1 = np.sinh(np.sqrt(-psi)) / np.sqrt(-psi)
        c2 = (1. - c0) / psi
        c3 = (1. - c1) / psi
        c4 = (1/2. - c2) / psi
        c5 = (1/6. - c3) / psi
    else:
        c0 = 1.
        c1 = 1.
        c2 = 1/2.
        c3 = 1/6.
        c4 = 1/24.
        c5 = 1/120.

    return c0, c1, c2, c3, c4, c5