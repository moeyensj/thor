import numpy as np
from adam_core.constants import Constants as c

__all__ = ["calcGibbs"]

MU = c.MU


def calcGibbs(r1, r2, r3):
    """
    Calculates the velocity vector at the location of the second position vector (r2) using the
    Gibbs method.

    .. math::
        \vec{D} = \vec{r}_1 \times \vec{r}_2  +  \vec{r}_2 \times \vec{r}_3 +  \vec{r}_3 \times \vec{r}_1

        \vec{N} = r_1 (\vec{r}_2 \times \vec{r}_3) + r_2 (\vec{r}_3 \times \vec{r}_1) + r_3 (\vec{r}_1 \times \vec{r}_2)

        \vec{B} \equiv \vec{D} \times \vec{r}_2

        L_g \equiv \sqrt{\frac{\mu}{ND}}

        \vec{v}_2 = \frac{L_g}{r_2} \vec{B} + L_g \vec{S}

    For more details on theory see Chapter 4 in David A. Vallado's "Fundamentals of Astrodynamics
    and Applications".

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

    Returns
    -------
    v2 : `~numpy.ndarray` (3)
        Velocity of object at position r2 at time t2 in units of AU per day.
    """
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    r3_mag = np.linalg.norm(r3)
    Z12 = np.cross(r1, r2)
    Z23 = np.cross(r2, r3)
    Z31 = np.cross(r3, r1)

    # coplanarity = np.arcsin(np.dot(Z23, r1) / (np.linalg.norm(Z23) * r1_mag))

    N = r1_mag * Z23 + r2_mag * Z31 + r3_mag * Z12
    N_mag = np.linalg.norm(N)
    D = Z12 + Z23 + Z31
    D_mag = np.linalg.norm(D)
    S = (r2_mag - r3_mag) * r1 + (r3_mag - r1_mag) * r2 + (r1_mag - r2_mag) * r3
    # S_mag = np.linalg.norm(S)
    B = np.cross(D, r2)
    Lg = np.sqrt(MU / N_mag / D_mag)
    v2 = Lg / r2_mag * B + Lg * S
    return v2
