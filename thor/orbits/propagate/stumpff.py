import numpy as np
from numba import jit

__all__ = [
    "calcC2C3"
]

@jit(nopython=True)
def calcC2C3(psi):
    """
    Calculate the second and third Stumpff functions. 
    
    .. math::
        
        c_2(\Psi) = \begin{cases}
            \frac{1 - \cos{\sqrt{\Psi}}}{\Psi} & \text{ if } \Psi > 0 \\ 
            \frac{1 - \cosh{\sqrt{-\Psi}}}{\Psi} & \text{ if } \Psi < 0 \\ 
            \frac{1}{2} & \text{ if } \Psi= 0
        \end{cases}
        
        c_3(\Psi) = \begin{cases}
            \frac{\sqrt{\Psi} - \sin{\sqrt{\Psi}}}{\sqrt{\Psi^3}} & \text{ if } \Psi > 0 \\ 
            \frac{\sinh{\sqrt{-\Psi}} - \sqrt{-\Psi}}{\sqrt{(-\Psi)^3}} & \text{ if } \Psi < 0 \\ 
            \frac{1}{6} & \text{ if } \Psi= 0
        \end{cases}
        
    For more details on theory see Chapter 2 in David A. Vallado's "Fundamentals of Astrodynamics
    and Applications" or Chapter 3 in Howard Curtis' "Orbital Mechanics for Engineering Students".
    
    Parameters
    ----------
    psi : float
        Dimensionless parameter at which to evaluate the Stumpff functions (equivalent to alpha * chi**2). 
        
    Returns
    -------
    c2, c3 : float, float
        Second and third Stumpff functions.
    """
    if psi > 0.0:
        c2 = (1 - np.cos(np.sqrt(psi))) / psi
        c3 = (np.sqrt(psi) - np.sin(np.sqrt(psi))) / np.sqrt(psi)**3
    elif psi < 0.0:
        c2 = (np.cosh(np.sqrt(-psi)) - 1) / (-psi)
        c3 = (np.sinh(np.sqrt(-psi)) - np.sqrt(-psi)) / np.sqrt(-psi)**3
    else:
        c2 = 1/2.
        c3 = 1/6.
    
    return c2, c3