from ..utils import _checkTime
from .ephemeris import getPerturberState

__all__ = [
    "shiftOrbitsOrigin"
]


def shiftOrbitsOrigin(orbits, t0, origin_in="heliocenter", origin_out="barycenter"):
    """
    Shift the origin of the given orbits. Orbits should be expressed in 
    ecliptic J2000 cartesian coordinates.
    
    Parameters
    ----------
    orbits : `~numpy.ndarray` (N, 6)
        Orbits to shift to a different coordinate frame.
    t0 : `~astropy.time.core.Time` (N)
        Epoch at which orbits are defined.
    origin_in : {'heliocenter', 'barycenter'}
        Origin of the input orbits. 
    origin_out : {'heliocenter', 'barycenter'}
        Desired origin of the output orbits.
    
    Returns
    -------
    orbits_shifted : `~numpy.ndarray` (N, 6)
        Orbits shifted to the desired output origin.
    """
    _checkTime(t0, "t0")
    
    orbits_shifted = orbits.copy()
    bary_to_helio = getPerturberState("sun", t0, origin="barycenter")
    
    if origin_in == origin_out:
        return orbits_shifted
    elif origin_in == "heliocenter" and origin_out == "barycenter":
        orbits_shifted -= bary_to_helio
    elif origin_in == "barycenter" and origin_out == "heliocenter":
        orbits_shifted += bary_to_helio
    else:
        err = (
            "origin_in and origin_out should be one of {'heliocenter', 'barycenter'}"
        )
        raise ValueError(err)
    
    return orbits_shifted