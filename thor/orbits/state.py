import numpy as np
import spiceypy as sp

from ..constants import Constants as c
from ..utils import setupSPICE
from ..utils import _checkTime

KM_P_AU = c.KM_P_AU
S_P_DAY = c.S_P_DAY

NAIF_MAPPING = {
    "solar system barycenter" : 0,
    "sun" : 10,
    "mercury" : 199,
    "venus" : 299,
    "earth" : 399,
    "mars" : 499,
    "jupiter" : 599,
    "saturn" : 699,
    "uranus" : 799,
    "neptune" : 899,
}

__all__ = ["getPerturberState"]

def getPerturberState(body_name, times, frame="ecliptic", origin="heliocenter"):
    """
    Query the JPL ephemeris files loaded in SPICE for the state vectors of desired perturbers.
    
    Major bodies and dynamical centers available: 
        'solar system barycenter', 'sun',
        'mercury', 'venus', 'earth', 
        'mars', 'jupiter', 'saturn', 
        'uranus', 'neptune'
    
    Parameters
    ----------
    body_name : str
        Name of major body.
    times : `~astropy.time.core.Time` (N)
        Times at which to get state vectors.
    frame : {'equatorial', 'ecliptic'}
        Return perturber state in the equatorial or ecliptic J2000 frames. 
    origin : {'barycenter', 'heliocenter'}
        Return perturber state with heliocentric or barycentric origin.
    
    Returns
    -------
    states : `~numpy.ndarray` (N, 6)
        Heliocentric ecliptic J2000 state vector with postion in AU 
        and velocity in AU per day.
    """
    if origin == "barycenter":
        center = 0 # Solar System Barycenter
    elif origin == "heliocenter":
        center = 10 # Heliocenter
    else: 
        err = ("origin should be one of 'heliocenter' or 'barycenter'")
        raise ValueError(err)

    if frame == "ecliptic":
        frame_spice = "ECLIPJ2000"
    elif frame == "equatorial":
        frame_spice = "J2000"
    else:
        err = (
            "frame should be one of {'equatorial', 'ecliptic'}"
        )
        raise ValueError(err)
    
    # Make sure SPICE is ready to roll
    setupSPICE(verbose=False)

    # Check that times is an astropy time object
    _checkTime(times, "times")

    # Convert MJD epochs in UTC to ET in TDB
    epochs_utc = times.utc
    epochs_et = np.array([sp.str2et('JD {:.16f} UTC'.format(i)) for i in epochs_utc.jd])
    
    # Get position of the body in heliocentric ecliptic J2000 coordinates 
    states = []
    for epoch in epochs_et:
        state, lt = sp.spkez(
            NAIF_MAPPING[body_name.lower()], 
            epoch, 
            frame_spice,
            'NONE',
            center
        )
        states.append(state)
    states = np.vstack(states)
    
    # Convert to AU and AU per day
    states = states / KM_P_AU
    states[:, 3:] = states[:, 3:] * S_P_DAY
    return states