import os
import numpy as np
import pandas as pd
import pyoorb as oo

from ..config import Config

__all__ = ["propagateOrbits"]

def propagateOrbits(elements,
                    mjdStart,
                    mjds,
                    elementType="cartesian",
                    mjdScale="UTC",
                    H=10,
                    G=0.15,
                    M1=1,
                    K1=1,
                    observatoryCode=Config.oorbObservatoryCode):
    """
    Propagate a test particle using its ecliptic coordinates and velocity to 
    a given epoch and generate an ephemeris. 
    
    Parameters
    ----------
    elements : `~np.ndarray` (N, 6)
        Orbital elements of type defined by elementType.
    mjdStart : float
        Epoch at which ecliptic coordinates and velocity are measured in MJD.
    mjds : '~np.ndarray' (N)
        List of mjds to which to propagate test particle. 
    elementType : {"cartesian", "keplerian", "cometary"}, optional
        The element type of the passed elements. 
        [Default = "cartesian"]
    mjdScale : {"UTC", "UT1", "TT", "TAI"}
        The mjd scale of the passed mjds.
    H : float or `~np.ndarray` (N), optional
        Absolute H magnitude, used if elements are given as cartesian or kelperian. 
        [Default = 10]
    G : float or `~np.ndarray` (N), optional
        HG phase function slope, used if elements are given as cartesian or kelperian. 
        [Default = 0.15]
    M1 : float or `~np.ndarray` (N), optional
    
    K1 : float or `~np.ndarray` (N), optional

    observatoryCode : str, optional
        Observatory from which to measure ephemerides.
        [Default = `~thor.Config.oorbObservatoryCode`]
    """
    if os.environ.get("OORB_DATA") == None:
        os.environ["OORB_DATA"] = os.path.join(os.environ["CONDA_PREFIX"], "share/openorb")
    # Prepare pyoorb
    ephfile = os.path.join(os.getenv('OORB_DATA'), 'de430.dat')
    oo.pyoorb.oorb_init(ephfile)

    if elements.shape == (6,):
        num_orbits = 1
    else:
        num_orbits = elements.shape[0]

    if elementType == "cartesian":
        orbitType = [1 for i in range(num_orbits)]
    elif elementType == "cometary":
        orbitType = [2 for i in range(num_orbits)]
        H = M1
        G = K1
    elif elementType == "keplerian":
        orbitType = [3 for i in range(num_orbits)]
    else:
        raise ValueError("elementType should be one of {'cartesian', 'keplerian', 'cometary'}")

    if mjdScale == "UTC":
        mjdScale = [1 for i in range(num_orbits)]
    elif mjdScale == "UT1": 
        mjdScale = [2 for i in range(num_orbits)]
    elif mjdScale == "TT":
        mjdScale = [3 for i in range(num_orbits)]
    elif mjdScale == "TAI":
        mjdScale = [4 for i in range(num_orbits)]
    else:
        raise ValueError("mjdScale should be one of {'UTC', 'UT1', 'TT', 'TAI'}")

    if type(G) != np.ndarray:
        G = [G for i in range(num_orbits)]
    if type(H) != np.ndarray:
        H = [H for i in range(num_orbits)]
    if type(mjdStart) != np.ndarray:
        mjdStart = [mjdStart for i in range(num_orbits)]

    ids = [i for i in range(num_orbits)]
    # Cater to pyoorb formatting
    if num_orbits > 1:
        orbits = np.array(
            np.array([ids, 
                     *list(elements.T), 
                     orbitType, 
                     mjdStart, 
                     mjdScale, 
                     H, 
                     G]).T, dtype=np.double, order='F')
    else:
        orbits = np.array([[ids[0], 
                     *list(elements.T), 
                     orbitType[0], 
                     mjdStart[0], 
                     mjdScale[0], 
                     H[0], 
                     G[0]]], dtype=np.double, order='F')
        mjdScale = [mjdScale[0] for i in mjds]

    epochs = np.array(list(np.vstack([mjds, mjdScale]).T), dtype=np.double, order='F')
    ephemeris, err = oo.pyoorb.oorb_ephemeris_full(in_orbits=orbits,
                                             in_obscode=observatoryCode,
                                             in_date_ephems=epochs,
                                             in_dynmodel='N', 
                                             )
    columns = ["mjd",
               "RA_deg",
               "Dec_deg",
               "dRAcosDec/dt_deg_p_day",
               "dDec/dt_deg_p_day",
               "PhaseAngle_deg",
               "SolarElon_deg",
               "r_au",
               "Delta_au",
               "VMag",
               "PosAngle_deg",
               "TLon_deg",
               "TLat_deg",
               "TOCLon_deg",
               "TOCLat_deg",
               "HLon_deg",
               "HLat_deg",
               "HOCLon_deg",
               "HOCLat_deg",
               "Alt_deg",
               "SolarAlt_deg",
               "LunarAlt_deg",
               "LunarPhase",
               "LunarElon_deg",
               "HEclObj_X_au",
               "HEclObj_Y_au",
               "HEclObj_Z_au",
               "HEclObj_dX/dt_au_p_day",
               "HEclObj_dY/dt_au_p_day",
               "HEclObj_dZ/dt_au_p_day",
               "HEclObsy_X_au",
               "HEclObsy_Y_au",
               "HEclObsy_Z_au",
               "TrueAnom"]

    eph = pd.DataFrame(np.vstack(ephemeris), 
                       columns=columns)
    eph["orbit_id"] = [i for i in ids for j in mjds]
    return eph[["orbit_id"] + columns]