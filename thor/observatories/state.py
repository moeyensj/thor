import numpy as np
import pandas as pd 
import spiceypy as sp

from ..constants import Constants as c
from ..utils import _checkTime
from ..utils import setupSPICE
from ..utils import readMPCObservatoryCodes
from ..orbits import getPerturberState

__all__ = ["getObserverState"]

R_EARTH = c.R_EARTH
OMEGA_EARTH = 2 * np.pi / 0.997269675925926 

def getObserverState(observatory_codes, observation_times, frame="ecliptic", origin="heliocenter"):
    """
    Find the heliocentric or barycentric ecliptic or equatorial J2000 state vectors for different observers or observatories at 
    the desired epochs. Currently only supports ground-based observers.
    
    The Earth body-fixed frame used for calculations is the standard ITRF93, which takes into account:
        - precession (IAU-1976)
        - nutation (IAU-1980 with IERS corrections)
        - polar motion
    This frame is retrieved through SPICE. 
    
    Parameters
    ----------
    observatory_codes : list or `~numpy.ndarray`
        MPC observatory codes. 
    observation_times : `~astropy.time.core.Time`
        Epochs for which to find the observatory locations.
    frame : {'equatorial', 'ecliptic'}
        Return observer state in the equatorial or ecliptic J2000 frames. 
    origin : {'barycenter', 'heliocenter'}
        Return observer state with heliocentric or barycentric origin.
        
    Returns
    -------
    `~pandas.DataFrame`
        Pandas DataFrame with a column of observatory codes, MJDs (in UTC), and the J2000 
        postion vector in three columns (obs_x, obs_y, obs_z) and J2000
        velocity in three columns (obs_vx, obs_vy, obs_vg). 
    """
    if type(observatory_codes) not in [list, np.ndarray]:
        err = (
            "observatory_codes should be a list or `~numpy.ndarray`."
        )
        raise TypeError(err)

    if frame == "ecliptic":
        frame_spice = "ECLIPJ2000"
    elif frame == "equatorial":
        frame_spice = "J2000"
    else:
        err = (
            "frame should be one of {'equatorial', 'ecliptic'}"
        )
        raise ValueError(err)

    setupSPICE()

    # Check that times is an astropy time object
    _checkTime(observation_times, "observation_times")

    observatories = readMPCObservatoryCodes()
    positions = {}
    
    for code in observatory_codes:
        if np.any(observatories[observatories.index == code][["longitude_deg", "cos", "sin"]].isna().values == True):
            err = (
                "{} is missing information on Earth-based geodetic coordinates. The MPC Obs Code\n"
                "file may be missing this information or the observer is a space-based observatory.\n"
                "Space observatories are currently not supported.\n"
            )
            raise ValueError(err.format(code))
                
        
        # Get observer location on Earth
        longitude = observatories[observatories.index == code]["longitude_deg"].values[0]
        sin_phi = observatories[observatories.index == code]["sin"].values[0]
        cos_phi = observatories[observatories.index == code]["cos"].values[0]
        sin_longitude = np.sin(np.radians(longitude))
        cos_longitude = np.cos(np.radians(longitude))
        
        # Calculate pointing vector from geocenter to observatory
        o_hat_ITRF93 = np.array([
            cos_longitude * cos_phi,
            sin_longitude * cos_phi,
            sin_phi
        ])
        
        # Multiply pointing vector with Earth radius to get actual vector
        o_vec_ITRF93 = np.dot(R_EARTH, o_hat_ITRF93)

        # Grab earth state vector
        state = getPerturberState("earth", observation_times, frame=frame, origin=origin)
        
        # Convert MJD epochs in TDB to ET in TDB
        epochs_utc = observation_times.tdb
        epochs_et = np.array([sp.str2et('JD {:.16f} TDB'.format(i)) for i in epochs_utc.jd])
        
        # Grab rotaton matrices from ITRF93 to ecliptic J2000
        # The ITRF93 high accuracy Earth rotation model takes into account:
        # Precession:  1976 IAU model from Lieske.
        # Nutation:  1980 IAU model, with IERS corrections due to Herring et al.
        # True sidereal time using accurate values of TAI-UT1
        # Polar motion
        rotation_matrices = np.array([sp.pxform('ITRF93', frame_spice, i) for i in epochs_et])
        
        # Add o_vec + r_geo to get r_obs
        r_obs = np.array([rg + rm @ o_vec_ITRF93 for rg, rm in zip(state[:, :3], rotation_matrices)])

        # Calculate velocity
        v_obs = np.array([vg + rm @ (- OMEGA_EARTH * R_EARTH * np.cross(o_hat_ITRF93, np.array([0, 0, 1]))) for vg, rm in zip(state[:, 3:], rotation_matrices)])

        # Create table of mjds and positions
        table = np.empty((len(observation_times), 7))
        table[:, 0] = observation_times.utc.mjd
        table[:, 1:4] = r_obs
        table[:, 4:] = v_obs
        
        # Add to dictionary
        positions[code] = table
    
    # Process dictionary into a clean pandas DataFrame
    dfs = []
    for code, table in positions.items():
        dfi = pd.DataFrame(table, columns=["mjd_utc", "obs_x", "obs_y", "obs_z", "obs_vx", "obs_vy", "obs_vz"])
        dfi["observatory_code"] = [code for i in range(len(dfi))]
        dfs.append(dfi)
                           
    df = pd.concat(dfs)
    df = df[["observatory_code", "mjd_utc", "obs_x", "obs_y", "obs_z", "obs_vx", "obs_vy", "obs_vz"]]
    df.reset_index(inplace=True, drop=True)
    return df