import numpy as np
import pandas as pd 
import spiceypy as sp
from astropy import units as u

from ..constants import Constants as c
from ..utils import _checkTime
from ..utils import setupSPICE
from .codes import readMPCObsCodeFile

__all__ = ["getObserverState"]

R_EARTH = c.R_EARTH
KM_TO_AU = u.km.to(u.AU)

def getObserverState(observatory_codes, observation_times):
    """
    Find the heliocentric ecliptic J2000 position vectors for different observers or observatories at 
    the desired epochs. Currently only supports ground-based observers.
    
    The Earth body-fixed frame used for calculations is the standard ITRF93, which takes into account:
        - precession (IAU-1976)
        - nutation (IAU-1980 with IERS corrections)
        - polar motion
    This frame is retrieved through SPICE. 
    
    TODO: We will want to eventually replace this function with getObserverState so we get the heliocentric
    velocity of the observer at each epoch as well. 
    
    Parameters
    ----------
    observatory_codes : list or `~numpy.ndarray`
        MPC observatory codes. 
    observation_times : `~astropy.time.core.Time`
        Epochs for which to find the observatory locations.
        
    Returns
    -------
    `~pandas.DataFrame`
        Pandas DataFrame with a column of observatory codes, MJDs (in UTC), and the heliocentric 
        ecliptic J2000 postion vector in three columns (obs_x_au, obs_y_au, obs_z_au). 
    """
    setupSPICE(verbose=False)

    # Check that times is an astropy time object
    _checkTime(observation_times, "observation_times")

    observatories = readMPCObsCodeFile()
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
        
        # Convert MJD epochs in UTC to ET in TDB
        epochs_utc = observation_times.utc
        epochs_et = np.array([sp.str2et("JD {:.16f} UTC".format(i)) for i in epochs_utc.jd])
        
        # Get position of the geocenter in ecliptic J2000 coordinates 
        state, lt = sp.spkpos('EARTH', epochs_et, 'ECLIPJ2000', 'NONE', 'SUN')
        
        ### Added spkez
        r_geo = [np.array(i) * KM_TO_AU for i in state]
        
        # Grab rotaton matrices from ITRF93 to ecliptic J2000
        # The ITRF93 high accuracy Earth rotation model takes into account:
        # Precession:  1976 IAU model due to Lieske.
        # Nutation:  1980 IAU model, with IERS corrections due to Herring et al.
        # True sidereal time using accurate values of TAI-UT1
        # Polar motion
        rotation_matrices = np.array([sp.pxform('ITRF93', 'ECLIPJ2000', i) for i in epochs_et])
        
        # Add o_vec + r_geo to get r_obs
        r_obs = np.array([rg + rm @ o_vec_ITRF93 for rg, rm in zip(r_geo, rotation_matrices)])

        # Create table of mjds and positions
        table = np.empty((len(observation_times), 4))
        table[:, 0] = observation_times.utc.mjd
        table[:, 1:] = r_obs
        
        # Add to dictionary
        positions[code] = table
    
    # Process dictionary into a clean pandas DataFrame
    dfs = []
    for code, table in positions.items():
        dfi = pd.DataFrame(table, columns=["mjd_utc", "obs_x", "obs_y", "obs_z"])
        dfi["code"] = [code for i in range(len(dfi))]
        dfs.append(dfi)
                           
    df = pd.concat(dfs)
    df = df[["code", "mjd_utc", "obs_x", "obs_y", "obs_z"]]
    df.reset_index(inplace=True, drop=True)
    return df